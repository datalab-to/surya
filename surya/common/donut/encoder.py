import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices,
    meshgrid,
    prune_linear_layer,
)
from transformers.utils import ModelOutput
from transformers import DonutSwinConfig

from surya.common.pretrained import SuryaPreTrainedModel
from surya.common.xla import mark_step

_EXPECTED_OUTPUT_SHAPE = [1, 49, 1024]


@dataclass
# Copied from transformers.models.swin.modeling_swin.SwinEncoderOutput with Swin->DonutSwin
class DonutSwinEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class DonutSwinModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None


# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        num_channels,
    )
    windows = (
        input_feature.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, num_channels)
    )
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(
        -1,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        num_channels,
    )
    windows = (
        windows.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, height, width, num_channels)
    )
    return windows


# Copied from transformers.models.swin.modeling_swin.SwinEmbeddings with Swin->DonutSwin
class DonutSwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = DonutSwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            if use_mask_token
            else None
        )

        self.position_embeddings = None
        self.row_embeddings = None
        self.column_embeddings = None
        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, num_patches + 1, config.embed_dim)
            )

        if hasattr(config, "use_2d_embeddings") and config.use_2d_embeddings:
            self.row_embeddings = nn.Parameter(
                torch.zeros(1, self.patch_grid[0] + 1, config.embed_dim)
            )
            self.column_embeddings = nn.Parameter(
                torch.zeros(1, self.patch_grid[1] + 1, config.embed_dim)
            )

        self.norm = nn.LayerNorm(config.embed_dim)

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor],
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Tuple[torch.Tensor]:
        _, num_channels, height, width = pixel_values.shape
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(
                    embeddings, height, width
                )
            else:
                embeddings = embeddings + self.position_embeddings[:, :seq_len]

        if self.row_embeddings is not None and self.column_embeddings is not None:
            # Repeat the x position embeddings across the y axis like 0, 1, 2, 3, 0, 1, 2, 3, ...
            row_embeddings = self.row_embeddings[
                :, : output_dimensions[0], :
            ].repeat_interleave(output_dimensions[1], dim=1)
            column_embeddings = self.column_embeddings[
                :, : output_dimensions[1], :
            ].repeat(1, output_dimensions[0], 1)

            embeddings = embeddings + row_embeddings + column_embeddings

        return embeddings, output_dimensions


# Copied from transformers.models.swin.modeling_swin.SwinPatchEmbeddings with Swin->DonutSwin
class DonutSwinPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(
        self, pixel_values: Optional[torch.FloatTensor]
    ) -> Tuple[torch.Tensor, Tuple[int]]:
        _, num_channels, height, width = pixel_values.shape
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, output_dimensions


# Copied from transformers.models.swin.modeling_swin.SwinPatchMerging
class DonutSwinPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(
        self,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(
        self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]
    ) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # batch_size height/2 width/2 4*num_channels
        input_feature = torch.cat(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1
        )
        input_feature = input_feature.view(
            batch_size, -1, 4 * num_channels
        )  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


# Copied from transformers.models.swin.modeling_swin.SwinSelfAttention with Swin->DonutSwin
class DonutSwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, num_kv_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_repeats = self.num_attention_heads // self.num_kv_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.kv_head_size = self.num_kv_heads * self.attention_head_size
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads
            )
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.query = nn.Linear(
            self.all_head_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            self.all_head_size, self.kv_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            self.all_head_size, self.kv_head_size, bias=config.qkv_bias
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_kv_for_scores(self, x, repeats):
        new_x_shape = x.size()[:-1] + (self.num_kv_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        x = x.repeat(
            1, 1, repeats, 1
        )  # repeat the values for each key-value head to match query dim
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        # Final is (batch_size, num_attention_heads, seq_len, attention_head_size)
        key_layer = self.transpose_kv_for_scores(
            self.key(hidden_states), self.kv_repeats
        )
        value_layer = self.transpose_kv_for_scores(
            self.value(hidden_states), self.kv_repeats
        )
        query_layer = self.transpose_for_scores(mixed_query_layer)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = (
            relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        )
        relative_position_bias = relative_position_bias.repeat(batch_size, 1, 1, 1)

        if attention_mask is None:
            attention_mask = relative_position_bias
        else:
            mask_shape = attention_mask.shape[0]
            repeat_count = batch_size // mask_shape
            attention_mask = attention_mask.repeat(repeat_count, 1, 1).unsqueeze(1)
            attention_mask = attention_mask + relative_position_bias

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.attention_head_size**-0.5,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, dim, num_channels)

        outputs = (attn_output,)
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput
class DonutSwinSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        return self.dense(hidden_states)


# Copied from transformers.models.swin.modeling_swin.SwinAttention with Swin->DonutSwin
class DonutSwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads, num_kv_heads, window_size):
        super().__init__()
        self.self = DonutSwinSelfAttention(
            config, dim, num_heads, num_kv_heads, window_size
        )
        self.output = DonutSwinSelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, output_attentions
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate
class DonutSwinIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinOutput
class DonutSwinOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden_states)


# Copied from transformers.models.swin.modeling_swin.SwinLayer with Swin->DonutSwin
class DonutSwinLayer(nn.Module):
    def __init__(
        self, config, dim, input_resolution, num_heads, num_kv_heads, shift_size=0
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = DonutSwinAttention(
            config, dim, num_heads, num_kv_heads, window_size=self.window_size
        )
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.intermediate = DonutSwinIntermediate(config, dim)
        self.output = DonutSwinOutput(config, dim)

    def set_shift_and_window_size(self, input_resolution):
        if min(input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = int(0)
            self.window_size = (
                torch.min(torch.tensor(input_resolution))
                if torch.jit.is_tracing()
                else min(input_resolution)
            )

    def get_attn_mask(self, height, width, dtype, device):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype, device=device)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)

        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(
                hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(
            shifted_hidden_states, self.window_size
        )
        hidden_states_windows = hidden_states_windows.view(
            -1, self.window_size * self.window_size, channels
        )
        attn_mask = self.get_attn_mask(
            height_pad,
            width_pad,
            dtype=hidden_states.dtype,
            device=hidden_states_windows.device,
        )

        attention_outputs = self.attention(
            hidden_states_windows,
            attn_mask,
            head_mask,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(
            -1, self.window_size, self.window_size, channels
        )
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(
                shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        hidden_states = shortcut + attention_windows

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        layer_outputs = (
            (layer_output, attention_outputs[1])
            if output_attentions
            else (layer_output,)
        )
        return layer_outputs


# Copied from transformers.models.swin.modeling_swin.SwinStage with Swin->DonutSwin
class DonutSwinStage(nn.Module):
    def __init__(
        self,
        config,
        layer_num,
        dim,
        input_resolution,
        depth,
        num_heads,
        num_kv_heads,
        downsample,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                DonutSwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=nn.LayerNorm
            )
        else:
            self.downsample = None

        self.pointing = False

        self.positional_encoding = None
        if config.use_positional_embeddings:
            self.positional_encoding = self.build_2d_sincos_position_embedding(
                input_resolution[1],
                input_resolution[0],
                embed_dim=dim,
            )

    @staticmethod
    def build_2d_sincos_position_embedding(
        width,
        height,
        embed_dim=256,
        temperature=10000.0,
        device="cpu",
        dtype=torch.float32,
    ):
        grid_w = torch.arange(int(width), dtype=dtype, device=device)
        grid_h = torch.arange(int(height), dtype=dtype, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError(
                "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
            )
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=dtype, device=device) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None, :, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions

        if self.positional_encoding is not None:
            hidden_states = hidden_states + self.positional_encoding.to(
                hidden_states.dtype
            ).to(hidden_states.device)

        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
                always_partition,
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(
                hidden_states_before_downsampling, input_dimensions
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_downsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


# Copied from transformers.models.swin.modeling_swin.SwinEncoder with Swin->DonutSwin
class DonutSwinEncoder(nn.Module):
    def __init__(self, config, grid_size):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        self.layers = nn.ModuleList(
            [
                DonutSwinStage(
                    config=config,
                    layer_num=i_layer,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    num_kv_heads=config.num_kv_heads[i_layer]
                    if hasattr(config, "num_kv_heads")
                    else config.num_heads[i_layer],
                    downsample=DonutSwinPatchMerging
                    if (i_layer < self.num_layers - 1)
                    else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, DonutSwinEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(
                batch_size, *input_dimensions, hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )

            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size,
                    *(output_dimensions[0], output_dimensions[1]),
                    hidden_size,
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return DonutSwinEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


# Copied from transformers.models.swin.modeling_swin.SwinPreTrainedModel with Swin->DonutSwin
class DonutSwinPreTrainedModel(SuryaPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DonutSwinConfig
    base_model_prefix = "swin"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DonutSwinStage"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
