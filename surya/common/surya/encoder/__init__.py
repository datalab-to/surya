import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from surya.common.pretrained import SuryaPreTrainedModel
from surya.common.surya.encoder.config import SuryaEncoderConfig
from surya.common.xla import get_nearest_pad
from surya.logging import get_logger
from surya.settings import settings

if settings.FOUNDATION_XLA:
    import torch_xla.experimental.custom_kernel

from surya.logging import get_logger
logger = get_logger()


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        bsz = hidden_states.shape[0]
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            bsz, -1, self.embed_dim
        )
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device="cpu", dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.mlp(self.ln_q(x).view(bsz, -1, self.hidden_size))
        return x


def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.layers.rotary import apply_rotary_emb

    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed


class Qwen2_5_VLVisionXLASdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, seq_length = hidden_states.shape[0], hidden_states.shape[1]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(bsz, seq_length, 3, self.num_heads, -1)
            .permute(0, 2, 1, 3, 4)
            .unbind(1)
        )
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.zeros([bsz, 1, seq_length, seq_length], dtype=torch.bool)
        cu_seqlens_cpu = cu_seqlens.cpu()
        for j in range(bsz):
            batch_seqlens = cu_seqlens_cpu[j]
            for i in range(1, len(batch_seqlens)):
                attention_mask[
                    j,
                    ...,
                    batch_seqlens[i - 1] : batch_seqlens[i],
                    batch_seqlens[i - 1] : batch_seqlens[i],
                ] = True

        attention_mask = attention_mask.to(q.device)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask,
            dropout_p=0.0,
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionXLAFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Note, this is faster than SDPA, but pretty memory inefficient
        # It also has significant accuracy issues

        bsz, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        # Single reshape to target layout - avoid multiple operations
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(bsz, seq_length, 3, self.num_heads, -1)
            .permute(0, 2, 1, 3, 4)
            .unbind(1)
        )

        # Apply rotary embeddings if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Single reshape to flash attention format [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        total_seqlen = q.shape[2]
        # from cu_seqlens to segment ids for each position in dim 0
        additive_bias = torch.zeros((bsz, 1, total_seqlen, total_seqlen), dtype=q.dtype)
        min_val = torch.finfo(q.dtype).min

        for i in range(bsz):
            padding_end = cu_seqlens[i][1].item()
            additive_bias[i, :, :, :padding_end] = min_val

        additive_bias = additive_bias.to(hidden_states.device)

        attn_scale = 1 / math.sqrt(self.head_dim)
        attn_output = torch_xla.experimental.custom_kernel.flash_attention(
            q, k, v, sm_scale=attn_scale, ab=additive_bias
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_length, -1)
        )
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func

        bsz = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(bsz, seq_length, 3, self.num_heads, -1)
            .permute(0, 2, 1, 3, 4)
            .unbind(1)
        )
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings

        q, k = apply_rotary_pos_emb_flashatt(q, k, cos.squeeze(0), sin.squeeze(0))

        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)
        cu_seqlens = cu_seqlens.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        ).reshape(bsz, seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, seq_length = hidden_states.shape[0], hidden_states.shape[1]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(bsz, seq_length, 3, self.num_heads, -1)
            .permute(0, 2, 1, 3, 4)
            .unbind(1)
        )
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings

        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.full(
            [bsz, 1, seq_length, seq_length],
            torch.finfo(q.dtype).min,
            device=q.device,
            dtype=q.dtype,
        )
        for j in range(bsz):
            batch_seqlens = cu_seqlens[j]
            for i in range(1, len(batch_seqlens)):
                attention_mask[
                    j,
                    ...,
                    batch_seqlens[i - 1] : batch_seqlens[i],
                    batch_seqlens[i - 1] : batch_seqlens[i],
                ] = 0

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def unpack_qkv_with_mask(self, q, k, v, cu_seqlens):
        """
        Unpacks q, k, v sequences into batch-major form and constructs an additive attention mask.

        Args:
            q, k, v: Tensors of shape (total_seq_len, num_heads, head_dim)
            cu_seqlens: Tensor of shape (batch_size + 1,) with cumulative sequence lengths

        Returns:
            batched_q: Tensor of shape (batch_size, max_seq_len, num_heads, head_dim)
            batched_k: Tensor of shape (batch_size, max_seq_len, num_heads, head_dim)
            batched_v: Tensor of shape (batch_size, max_seq_len, num_heads, head_dim)
            attention_mask: Tensor of shape (batch_size, 1, max_seq_len, max_seq_len)
                            with 0 for valid tokens and -inf for padding (for additive attention)
        """
        device = q.device
        dtype = q.dtype

        batch_size = cu_seqlens.shape[0] - 1
        num_heads = q.shape[1]
        head_dim = q.shape[2]

        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]  # Keep as tensor
        max_seq_len = seq_lengths.max().item()  # Use .max() on tensor

        if settings.FOUNDATION_STATIC_CACHE:
            # Pad max_seq_len to the nearest multiple for compilation
            max_seq_len = get_nearest_pad(max_seq_len, pad_multiple=16)

            # Pad batch_size to the nearest multiple for compilation
            batch_size = get_nearest_pad(batch_size, pad_multiple=2)

            # Ensure seq_lengths is a tensor of the correct size
            seq_lengths = F.pad(
                seq_lengths, (0, batch_size - seq_lengths.size(0)), "constant", 0
            )

        # some day, you may look at this, and think: "what if I used repeat_interlave or some other fancy torch instead"?
        # don't do this - it's a path to madness.  For some reason, this loop is optimal

        batch_indices = []
        position_indices = []

        for i, seq_len in enumerate(
            seq_lengths.tolist()
        ):  # Convert to list only for iteration
            batch_indices.extend([i] * seq_len)
            position_indices.extend(list(range(seq_len)))

        batch_indices = torch.tensor(batch_indices, device=device)
        position_indices = torch.tensor(position_indices, device=device)

        batched_q = torch.zeros(
            (batch_size, max_seq_len, num_heads, head_dim), device=device, dtype=dtype
        )
        batched_k = torch.zeros_like(batched_q)
        batched_v = torch.zeros_like(batched_q)

        # Create additive attention mask
        attention_mask = torch.full(
            (batch_size, max_seq_len, max_seq_len),
            fill_value=float("-inf"),
            device=device,
            dtype=dtype,
        )

        # Create mask for valid positions
        seq_range = torch.arange(max_seq_len, device=device)
        valid_mask = seq_range.unsqueeze(0) < seq_lengths.unsqueeze(
            1
        )  # (batch_size, max_seq_len)
        valid_2d = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(
            1
        )  # (batch_size, max_seq_len, max_seq_len)

        # Simply use boolean indexing to set valid positions to 0
        attention_mask[valid_2d] = 0

        attention_mask = attention_mask.unsqueeze(
            1
        )  # (batch_size, 1, max_seq_len, max_seq_len)

        batched_q[batch_indices, position_indices] = q
        batched_k[batch_indices, position_indices] = k
        batched_v[batch_indices, position_indices] = v

        return (
            batched_q,
            batched_k,
            batched_v,
            attention_mask,
            batch_indices,
            position_indices,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.squeeze(0)
        cu_seqlens = cu_seqlens.squeeze(0)

        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        q, k, v, attention_mask, batch_indices, position_indices = (
            self.unpack_qkv_with_mask(q, k, v, cu_seqlens)
        )
        batch_size, max_seqlen = q.shape[:2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask,
            dropout_p=0.0,
        )
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(
            batch_size, max_seqlen, -1
        )  # Bring back to (batch_size, max_seqlen, hidden_dim)
        attn_output = attn_output[batch_indices, position_indices]
        attn_output = self.proj(attn_output)

        return attn_output.unsqueeze(0)


QWEN2_5_VL_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLVisionAttention,
    "flash_attention_2": Qwen2_5_VLVisionXLAFlashAttention2
    if settings.FOUNDATION_XLA
    else Qwen2_5_VLVisionFlashAttention2,
    "sdpa": Qwen2_5_VLVisionXLASdpaAttention
    if settings.FOUNDATION_XLA
    else Qwen2_5_VLVisionSdpaAttention,
}


class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = QWEN2_5_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


Qwen2_5_VL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2_5_VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


class Qwen2_5_VLPreTrainedModel(SuryaPreTrainedModel):
    config_class = SuryaEncoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False  # TODO (joao): fix. torch.compile failing probably due to `cache_positions`

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
    config_class = SuryaEncoderConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VLVisionBlock(config, config._attn_implementation)
                for _ in range(config.depth)
            ]
        )
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        rotary_pos_emb = []
        grid_thw_list = grid_thw.cpu().tolist()
        for batch_item in grid_thw_list:
            row_pos_ids = []
            heights = [h for _, h, _ in batch_item]
            widths = [w for _, _, w in batch_item]
            max_grid_size = max(heights + widths)
            for t, h, w in batch_item:
                hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
                hpos_ids = hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                hpos_ids = hpos_ids.permute(0, 2, 1, 3)
                hpos_ids = hpos_ids.flatten()

                wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
                wpos_ids = wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                wpos_ids = wpos_ids.permute(0, 2, 1, 3)
                wpos_ids = wpos_ids.flatten()
                # shape: token_count, 2
                row_pos_ids.append(
                    torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
                )
            # shape: token_count, 2
            pos_ids = torch.cat(row_pos_ids, dim=0)
            rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
            rotary_pos_emb_row = rotary_pos_emb_full[pos_ids].flatten(1)
            rotary_pos_emb.append(rotary_pos_emb_row)
        rotary_pos_emb = torch.stack(rotary_pos_emb, dim=0)
        return rotary_pos_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(bsz, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(bsz, num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        bsz, seq_len, _ = hidden_states.size()
        hidden_states = self.patch_embed(hidden_states)  # (bsz, seq_len, hidden_dim)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # hidden_states = hidden_states.reshape(bsz, seq_len, -1)
        # rotary_pos_emb = rotary_pos_emb.reshape(bsz, seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1).to(
            hidden_states.device
        )
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = (grid_thw[:, :, 1] * grid_thw[:, :, 2]).cumsum(
            dim=1,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for layer_num, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__,
                    hidden_states,
                    cu_seqlens,
                    None,
                    position_embeddings,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )

        hidden_states = self.merger(hidden_states)
        return hidden_states


class SuryaEncoderModel(Qwen2_5_VisionTransformerPretrainedModel):
    @property
    def image_size(self) -> int:
        config: SuryaEncoderConfig = self.config
        if isinstance(config.image_size, tuple) and len(config.image_size) == 2:
            return config.image_size
        elif isinstance(config.image_size, int):
            return (config.image_size, config.image_size)

        raise ValueError(
            f"The `image_size` for SwinConfig should be a tuple of (int, int) or a single int but found {type(config.image_size)}"
        )

    @property
    def hidden_size(self) -> int:
        config: SuryaEncoderConfig = self.config
        return config.hidden_size

    def embed_images(
        self,
        image_batch: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(
            hidden_states=image_batch,
            grid_thw=grid_thw,
        )
