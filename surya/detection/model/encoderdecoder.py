"""
This is an implementation of efficientvit, with some modifications (decode head, etc).

Original paper at https://arxiv.org/abs/2205.14756

Code adapted from timm, https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientvit_mit.py
Original code (that timm adapted from) at https://github.com/mit-han-lab/efficientvit

License: Apache 2
"""

from __future__ import annotations

from typing import Optional, Union, Tuple, List, Any
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import SemanticSegmenterOutput

from surya.common.pretrained import SuryaPreTrainedModel
from surya.common.s3 import S3DownloaderMixin
from surya.detection.model.config import EfficientViTConfig


def val2list(x: Union[List, Tuple, Any], repeat_time=1):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: Union[List, Tuple, Any], min_len: int = 1, idx_repeat: int = -1):
    # repeat elements if necessary
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def get_same_padding(
    kernel_size: Union[int, Tuple[int, ...]],
) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        dropout=0.0,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ):
        super(ConvNormAct, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=False)
        padding = get_padding(kernel_size, stride, dilation)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )
        self.norm = (
            norm_layer(num_features=out_channels) if norm_layer else nn.Identity()
        )
        self.act = act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm_layer=(nn.BatchNorm2d, nn.BatchNorm2d),
        act_layer=(nn.ReLU6, None),
    ):
        super(DSConv, self).__init__()
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)

        self.depth_conv = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.point_conv = ConvNormAct(
            in_channels,
            out_channels,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm_layer=(nn.BatchNorm2d, nn.BatchNorm2d),
        act_layer=(nn.ReLU6, None),
    ):
        super(ConvBlock, self).__init__()
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvNormAct(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.conv2 = ConvNormAct(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm_layer=(nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d),
        act_layer=(nn.ReLU6, nn.ReLU6, None),
    ):
        super(MBConv, self).__init__()
        use_bias = val2tuple(use_bias, 3)
        norm_layer = val2tuple(norm_layer, 3)
        act_layer = val2tuple(act_layer, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvNormAct(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.depth_conv = ConvNormAct(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )
        self.point_conv = ConvNormAct(
            mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer[2],
            act_layer=act_layer[2],
            bias=use_bias[2],
        )

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm_layer=(nn.BatchNorm2d, nn.BatchNorm2d),
        act_layer=(nn.ReLU6, None),
    ):
        super(FusedMBConv, self).__init__()
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvNormAct(
            in_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.point_conv = ConvNormAct(
            mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class LiteMLA(nn.Module):
    """Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Union[int, None] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm_layer=(None, nn.BatchNorm2d),
        act_layer=(None, None),
        kernel_func=nn.ReLU,
        scales=(5,),
        eps=1e-5,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)

        self.dim = dim
        self.qkv = ConvNormAct(
            in_channels,
            3 * total_dim,
            1,
            bias=use_bias[0],
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        1,
                        groups=3 * heads,
                        bias=use_bias[0],
                    ),
                )
                for scale in scales
            ]
        )
        self.kernel_func = kernel_func(inplace=False)

        self.proj = ConvNormAct(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            bias=use_bias[1],
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
        )

    def _attn(self, q, k, v):
        dtype = v.dtype
        q, k, v = q.float(), k.float(), v.float()
        kv = k.transpose(-1, -2) @ v
        out = q @ kv
        out = out[..., :-1] / (out[..., -1:] + self.eps)
        return out.to(dtype)

    def forward(self, x):
        # Shape is B, C, H, W
        B, _, H, W = x.shape

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        multi_scale_qkv = multi_scale_qkv.reshape(B, -1, 3 * self.dim, H * W).transpose(
            -1, -2
        )
        # Shape for each is B, C, HW, head_dim
        q, k, v = multi_scale_qkv.chunk(3, dim=-1)

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v = F.pad(v, (0, 1), mode="constant", value=1.0)

        out = self._attn(q, k, v)

        # final projection
        out = out.transpose(-1, -2).reshape(B, -1, H, W)
        out = self.proj(out)
        return out


class EfficientVitBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        heads_ratio=1.0,
        head_dim=32,
        expand_ratio=4,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
    ):
        super(EfficientVitBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=head_dim,
                norm_layer=(None, norm_layer),
            ),
            nn.Identity(),
        )
        self.local_module = ResidualBlock(
            MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm_layer=(None, None, norm_layer),
                act_layer=(act_layer, act_layer, None),
            ),
            nn.Identity(),
        )

    def forward(self, x):
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module] = None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()
        self.pre_norm = pre_norm if pre_norm is not None else nn.Identity()
        self.main = main
        self.shortcut = shortcut

    def forward(self, x):
        res = self.main(self.pre_norm(x))
        if self.shortcut is not None:
            res = res + self.shortcut(x)
        return res


def build_local_block(
    in_channels: int,
    out_channels: int,
    stride: int,
    kernel_size: int,
    expand_ratio: float,
    norm_layer: str,
    act_layer: str,
    fewer_norm: bool = False,
    block_type: str = "default",
):
    assert block_type in ["default", "large", "fused"]
    if expand_ratio == 1:
        if block_type == "default":
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
        else:
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
    else:
        if block_type == "default":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm_layer=(None, None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, act_layer, None),
            )
        else:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
    return block


class Stem(nn.Sequential):
    def __init__(
        self,
        in_chs,
        out_chs,
        depth,
        stride,
        norm_layer,
        act_layer,
        block_type="default",
    ):
        super().__init__()
        self.stride = stride

        self.add_module(
            "in_conv",
            ConvNormAct(
                in_chs,
                out_chs,
                kernel_size=stride + 1,
                stride=stride,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
        )
        stem_block = 0
        for _ in range(depth):
            self.add_module(
                f"res{stem_block}",
                ResidualBlock(
                    build_local_block(
                        in_channels=out_chs,
                        out_channels=out_chs,
                        stride=1,
                        kernel_size=3,
                        expand_ratio=1,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        block_type=block_type,
                    ),
                    nn.Identity(),
                ),
            )
            stem_block += 1


class EfficientVitLargeStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        depth,
        stride,
        norm_layer,
        act_layer,
        head_dim,
        vit_stage=False,
        fewer_norm=False,
    ):
        super(EfficientVitLargeStage, self).__init__()
        blocks = [
            ResidualBlock(
                build_local_block(
                    in_channels=in_chs,
                    out_channels=out_chs,
                    stride=stride,
                    kernel_size=stride + 1,
                    expand_ratio=24 if vit_stage else 16,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    fewer_norm=vit_stage or fewer_norm,
                    block_type="default" if fewer_norm else "fused",
                ),
                None,
            )
        ]
        in_chs = out_chs

        if vit_stage:
            # for stage 4
            for _ in range(depth):
                blocks.append(
                    EfficientVitBlock(
                        in_channels=in_chs,
                        head_dim=head_dim,
                        expand_ratio=6,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    )
                )
        else:
            # for stage 1, 2, 3
            for i in range(depth):
                blocks.append(
                    ResidualBlock(
                        build_local_block(
                            in_channels=in_chs,
                            out_channels=out_chs,
                            stride=1,
                            kernel_size=3,
                            expand_ratio=4,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            fewer_norm=fewer_norm,
                            block_type="default" if fewer_norm else "fused",
                        ),
                        nn.Identity(),
                    )
                )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class EfficientVitLarge(nn.Module):
    def __init__(
        self,
        config: EfficientViTConfig,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
    ):
        super(EfficientVitLarge, self).__init__()
        self.grad_checkpointing = False
        self.num_classes = config.num_classes
        self.norm_eps = config.layer_norm_eps
        norm_layer = partial(norm_layer, eps=self.norm_eps)

        # input stem
        self.stem = Stem(
            config.num_channels,
            config.widths[0],
            config.depths[0],
            config.strides[0],
            norm_layer,
            act_layer,
            block_type="large",
        )
        stride = config.strides[0]

        # stages
        self.feature_info = []
        self.stages = nn.Sequential()
        in_channels = config.widths[0]
        for i, (w, d, s) in enumerate(
            zip(config.widths[1:], config.depths[1:], config.strides[1:])
        ):
            self.stages.append(
                EfficientVitLargeStage(
                    in_channels,
                    w,
                    depth=d,
                    stride=s,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    head_dim=config.head_dim,
                    vit_stage=i >= 3,
                    fewer_norm=i >= 2,
                )
            )
            stride *= s
            in_channels = w
            self.feature_info += [
                dict(num_chs=in_channels, reduction=stride, module=f"stages.{i}")
            ]

        self.num_features = in_channels

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, x):
        x = self.stem(x)
        encoder_hidden_states = []
        for i, module in enumerate(self.stages):
            x = module(x)
            encoder_hidden_states.append(x)

        return encoder_hidden_states


class EfficientViTPreTrainedModel(SuryaPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EfficientViTConfig
    base_model_prefix = "efficientvit"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class DecodeMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor):
        # Input is B, C, H, W
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        # Output is B, HW, C
        hidden_states = self.proj(hidden_states)
        return hidden_states


class DecodeHead(EfficientViTPreTrainedModel):
    def __init__(self, config: EfficientViTConfig):
        super().__init__(config)

        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for width in config.widths[1:]:
            mlp = DecodeMLP(
                input_dim=width, output_dim=config.decoder_layer_hidden_size
            )
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_layer_hidden_size * config.num_stages,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(
            config.decoder_hidden_size, config.num_labels, kernel_size=1
        )

        self.config = config

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)  # Output is B, HW, C
            # Permute to B, C, HW
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width
            )
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


class EfficientViTForSemanticSegmentation(
    S3DownloaderMixin, EfficientViTPreTrainedModel
):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.vit = EfficientVitLarge(config)
        self.decode_head = DecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, pixel_values: torch.FloatTensor
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        # Pixel values should be B,C,H,W
        encoder_hidden_states = self.vit(
            pixel_values,
        )

        logits = self.decode_head(encoder_hidden_states)

        # Apply sigmoid to get 0-1 output
        logits = torch.special.expit(logits)

        return SemanticSegmenterOutput(
            loss=None, logits=logits, hidden_states=encoder_hidden_states
        )


class EfficientViTForSemanticLayoutSegmentation(EfficientViTPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.vit = EfficientVitLarge(config)
        self.decode_head = DecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, pixel_values: torch.FloatTensor
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        # Pixel values should be B,C,H,W
        encoder_hidden_states = self.vit(
            pixel_values,
        )

        logits = self.decode_head(encoder_hidden_states)

        # Apply sigmoid to get 0-1 output
        logits = torch.special.expit(logits)

        return SemanticSegmenterOutput(
            loss=None, logits=logits, hidden_states=encoder_hidden_states
        )
