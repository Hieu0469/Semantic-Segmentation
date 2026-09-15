"""Shared building blocks: ConvBNAct, build_norm_layer, build_act_layer.

Imported by both backbones and decode heads so everything uses
the same ConvBNAct with configurable norm/act.
"""

from typing import Optional, Union

import torch
import torch.nn as nn


# ── Activation builder ────────────────────────────────────────────────────────

def build_act_layer(act_cfg: dict) -> nn.Module:
    cfg = act_cfg.copy()
    act_type = cfg.pop('type', 'ReLU')

    if act_type == 'ReLU':
        return nn.ReLU(inplace=cfg.get('inplace', True))
    elif act_type == 'GELU':
        return nn.GELU()
    elif act_type == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=cfg.get('negative_slope', 0.01),
                            inplace=cfg.get('inplace', True))
    elif act_type == 'SiLU':
        return nn.SiLU(inplace=cfg.get('inplace', True))
    elif act_type == 'Hardswish':
        return nn.Hardswish(inplace=cfg.get('inplace', True))
    elif act_type == 'PReLU':
        return nn.PReLU()
    else:
        raise ValueError(f"Unknown activation type: '{act_type}'")


# ── Norm builder ──────────────────────────────────────────────────────────────

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm cho CNN feature map (B, C, H, W).

    Permute → (B, H, W, C) → LayerNorm → permute lại.
    Dùng trong ViT-style backbone như EfficientViT.
    """
    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def build_norm_layer(
    norm_cfg: dict,
    num_channels: int,
    postfix: Union[int, str] = '',
):
    """Returns (name, layer) — mirrors mmcv.build_norm_layer API."""
    cfg = norm_cfg.copy()
    norm_type = cfg.pop('type', 'BN')
    requires_grad = cfg.pop('requires_grad', True)

    abbr_map = {
        'BN':     'bn',
        'GN':     'gn',
        'SyncBN': 'syncbn',
        'LN':     'ln',
    }

    if norm_type == 'BN':
        layer = nn.BatchNorm2d(num_channels, **cfg)
    elif norm_type == 'GN':
        layer = nn.GroupNorm(num_channels=num_channels, **cfg)
    elif norm_type == 'SyncBN':
        layer = nn.SyncBatchNorm(num_channels, **cfg)
    elif norm_type == 'LN':
        layer = LayerNorm2d(num_channels, **cfg)
    else:
        raise ValueError(f"Unknown norm type: '{norm_type}'")

    for p in layer.parameters():
        p.requires_grad = requires_grad

    name = abbr_map[norm_type] + str(postfix)
    return name, layer


# ── ConvBNAct — shared building block ────────────────────────────────────────

class ConvBNAct(nn.Sequential):
    """Conv2d → Norm → Activation.

    Drop-in replacement for mmcv's ConvModule with configurable
    norm_cfg and act_cfg.

    Args:
        in_channels:  Input channels.
        out_channels: Output channels.
        kernel_size:  Default 3.
        stride:       Default 1.
        padding:      Default kernel_size//2 (auto).
        dilation:     Default 1.
        groups:       Default 1.
        bias:         Default False (norm makes bias redundant).
        norm_cfg:     e.g. dict(type='BN') or dict(type='LN').
        act_cfg:      e.g. dict(type='ReLU') or dict(type='GELU').
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_cfg: dict = dict(type='BN'),
        act_cfg: dict = dict(type='ReLU'),
    ):
        if padding is None:
            padding = (kernel_size // 2) * dilation

        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_act_layer(act_cfg),
        ]
        super().__init__(*layers)

def build_conv_layer(conv_cfg: Optional[dict], *args, **kwargs) -> nn.Conv2d:
    """Returns a Conv2d (conv_cfg=None → standard Conv2d, matching mmcv API)."""
    if conv_cfg is None or conv_cfg.get('type') == 'Conv2d':
        return nn.Conv2d(*args, **kwargs)
    raise NotImplementedError(f"conv_cfg type '{conv_cfg['type']}' not yet supported.")