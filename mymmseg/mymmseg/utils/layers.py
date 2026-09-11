import torch.nn as nn
import torch
from typing import Iterable, List, Optional, Union


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm cho feature map (B, C, H, W).
    
    Permute → (B, H, W, C) → LayerNorm → permute lại.
    Dùng trong ViT-style backbone như EfficientViT.
    """
    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H, W, C) → norm → (B, C, H, W)
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

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
    else:
        raise ValueError(f"Unknown activation type: {act_type}")

def build_norm_layer(norm_cfg: dict, num_channels: int, postfix: Union[int, str] = ''):
    cfg = norm_cfg.copy()
    norm_type = cfg.pop('type', 'BN')
    requires_grad = cfg.pop('requires_grad', True)

    if norm_type == 'BN':
        layer = nn.BatchNorm2d(num_channels, **cfg)
    elif norm_type == 'GN':
        layer = nn.GroupNorm(num_channels=num_channels, **cfg)
    elif norm_type == 'SyncBN':
        layer = nn.SyncBatchNorm(num_channels, **cfg)
    elif norm_type == 'LN':
        layer = LayerNorm2d(num_channels, **cfg)   # ← thêm
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

    for p in layer.parameters():
        p.requires_grad = requires_grad

    abbr = {'BN': 'bn', 'GN': 'gn', 'SyncBN': 'syncbn', 'LN': 'ln'}[norm_type]
    name = abbr + str(postfix)
    return name, layer

def build_conv_layer(conv_cfg: Optional[dict], *args, **kwargs) -> nn.Conv2d:
    """Returns a Conv2d (conv_cfg=None → standard Conv2d, matching mmcv API)."""
    if conv_cfg is None or conv_cfg.get('type') == 'Conv2d':
        return nn.Conv2d(*args, **kwargs)
    raise NotImplementedError(f"conv_cfg type '{conv_cfg['type']}' not yet supported.")