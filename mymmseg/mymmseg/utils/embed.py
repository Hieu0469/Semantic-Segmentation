"""PatchEmbed and PatchMerging for Swin Transformer."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from ...utils.layers import build_norm_layer


def _build_norm_1d(norm_cfg: dict, num_channels: int) -> nn.Module:
    """Build norm layer cho token sequence (B, L, C) — dùng LayerNorm thường."""
    cfg = norm_cfg.copy()
    norm_type = cfg.pop('type', 'LN')
    cfg.pop('requires_grad', None)
    if norm_type in ('LN', 'LayerNorm'):
        return nn.LayerNorm(num_channels, **cfg)
    elif norm_type == 'BN':
        return nn.BatchNorm1d(num_channels, **cfg)
    else:
        raise ValueError(f"Unsupported norm for 1D tokens: {norm_type}")


class PatchEmbed(nn.Module):
    """Image → Patch token sequence.

    (B, C, H, W) → (B, H'*W', embed_dims)
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 96,
        conv_type: str = 'Conv2d',
        kernel_size: int = 4,
        stride: int = 4,
        padding: str = 'corner',
        norm_cfg: Optional[dict] = dict(type='LN'),
        init_cfg=None,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        if padding == 'corner':
            self.pad = nn.ZeroPad2d((0, kernel_size - 1, 0, kernel_size - 1))
            pad_val  = 0
        else:
            self.pad    = None
            pad_val     = int(padding) if isinstance(padding, (int, str)) else 0

        self.projection = nn.Conv2d(
            in_channels, embed_dims,
            kernel_size=kernel_size, stride=stride, padding=pad_val,
        )
        # Norm trên token sequence (B, L, C) → LayerNorm 1D
        self.norm = _build_norm_1d(norm_cfg, embed_dims) if norm_cfg else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if self.pad is not None:
            x = self.pad(x)
        x = self.projection(x)              # (B, embed_dims, H', W')
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)   # (B, H'*W', embed_dims)
        if self.norm is not None:
            x = self.norm(x)
        return x, (H, W)


class PatchMerging(nn.Module):
    """Patch Merging — downsampling between Swin stages.

    (B, H*W, C) → (B, H/2*W/2, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 2,
        norm_cfg: Optional[dict] = dict(type='LN'),
        init_cfg=None,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels or 2 * in_channels

        self.norm      = _build_norm_1d(norm_cfg, 4 * in_channels) if norm_cfg else nn.Identity()
        self.reduction = nn.Linear(4 * in_channels, self.out_channels, bias=False)

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]):
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        if H % 2 != 0 or W % 2 != 0:
            import torch.nn.functional as F
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
        H2, W2 = x.shape[1], x.shape[2]
        x = x.view(B, H2 * W2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, (H2, W2)
