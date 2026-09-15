"""ASPPHead — DeepLabV3 decode head.

Reference: Chen et al. "Rethinking Atrous Convolution for Semantic
           Image Segmentation" (DeepLabV3) — https://arxiv.org/abs/1706.05587

Architecture:
    input feature
        ├── Global Average Pooling → 1×1 conv → upsample
        ├── Conv 1×1  (dilation=1)
        ├── Conv 3×3  (dilation=d1)
        ├── Conv 3×3  (dilation=d2)
        └── Conv 3×3  (dilation=d3)
              ↓ concat tất cả
        Bottleneck Conv 3×3  (channels*(len+1) → channels)
              ↓
        dropout → cls_seg(1×1) → upsample
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import DECODE_HEADS
from ...utils.layers import ConvBNAct
from .base_decode_head import BaseDecodeHead


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling module.

    Tạo nhiều nhánh conv với dilation khác nhau để capture
    multi-scale context. Với dilation=1 dùng conv 1×1,
    các dilation khác dùng conv 3×3.

    Args:
        dilations:   Tuple các dilation rate, e.g. (1, 6, 12, 18).
        in_channels: Channels của input feature.
        channels:    Output channels mỗi nhánh.
        norm_cfg:    Config norm layer.
        act_cfg:     Config activation.
    """

    def __init__(
        self,
        dilations: tuple,
        in_channels: int,
        channels: int,
        norm_cfg: dict = dict(type='BN'),
        act_cfg: dict = dict(type='ReLU'),
    ):
        super().__init__()
        self.dilations   = dilations
        self.in_channels = in_channels
        self.channels    = channels

        for dilation in dilations:
            # dilation=1 → conv 1×1 (no padding needed)
            # dilation>1 → conv 3×3 với padding=dilation
            self.append(
                ConvBNAct(
                    in_channels,
                    channels,
                    kernel_size=1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Chạy tất cả nhánh ASPP, trả về list feature maps."""
        return [branch(x) for branch in self]


@DECODE_HEADS.register_module()
class ASPPHead(BaseDecodeHead):
    """DeepLabV3 decode head với ASPP.

    Args:
        dilations:  Dilation rates cho các nhánh ASPP.
                    Default: (1, 6, 12, 18).
        in_channels: Channels của backbone feature dùng làm input.
        in_index:   Index của backbone feature. Default: -1 (layer4).
        channels:   Internal channels. Default: 256.
        num_classes: Số class. Default: 19.
        dropout_ratio: Dropout trước cls_seg. Default: 0.1.
        norm_cfg:   Config norm. Default: BN.
        act_cfg:    Config activation. Default: ReLU.
        align_corners: Cho F.interpolate. Default: False.

    Architecture chi tiết::

        x (B, in_channels, H, W)
        ├─ GAP → Conv1×1 → upsample → (B, channels, H, W)
        ├─ Conv1×1 dilation=1   → (B, channels, H, W)
        ├─ Conv3×3 dilation=6   → (B, channels, H, W)
        ├─ Conv3×3 dilation=12  → (B, channels, H, W)
        └─ Conv3×3 dilation=18  → (B, channels, H, W)
              ↓ cat → (B, channels*5, H, W)
        bottleneck Conv3×3 → (B, channels, H, W)
              ↓
        dropout → cls_seg → upsample → (B, num_classes, H, W)
    """

    def __init__(
        self,
        dilations: tuple = (1, 6, 12, 18),
        in_channels: int = 2048,
        in_index: int = -1,
        channels: int = 256,
        num_classes: int = 19,
        dropout_ratio: float = 0.1,
        norm_cfg: dict = dict(type='BN'),
        act_cfg: dict = dict(type='ReLU'),
        align_corners: bool = False,
        init_cfg: Optional[dict] = dict(type='Normal', std=0.01, layer='Conv2d'),
    ):
        assert isinstance(dilations, (list, tuple)), \
            f'dilations phải là list hoặc tuple, nhận được {type(dilations)}'

        # Lưu trước khi gọi super() vì _build_head() cần chúng
        self._dilations = dilations
        self._norm_cfg  = norm_cfg
        self._act_cfg   = act_cfg

        super().__init__(
            in_channels=in_channels,
            in_index=in_index,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            align_corners=align_corners,
            init_cfg=init_cfg,
        )

    # ── BaseDecodeHead interface ──────────────────────────────────────────────

    def _build_head(self) -> None:
        """Build: image_pool + aspp_modules + bottleneck."""

        # Nhánh Global Average Pooling
        # GAP → (B, in_channels, 1, 1) → Conv1×1 → (B, channels, 1, 1)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(
                self.in_channels,
                self.channels,
                kernel_size=1,
                norm_cfg=self._norm_cfg,
                act_cfg=self._act_cfg,
            ),
        )

        # Các nhánh ASPP với dilation khác nhau
        self.aspp_modules = ASPPModule(
            dilations=self._dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            norm_cfg=self._norm_cfg,
            act_cfg=self._act_cfg,
        )

        # Bottleneck: concat (len(dilations)+1) nhánh → channels
        # +1 vì có thêm nhánh image_pool
        self.bottleneck = ConvBNAct(
            (len(self._dilations) + 1) * self.channels,
            self.channels,
            kernel_size=3,
            norm_cfg=self._norm_cfg,
            act_cfg=self._act_cfg,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone feature (B, in_channels, H, W).

        Returns:
            Decoded feature (B, channels, H, W).
        """
        H, W = x.shape[2], x.shape[3]

        # Nhánh GAP: upsample về cùng kích thước với x
        gap_out = F.interpolate(
            self.image_pool(x),
            size=(H, W),
            mode='bilinear',
            align_corners=self.align_corners,
        )

        # Các nhánh ASPP
        aspp_outs = self.aspp_modules(x)

        # Concat tất cả: [gap, aspp1, aspp2, ...]
        # → (B, channels*(len+1), H, W)
        out = torch.cat([gap_out] + aspp_outs, dim=1)

        # Bottleneck fusion
        return self.bottleneck(out)