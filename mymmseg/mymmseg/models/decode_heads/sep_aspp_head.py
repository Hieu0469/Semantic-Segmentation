"""DepthwiseSeparableASPPHead — DeepLabV3+ decode head.

Reference: Chen et al. "Encoder-Decoder with Atrous Separable Convolution
           for Semantic Image Segmentation" — https://arxiv.org/abs/1802.02611

So với DeepLabV3 (ASPPHead):
    1. Các nhánh ASPP có dilation > 1 dùng depthwise-separable conv.
    2. Thêm decoder: lấy feature low-level (stride 4) để khôi phục ranh giới.

Luồng dữ liệu (ảnh 512×1024, backbone output stride 8)::

    layer4 (B, 2048, 64, 128) ─► ASPP (5 nhánh) ─► concat ─► bottleneck 3×3
                                                         │ (B, 512, 64, 128)
                                                         ▼ upsample ×2
    layer1 (B, 256, 128, 256) ─► c1_bottleneck 1×1 ─► (B, 48, 128, 256)
                                                         │
                                          concat ◄───────┘ (B, 560, 128, 256)
                                            ▼
                         sep_bottleneck: 2 × DepthwiseSeparableConv 3×3
                                            ▼ (B, 512, 128, 256)
                         dropout → cls_seg → (B, 19, 128, 256)
                                            ▼ upsample ×4 (base class)
                                       (B, 19, 512, 1024)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import DECODE_HEADS
from ...utils.layers import ConvBNAct, DepthwiseSeparableConvBNAct
from .aspp_head import ASPPHead, ASPPModule


class DepthwiseSeparableASPPModule(ASPPModule):
    """ASPPModule mà các nhánh dilation > 1 dùng depthwise-separable conv.

    Nhánh dilation == 1 giữ nguyên conv 1×1 thường (như ASPPModule gốc).
    """

    def __init__(
        self,
        dilations: tuple,
        in_channels: int,
        channels: int,
        norm_cfg: dict = dict(type='BN'),
        act_cfg: dict = dict(type='ReLU'),
    ):
        super().__init__(dilations, in_channels, channels, norm_cfg, act_cfg)
        for i, dilation in enumerate(dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvBNAct(
                    in_channels,
                    channels,
                    kernel_size=3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )


@DECODE_HEADS.register_module()
class DepthwiseSeparableASPPHead(ASPPHead):
    """DeepLabV3+ decode head.

    Args:
        c1_in_channels: Channels của feature low-level (layer1 của ResNet-50/101
                        là 256). Đặt 0 để tắt nhánh low-level (→ giống DeepLabV3
                        nhưng dùng sep conv).
        c1_channels:    Channels của nhánh low-level sau conv 1×1. Default: 48.
        c1_index:       Index trong list backbone feature dùng làm low-level.
                        Default: 0 (stage đầu tiên, stride 4).
        dilations:      Dilation các nhánh ASPP. Dùng (1, 12, 24, 36) khi
                        backbone output stride 8, (1, 6, 12, 18) khi stride 16.
        in_channels:    Channels feature high-level (layer4 → 2048).
        in_index:       Index feature high-level. Default: -1.
        channels:       Channels nội bộ của head.
        num_classes:    Số class.
        dropout_ratio:  Dropout trước cls_seg.
        norm_cfg:       Config norm (BN / SyncBN / GN / LN ...).
        act_cfg:        Config activation.
        align_corners:  Cho F.interpolate.

    Lưu ý: backbone phải trả về cả feature low-level, ví dụ
    ``out_indices=(0, 1, 2, 3)`` rồi ``c1_index=0, in_index=3``.
    """

    def __init__(
        self,
        c1_in_channels: int,
        c1_channels: int = 48,
        c1_index: int = 0,
        dilations: tuple = (1, 6, 12, 18),
        in_channels: int = 2048,
        in_index: int = -1,
        channels: int = 512,
        num_classes: int = 19,
        dropout_ratio: float = 0.1,
        norm_cfg: dict = dict(type='BN'),
        act_cfg: dict = dict(type='ReLU'),
        align_corners: bool = False,
        init_cfg: Optional[dict] = dict(type='Normal', std=0.01, layer='Conv2d'),
    ):
        assert c1_in_channels >= 0, 'c1_in_channels phải >= 0'

        # Lưu trước super().__init__() vì _build_head() cần dùng
        self._c1_in_channels = c1_in_channels
        self._c1_channels    = c1_channels
        self.c1_index        = c1_index

        super().__init__(
            dilations=dilations,
            in_channels=in_channels,
            in_index=in_index,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners,
            init_cfg=init_cfg,
        )

    # ── build ─────────────────────────────────────────────────────────────────

    def _build_head(self) -> None:
        # image_pool + aspp_modules + bottleneck (3×3) của ASPPHead
        super()._build_head()

        # Thay các nhánh ASPP bằng bản depthwise-separable
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self._dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            norm_cfg=self._norm_cfg,
            act_cfg=self._act_cfg,
        )

        # Nhánh low-level: giảm channels để không lấn át feature high-level
        if self._c1_in_channels > 0:
            self.c1_bottleneck = ConvBNAct(
                self._c1_in_channels,
                self._c1_channels,
                kernel_size=1,
                norm_cfg=self._norm_cfg,
                act_cfg=self._act_cfg,
            )
            sep_in = self.channels + self._c1_channels
        else:
            self.c1_bottleneck = None
            sep_in = self.channels

        # Decoder: 2 × depthwise-separable conv 3×3
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvBNAct(
                sep_in, self.channels, kernel_size=3,
                norm_cfg=self._norm_cfg, act_cfg=self._act_cfg),
            DepthwiseSeparableConvBNAct(
                self.channels, self.channels, kernel_size=3,
                norm_cfg=self._norm_cfg, act_cfg=self._act_cfg),
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        inputs: List[torch.Tensor],
        img_size: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Override forward của base class vì cần cả feature low-level
        (inputs[c1_index]) chứ không chỉ feature được chọn bởi in_index.
        """
        x = self._select_features(inputs)
        out = self.forward_features(x)     # ASPP → bottleneck, stride của x

        if self.c1_bottleneck is not None:
            c1 = self.c1_bottleneck(inputs[self.c1_index])
            out = F.interpolate(
                out, size=c1.shape[2:],
                mode='bilinear', align_corners=self.align_corners,
            )
            out = torch.cat([out, c1], dim=1)

        out = self.sep_bottleneck(out)
        out = self.dropout(out)
        out = self.cls_seg(out)

        if img_size is not None:
            out = F.interpolate(
                out, size=img_size,
                mode='bilinear', align_corners=self.align_corners,
            )
        return out