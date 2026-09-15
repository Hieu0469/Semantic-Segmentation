"""FCN (Fully Convolutional Network) decode head.

Reference: Long et al. "Fully Convolutional Networks for Semantic Segmentation"
           CVPR 2015 — https://arxiv.org/abs/1411.4038

Architecture:
    backbone_feature → [conv-BN-ReLU] x num_convs → dropout → cls_seg(1x1) → upsample
"""

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import DECODE_HEADS
from ...utils.layers import build_norm_layer,build_act_layer


from .base_decode_head import BaseDecodeHead

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 padding=1, dilation=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                      padding * dilation, dilation=dilation, bias=False),
            build_norm_layer(norm_cfg, out_ch)[1],
            build_act_layer(act_cfg),
        ]
        super().__init__(*layers)


@DECODE_HEADS.register_module()
class FCNHead(BaseDecodeHead):
    """FCN decode head with N × (Conv-BN-ReLU) blocks.

    Args:
        in_channels: Channels of the single backbone feature used.
        in_index:    Index of the backbone feature to use (default: -1, last).
        channels:    Internal channel width (default: 256).
        num_classes: Number of segmentation classes.
        num_convs:   Number of ConvBNReLU blocks (default: 2).
        kernel_size: Kernel size of each conv (default: 3).
        dilation:    Dilation of each conv (default: 1).
        dropout_ratio: Dropout before cls_seg (default: 0.1).
        align_corners: For bilinear upsample (default: False).

    Input → Output shapes:
        backbone_feat: (B, in_channels, H/s, W/s)
        output logits: (B, num_classes, H, W)   (after upsampling to img_size)
    """

    def __init__(
        self,
        in_channels: int = 2048,
        in_index: int = -1,
        channels: int = 256,
        num_classes: int = 19,
        num_convs: int = 2,
        concat_input: bool = False,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU')
    ):
        # Store before super().__init__ so _build_head() can access them
        self._num_convs = num_convs
        self._kernel_size = kernel_size
        self._dilation = dilation
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self._concat_input = concat_input

        super().__init__(
            in_channels=in_channels,
            in_index=in_index,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            align_corners=align_corners,
        )

    # ── BaseDecodeHead interface ──────────────────────────────────────────────

    def _build_head(self) -> None:
        """Build the stack of ConvBNReLU blocks."""
        convs: List[nn.Module] = []
        _in_ch = self.in_channels

        for i in range(self._num_convs):
            convs.append(
                ConvBNAct(
                    _in_ch,
                    self.channels,
                    kernel_size=self._kernel_size,
                    dilation=self._dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
            _in_ch = self.channels  # subsequent blocks: channels → channels

        # num_convs=0 is valid (linear probe on raw features)
        if self._num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        # When num_convs=0 the cls_seg receives in_channels directly,
        # so override the parent's cls_seg channel expectation.
        if self._num_convs == 0:
            self.cls_seg = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)

        # Thêm conv fusion nếu concat_input=True
        if self._concat_input:
            self.conv_cat = ConvBNAct(
                self.in_channels + self.channels,  # concat 2 feature
                self.channels,
                kernel_size=1,                     # 1x1 để fusion
                norm_cfg=self._norm_cfg,
                act_cfg=self._act_cfg,
            )

    def forward_features(self, x):
        input_ = x                  # giữ lại input gốc
        out = self.convs(x)

        if self._concat_input:
            out = self.conv_cat(torch.cat([input_, out], dim=1))

        return out