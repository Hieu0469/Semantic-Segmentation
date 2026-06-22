import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.base import SegmentationHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackbone

class FlexibleDeepLabV3PlusDecoder(DeepLabV3PlusDecoder):
    """Override forward để tự resize thay vì dựa vào spatial size cố định."""
    def forward(self, features):
        aspp_features = self.aspp(features[-1])           # high-level, spatial nhỏ
        high_res_features = self.block1(features[2])      # low-level, spatial lớn hơn

        # Resize aspp_features về cùng size với high_res_features
        aspp_features = F.interpolate(
            aspp_features,
            size=high_res_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features

class EfficientViT_DeepLabV3Plus(nn.Module):
    def __init__(self,
                 encoder,
                 decoder, 
                 seg_head
                ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.segmentation_head = seg_head

    def forward(self, x):
        feats_dict = self.encoder(x)
        feats_list = list(feats_dict.values())  # [input, stage0, stage1, stage2, stage3, stage4, stage_final]

        dec_out = self.decoder(feats_list)
        out = self.segmentation_head(dec_out)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out

def EfficientvitL1_DeeplabV3Plus() -> EfficientViT_DeepLabV3Plus:
    encoder = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        act_func="relu"
    )
    decoder = FlexibleDeepLabV3PlusDecoder(
        encoder_channels=[3, 32, 64, 128, 256, 512, 512],
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
        encoder_depth=6,        # len(encoder_channels) - 1 = 6
        aspp_separable=True,    # dùng separable conv trong ASPP (nhẹ hơn)
        aspp_dropout=0.5,
    )
    seg_head = SegmentationHead(
        in_channels=256,
        out_channels=19,
        kernel_size=1,
        upsampling=1,
    )
    return EfficientViT_DeepLabV3Plus(
        encoder=encoder,
        decoder=decoder,
        seg_head=seg_head
    )