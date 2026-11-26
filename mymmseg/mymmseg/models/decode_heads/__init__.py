from .base_decode_head import BaseDecodeHead
from .fcn_head import FCNHead  # ← decorator @DECODE_HEADS.register_module() chạy ở đây
from .aspp_head import ASPPHead
from .sep_aspp_head import DepthwiseSeparableASPPHead


__all__ = ["BaseDecodeHead", "FCNHead", "ASPPHead", "DepthwiseSeparableASPPHead"]