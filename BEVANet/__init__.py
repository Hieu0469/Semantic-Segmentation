from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import BEVANet.BEVANet
import BEVANet.LSKA
import BEVANet.PPM
import BEVANet.conv
import BEVANet.FB

module_dict_LKA = {
    "LSKA": BEVANet.LSKA.LSKA,
    "SDLSKA": BEVANet.LSKA.SDLSKA,
    "SLAK": BEVANet.LSKA.SLAK
}

module_dict_conv = {
    "DWConv": BEVANet.conv.DWConv,
    "PDWConv": BEVANet.conv.PDWConv,
    "PWConv": BEVANet.conv.PWConv,
    "GSPWConv": BEVANet.conv.GSPWConv,
    "PGSPWConv": BEVANet.conv.PGSPWConv,
}

module_dict_PPM = {
    "DAPPM": BEVANet.PPM.DAPPM,
    "PAPPM": BEVANet.PPM.PAPPM,
    "DLKPPM": BEVANet.PPM.DLKPPM,
    "PLKPPM": BEVANet.PPM.PLKPPM,
}
module_dict_FB = {
    "Bag": BEVANet.FB.Bag,
    "Light_Bag": BEVANet.FB.Light_Bag,
    "BGAF": BEVANet.FB.BGAF,
}

module_dict = {
    "LKA": module_dict_LKA,
    "conv": module_dict_conv,
    "PPM": module_dict_PPM,
    "FB": module_dict_FB,
}