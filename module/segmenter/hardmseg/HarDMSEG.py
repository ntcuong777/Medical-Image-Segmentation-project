import torch.nn as nn
import torch.nn.functional as F
from module.baseline_network import BaselineFactory
from module.useful_blocks import Aggregation, RFB_modified

# This class is copied from the original author.
# It has gone through several modifications since the author
# make this class overloaded with stuffs that are unnecessary.
class HarDMSEG(nn.Module):
    # res2net based encoder decoder
    def __init__(self, model_variant='HarDNet68', activation='relu', channel=32):
        super(HarDMSEG, self).__init__()

        assert model_variant in ['HarDNet39ds', 'HarDNet68ds', 'HarDNet68', 'HarDNet85']
        
        arch = 68 # default arch
        if '39' in model_variant:
            arch = 39
        elif '85' in model_variant:
            arch = 85

        self.relu = nn.ReLU(True)

        # ---- Receptive Field Block like module ----
        ch1, ch2, ch3 = 320, 720, 1280
        if arch == 68:
            ch1, ch2, ch3 = 320, 640, 1024

        self.rfb2_1 = RFB_modified(ch1, channel, activation=activation)
        self.rfb3_1 = RFB_modified(ch2, channel, activation=activation)
        self.rfb4_1 = RFB_modified(ch3, channel, activation=activation)
        
        # ---- Partial Decoder ----
        self.agg1 = Aggregation(channel)

        # ---- define baseline network ----
        self.base_net = BaselineFactory.create_baseline_as(baseline_model='hardnet', model_variant=model_variant,
                                                            activation=activation)
        

    def forward(self, x, use_sigmoid=False):
        [x2, x3, x4] = self.base_net(x)

        x2 = self.rfb2_1(x2)        # channel -> 32
        x3 = self.rfb3_1(x3)        # channel -> 32
        x4 = self.rfb4_1(x4)        # channel -> 32
        
        out = self.agg1(x4, x3, x2)

        out = F.interpolate(out, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        if use_sigmoid:
            return out.sigmoid()
        else:
            return out