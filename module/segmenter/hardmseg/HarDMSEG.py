import torch.nn as nn
import torch.nn.functional as F
from module.baseline_network.baseline_factory import BaselineFactory
from module.useful_blocks.aggregation import Aggregation
from module.useful_blocks.rfb_modified import RFB_modified

class HarDMSEG(nn.Module):
    # res2net based encoder decoder
    def __init__(self, model_variant='HarDNet68ds', activation='relu', channel=32, use_attention=False, w_net_style=False):
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
        self.agg1 = Aggregation(channel, w_net_style=w_net_style)

        # ---- define baseline network ----
        self.base_net = BaselineFactory.create_baseline_as(baseline_model='hardnet', model_variant=model_variant,
                                                            use_attention=use_attention, activation=activation)
        

    def forward(self, x, get_segmentation_result=True):
        #print("input",x.size())
        
        base_net_out = self.base_net(x)
        
        # _ = base_net_out[0]
        x2 = base_net_out[-3]
        x3 = base_net_out[-2]
        x4 = base_net_out[-1]
        
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32
        
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)

        if get_segmentation_result:
            lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        if get_segmentation_result:
            return lateral_map_5 #, lateral_map_4, lateral_map_3, lateral_map_2
        else:
            return lateral_map_5, base_net_out