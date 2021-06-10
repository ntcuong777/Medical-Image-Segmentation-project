import torch
import torch.nn as nn
import torch.nn.functional as F
from module.baseline_network.baseline_factory import BaselineFactory
from .useful_blocks.aggregation import Aggregation
from .useful_blocks.rfb_modified import RFB_modified
from .useful_blocks.basic_conv2d import BasicConv2d

class HarDMSEG(nn.Module):
    # res2net based encoder decoder
    def __init__(self, model_variant='HarDNet68ds', activation='relu', channel=32, use_attention=False):
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
        
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(1024, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(640, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(320, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(320, 32, kernel_size=1)
        self.conv3 = BasicConv2d(640, 32, kernel_size=1)
        self.conv4 = BasicConv2d(1024, 32, kernel_size=1)
        self.conv5 = BasicConv2d(1024, 1024, 3, padding=1)
        self.conv6 = nn.Conv2d(1024, 1, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        
        # ---- define baseline network ----
        self.base_net = BaselineFactory.create_baseline_as(baseline_model='hardnet', model_variant=model_variant,
                                                            use_attention=use_attention, activation=activation)
        

    def forward(self, x):
        #print("input",x.size())
        
        base_net_out = self.base_net(x)
        
        x2 = base_net_out[0]
        x3 = base_net_out[1]
        x4 = base_net_out[2]
        
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32
        
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)

        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5 #, lateral_map_4, lateral_map_3, lateral_map_2
