import torch
import torch.nn as nn
from module.segmenter.segmenter_factory import SegmenterFactory

class DoubleUnet(nn.Module):
    def __init__(self):
        # I will use default settings of all class because I want to keep things simple
        self.first_network = SegmenterFactory.create_segmenter_as(baseline_model='HarDMEG')
        self.second_network = SegmenterFactory.create_segmenter_as(baseline_model='DCUnet')
        self.last_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding='same')
    
    def forward(self, inputs, use_sigmoid=False):
        first_net_out, first_net_enc = self.first_network(inputs)
        x = torch.cat((first_net_out, inputs), dim=1) # I will concatenate by depth instead of multiplying
        
        second_net_out = self.second_network(x, first_net_enc)
        out = torch.cat((first_net_out, second_net_out), dim=1) # Concatenate by depth
        out = self.last_conv(out)
        if use_sigmoid:
            return out.sigmoid()
        else:
            return out