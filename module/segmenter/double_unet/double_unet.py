import torch
import torch.nn as nn
from module.segmenter.dcunet.DCUnet import DcUnet
from module.segmenter.hardmseg.HarDMSEG import HarDMSEG

class DoubleUnet(nn.Module):
    def __init__(self):
        super(DoubleUnet, self).__init__()
        # I will use default settings of all class because I want to keep things simple
        self.first_network = HarDMSEG(activation='relu', use_attention=False)
        self.second_network = DcUnet(input_channels=4)
        self.last_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
    
    def forward(self, inputs, use_sigmoid=False):
        # first_net_out, first_net_enc = self.first_network(inputs, double_unet_style=True)
        first_net_out = self.first_network(inputs, double_unet_style=False)
        x = torch.cat((first_net_out, inputs), dim=1) # I will concatenate by depth instead of multiplying
        
        # second_net_out = self.second_network(x, first_net_enc)
        second_net_out = self.second_network(x)
        out = torch.cat((first_net_out, second_net_out), dim=1) # Concatenate by depth
        out = self.last_conv(out)
        if use_sigmoid:
            return out.sigmoid()
        else:
            return out