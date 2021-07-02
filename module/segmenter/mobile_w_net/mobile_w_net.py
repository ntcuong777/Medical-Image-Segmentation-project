import torch
import torch.nn as nn
from module.baseline_network.mobilenetv3.mobilenetv3 import MobileNetV3
from module.segmenter.hardmseg.HarDMSEG import HarDMSEG
from module.useful_blocks.dw_conv import DWSepConvBlock

class MobileWnet(nn.Module):
    def __init__(self, activation='relu'):
        def double_dw_conv(in_channels, out_channels):
            return nn.Sequential(DWSepConvBlock(in_channels, out_channels, activation=activation),
                                DWSepConvBlock(out_channels, out_channels, activation=activation))

        super(MobileWnet, self).__init__()
        # I will use default settings of all class because I want to keep things simple
        self.first_network = HarDMSEG(activation=activation, w_net_style=True)
        self.second_network = MobileNetV3(True)

        ch1, ch2, ch3, ch4, ch5, ch6 = 32, 64, 128, 40, 112, 960

        # ---- Decoder - Second part of Wnet is Unet-like
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest', align_corners=False)
        self.decoder1 = double_dw_conv(ch6+ch5, 512)
        self.decoder2 = double_dw_conv(ch4+512, 256)
        self.decoder3 = double_dw_conv(ch3+256, 128)
        self.decoder4 = double_dw_conv(ch2+128, 64)
        self.decoder5 = double_dw_conv(ch1+64, 32)
        self.conv_last = DWSepConvBlock(32, 1, activation=activation)


    def forward(self, inputs, use_sigmoid=False):
        first_net_out, first_net_enc = self.first_network(inputs, get_segmentation_result=False)
        first_net_enc_1 = first_net_enc[0]
        first_net_enc_2 = first_net_enc[1]
        first_net_enc_3 = first_net_enc[2]
        
        second_net_out = self.second_network(first_net_out)
        x2 = second_net_out[0]
        x3 = second_net_out[1]
        x4 = second_net_out[2]

        out = self.upsample(x4)
        
        out = torch.cat((out, x3), dim=1)
        out = self.decoder1(out)

        out = torch.cat((out, x2), dim=1)
        out = self.decoder2(out)

        out = torch.cat((out, first_net_enc_3), dim=1)
        out = self.decoder3(out)

        out = torch.cat((out, first_net_enc_2), dim=1)
        out = self.decoder4(out)

        out = torch.cat((out, first_net_enc_1), dim=1)
        out = self.decoder5(out)

        out = self.conv_last(out)
        if use_sigmoid:
            return out.sigmoid()
        else:
            return out