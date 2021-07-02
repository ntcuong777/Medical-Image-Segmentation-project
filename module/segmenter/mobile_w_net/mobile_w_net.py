import torch
import torch.nn as nn
from module.baseline_network.mobilenetv3.mobilenetv3 import MobileNetV3
from module.segmenter.hardmseg.HarDMSEG import HarDMSEG
from module.useful_blocks.dw_conv import DWSepConvBlock

"""
The class below is Unet & Double-Unet (UUnet) inspired Wnet.
*Note* that W has low middle point, unlike UU, which has high middle point.
  In this Wnet model, first, I use HarDMSEG to get the decoded feature map (code from author, minor modification)
  of size (original_size / 2^3). Then, I encode that feature map again with the 3 last encoder blocks of MobileNetV3
  (without the classification head). The decoder of the second part of Wnet has the same skip connections as Unet
  with the concat "feature fusion operator" (Eh hmm, Chinese words here). Unlike HarDMSEG's decoder, whose author
  say that they use Cascade Partial Decoder because it yields good result.
  
  In addition, even though I place activation='relu' here, but I prefer 'hard_swish' because it's better in
  optimizing model parameters (I set activation='hard_swish' by default in factory classes, so the default
  here will have no effect).

FAQ:
  Q: Why Mobile in the name?
  A: Because I make all convolution layers are depthwise separable by default.
    Dont go search for it. However, if you really want to, use VSCode to search
    some keywords such as depth_wise, dw_conv, or dw. You will see that most of
    them are function parameters with default value = True. I cannot figure out
    how to refactor this thing. Sorry :(
"""
class MobileWnet(nn.Module):
    def __init__(self, activation='relu'):
        def double_dw_conv(in_channels, out_channels):
            return nn.Sequential(DWSepConvBlock(in_channels, out_channels, activation=activation),
                                DWSepConvBlock(out_channels, out_channels, activation=activation))

        super(MobileWnet, self).__init__()
        # I will use default settings of all class because I want to keep things simple
        self.first_network = HarDMSEG(activation=activation, w_net_style=True)
        self.second_network = MobileNetV3(True)

        _, ch2, ch3, ch4, ch5, ch6 = 32, 64, 128, 40, 112, 960

        # ---- Decoder - Second part of Wnet is Unet-like
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder1 = double_dw_conv(ch6+ch5, 512)
        self.decoder2 = double_dw_conv(ch4+512, 256)
        self.decoder3 = double_dw_conv(ch3+256, 128)
        self.decoder4 = double_dw_conv(ch2+128, 64)
        self.decoder5 = double_dw_conv(64, 32)
        self.conv_last = DWSepConvBlock(32, 1, activation=activation)


    def forward(self, inputs, use_sigmoid=False):
        out, [first_net_enc_1, first_net_enc_2] = self.first_network(inputs, get_segmentation_result=False)
        
        [x2, x3, x4] = self.second_network(out)

        out = self.upsample(x4)
        
        out = torch.cat((out, x3), dim=1)
        out = self.decoder1(out)
        out = self.upsample(out)

        out = torch.cat((out, x2), dim=1)
        out = self.decoder2(out)
        out = self.upsample(out)

        out = torch.cat((out, first_net_enc_2), dim=1)
        out = self.decoder3(out)
        out = self.upsample(out)

        out = torch.cat((out, first_net_enc_1), dim=1)
        out = self.decoder4(out)
        out = self.upsample(out)

        # There is no encoder layer with the same dimension as the input due to
        # HarDNet's design, no concat operator here, sadly. Specifically, HarDNet's
        # first conv layer is also a downsampling layer, so I cannot add a skip
        # connection to the last decoder block. However, Unet, whose input
        # will be transformed by some conv layer with approximately the same size as
        # input before the first downsampling layer. Thus, they can add a skip connection
        # at the top (the highest arrow - literally, horizontally - in there model).
        #
        # For more descriptive image, see Unet's architecture, and delete the highest gray
        # arrow & the first 2 convolution layers. You will know what I'm talking about
        # righaway.
        out = self.decoder5(out)
        out = self.conv_last(out)

        if use_sigmoid:
            return out.sigmoid()
        else:
            return out