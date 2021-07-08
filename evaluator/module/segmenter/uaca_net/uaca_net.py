import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .losses import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s

class UACANet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, opt):
        super(UACANet, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=opt.pretrained, output_stride=opt.output_stride)

        self.context2 = PAA_e(512, opt.channel)
        self.context3 = PAA_e(1024, opt.channel)
        self.context4 = PAA_e(2048, opt.channel)

        self.decoder = PAA_d(opt.channel)

        self.attention2 = UACA(opt.channel * 2, opt.channel)
        self.attention3 = UACA(opt.channel * 2, opt.channel)
        self.attention4 = UACA(opt.channel * 2, opt.channel)

        self.loss_fn = bce_iou_loss

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)


    def forward(self, x, use_sigmoid=True):
        base_size = x.shape[-2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)

        # This class will be used for inference, this means that
        # the output of out5, out4, and out3 are not necessary so
        # I deleted them 
        f5, a5 = self.decoder(x4, x3, x2)

        f4, a4 = self.attention4(torch.cat([x4, self.ret(f5, x4)], dim=1), a5)

        f3, a3 = self.attention3(torch.cat([x3, self.ret(f4, x3)], dim=1), a4)

        _, a2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), a3)
        out2 = self.res(a2, base_size)

        if use_sigmoid:
            return out2.sigmoid()
        else:
            return out2