import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.losses import get_loss_fn
from lib.segmenter.modules.layers import *
from lib.segmenter.modules.context_module import *
from lib.segmenter.modules.attention_module import *
from lib.segmenter.modules.decoder_module import *

from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s

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

        self.loss_fn = get_loss_fn(opt.loss_fn)

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x, y=None):
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

        f5, a5 = self.decoder(x4, x3, x2)
        out5 = self.res(a5, base_size)

        f4, a4 = self.attention4(torch.cat([x4, self.ret(f5, x4)], dim=1), a5)
        out4 = self.res(a4, base_size)

        f3, a3 = self.attention3(torch.cat([x3, self.ret(f4, x3)], dim=1), a4)
        out3 = self.res(a3, base_size)

        _, a2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), a3)
        out2 = self.res(a2, base_size)


        if y is not None:
            loss5 = self.loss_fn(out5, y)
            loss4 = self.loss_fn(out4, y)
            loss3 = self.loss_fn(out3, y)
            loss2 = self.loss_fn(out2, y)

            loss = loss2 + loss3 + loss4 + loss5
        else:
            loss = 0

        return {'pred': out2, 'loss': loss}