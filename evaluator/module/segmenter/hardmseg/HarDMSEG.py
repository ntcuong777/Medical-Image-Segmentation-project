import torch
import torch.nn as nn
import torch.nn.functional as F
from module.baseline_network.hardnet import hardnet


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        
        
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
        

    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class HarDMSEG(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(HarDMSEG, self).__init__()
        # ---- ResNet Backbone ----
        #self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.relu = nn.ReLU(True)
        # ---- Receptive Field Block like module ----
        
        self.rfb2_1 = RFB_modified(320, channel)
        self.rfb3_1 = RFB_modified(640, channel)
        self.rfb4_1 = RFB_modified(1024, channel)
        # ---- Partial Decoder ----
        #self.agg1 = aggregation(channel)
        self.agg1 = aggregation(32)
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
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.hardnet = hardnet(arch=68, pretrained=False)
        
    def forward(self, x, use_sigmoid=True):        
        hardnetout = self.hardnet(x)
        
        x2 = hardnetout[0]
        x3 = hardnetout[1]
        x4 = hardnetout[2]
        
        x2 = self.rfb2_1(x2)        # channel -> 32
        x3 = self.rfb3_1(x3)        # channel -> 32
        x4 = self.rfb4_1(x4)        # channel -> 32
        
        out = self.agg1(x4, x3, x2)
        
        out = F.interpolate(out, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        if use_sigmoid:
            return out.sigmoid()
        else:
            return out


    def delete_unnecessary_layers(self):
        """
            Actually, the author define a lot of layers that are unnecessary.
            Thus, I make this method to delete all of them to free memory after
            loading the pretrained model. I wanted to refactor this code, but
            it turns out to be really challenging...

            You might find that it might be not that challenging, we can refactor
            the code and then train the model from scratch. However, the situation
            is actually a little different. I tried training the model from scratch
            but I cannot get the same accuracy as the author reported. However, the
            author's pretrained weights is actually very good. So, I make this function
            just in case we are allowed to use the author's pretrained weights for
            the contest or something :)
        """
        self.hardnet.delete_classification_head()
        del self.ra4_conv1, self.ra4_conv2, self.ra4_conv3, self.ra4_conv4, self.ra4_conv5
        del self.ra3_conv1, self.ra3_conv2, self.ra3_conv3, self.ra3_conv4
        del self.ra2_conv1, self.ra2_conv2, self.ra2_conv3, self.ra2_conv4, self.conv2
        del self.conv3, self.conv4, self.conv5, self.conv6, self.upsample