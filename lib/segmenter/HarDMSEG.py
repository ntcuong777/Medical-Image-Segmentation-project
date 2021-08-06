import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.backbones.hardnet import get_hardnet_baseline
from lib.segmenter.modules.hardmseg_custom_layers import aggregation, RFB_modified
from lib.losses import get_loss_fn

class HarDMSEG(nn.Module):
    def __init__(self, opt):
        """
            Init HarDMSEG framework.
            channel: author's default parameter.
            load_pretrained_baseline: If you want to train the model and initialize the baseline network
                with ImageNet pretrained weights, set this flag to `True`. If you want to load entire HarDMSEG
                pretrained model for inference, set this flag to `False`.
        """
        super(HarDMSEG, self).__init__()

        channel = opt.decoder_channel

        arch = opt.baseline_model.arch

        # ---- Receptive Field Block like module ----
        ch1, ch2, ch3 = 320, 720, 1280 # iff arch == 85 only
        if arch == 68 or arch == 39:
            ch1, ch2, ch3 = 320, 640, 1024

        self.rfb2_1 = RFB_modified(ch1, channel)
        self.rfb3_1 = RFB_modified(ch2, channel)
        self.rfb4_1 = RFB_modified(ch3, channel)

        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)

        if opt.pretrained_hardmseg: # no need to load the baseline pretrained weights if we want to load whole HarDMSEG weights
            opt.baseline_model.pretrained = False

        self.hardnet = get_hardnet_baseline(opt.baseline_model)

        if opt.pretrained_hardmseg:
            self.load_state_dict(torch.load(opt.pretrained_path))

        self.loss_fn = get_loss_fn(opt.loss_fn)


    def forward(self, x, targets=None):
        [x2, x3, x4] = self.hardnet(x)

        x2 = self.rfb2_1(x2)        # channel -> 32
        x3 = self.rfb3_1(x3)        # channel -> 32
        x4 = self.rfb4_1(x4)        # channel -> 32
        
        out = self.agg1(x4, x3, x2)
        
        out = F.interpolate(out, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        loss = 0
        if targets is not None:
            loss = self.loss_fn(out, targets)

        return {'pred': out, 'loss': loss}