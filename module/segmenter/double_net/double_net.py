from config import TrainConfig
import torch
import torch.nn as nn
from module.segmenter.hardmseg import HarDMSEG
from module.segmenter.medical_transformer import MedT

class DoubleNet(nn.Module):
    def __init__(self, img_size=512, img_channels=3, hardnet_channel=32, pretrained_hardmseg=None):
        super(DoubleNet, self).__init__()

        # Try to use 64 channels for RFB module to see if it will improve
        self.first_network = HarDMSEG(activation='relu', channel=hardnet_channel)
        if pretrained_hardmseg is not None:
            self.first_network.load_state_dict(torch.load(pretrained_hardmseg))
            for param in self.first_network.parameters():
                param.requires_grad = False # freeze HarDMSEG

        # Input is concatenated with the segmentation mask depthwise. Thus, imgchan = img_channels + 1
        self.second_network = MedT(img_size=img_size, imgchan=img_channels+1)


    def forward(self, inputs, use_sigmoid=False):
        out = self.first_network.forward(inputs, use_sigmoid=True)
        out = torch.cat((inputs, out), dim=1) # Concatenate inputs and out depthwise
        out = self.second_network(out, use_sigmoid=use_sigmoid)
        return out
    

    def set_requires_grad_medt(self, value=True):
        for param in self.second_network.parameters():
            param.requires_grad = True