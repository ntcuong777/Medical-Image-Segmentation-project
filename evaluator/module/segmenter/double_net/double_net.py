import torch
import torch.nn as nn
from module.segmenter.hardmseg import HarDMSEG
from module.segmenter.uaca_net import UACANet
from easydict import EasyDict as ed
import yaml

class DoubleNet(nn.Module):
    """
        This class basically get the output from HarDMSEG. Then, it get the outputs from 2 models of UACANet.
        The final output will be based on the voting results. The voting works as follows:
            - If at least 2 models classify a pixel (x, y) as 1, the (x, y) pixel in final output will be set to 1
            - Else, 0.
        This is actually a heuristic. This module will not be trained as HarDMSEG & UACANet will be trained
        separately.
    """
    def __init__(self, pretrained_hardmseg='snapshots/HarDMSEG/best.pth'):
        super(DoubleNet, self).__init__()

        assert pretrained_hardmseg is not None, 'Must provide HarDMSEG weights to DoubleNet'

        self.first_network = HarDMSEG()
        self.first_network.load_state_dict(torch.load(pretrained_hardmseg))
        self.first_network.delete_unnecessary_layers() # Delete unused layers to free memory
        for param in self.first_network.parameters():
            param.requires_grad = False # freeze HarDMSEG
        self.first_network.eval()

        uaca_opt = ed(yaml.load(open('config/uaca_config/UACANet-L.yaml'), yaml.FullLoader))
        self.second_network = UACANet(uaca_opt.Model)
        self.second_network.load_state_dict(torch.load(uaca_opt.Test.pth_path))
        for param in self.second_network.parameters():
            param.requires_grad = False # freeze UACANet-L
        self.second_network.eval()

        uaca_opt = ed(yaml.load(open('config/uaca_config/UACANet-S.yaml'), yaml.FullLoader))
        self.third_network = UACANet(uaca_opt.Model)
        self.third_network.load_state_dict(torch.load(uaca_opt.Test.pth_path))
        for param in self.third_network.parameters():
            param.requires_grad = False # freeze UACANet-S
        self.third_network.eval()


    def forward(self, inputs, threshold=0.5):
        out1 = self.first_network.forward(inputs, use_sigmoid=True)
        out1 = (out1 > threshold).float()
        out2 = self.second_network(inputs, use_sigmoid=True)
        out2 = (out2 > threshold).float()
        out3 = self.third_network(inputs, use_sigmoid=True)
        out3 = (out3 > threshold).float()

        out = out1 + out2 + out3
        out = (out > 1).float() # Voting result
        return out