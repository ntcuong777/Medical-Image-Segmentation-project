from typing import Container
import torch.nn as nn
from module.activations.mish.mish import Mish

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, activation=None, dilation=1):
        super(BasicConv2d, self).__init__()
        
        
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

        self.contains_activation = False
        if not (activation is None):
            if activation == 'relu':
                self.activation = nn.ReLU(inplace=True)
            elif activation == 'mish':
                self.activation = Mish()
            else:
                raise NotImplementedError("Activation is not supported.")

            self.contains_activation = True


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.contains_activation:
            x = self.activation(x)
        
        return x