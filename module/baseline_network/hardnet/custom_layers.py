import torch.nn as nn
from module.activations.mish.mish import Mish

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.data.size(0),-1)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, activation='relu', bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))

        if activation == 'relu':
            self.add_module('relu', nn.ReLU6(True))
        elif activation == 'mish':
            self.add_module('mish', Mish())
        else:
            raise NotImplementedError("Activation not supported")

    def forward(self, x):
        return super().forward(x)


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, activation='relu', bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel, activation=activation, bias=bias))
        self.add_module('layer2', DWConvLayer(out_channels, stride=stride))
        
    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, kernel=3, stride=1, bias=False):
        super().__init__()
        
        groups = in_channels
        
        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=kernel,
                                          stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))


    def forward(self, x):
        return super().forward(x)