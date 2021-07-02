import torch.nn as nn
from module.activations.mish.mish import Mish

class DWSepConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 3, stride = 1,
                padding = 1, dilation = 1, activation='relu', bias=False):
        super(DWSepConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, groups=input_channels, bias=bias)
        self.pointwise = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
        if activation == 'relu':
            self.activation = nn.ReLU(True)
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'hard_swish':
            self.activation = nn.HardSwish(True)
        else:
            raise NotImplementedError("Activation not implemented")
        self.input_channels = input_channels
        self.output_channels = output_channels


    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.activation(out)
        return out