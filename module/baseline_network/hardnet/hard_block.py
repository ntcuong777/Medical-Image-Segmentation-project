import torch
import torch.nn as nn
from .custom_layers import *

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul

        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0

        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch

        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keep_base=False, dwconv=False, activation='relu'):
        super().__init__()
        self.keep_base = keep_base
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)

            if dwconv:
                layers_.append(CombConvLayer(inch, outch, activation=activation))
            else:
                layers_.append(ConvLayer(inch, outch, activation=activation))
          
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch

        self.layers = nn.ModuleList(layers_)

        
    def forward(self, x):
        layers_ = [x]
        
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:            
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
            
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keep_base) or (i == t-1) or (i%2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)

        return out