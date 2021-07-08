import os
import torch
import torch.nn as nn
import torch.nn.functional as F

hardnet_weight_paths = {
    "HarDNet39ds": "weights/hardnet39ds.pth",
    "HarDNet68": "weights/hardnet68.pth",
    "HarDNet68ds": "weights/hardnet68ds.pth",
    "HarDNet85": "weights/hardnet85.pth"
}

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)
    
class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,          
                                          stride=stride, padding=kernel//2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))                                          
    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  stride=1,  bias=False):
        super().__init__()
        out_ch = out_channels
        
        groups = in_channels
        kernel = 3
        #print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')
        
        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                          stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))
    def forward(self, x):
        return super().forward(x)  


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1',ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2',DWConvLayer(out_channels, out_channels, stride=stride))
        
    def forward(self, x):
        return super().forward(x)


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

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          if dwconv:
            layers_.append(CombConvLayer(inch, outch))
          else:
            layers_.append(ConvLayer(inch, outch))
          
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
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
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out

        


# +
class HarDNet(nn.Module):
    def __init__(self, depth_wise=False, arch=85, pretrained=True, weight_path=''):
        super().__init__()
        first_ch  = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = 0.1
        
        #HarDNet68
        ch_list = [  128, 256, 320, 640, 1024]
        gr       = [  14, 16, 20, 40,160]
        n_layers = [   8, 16, 16, 16,  4]
        downSamp = [   1,  0,  1,  1,  0]
        
        if arch==85:
          #HarDNet85
          first_ch  = [48, 96]
          ch_list = [  192, 256, 320, 480, 720, 1280]
          gr       = [  24,  24,  28,  36,  48, 256]
          n_layers = [   8,  16,  16,  16,  16,   4]
          downSamp = [   1,   0,   1,   0,   1,   0]
          drop_rate = 0.2
        elif arch==39:
          #HarDNet39
          first_ch  = [24, 48]
          ch_list = [  96, 320, 640, 1024]
          grmul = 1.6
          gr       = [  16,  20, 64, 160]
          n_layers = [   4,  16,  8,   4]
          downSamp = [   1,   1,  1,   0]
          
        if depth_wise:
          second_kernel = 1
          max_pool = False
          drop_rate = 0.05
        
        blks = len(n_layers)
        self.base = nn.ModuleList([])

        count_layers = 0
        # First Layer: Standard Conv3x3, Stride=2
        self.base.append (
             ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                       stride=2,  bias=False) )
        count_layers += 1 # one layer added
  
        # Second Layer
        self.base.append ( ConvLayer(first_ch[0], first_ch[1],  kernel=second_kernel) )
        count_layers += 1 # one layer added
        self.encoder_block_end_indices.append(count_layers - 1)
        
        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append ( DWConvLayer(first_ch[1], first_ch[1], stride=2) )
        count_layers += 1 # one layer added

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append ( blk )
            count_layers += 1 # one layer added - one HarDBlock
            
            if i == blks-1 and arch == 85:
                self.base.append ( nn.Dropout(0.1))
                count_layers += 1 # dropout is added

            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1) )
            count_layers += 1 # one layer is added

            ch = ch_list[i]
            if downSamp[i] == 1:
                self.encoder_block_end_indices.append(count_layers - 1)

                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append ( DWConvLayer(ch, ch, stride=2) )
                count_layers += 1 # one layer is added - either dwconv or maxpool
            
        
        # This is the end of the baseline, the next layers are classification layers for ImageNet
        #
        # After the last [HarDBlock - Conv] block, there is no downsampling layer, but this block
        # is still considered as an encoder block. Here, I consider an encoder block as a list of
        # layers that have the same output size (counting even the downsampler). See below for
        # illustration.
        #
        # e.g.: [1st_downsampler (out_size = 256x256) -> some_conv_layer_or_block_2 (out_size = 256x256)]
        #       -> [2nd_downsampler (out_size = 128x128) -> some_conv_layer_or_block_2 (out_size = 128x128)]
        # Here, the [] pair denotes an encoder block as considered by me.
        #
        # The reason for this is because of the properties of Cascade Partial Decoder module.
        # That module needs the output of the 3 previous encoder blocks so I have the save the
        # indices where an encoder block ends so that I can save its output & return it to the
        # main pipeline which uses Cascade Partial Decoder.
        self.encoder_block_end_indices.append(count_layers - 1)

        ch = ch_list[blks-1]
        self.base.append (
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(ch, 1000) ))


    def forward(self, x):
        # Assume arch=68
        out_branch =[]
        for i in range(len(self.base)):
            x = self.base[i](x)

            # HarDMSEG need the output of the last 3 encoder blocks for further processing
            # I need all encoder blocks to append attention layer
            if i in self.encoder_block_end_indices[-3:]:
                out_branch.append(x)

        return out_branch
    

    def delete_classification_head(self):
        self.base = nn.ModuleList([self.base[i] for i in range(len(self.base) - 1)])


def hardnet(arch=68, pretrained=True, depth_wise=False, **kwargs):
    model = HarDNet(arch=arch, depth_wise=depth_wise, **kwargs)
    if pretrained:
        #### DONE DEFINING NETWORK - STARTS LOADING SAVED WEIGHTS ####
        WORKDIR = os.getcwd()

        # Changing the workdir is necessary to load the saved HarDNet weights
        TEMPDIR = os.path.join(os.getcwd(), 'module/baseline_network/hardnet')
        os.chdir(TEMPDIR)

        model_variant = 'HarDNet68'
        if arch==39:
            model_variant = 'HarDNet39ds'
        elif arch==85:
            model_variant = 'HarDNet85'
        elif arch==68 and depth_wise:
            model_variant = 'HarDNet68ds'

        weights = torch.load(hardnet_weight_paths[model_variant])

        os.chdir(WORKDIR) # back to working directory

        model.load_state_dict(weights)

        model.delete_classification_head() # remove the classification head to free memory

    return model
