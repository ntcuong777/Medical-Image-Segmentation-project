from typing import Counter
import torch
import torch.nn as nn
from .custom_layers import *
from .hard_block import *
import os

weight_paths = {
    "HarDNet39ds": "weights/hardnet39ds.pth",
    "HarDNet68": "weights/hardnet68.pth",
    "HarDNet68ds": "weights/hardnet68ds.pth",
    "HarDNet85": "weights/hardnet85.pth"
}

class HarDNet(nn.Module):
    def __init__(self, model_name, activation='relu'):
        super().__init__()

        assert model_name in ['HarDNet39ds', 'HarDNet68ds', 'HarDNet68', 'HarDNet85']

        first_ch  = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = 0.1
        depth_wise = model_name.endswith('ds') # 'ds' denotes depthwise separable is used

        self.arch = 68 # Cuong need to save the `arch`
        if model_name == 'HarDNet39ds':
            self.arch = 39
        elif model_name == 'HarDNet85':
            self.arch = 85

        self.har_d_block_indices = []
        self.encoder_block_end_indices = [] # save the indices of the layer which has the last output
                                            # of each encoder block before the downsampling layer next to it

        ######## CONFIGURATION PARAMETERS OF DIFFERENT VARIANTS ########
        # HarDNet68
        ch_list = [  128, 256, 320, 640, 1024]
        gr       = [  14, 16, 20, 40,160]
        n_layers = [   8, 16, 16, 16,  4]
        downSamp = [   1,  0,  1,  1,  0]
        
        if self.arch == 85:
            # HarDNet85
            first_ch  = [48, 96]
            ch_list = [  192, 256, 320, 480, 720, 1280]
            gr       = [  24,  24,  28,  36,  48, 256]
            n_layers = [   8,  16,  16,  16,  16,   4]
            downSamp = [   1,   0,   1,   0,   1,   0]
            drop_rate = 0.2
        elif self.arch == 39:
            # HarDNet39
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


        ######## BEGIN BUILDING HarDNet ########
        blks = len(n_layers)
        self.base = nn.ModuleList([])

        count_layers = 0
        # First Encoder block: Standard Conv3x3, Stride=2
        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                            stride=2, activation=activation, bias=False))
        count_layers += 1
        self.encoder_block_end_indices.append(count_layers - 1)
  
        # Second Encoder block
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel, activation=activation))
        count_layers += 1
        self.encoder_block_end_indices.append(count_layers - 1)

        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append ( DWConvLayer(first_ch[1], first_ch[1], stride=2) )

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise, activation=activation)
            ch = blk.get_out_ch()
            self.base.append ( blk )
            count_layers += 1
            self.har_d_block_indices.append(count_layers - 1)
            
            if i == blks-1 and self.arch == 85:
                self.base.append ( nn.Dropout(0.1))
            
            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1, activation=activation) )
            count_layers += 1
            ch = ch_list[i]
            if downSamp[i] == 1:
                self.encoder_block_end_indices.append(count_layers - 1)
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append ( DWConvLayer(ch, ch, stride=2) )
            
        
        ch = ch_list[blks-1]
        self.base.append (
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(ch, 1000) ))
        

        #### DONE DEFINING NETWORK - STARTS LOADING SAVED WEIGHTS ####
        WORKDIR = os.getcwd()

        # Changing the workdir is necessary to load the saved HarDNet weights
        TEMPDIR = os.path.join(os.getcwd(), 'module/baseline_network/hardnet')
        os.chdir(TEMPDIR)

        self.load_state_dict(torch.load(weight_paths[model_name]))

        os.chdir(WORKDIR) # back to working directory


    def forward(self, x):
        out_branch =[]
        for i in range(len(self.base)-1):
            x = self.base[i](x)

            # HarDMSEG need the output of the last 3 encoder blocks for further processing
            # I need all encoder blocks to append attention layer
            if i in self.encoder_block_end_indices[len(self.encoder_block_end_indices) - 3:len(self.encoder_block_end_indices)]:
                out_branch.append(x)

        return out_branch
    

def load_hardnet_baseline(arch=68, depth_wise=False, pretrained=True):
    if arch == 39:
        raise NotImplementedError("HarDNet39ds is not supported!")
    elif arch == 68:
        print("Loading HarDNet68{:s}".format("ds" if depth_wise else ""))
        model = HarDNet(arch=68, depth_wise=depth_wise)
        if pretrained:
            if not depth_wise:
                weights = torch.load(weight_paths['HarDNet68'])
            else:
                weights = torch.load(weight_paths['HarDNet68ds'])

            model.load_state_dict(weights)
            print("68 LOADED READY")
    elif arch == 85:
        print("85 LOADED")
        model = HarDNet(arch=85)
        if pretrained:
            weights = torch.load(weight_paths['HarDNet85'])
            model.load_state_dict(weights)

    return model
