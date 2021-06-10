import torch.nn as nn
from module.baseline_network.hardnet.hardnet import HarDNet
from module.baseline_network.hardnet_cbam.CBAM.cbam import CBAM

class HarDNet_CBAM(HarDNet): # Adapter design pattern is great
    def __init__(self, hardnet_model_name, activation='relu'):
        super().__init__(model_name=hardnet_model_name, activation=activation)

        self.cbam_list = nn.ModuleList()

        for i in self.har_d_block_indices:
            out_channels = self.base[i].get_out_ch()
            self.cbam_list.append(CBAM(out_channels))
    
    def forward(self, x):
        out_branch = []
        cbam_idx = 0
        for i in range(len(self.base) - 1):
            x = self.base[i](x)

            if i in self.har_d_block_indices:
                x = self.cbam_list[cbam_idx](x)
                cbam_idx += 1
            
            if i in self.encoder_block_end_indices[len(self.encoder_block_end_indices) - 3:len(self.encoder_block_end_indices)]:
                out_branch.append(x)
        
        return out_branch