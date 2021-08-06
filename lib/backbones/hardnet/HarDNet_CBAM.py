import torch.nn as nn
from lib.backbones.hardnet.HarDNet import HarDNet
from lib.backbones.hardnet.modules.cbam_module import CBAM

class HarDNet_CBAM(HarDNet): # Adapter design pattern is great
    def __init__(self, arch=85, depth_wise=False, pretrained_hardnet=True):
        super().__init__(arch=arch, depth_wise=depth_wise)

        if pretrained_hardnet:
            model_name = 'HarDNet' + str(arch) + ('ds' if depth_wise else '')
            self.load_pretrained(model_name=model_name)

        self.cbam_list = nn.ModuleList()

        for i in self.har_d_block_indices:
            out_channels = self.base[i].get_out_ch()
            self.cbam_list.append(CBAM(out_channels))


    def forward(self, x):
        out_branch = []
        cbam_idx = 0
        for i in range(len(self.base)):
            x = self.base[i](x)

            if i in self.har_d_block_indices:
                x = self.cbam_list[cbam_idx](x)
                cbam_idx += 1
            
            if i in self.encoder_block_end_indices[-3:]:
                out_branch.append(x)

        return out_branch