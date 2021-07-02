import torch.nn as nn
from torchvision.models import mobilenet_v3_large

# MobileNetV3's input size in each layer, anything that is not the
# image size is the index of the layer here (as presented in a table
# in their paper). For example, [0, 1] means layer 0 to layer 1.
#
# No reduction: Input img (512x512)
# Reduced by 2: [0, 1] (512x512) at 0 but size is (256x256) after 0
# Reduced by 2^2: [2, 3] (256x256) at 2 but size is (128x128) after 2
# Reduced by 2^3: [4, 6] (128x128) at 4 but size is (64x64) after 4
# Reduced by 2^4: [7, 12] (64x64) at 7 but size is (32x32) after 7
# Reduced by 2^6: [13, 16] (32x32) at 13 but size is (16x16) after 13

"""
As the name suggest, I wrap the MobileNetV3-Large model in this class :)

Basically, I have eliminated the classification head which is not
  needed. Also, the `get_only_3_last_encoder` is a parameter to tell
  the class to use the whole MobileNetV3 model as backbone or not.
  `False` means that the input will be forwarded over all conv layers.
  Otherwise, the input will only be forwarded through the last 3 encoder
  blocks (forwarded through the last 3 green lines above, except the layer 4).
  The whole idea of forwarding through only 3 last encoder blocks serves
  to build the Wnet architecture (with low middle tipping point).

I initialize the conv layers' weights with the pretrained weights.

"""
class MobileNetV3(nn.Module):
    def __init__(self, get_only_3_last_encoder=False):
        super(MobileNetV3, self).__init__()
        self.get_only_3_last_encoder = get_only_3_last_encoder
        
        baseline = mobilenet_v3_large(pretrained=True).features # Omit the classification layer
        if get_only_3_last_encoder:
            # get the non-linear transformation after the 3rd downsampling layer
            self.enc_1 = nn.Sequential(*[baseline[i] for i in range(5, 7)])
            self.enc_2 = nn.Sequential(*[baseline[i] for i in range(7, 13)])
            self.enc_3 = nn.Sequential(*[baseline[i] for i in range(13, 17)])
        else:
            self.enc_1 = nn.Sequential(*[baseline[i] for i in range(0, 2)])
            self.enc_2 = nn.Sequential(*[baseline[i] for i in range(2, 4)])
            self.enc_3 = nn.Sequential(*[baseline[i] for i in range(4, 7)])
            self.enc_4 = nn.Sequential(*[baseline[i] for i in range(7, 13)])
            self.enc_5 = nn.Sequential(*[baseline[i] for i in range(13, 17)])

    def forward(self, x):
        if not self.get_only_3_last_encoder:
            out1 = self.enc_1(x)
            out2 = self.enc_2(out1)
            out3 = self.enc_3(out2)
            out4 = self.enc_4(out3)
            out5 = self.enc_5(out4)
            return [out1, out2, out3, out4, out5] # Return all encoder results for Unet-like architecture
        else:
            out1 = self.enc_1(x)
            out2 = self.enc_2(out1)
            out3 = self.enc_3(out2)
            return [out1, out2, out3]