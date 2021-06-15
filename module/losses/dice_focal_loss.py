from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    def __init__(self): # temporarily, no parameters
        super(DiceFocalLoss, self).__init__()
        self.my_focal = FocalLoss()
        self.my_dice = DiceLoss()
    
    def forward(self, inputs, targets):
        return self.my_focal(inputs, targets) + self.my_dice(inputs, targets)