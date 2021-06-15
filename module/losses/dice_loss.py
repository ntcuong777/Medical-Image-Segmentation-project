import torch.nn as nn
import torch.nn.function as F

class DiceLoss(nn.Module):
    def __init__(self, apply_sigmoid=True, smooth=1e-5, weight=None, reduce=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        if not (weight is None):
            self.weight = weight
        self.reduce = reduce
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.apply_sigmoid:
            inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  

        if not self.reduce:
            return 1 - dice
        else:
            return (1 - dice).mean()