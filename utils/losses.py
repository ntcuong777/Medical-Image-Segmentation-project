"""
    This source code is borrowed from https://github.com/Latterlig96/DCUnet/blob/main/loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init_()
    
    def forward(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
    
        return (wbce + wiou).mean()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor, 
                smooth: int=1
                ) -> torch.Tensor:
        
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self,
                inputs: torch.Tensor, 
                targets: torch.Tensor, 
                smooth: int=1
                ) -> torch.Tensor:
        
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor, 
                alpha: int=0.8, 
                gamma: int=2
                ) -> torch.Tensor:
        
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


class DiceFocalLoss(nn.Module):
    def __init__(self): # temporarily, no parameters
        super(DiceFocalLoss, self).__init__()
        self.my_focal = FocalLoss()
        self.my_dice = DiceLoss()
    
    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor
                ) -> torch.Tensor:
    
        return self.my_focal(inputs, targets) + self.my_dice(inputs, targets)


class TverskyLoss(nn.Module):
    def __init__(self):
        super(TverskyLoss, self).__init__()

    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor, 
                smooth: int=1, 
                alpha: int=0.75
                ) -> torch.Tensor:
        
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FN + (1-alpha)*FP + smooth)  
        
        return 1 - Tversky

class FocalTverskyLoss(nn.Module):
    def __init__(self):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor, 
                smooth: int=1, 
                alpha: int=0.75, 
                gamma: int=0.75
                ) -> torch.Tensor:
        
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FN + (1-alpha)*FP + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky