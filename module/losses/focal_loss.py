import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, apply_logits=True, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.apply_logits = apply_logits

    def forward(self, inputs, targets):
        BCE_loss = None
        if self.apply_logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()