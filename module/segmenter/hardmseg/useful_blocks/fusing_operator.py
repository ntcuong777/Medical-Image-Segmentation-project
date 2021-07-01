import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAddition(nn.Module):
    def __init__(self):
        super(LinearAddition, self).__init__()

        self.w = nn.Parameter(torch.rand(2)) # Learnable linear weights

    def forward(self, tensor1, tensor2):
        return self.w[0] * tensor1 + w[1] * self.tensor2 # Learnable linear addtion operator