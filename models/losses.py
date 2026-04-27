import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    def __init__(self, l1_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target):
        return self.l1_weight * self.l1(pred, target) + \
               (1 - self.l1_weight) * self.l2(pred, target)
