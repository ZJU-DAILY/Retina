import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss

class CERerankerLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, logits, label):
        
        loss = self.ce_loss(logits, label)
        return loss