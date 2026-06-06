import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE loss with label smoothing (without reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Get true class probabilities
        pt = torch.exp(-ce_loss)
        
        # Focal factor
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            focal_term = alpha_t * focal_term
            
        loss = focal_term * ce_loss
        return loss.mean()
