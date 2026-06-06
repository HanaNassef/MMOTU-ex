import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
