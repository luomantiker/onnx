import torch
import torch.nn as nn


class ModelTorch(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y
