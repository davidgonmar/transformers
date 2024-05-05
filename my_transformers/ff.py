import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, inner_dim: int):
        """
        Args:
            d_model: dimensionality of the input and output
            inner_dim: dimensionality of the inner layer, also called d_ff in the paper
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, inner_dim)
        self.fc2 = nn.Linear(inner_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(x)))
