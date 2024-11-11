import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()

        scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", scale)

    def forward(self, x):
        return self.scale * x / (torch.norm(x, 2, dim=-1, keepdim=True) ** (-1.0 / 2))
