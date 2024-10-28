import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, n_dim, inter_dim):
        super().__init__()

        self.linear1 = nn.Linear(n_dim, inter_dim)
        self.linear2 = nn.Linear(n_dim, inter_dim)
        self.silu = nn.SiLU()


    def forward(self, x):
        b, length, n_dim = x.shape

        return self.silu(self.linear1(x)) * self.linear2(x)
