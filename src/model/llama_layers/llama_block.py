import torch
import xformers.ops as xops
import numpy as np

from torch import nn
from .swish import SwiGLU


class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, d_model, d_head, seq_len):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_head)
        self.w_k = nn.Linear(d_model, d_head)
        self.w_v = nn.Linear(d_model, d_head)

        self.sin, self.cos = self._get_rotary_vectors(seq_len, d_head)

    def _get_rotary_vectors(self, seq_len, d_head):

        sin = torch.zeros((seq_len, d_head), requires_grad=False)
        cos = torch.zeros((seq_len, d_head), requires_grad=False)
        
        for position in range(seq_len):
            for i in range(d_head // 2):
                theta = 10000. ** (-2.0 * (i - 1) / d_head)
                m_theta = position * theta
                cos[position, 2 * i] = np.cos(m_theta)
                cos[position, 2 * i + 1] = np.cos(m_theta)
                sin[position, 2 * i] = -np.sin(m_theta)
                sin[position, 2 * i + 1] = np.cos(m_theta)
        return sin, cos

    def forward(self, x):
        b, seq_len, d_head = x.shape  

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = q * self.cos
        q[:, ::2, :], q[:, 1::2, :] = q[:, 1::2, :], q[:, 0::2, :] # x2, x1, x4, x3 ...
        q_rotated += q * self.sin

        k_rotated = k * self.cos
        k[:, ::2, :], k[:, 1::2, :] = k[:, 1::2, :], k[:, 0::2, :] # x2, x1, x4, x3 ...
        k_rotated += k * self.sin

        attn_bias = xops.LowerTriangularMask()
        out = xops.memory_efficient_attention(q_rotated, k_rotated, v, attn_bias=attn_bias)

        return out
    

class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, seq_len):
        super().__init__()

        d_head = d_model // n_heads
        self.heads = nn.ModuleList(
            [RoPEMaskedAttentionHead(d_model=d_model, d_head=d_head, seq_len=seq_len) for _ in range(n_heads)]
        )
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):

        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        
        x = self.linear(x)
        return x


        
class LlamaBlock(nn.Module):
    def __init__(self, n_heads, d_model, seq_len, inter_dim):
        super().__init__()

        self.rms = nn.RMSNorm([seq_len, d_model])
        self.attention = RoPEMaskedMultiheadAttention(n_heads, d_model, seq_len)

        self.feedforward = nn.Sequential(
            SwiGLU(d_model, inter_dim),
            nn.Linear(inter_dim, d_model),
        )

    def forward(self, x):
        tmp = self.rms(x)
        x = x + self.attention(tmp)

        tmp = self.rms(x) 
        x = x + self.feedforward(tmp)
        return x