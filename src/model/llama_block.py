import numpy as np
import torch
import torch.nn.functional as F
import xformers.ops as xops
from torch import nn

from .swish import SwiGLU


class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, d_model, d_head, pre_train_seq_len, expected_seq_len, use_xformers):
        super().__init__()
        self.pre_train_seq_len = pre_train_seq_len
        self.d_head = d_head
        self.expected_seq_len = expected_seq_len
        self.use_xformers = use_xformers
        self.w_q = nn.Linear(d_model, d_head)
        self.w_k = nn.Linear(d_model, d_head)
        self.w_v = nn.Linear(d_model, d_head)

        sin, cos = self._get_rotary_vectors()

        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def _get_rotary_vectors(self):
        sin = torch.zeros((self.expected_seq_len, self.d_head))
        cos = torch.zeros((self.expected_seq_len, self.d_head))

        for position in range(self.expected_seq_len):
            for i in range(self.d_head // 2):
                theta = 10000.0 ** (-2.0 * (i - 1) / self.d_head)
                m_theta = position * theta * self.pre_train_seq_len / self.expected_seq_len
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
        q[:, ::2, :], q[:, 1::2, :] = q[:, 1::2, :], q[:, 0::2, :]  # x2, x1, x4, x3 ...
        q_rotated += q * self.sin

        k_rotated = k * self.cos
        k[:, ::2, :], k[:, 1::2, :] = k[:, 1::2, :], k[:, 0::2, :]  # x2, x1, x4, x3 ...
        k_rotated += k * self.sin

        if self.use_xformers:
            out = xops.memory_efficient_attention(
                q_rotated.to(v.dtype),
                k_rotated.to(v.dtype),
                v,
                attn_bias=xops.LowerTriangularMask(),
            )
        else:
            out = F.scaled_dot_product_attention(
                q_rotated, k_rotated, v, is_causal=True
            )

        return out


class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, pre_train_seq_len, expected_seq_len, use_xformers):
        super().__init__()

        d_head = d_model // n_heads
        self.heads = nn.ModuleList(
            [
                RoPEMaskedAttentionHead(
                    d_model, d_head, pre_train_seq_len, expected_seq_len, use_xformers
                )
                for _ in range(n_heads)
            ]
        )
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)

        x = self.linear(x)
        return x


class LlamaBlock(nn.Module):
    def __init__(
        self, n_heads, d_model, pre_train_seq_len, inter_dim, expected_seq_len, use_xformers
    ):
        super().__init__()

        self.rms = nn.RMSNorm([d_model])
        self.attention = RoPEMaskedMultiheadAttention(
            n_heads, d_model, pre_train_seq_len, expected_seq_len, use_xformers
        )

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
