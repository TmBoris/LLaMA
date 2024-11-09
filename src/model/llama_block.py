import numpy as np
import torch
import torch.nn.functional as F
# import xformers.ops as xops
from torch import nn

from .swish import SwiGLU


class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, d_model, d_head, n_ropes, use_xformers, rope_coef):
        super().__init__()
        self.d_head = d_head
        self.n_ropes = n_ropes
        self.rope_coef = rope_coef
        self.use_xformers = use_xformers
        self.w_q = nn.Linear(d_model, d_head)
        self.w_k = nn.Linear(d_model, d_head)
        self.w_v = nn.Linear(d_model, d_head)

        sin, cos = self._get_rotary_vectors()

        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
        self.sin.requires_grad_(False)
        self.cos.requires_grad_(False)

    def _get_rotary_vectors(self):
        sin = torch.zeros((self.n_ropes, self.d_head))
        cos = torch.zeros((self.n_ropes, self.d_head))

        for position in range(self.n_ropes):
            for i in range(self.d_head // 2):
                theta = 10000.0 ** (-2.0 * (i - 1) / self.d_head)
                m_theta = position * theta * self.rope_coef
                cos[position, 2 * i] = np.cos(m_theta)
                cos[position, 2 * i + 1] = np.cos(m_theta)
                sin[position, 2 * i] = -np.sin(m_theta)
                sin[position, 2 * i + 1] = np.cos(m_theta)

        return sin, cos

    def forward(self, x):
        b, seq_len, d_model = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = q * self.cos[:q.shape[1]]
        q[:, :, ::2], q[:, :, 1::2] = q[:, :, 1::2], q[:, :, 0::2]  # d2, d1, d4, d3 ...
        q_rotated += q * self.sin[:q.shape[1]]

        k_rotated = k * self.cos[:q.shape[1]]
        k[:, :, ::2], k[:, :, 1::2] = k[:, :, 1::2], k[:, :, 0::2]  # d2, d1, d4, d3 ...
        k_rotated += k * self.sin[:q.shape[1]]

        if self.use_xformers:
            # out = xops.memory_efficient_attention(
            #     q_rotated.to(v.dtype),
            #     k_rotated.to(v.dtype),
            #     v,
            #     attn_bias=xops.LowerTriangularMask(),
            # )
            pass
        else:
            out = F.scaled_dot_product_attention(
                q_rotated, k_rotated, v, is_causal=True
            )

        return out


class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, n_ropes, use_xformers, rope_coef):
        super().__init__()

        d_head = d_model // n_heads
        self.heads = nn.ModuleList(
            [
                RoPEMaskedAttentionHead(
                    d_model, d_head, n_ropes, use_xformers, rope_coef
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
        self, n_heads, d_model, inter_dim, n_ropes, use_xformers, rope_coef
    ):
        super().__init__()

        self.rms = nn.RMSNorm([d_model])
        self.attention = RoPEMaskedMultiheadAttention(
            n_heads, d_model, n_ropes, use_xformers, rope_coef
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
