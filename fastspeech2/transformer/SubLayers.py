import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask=None):
        sz_b, len_q, _ = q.size()
        len_k = k.size(1)
        len_v = v.size(1)
        residual = q

        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_v).transpose(1, 2)

        output = self.attention(q, k, v, key_padding_mask=key_padding_mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
