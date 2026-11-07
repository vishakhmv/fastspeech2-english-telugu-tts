import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, key_padding_mask=None):
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask[:, None, None, :]
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=1.0 / self.temperature
        )
