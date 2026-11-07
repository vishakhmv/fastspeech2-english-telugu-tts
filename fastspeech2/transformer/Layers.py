import torch
import torch.nn as nn
import torch.nn.functional as F

from .SubLayers import MultiHeadAttention, PositionwiseFeedForward


class FFTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask):
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input, key_padding_mask=mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        return enc_output


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        return self.conv(signal)


class PostNet(nn.Module):
    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):
        super().__init__()
        pad = int((postnet_kernel_size - 1) / 2)
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    padding=pad,
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )
        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        padding=pad,
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    padding=pad,
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x, mask=None):
        x = x.contiguous().transpose(1, 2)
        keep = None if mask is None else (~mask).unsqueeze(1).to(x.dtype)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
            if keep is not None:
                x = x * keep
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x.contiguous().transpose(1, 2)
