import numpy as np
import torch
import torch.nn as nn

from . import Constants
from .Layers import FFTBlock


def get_sinusoid_encoding_table(n_position, d_hid):
    positions = np.arange(n_position)[:, None]
    dims = np.arange(d_hid)[None, :]
    angles = positions / np.power(10000, 2 * (dims // 2) / d_hid)
    table = np.zeros((n_position, d_hid))
    table[:, 0::2] = np.sin(angles[:, 0::2])
    table[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.FloatTensor(table)


class Encoder(nn.Module):
    def __init__(self, config, n_symbols):
        super().__init__()
        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = d_word_vec // n_head
        d_model = d_word_vec
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(n_symbols, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask):
        max_len = src_seq.shape[1]
        if max_len > self.max_seq_len:
            position = get_sinusoid_encoding_table(max_len, self.d_model).to(src_seq.device)
            position = position.unsqueeze(0)
        else:
            position = self.position_enc[:, :max_len, :]
        enc_output = self.src_word_emb(src_seq) + position
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, mask)
        return enc_output


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = d_word_vec // n_head
        d_model = d_word_vec
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask):
        max_len = enc_seq.shape[1]
        if max_len > self.max_seq_len:
            position = get_sinusoid_encoding_table(max_len, self.d_model).to(enc_seq.device)
            position = position.unsqueeze(0)
        else:
            position = self.position_enc[:, :max_len, :]
        dec_output = enc_seq + position
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, mask)
        return dec_output, mask
