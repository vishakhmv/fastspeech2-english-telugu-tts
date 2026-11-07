import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d

LRELU_SLOPE = 0.1


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
                for _ in dilation
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(
                    h.upsample_initial_channel // (2 ** i),
                    h.upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)


def fuse_weight_norm(state_dict):
    fused = {}
    for key, value in state_dict.items():
        if key.endswith(".weight_g"):
            continue
        if key.endswith(".weight_v"):
            prefix = key[: -len(".weight_v")]
            g = state_dict[prefix + ".weight_g"]
            norm = value.flatten(1).norm(dim=1).view(-1, *([1] * (value.dim() - 1)))
            fused[prefix + ".weight"] = value * (g / norm)
        else:
            fused[key] = value
    return fused


def load_generator(checkpoint_path, config_path=None, device="cpu"):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = AttrDict(json.load(f))
    generator = Generator(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["generator"] if "generator" in checkpoint else checkpoint
    if any(k.endswith(".weight_g") for k in state):
        state = fuse_weight_norm(state)
    generator.load_state_dict(state)
    generator.eval()
    return generator.to(device)
