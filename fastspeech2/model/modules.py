from collections import OrderedDict

import torch
import torch.nn as nn

from fastspeech2.utils.tools import get_mask_from_lengths


def scale_gradient(x, scale):
    if scale >= 1.0:
        return x
    detached = x.detach()
    return detached + scale * (x - detached)


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config, pitch_bounds, energy_bounds):
        super().__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        embedding_config = model_config["variance_embedding"]
        self.embedding_kind = embedding_config.get("kind", "quantized")
        self.encoder_grad = model_config["variance_predictor"].get("encoder_grad", 1.0)
        hidden = model_config["transformer"]["encoder_hidden"]

        if self.embedding_kind == "continuous":
            kernel = embedding_config.get("kernel_size", 3)
            padding = (kernel - 1) // 2
            self.register_buffer("pitch_bounds", torch.tensor(pitch_bounds, dtype=torch.float32))
            self.register_buffer("energy_bounds", torch.tensor(energy_bounds, dtype=torch.float32))
            self.pitch_embedding = nn.Conv1d(1, hidden, kernel, padding=padding)
            self.energy_embedding = nn.Conv1d(1, hidden, kernel, padding=padding)
        else:
            n_bins = embedding_config["n_bins"]
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_bounds[0], pitch_bounds[1], n_bins - 1),
                requires_grad=False,
            )
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_bounds[0], energy_bounds[1], n_bins - 1),
                requires_grad=False,
            )
            self.pitch_embedding = nn.Embedding(n_bins, hidden)
            self.energy_embedding = nn.Embedding(n_bins, hidden)

    def embed_continuous(self, values, mask, embedding, bounds):
        clipped = torch.clamp(values.detach().float(), bounds[0].item(), bounds[1].item())
        output = embedding(clipped.unsqueeze(1)).transpose(1, 2)
        return output.masked_fill(mask.unsqueeze(-1), 0.0)

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            values = target
        else:
            prediction = prediction * control
            values = prediction
        if self.embedding_kind == "continuous":
            return prediction, self.embed_continuous(values, mask, self.pitch_embedding, self.pitch_bounds)
        index = torch.bucketize(values.detach().float(), self.pitch_bins)
        return prediction, self.pitch_embedding(index)

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            values = target
        else:
            prediction = prediction * control
            values = prediction
        if self.embedding_kind == "continuous":
            return prediction, self.embed_continuous(values, mask, self.energy_embedding, self.energy_bounds)
        index = torch.bucketize(values.detach().float(), self.energy_bins)
        return prediction, self.energy_embedding(index)

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        predictor_input = scale_gradient(x, self.encoder_grad)
        log_duration_prediction = self.duration_predictor(predictor_input, src_mask)

        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            predictor_input, pitch_target, src_mask, p_control
        )
        x = x + pitch_embedding
        energy_prediction, energy_embedding = self.get_energy_embedding(
            scale_gradient(x, self.encoder_grad), energy_target, src_mask, e_control
        )
        x = x + energy_embedding

        if duration_target is not None:
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                torch.round((torch.exp(log_duration_prediction.float()) - 1) * d_control),
                min=0,
            ).long()
            duration_rounded = duration_rounded.masked_fill(src_mask, 0)

        x, mel_len = self.length_regulator(x, duration_rounded)
        if duration_target is None:
            mel_mask = get_mask_from_lengths(mel_len, x.size(1))

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    def forward(self, x, duration, max_len=None):
        duration = duration.long()
        cumulative = torch.cumsum(duration, dim=1)
        mel_len = cumulative[:, -1]
        if max_len is None:
            max_len = max(int(mel_len.max().item()), 1)
        positions = torch.arange(max_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        index = torch.searchsorted(cumulative, positions.contiguous(), right=True)
        index = index.clamp(max=x.size(1) - 1)
        output = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        valid = positions < mel_len.unsqueeze(1)
        output = output * valid.unsqueeze(-1).to(output.dtype)
        return output, mel_len


class VariancePredictor(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = self.filter_size
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )
        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        if mask is not None:
            keep = (~mask).unsqueeze(-1).to(encoder_output.dtype)
            out = encoder_output * keep
        else:
            keep = None
            out = encoder_output
        for layer in self.conv_layer:
            out = layer(out)
            if keep is not None and isinstance(layer, nn.Dropout):
                out = out * keep
        out = self.linear_layer(out).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        return x.contiguous().transpose(1, 2)
