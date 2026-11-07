import numpy as np
import torch
import torch.nn.functional as F


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0)
    return ids >= lengths.unsqueeze(1)


def pad_1D(inputs, pad_value=0):
    max_len = max(len(x) for x in inputs)
    padded = [
        np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)
        for x in inputs
    ]
    return np.stack(padded)


def pad_2D(inputs, pad_value=0):
    max_len = max(x.shape[0] for x in inputs)
    padded = [
        np.pad(
            x,
            ((0, max_len - x.shape[0]), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
        for x in inputs
    ]
    return np.stack(padded)


def pad_sequence_tensors(sequences):
    max_len = max(seq.size(0) for seq in sequences)
    out = []
    for seq in sequences:
        if seq.dim() == 1:
            out.append(F.pad(seq, (0, max_len - seq.size(0))))
        else:
            out.append(F.pad(seq, (0, 0, 0, max_len - seq.size(0))))
    return torch.stack(out)


def expand_by_duration(values, durations):
    out = []
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)
