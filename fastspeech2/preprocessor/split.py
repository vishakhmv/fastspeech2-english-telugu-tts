import numpy as np
import pandas as pd


def split_groups(frame, train_hours, val_hours, seed, key="words"):
    rng = np.random.default_rng(seed)
    groups = [group for _, group in frame.groupby(key, sort=True)]
    order = rng.permutation(len(groups))
    val_target = val_hours * 3600
    train_target = train_hours * 3600
    val_parts, train_parts = [], []
    val_seconds = train_seconds = 0.0
    for position in order:
        group = groups[position]
        seconds = float(group["trimmed_seconds"].sum())
        if val_seconds < val_target:
            val_parts.append(group)
            val_seconds += seconds
        elif train_seconds < train_target:
            train_parts.append(group)
            train_seconds += seconds
    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)


def split_per_speaker(frame, val_fraction, seed, min_speaker_seconds, key="words"):
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for speaker, speaker_frame in frame.groupby("speaker", sort=True):
        total = float(speaker_frame["trimmed_seconds"].sum())
        groups = [group for _, group in speaker_frame.groupby(key, sort=True)]
        if total < min_speaker_seconds or len(groups) < 3:
            train_parts.extend(groups)
            continue
        target = val_fraction * total
        val_seconds = 0.0
        for position in rng.permutation(len(groups)):
            group = groups[position]
            seconds = float(group["trimmed_seconds"].sum())
            remaining_train = total - val_seconds - seconds
            if val_seconds < target and remaining_train > 0.5 * total:
                val_parts.append(group)
                val_seconds += seconds
            else:
                train_parts.append(group)
    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)
