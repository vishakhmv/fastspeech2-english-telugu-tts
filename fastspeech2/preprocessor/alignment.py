import os
import sqlite3

import numpy as np
import pandas as pd
import tgt

SILENCE = "sp"
SILENCE_LABELS = {"", "sil", "sp"}


def read_phone_intervals(textgrid_path):
    textgrid = tgt.io.read_textgrid(textgrid_path, include_empty_intervals=True)
    tier = textgrid.get_tier_by_name("phones")
    return [(interval.start_time, interval.end_time, interval.text.strip()) for interval in tier]


def phone_alignment(intervals, sampling_rate, hop_length):
    labels = []
    starts = []
    ends = []
    for start, end, label in intervals:
        label = SILENCE if label in SILENCE_LABELS else label
        if labels and label == SILENCE and labels[-1] == SILENCE:
            ends[-1] = end
            continue
        labels.append(label)
        starts.append(start)
        ends.append(end)
    first = 0
    while first < len(labels) and labels[first] == SILENCE:
        first += 1
    last = len(labels)
    while last > first and labels[last - 1] == SILENCE:
        last -= 1
    if first >= last:
        return None
    labels = labels[first:last]
    starts = starts[first:last]
    ends = ends[first:last]
    origin = starts[0]
    scale = sampling_rate / hop_length
    boundaries = [int(np.round((t - origin) * scale)) for t in starts + [ends[-1]]]
    boundaries[0] = 0
    durations = [boundaries[i + 1] - boundaries[i] for i in range(len(labels))]
    return {
        "phones": labels,
        "durations": durations,
        "start": float(origin),
        "end": float(ends[-1]),
    }


def alignment_scores(mfa_root, corpus_name):
    candidates = [
        os.path.join(mfa_root, corpus_name, corpus_name + ".db"),
        os.path.join(mfa_root, corpus_name + ".db"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None
    connection = sqlite3.connect(path)
    try:
        frame = pd.read_sql_query(
            "SELECT f.name AS utt_id, u.alignment_log_likelihood AS log_likelihood, "
            "u.speech_log_likelihood AS speech_log_likelihood, u.snr AS mfa_snr "
            "FROM utterance u JOIN file f ON f.id = u.file_id",
            connection,
        )
    finally:
        connection.close()
    return frame
