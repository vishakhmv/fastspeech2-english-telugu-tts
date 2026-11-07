import glob
import os

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from fastspeech2.audio import tools as audio_tools


def parquet_files(dataset_dir):
    return sorted(glob.glob(os.path.join(dataset_dir, "data", "*.parquet")))


def read_metadata(dataset_dir, audio_column, columns):
    frames = []
    for path in parquet_files(dataset_dir):
        table = pq.read_table(path, columns=[audio_column + ".path"] + list(columns))
        frames.append(table.to_pandas())
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.rename(columns={"path": "orig_path"})
    frame.insert(0, "index", range(len(frame)))
    return frame


def iter_audio(dataset_dir, audio_column, wanted):
    wanted = sorted(set(int(i) for i in wanted))
    cursor = 0
    base = 0
    for path in parquet_files(dataset_dir):
        parquet_file = pq.ParquetFile(path)
        for group in range(parquet_file.num_row_groups):
            rows = parquet_file.metadata.row_group(group).num_rows
            hits = []
            while cursor < len(wanted) and wanted[cursor] < base + rows:
                hits.append(wanted[cursor])
                cursor += 1
            if hits:
                column = parquet_file.read_row_group(group, columns=[audio_column]).column(0)
                for index in hits:
                    value = column[index - base].as_py()
                    yield index, value["bytes"]
            base += rows


def write_wav(path, wav, sampling_rate):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav, sampling_rate, subtype="PCM_16")


def decode_and_prepare(audio_bytes, tts_rate, mfa_rate, target_db, peak_limit):
    wav, source_rate = audio_tools.decode_audio(audio_bytes)
    wav_tts = audio_tools.resample(wav, source_rate, tts_rate)
    wav_tts, gain = audio_tools.normalize_loudness(wav_tts, tts_rate, target_db, peak_limit)
    wav_mfa = audio_tools.resample(wav * gain, source_rate, mfa_rate)
    wav_mfa = wav_mfa.clip(-1.0, 1.0)
    return wav_tts, wav_mfa, gain, len(wav) / source_rate
