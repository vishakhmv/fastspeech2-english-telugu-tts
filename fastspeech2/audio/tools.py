import io

import librosa
import numpy as np
import soundfile as sf
import torch


def decode_audio(audio_bytes):
    wav, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    return wav.mean(axis=1), sampling_rate


def resample(wav, orig_sr, target_sr):
    if orig_sr == target_sr:
        return wav.astype(np.float32)
    return librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def frame_levels_db(wav, sampling_rate, frame_ms=25, hop_ms=10):
    frame = int(sampling_rate * frame_ms / 1000)
    hop = int(sampling_rate * hop_ms / 1000)
    if len(wav) < frame:
        return np.array([10 * np.log10(np.mean(wav.astype(np.float64) ** 2) + 1e-12)])
    cumulative = np.cumsum(np.insert(wav.astype(np.float64) ** 2, 0, 0.0))
    power = (cumulative[frame:] - cumulative[:-frame])[::hop] / frame
    return 10 * np.log10(np.maximum(power, 0) + 1e-12)


def active_speech_level_db(wav, sampling_rate, window_db=25.0):
    levels = frame_levels_db(wav, sampling_rate)
    threshold = np.percentile(levels, 95) - window_db
    active = levels[levels > threshold]
    return 10 * np.log10(np.mean(10 ** (active / 10)) + 1e-12)


def normalize_loudness(wav, sampling_rate, target_db=-23.0, peak_limit=0.95):
    level = active_speech_level_db(wav, sampling_rate)
    gain = 10 ** ((target_db - level) / 20)
    peak = float(np.max(np.abs(wav))) + 1e-9
    gain = min(gain, peak_limit / peak)
    return (wav * gain).astype(np.float32), gain


def mel_and_energy(wav, stft):
    audio = torch.clip(torch.from_numpy(wav.astype(np.float32)).unsqueeze(0), -1, 1)
    with torch.no_grad():
        mel, energy = stft.mel_spectrogram(audio)
    return mel.squeeze(0).numpy().astype(np.float32), energy.squeeze(0).numpy().astype(np.float32)
