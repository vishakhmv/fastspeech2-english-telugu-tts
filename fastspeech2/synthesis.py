import os

import numpy as np
import torch
from scipy.io import wavfile

from fastspeech2.audio import TacotronSTFT
from fastspeech2.audio import tools as audio_tools
from fastspeech2.dataset import collate_fn, to_device
from fastspeech2.model import FastSpeech2
from fastspeech2.plots import plot_mel_comparison
from fastspeech2.text import phones_to_sequence


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = FastSpeech2(
        config["preprocessing"],
        config["model"],
        len(checkpoint["symbols"]),
        len(checkpoint["speakers"]),
        checkpoint["stats"],
    )
    state = {k: v.float() if v.is_floating_point() else v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model.to(device), checkpoint


@torch.no_grad()
def predict(model, phones, language, speaker_id=0, device="cpu", p_control=1.0, e_control=1.0, d_control=1.0):
    ids = np.array(phones_to_sequence(phones, language), dtype=np.int64)
    texts = torch.from_numpy(ids).unsqueeze(0).to(device)
    src_lens = torch.tensor([len(ids)], device=device)
    speakers = torch.tensor([speaker_id], device=device)
    output = model(speakers, texts, src_lens, p_control=p_control, e_control=e_control, d_control=d_control)
    mel_len = int(output[9][0].item())
    return {
        "mel": output[1][0, :mel_len].cpu().numpy(),
        "pitch": output[2][0].cpu().numpy(),
        "energy": output[3][0].cpu().numpy(),
        "log_duration": output[4][0].cpu().numpy(),
        "duration": output[5][0].cpu().numpy().astype(int),
    }


def denormalize(prediction, stats, speaker_name):
    speaker = stats["speakers"][speaker_name]
    f0_hz = np.exp(prediction["pitch"] * speaker["log_f0_std"] + speaker["log_f0_mean"])
    energy = prediction["energy"] * speaker["energy_std"] + speaker["energy_mean"]
    return f0_hz, energy


@torch.no_grad()
def vocode(vocoder, mel, device="cpu", max_wav_value=32768.0):
    tensor = torch.from_numpy(mel.T).unsqueeze(0).float().to(device)
    wav = vocoder(tensor).squeeze().cpu().numpy()
    return np.clip(wav * max_wav_value, -max_wav_value, max_wav_value - 1).astype(np.int16)


def save_wav(path, wav, sampling_rate):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, sampling_rate, wav)


def measured_contours(wav_int16, preprocessing, f0_floor, f0_ceiling):
    from fastspeech2.preprocessor import pitch as pitch_tools

    sampling_rate = preprocessing["audio"]["sampling_rate"]
    hop = preprocessing["stft"]["hop_length"]
    wav = wav_int16.astype(np.float32) / preprocessing["audio"]["max_wav_value"]
    n_frames = len(wav) // hop + 1
    f0 = pitch_tools.f0_on_frames(wav, sampling_rate, hop, n_frames, f0_floor, f0_ceiling)
    stft = TacotronSTFT(
        preprocessing["stft"]["filter_length"],
        hop,
        preprocessing["stft"]["win_length"],
        preprocessing["mel"]["n_mel_channels"],
        sampling_rate,
        preprocessing["mel"]["mel_fmin"],
        preprocessing["mel"]["mel_fmax"],
    )
    _, energy = audio_tools.mel_and_energy(wav, stft)
    times = np.arange(n_frames) * hop / sampling_rate
    return (times, f0), (np.arange(len(energy)) * hop / sampling_rate, energy)


@torch.no_grad()
def render_validation_samples(model, vocoder, dataset, device, out_dir, config, stats, count=2):
    os.makedirs(out_dir, exist_ok=True)
    preprocessing = config["preprocessing"]
    sampling_rate = preprocessing["audio"]["sampling_rate"]
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, count).astype(int)
    batch = to_device(collate_fn([dataset[i] for i in indices]), device)
    output = model(
        batch["speakers"], batch["texts"], batch["src_lens"], batch["mel_lens"],
        batch["pitches"], batch["energies"], batch["durations"],
    )
    for k, utt_id in enumerate(batch["ids"]):
        mel_len = int(output[9][k].item())
        predicted = output[1][k, :mel_len].float().cpu().numpy()
        target = batch["mels"][k, :mel_len].cpu().numpy()
        save_wav(os.path.join(out_dir, utt_id + "_reconstructed.wav"), vocode(vocoder, target, device), sampling_rate)
        save_wav(os.path.join(out_dir, utt_id + "_synthesized.wav"), vocode(vocoder, predicted, device), sampling_rate)
        plot_mel_comparison(os.path.join(out_dir, utt_id + ".png"), target, predicted, utt_id)
