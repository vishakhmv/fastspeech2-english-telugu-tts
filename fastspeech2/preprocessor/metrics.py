import numpy as np
from scipy import signal

from fastspeech2.audio import tools as audio_tools


def clip_metrics(wav, sampling_rate):
    levels = audio_tools.frame_levels_db(wav, sampling_rate)
    active = levels[levels > -90]
    speech_level = float(np.percentile(levels, 95))
    floor = float(np.percentile(active, 10)) if len(active) else -120.0
    metrics = {
        "peak": float(np.max(np.abs(wav))) if len(wav) else 0.0,
        "clip_fraction": float(np.mean(np.abs(wav) >= 0.999)) if len(wav) else 0.0,
        "level_db": float(audio_tools.active_speech_level_db(wav, sampling_rate)),
        "noise_floor_db": floor,
        "snr_proxy_db": speech_level - floor,
    }
    nperseg = 512
    if len(wav) >= nperseg:
        frequencies, power = signal.welch(wav, sampling_rate, nperseg=nperseg)
        low = power[(frequencies >= 100) & (frequencies < 4000)].sum() + 1e-12
        high = power[(frequencies >= 4000) & (frequencies < min(8000, sampling_rate / 2))].sum() + 1e-12
        metrics["hf_ratio_db"] = float(10 * np.log10(high / low))
    else:
        metrics["hf_ratio_db"] = -60.0
    return metrics


class SquimEstimator:
    def __init__(self, num_threads=None):
        import torch
        import torchaudio

        if num_threads:
            torch.set_num_threads(num_threads)
        self.torch = torch
        bundle = torchaudio.pipelines.SQUIM_OBJECTIVE
        self.model = bundle.get_model().eval()
        self.sampling_rate = bundle.sample_rate

    def estimate(self, wav, sampling_rate):
        torch = self.torch
        if sampling_rate != self.sampling_rate:
            wav = audio_tools.resample(wav, sampling_rate, self.sampling_rate)
        tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        with torch.inference_mode():
            stoi, pesq, si_sdr = self.model(tensor)
        return float(stoi.item()), float(pesq.item()), float(si_sdr.item())
