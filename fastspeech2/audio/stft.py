import librosa
import torch


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate

        self.register_buffer("window", torch.hann_window(win_length, periodic=True))
        mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

    def spectral_normalize(self, magnitudes):
        return torch.log(torch.clamp(magnitudes, min=1e-5))

    def mel_spectrogram(self, y):
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1
        spectrum = torch.stft(
            y,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitudes = spectrum.abs()
        mel_output = self.spectral_normalize(torch.matmul(self.mel_basis, magnitudes))
        energy = torch.norm(magnitudes, dim=1)
        return mel_output, energy
