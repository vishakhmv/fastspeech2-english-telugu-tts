import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, batch, predictions):
        mel_targets = batch["mels"]
        pitch_targets = batch["pitches"]
        energy_targets = batch["energies"]
        duration_targets = batch["durations"]

        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions

        src_valid = ~src_masks
        mel_valid = ~mel_masks
        mel_targets = mel_targets[:, : mel_valid.shape[1], :]

        log_duration_targets = torch.log(duration_targets.float() + 1)

        pitch_predictions = pitch_predictions.float().masked_select(src_valid)
        pitch_targets = pitch_targets.masked_select(src_valid)
        energy_predictions = energy_predictions.float().masked_select(src_valid)
        energy_targets = energy_targets.masked_select(src_valid)
        log_duration_predictions = log_duration_predictions.float().masked_select(src_valid)
        log_duration_targets = log_duration_targets.masked_select(src_valid)

        mel_valid3 = mel_valid.unsqueeze(-1)
        mel_predictions = mel_predictions.float().masked_select(mel_valid3)
        postnet_mel_predictions = postnet_mel_predictions.float().masked_select(mel_valid3)
        mel_targets = mel_targets.masked_select(mel_valid3)

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        return total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss
