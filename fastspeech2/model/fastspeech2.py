import torch.nn as nn

from fastspeech2.transformer import Encoder, Decoder, PostNet
from fastspeech2.utils.tools import get_mask_from_lengths

from .modules import VarianceAdaptor


class FastSpeech2(nn.Module):
    def __init__(self, preprocess_config, model_config, n_symbols, n_speakers, stats):
        super().__init__()
        self.model_config = model_config
        n_mels = preprocess_config["mel"]["n_mel_channels"]

        self.encoder = Encoder(model_config, n_symbols)
        self.variance_adaptor = VarianceAdaptor(
            model_config,
            (stats["pitch"]["min"], stats["pitch"]["max"]),
            (stats["energy"]["min"], stats["energy"]["max"]),
        )
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(model_config["transformer"]["decoder_hidden"], n_mels)
        self.postnet = PostNet(n_mel_channels=n_mels)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            self.speaker_emb = nn.Embedding(
                n_speakers, model_config["transformer"]["encoder_hidden"]
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        mel_lens=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        max_src_len = texts.shape[1]
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = None
        if mel_lens is not None:
            mel_masks = get_mask_from_lengths(mel_lens, int(d_targets.sum(dim=1).max().item()))

        output = self.encoder(texts, src_masks)
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output).masked_fill(mel_masks.unsqueeze(-1), 0.0)
        postnet_output = (self.postnet(output, mel_masks) + output).masked_fill(mel_masks.unsqueeze(-1), 0.0)

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
