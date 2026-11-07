import json
import os

import numpy as np
import torch

from fastspeech2.hifigan import load_generator
from fastspeech2.plots import plot_loss_curves, plot_variance
from fastspeech2.synthesis import (
    denormalize,
    load_checkpoint,
    measured_contours,
    predict,
    save_wav,
    vocode,
)
from fastspeech2.text import english, telugu


def resolve_speaker(config, checkpoint, speaker):
    speakers = checkpoint["speakers"]
    if speaker is None:
        speaker = config["inference"].get("speaker", next(iter(speakers)))
    if speaker not in speakers:
        raise KeyError("speaker %s not in %s" % (speaker, sorted(speakers)))
    return speaker, speakers[speaker]


def sentence_details(phones, words, prediction, f0_hz, energy, hop_length, sampling_rate):
    durations = prediction["duration"]
    ms = durations * hop_length / sampling_rate * 1000
    pause = np.array([p == "sp" for p in phones])
    detail = {
        "phones": phones,
        "num_phones": len(phones),
        "total_duration_s": float(durations.sum() * hop_length / sampling_rate),
        "phone_duration_ms": [round(float(v), 1) for v in ms],
        "predicted_f0_hz": [round(float(v), 1) for v in f0_hz],
        "predicted_energy": [round(float(v), 3) for v in energy],
        "words": [
            {
                "word": w,
                "start_phone": s,
                "end_phone": e,
                "duration_ms": round(float(ms[s:e].sum()), 1),
                "mean_f0_hz": round(float(np.mean(f0_hz[s:e])), 1),
                "mean_energy": round(float(np.mean(energy[s:e])), 3),
            }
            for w, s, e in words
        ],
    }
    speech = ~pause
    detail["f0_range_hz"] = [round(float(f0_hz[speech].min()), 1), round(float(f0_hz[speech].max()), 1)]
    detail["pause_ms"] = [round(float(v), 1) for v in ms[pause]]
    return detail


def run_inference(
    config,
    checkpoint_path,
    vocoder_path,
    output_dir,
    front_end,
    speaker=None,
    device="cpu",
    loss_log_dir=None,
    controls=(1.0, 1.0, 1.0),
):
    language = config["language"]
    preprocessing = config["preprocessing"]
    sampling_rate = preprocessing["audio"]["sampling_rate"]
    hop = preprocessing["stft"]["hop_length"]
    max_wav_value = preprocessing["audio"]["max_wav_value"]

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    stats = checkpoint["stats"]
    speaker_name, speaker_id = resolve_speaker(config, checkpoint, speaker)
    vocoder = load_generator(vocoder_path, device=device)
    f0_floor = stats["speakers"][speaker_name]["f0_floor"]
    f0_ceiling = stats["speakers"][speaker_name]["f0_ceiling"]

    audio_dir = os.path.join(output_dir, "test_generated_audios")
    plot_dirs = {kind: os.path.join(output_dir, "variance_adaptor_plots", kind) for kind in ("duration", "pitch", "energy")}
    os.makedirs(audio_dir, exist_ok=True)
    for path in plot_dirs.values():
        os.makedirs(path, exist_ok=True)

    sentences = list(config["inference"]["sentences"]) + list(config["inference"].get("extra_sentences", []))
    speech_scale = float(config["inference"].get("speech_duration_scale", 1.0))
    pause_scale = float(config["inference"].get("pause_duration_scale", 1.0))
    report = {
        "checkpoint": os.path.basename(checkpoint_path),
        "step": int(checkpoint.get("step", -1)),
        "speaker": speaker_name,
        "duration_scale": {"speech": speech_scale, "pause": pause_scale},
        "sentences": {},
    }
    for item in sentences:
        sentence_id = item["id"]
        text = item["text"]
        phones, words = front_end(text, bool(item.get("repair_broken_words", False)))
        duration_control = torch.tensor(
            [[pause_scale if p == "sp" else speech_scale for p in phones]], dtype=torch.float32, device=device
        ) * controls[2]
        prediction = predict(model, phones, language, speaker_id, device, controls[0], controls[1], duration_control)
        wav = vocode(vocoder, prediction["mel"], device, max_wav_value)
        padding = np.zeros(int(0.15 * sampling_rate), dtype=wav.dtype)
        save_wav(os.path.join(audio_dir, sentence_id + ".wav"), np.concatenate([padding, wav, padding]), sampling_rate)
        f0_hz, energy = denormalize(prediction, stats, speaker_name)
        f0_measured, energy_measured = measured_contours(wav, preprocessing, f0_floor, f0_ceiling)
        plot_words = words
        plot_text = text
        if language == "telugu":
            plot_words = [(telugu.romanize(phones[s:e]), s, e) for _, s, e in words]
            plot_text = " ".join(w for w, _, _ in plot_words)
        elif language == "english":
            prefix = english.LETTER_WORD_PREFIX
            plot_words = [(w[len(prefix):].upper() if w.startswith(prefix) else w, s, e) for w, s, e in words]
        title = "%s | %s" % (sentence_id, plot_text)
        common = dict(
            title=title,
            phones=phones,
            durations=prediction["duration"],
            mel=prediction["mel"],
            hop_length=hop,
            sampling_rate=sampling_rate,
            words=plot_words,
        )
        plot_variance(os.path.join(plot_dirs["duration"], sentence_id + ".png"), "duration", values=None, **common)
        plot_variance(os.path.join(plot_dirs["pitch"], sentence_id + ".png"), "pitch", values=f0_hz, measured=f0_measured, **common)
        plot_variance(os.path.join(plot_dirs["energy"], sentence_id + ".png"), "energy", values=energy, measured=energy_measured, **common)
        detail = sentence_details(phones, words, prediction, f0_hz, energy, hop, sampling_rate)
        detail["text"] = text
        detail["audio_seconds"] = round(len(wav) / sampling_rate, 3)
        report["sentences"][sentence_id] = detail
        print("%s: %d phones, %.2f s" % (sentence_id, len(phones), len(wav) / sampling_rate), flush=True)

    with open(os.path.join(output_dir, "inference_details.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    if loss_log_dir:
        train_csv = os.path.join(loss_log_dir, "train_log.csv")
        val_csv = os.path.join(loss_log_dir, "val_log.csv")
        if os.path.exists(train_csv):
            plot_loss_curves(train_csv, os.path.join(output_dir, "training_loss.png"), "%s training loss" % language.capitalize())
        if os.path.exists(val_csv):
            best = checkpoint.get("best", {}).get("step")
            plot_loss_curves(val_csv, os.path.join(output_dir, "validation_loss.png"), "%s validation loss" % language.capitalize(), best_step=best)
    return report
