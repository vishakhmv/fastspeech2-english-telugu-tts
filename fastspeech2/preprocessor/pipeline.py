import json
import os
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf

from fastspeech2.audio import TacotronSTFT
from fastspeech2.audio import tools as audio_tools
from fastspeech2.preprocessor import alignment
from fastspeech2.preprocessor import pitch as pitch_tools
from fastspeech2.text import format_phone_field, symbol_to_id

_STFT_CACHE = {}


def get_stft(preprocessing):
    key = json.dumps(preprocessing["stft"], sort_keys=True) + json.dumps(preprocessing["mel"], sort_keys=True)
    if key not in _STFT_CACHE:
        stft = preprocessing["stft"]
        mel = preprocessing["mel"]
        _STFT_CACHE[key] = TacotronSTFT(
            stft["filter_length"],
            stft["hop_length"],
            stft["win_length"],
            mel["n_mel_channels"],
            preprocessing["audio"]["sampling_rate"],
            mel["mel_fmin"],
            mel["mel_fmax"],
        )
    return _STFT_CACHE[key]


def language_module(language):
    if language == "telugu":
        from fastspeech2.preprocessor import telugu as module
    else:
        from fastspeech2.preprocessor import english as module
    return module


def collect_candidates(config, language):
    workdir = config["paths"]["workdir"]
    preprocessing = config["preprocessing"]
    sampling_rate = preprocessing["audio"]["sampling_rate"]
    hop = preprocessing["stft"]["hop_length"]
    max_len = config["model"]["max_seq_len"]
    table = symbol_to_id(language)
    manifest = pd.read_csv(os.path.join(workdir, "manifest.csv"), keep_default_na=False)
    records = []
    reasons = {"no_textgrid": 0, "parse": 0, "unknown_phone": 0, "too_long": 0, "spoken_noise": 0}
    for row in manifest.to_dict("records"):
        path = os.path.join(workdir, "mfa", "textgrids", row["speaker"], row["utt_id"] + ".TextGrid")
        if not os.path.exists(path):
            reasons["no_textgrid"] += 1
            continue
        try:
            result = alignment.phone_alignment(alignment.read_phone_intervals(path), sampling_rate, hop)
        except Exception:
            reasons["parse"] += 1
            continue
        if result is None:
            reasons["parse"] += 1
            continue
        if "spn" in result["phones"]:
            reasons["spoken_noise"] += 1
            continue
        if any(p not in table for p in result["phones"]):
            reasons["unknown_phone"] += 1
            continue
        frames = int(sum(result["durations"]))
        if frames > max_len or len(result["phones"]) > max_len:
            reasons["too_long"] += 1
            continue
        row.update(result)
        row["frames"] = frames
        row["trimmed_seconds"] = frames * hop / sampling_rate
        records.append(row)
    frame = pd.DataFrame(records)
    scores = alignment.alignment_scores(os.path.join(workdir, "mfa_root"), "mfa_corpus")
    if scores is not None and len(frame):
        frame = frame.merge(scores[["utt_id", "log_likelihood", "speech_log_likelihood", "mfa_snr"]], on="utt_id", how="left")
    print("aligned candidates:", len(frame), "of", len(manifest), "dropped:", reasons, flush=True)
    return frame


def remove_outliers(values):
    values = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(values, [25, 75])
    spread = q3 - q1
    return values[(values > q1 - 1.5 * spread) & (values < q3 + 1.5 * spread)]


def extract_job(job):
    import torch

    torch.set_num_threads(1)
    preprocessing = job["preprocessing"]
    sampling_rate = preprocessing["audio"]["sampling_rate"]
    hop = preprocessing["stft"]["hop_length"]
    stft = get_stft(preprocessing)
    wav, rate = sf.read(job["wav_path"], dtype="float32")
    assert rate == sampling_rate
    wav = wav[int(round(job["start"] * rate)) : int(round(job["end"] * rate))]
    durations = np.array(job["durations"], dtype=np.int64)
    total = int(durations.sum())
    mel, energy = audio_tools.mel_and_energy(wav, stft)
    if mel.shape[1] < total:
        pad = total - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad)), mode="edge")
        energy = np.pad(energy, (0, pad), mode="edge")
    mel = mel[:, :total].T.astype(np.float32)
    energy = energy[:total]
    f0 = pitch_tools.f0_on_frames(wav, rate, hop, total, job["f0_floor"], job["f0_ceiling"])
    voiced = f0 > 0
    if voiced.sum() < 5:
        return {"utt_id": job["utt_id"], "ok": False}
    log_f0 = np.zeros_like(f0)
    log_f0[voiced] = np.log(f0[voiced])
    filled = pitch_tools.interpolate_unvoiced(log_f0, voiced)
    phone_pitch = pitch_tools.phone_average(filled, durations)
    phone_energy = pitch_tools.phone_average(energy.astype(np.float64), durations)
    name = "{}-mel-{}.npy".format(job["speaker"], job["utt_id"])
    np.save(os.path.join(job["out_dir"], "mel", name), mel)
    return {
        "utt_id": job["utt_id"],
        "ok": True,
        "pitch": phone_pitch,
        "energy": phone_energy,
        "voiced_fraction": float(voiced.mean()),
    }


def estimate_ranges(frame, config, workdir):
    preprocessing = config["preprocessing"]
    pitch_config = preprocessing["pitch"]
    hop = preprocessing["stft"]["hop_length"]
    rng = np.random.default_rng(config["data"]["seed"])
    ranges = {}
    for speaker, group in frame.groupby("speaker"):
        chosen = group.sample(n=min(len(group), 40), random_state=int(rng.integers(1_000_000)))
        waves = []
        for row in chosen.to_dict("records"):
            wav, rate = sf.read(os.path.join(workdir, "wav22", row["folder"], row["utt_id"] + ".wav"), dtype="float32")
            waves.append(wav[int(round(row["start"] * rate)) : int(round(row["end"] * rate))])
        ranges[speaker] = pitch_tools.estimate_speaker_range(
            waves,
            preprocessing["audio"]["sampling_rate"],
            hop,
            pitch_config["first_pass_range"],
            pitch_config["floor_factor"],
            pitch_config["ceiling_factor"],
        )
    return ranges


def build_features(config, language, workers=None):
    workdir = config["paths"]["workdir"]
    out_dir = config["paths"]["preprocessed"]
    preprocessing = config["preprocessing"]
    for sub in ("mel", "pitch", "energy", "duration"):
        shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    candidates = collect_candidates(config, language)
    candidates["folder"] = candidates["speaker"]
    module = language_module(language)
    if hasattr(module, "assign_speakers"):
        candidates = module.assign_speakers(candidates, config)
    train, val = module.finalize(candidates, config)
    selected = pd.concat([train.assign(split="train"), val.assign(split="val")], ignore_index=True)
    print(
        "selected train %.3f h (%d clips), val %.3f h (%d clips)"
        % (train["trimmed_seconds"].sum() / 3600, len(train), val["trimmed_seconds"].sum() / 3600, len(val)),
        flush=True,
    )

    ranges = estimate_ranges(selected, config, workdir)
    print("speaker f0 ranges:", {k: [round(x) for x in v] for k, v in list(ranges.items())[:6]}, "...", flush=True)

    jobs = [
        {
            "utt_id": row["utt_id"],
            "speaker": row["speaker"],
            "wav_path": os.path.join(workdir, "wav22", row["folder"], row["utt_id"] + ".wav"),
            "start": row["start"],
            "end": row["end"],
            "durations": row["durations"],
            "f0_floor": ranges[row["speaker"]][0],
            "f0_ceiling": ranges[row["speaker"]][1],
            "preprocessing": preprocessing,
            "out_dir": out_dir,
        }
        for row in selected.to_dict("records")
    ]
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    started = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, result in enumerate(pool.map(extract_job, jobs, chunksize=4), 1):
            results[result["utt_id"]] = result
            if count % 500 == 0:
                print("features %d/%d, %.0f s" % (count, len(jobs), time.time() - started), flush=True)
    failed = [u for u, r in results.items() if not r["ok"]]
    print("feature extraction failures:", len(failed), flush=True)
    selected = selected[~selected["utt_id"].isin(failed)].reset_index(drop=True)

    speaker_stats = {}
    all_pitch = []
    all_energy = []
    for speaker, group in selected[selected["split"] == "train"].groupby("speaker"):
        pitch_values = []
        energy_values = []
        for row in group.to_dict("records"):
            keep = np.array([p != alignment.SILENCE for p in row["phones"]]) & (np.array(row["durations"]) > 0)
            pitch_values.append(results[row["utt_id"]]["pitch"][keep])
            energy_values.append(results[row["utt_id"]]["energy"][keep])
        pitch_values = remove_outliers(np.concatenate(pitch_values))
        energy_values = remove_outliers(np.concatenate(energy_values))
        speaker_stats[speaker] = {
            "log_f0_mean": float(pitch_values.mean()),
            "log_f0_std": float(pitch_values.std()),
            "energy_mean": float(energy_values.mean()),
            "energy_std": float(energy_values.std()),
            "f0_floor": ranges[speaker][0],
            "f0_ceiling": ranges[speaker][1],
        }

    lines = {"train": [], "val": []}
    for row in selected.to_dict("records"):
        stats = speaker_stats[row["speaker"]]
        result = results[row["utt_id"]]
        pitch = (result["pitch"] - stats["log_f0_mean"]) / stats["log_f0_std"]
        energy = (result["energy"] - stats["energy_mean"]) / stats["energy_std"]
        keep = np.array([p != alignment.SILENCE for p in row["phones"]])
        empty = np.array(row["durations"]) == 0
        pitch[empty] = 0.0
        energy[empty] = 0.0
        keep &= ~empty
        if row["split"] == "train":
            all_pitch.append(pitch[keep])
            all_energy.append(energy[keep])
        base = "{}-{{}}-{}.npy".format(row["speaker"], row["utt_id"])
        np.save(os.path.join(out_dir, "pitch", base.format("pitch")), pitch.astype(np.float32))
        np.save(os.path.join(out_dir, "energy", base.format("energy")), energy.astype(np.float32))
        np.save(os.path.join(out_dir, "duration", base.format("duration")), np.array(row["durations"], dtype=np.int64))
        raw_text = str(row["text"]).replace("|", " ").replace("\n", " ")
        lines[row["split"]].append(
            "|".join([row["utt_id"], row["speaker"], format_phone_field(row["phones"]), raw_text])
        )

    low, high = preprocessing["stats_percentile"]
    all_pitch = np.concatenate(all_pitch)
    all_energy = np.concatenate(all_energy)
    stats = {
        "pitch": {"min": float(np.percentile(all_pitch, low)), "max": float(np.percentile(all_pitch, high))},
        "energy": {"min": float(np.percentile(all_energy, low)), "max": float(np.percentile(all_energy, high))},
        "speakers": speaker_stats,
    }
    speakers = sorted(speaker_stats)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(out_dir, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump({name: index for index, name in enumerate(speakers)}, f, indent=2, ensure_ascii=False)
    for split, items in lines.items():
        with open(os.path.join(out_dir, split + ".txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(items) + "\n")

    summary = summarize(selected, language)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    selected.drop(columns=["phones", "durations"]).to_csv(os.path.join(out_dir, "selected_clips.csv"), index=False, encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "phone_counts"}, indent=2, ensure_ascii=False), flush=True)
    return summary


def summarize(selected, language):
    summary = {}
    for split in ("train", "val"):
        part = selected[selected["split"] == split]
        summary[split] = {
            "clips": int(len(part)),
            "hours": round(float(part["trimmed_seconds"].sum() / 3600), 3),
            "speakers": int(part["speaker"].nunique()),
            "mean_seconds": round(float(part["trimmed_seconds"].mean()), 2),
        }
    counts = {}
    for row in selected[selected["split"] == "train"].to_dict("records"):
        for phone in row["phones"]:
            counts[phone] = counts.get(phone, 0) + 1
    summary["phone_counts"] = dict(sorted(counts.items(), key=lambda item: item[1]))
    train_texts = set(selected[selected["split"] == "train"]["words"])
    val_texts = set(selected[selected["split"] == "val"]["words"])
    summary["text_overlap"] = len(train_texts & val_texts)
    train_ids = set(selected[selected["split"] == "train"]["utt_id"])
    val_ids = set(selected[selected["split"] == "val"]["utt_id"])
    summary["clip_overlap"] = len(train_ids & val_ids)
    summary["language"] = language
    return summary


def pack(config, output_path=None):
    out_dir = config["paths"]["preprocessed"]
    output_path = output_path or out_dir.rstrip("/\\") + ".tar"
    with tarfile.open(output_path, "w") as archive:
        archive.add(out_dir, arcname=os.path.basename(out_dir.rstrip("/\\")))
    print("packed", output_path, round(os.path.getsize(output_path) / 1e6, 1), "MB", flush=True)
