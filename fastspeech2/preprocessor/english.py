import os
import time

import numpy as np
import pandas as pd

from fastspeech2.audio import tools as audio_tools
from fastspeech2.preprocessor import dataio, lexicon, metrics, split
from fastspeech2.preprocessor import pitch as pitch_tools
from fastspeech2.text import english

METADATA_COLUMNS = [
    "duration",
    "text",
    "gender",
    "age-group",
    "primary_language",
    "native_place_state",
    "native_place_district",
    "highest_qualification",
    "job_category",
    "occupation_domain",
]
SPEAKER_COLUMNS = METADATA_COLUMNS[2:]
MFA_RATE = 16000


def speaker_key(row):
    return "|".join(str(row[c]) for c in SPEAKER_COLUMNS)


def eligible_rows(dataset_dir, config):
    data = config["data"]
    frame = dataio.read_metadata(dataset_dir, data["audio_column"], METADATA_COLUMNS)
    frame["speaker_key"] = frame.apply(speaker_key, axis=1)
    frame["word_list"] = frame["text"].map(english.training_words)
    frame = frame[frame["word_list"].notna()].copy()
    frame = frame[(frame["duration"] >= data["min_duration"]) & (frame["duration"] <= data["max_duration"])]
    frame["words"] = frame["word_list"].map(" ".join)
    frame["words_per_second"] = frame["word_list"].map(len) / frame["duration"]
    frame = frame[
        (frame["words_per_second"] >= data["words_per_second"][0])
        & (frame["words_per_second"] <= data["words_per_second"][1])
    ]
    return frame.drop(columns=["word_list"]).reset_index(drop=True)


def analyze(config, dataset_dir, workdir, limit=None, threads=8):
    data = config["data"]
    os.makedirs(os.path.join(workdir, "analysis"), exist_ok=True)
    clips_path = os.path.join(workdir, "analysis", "clips.csv")
    rows = eligible_rows(dataset_dir, config)
    if limit:
        rows = rows.iloc[:: max(1, len(rows) // limit)].head(limit)
    done = pd.read_csv(clips_path) if os.path.exists(clips_path) else pd.DataFrame()
    finished = set(done["index"]) if len(done) else set()
    todo = rows[~rows["index"].isin(finished)]
    print("eligible rows: %d (%.2f h), already analysed: %d" % (len(rows), rows["duration"].sum() / 3600, len(finished)), flush=True)
    if todo.empty:
        return done
    estimator = metrics.SquimEstimator(num_threads=threads)
    by_index = todo.set_index("index")
    results = []
    started = time.time()
    for count, (index, audio_bytes) in enumerate(dataio.iter_audio(dataset_dir, data["audio_column"], by_index.index), 1):
        wav, rate = audio_tools.decode_audio(audio_bytes)
        record = by_index.loc[index].to_dict()
        record["index"] = int(index)
        record.update(metrics.clip_metrics(wav, rate))
        stoi, pesq, si_sdr = estimator.estimate(wav, rate)
        record.update({"stoi": stoi, "pesq": pesq, "si_sdr": si_sdr})
        _, f0 = pitch_tools.raw_f0(wav, rate, 0.01, 50, 700)
        voiced = f0[f0 > 0]
        record["f0_median"] = float(np.median(voiced)) if len(voiced) > 10 else 0.0
        record["voiced_fraction"] = float((f0 > 0).mean())
        results.append(record)
        if count % 250 == 0:
            partial = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
            partial.to_csv(clips_path, index=False, encoding="utf-8")
            print("analysed %d/%d clips, %.0f s" % (count, len(todo), time.time() - started), flush=True)
    frame = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
    frame.to_csv(clips_path, index=False, encoding="utf-8")
    return frame


def quality_score(frame, weights):
    def z(column):
        values = frame[column]
        return (values - values.mean()) / (values.std() + 1e-9)

    return (
        weights["si_sdr"] * z("si_sdr")
        + weights["pesq"] * z("pesq")
        + weights["stoi"] * z("stoi")
        + weights["snr_proxy"] * z("snr_proxy_db")
        + weights["hf_ratio"] * z("hf_ratio_db")
    )


def name_speakers(frame):
    totals = frame.groupby("speaker_key")["duration"].sum().sort_values(ascending=False)
    names = {key: "S%03d" % (rank + 1) for rank, key in enumerate(totals.index)}
    frame["speaker"] = frame["speaker_key"].map(names)
    return names


def build_dictionaries(words, workdir, mfa_dictionary, runtime_lexicon_path):
    entries = lexicon.read_mfa_dictionary(mfa_dictionary)
    known = set(entries)
    spelling = dict(english.LETTER_WORDS)
    missing = sorted(w for w in words if w not in known and w not in spelling)
    g2p = english.Lexicon(runtime_lexicon_path)
    extra = {}
    for word in missing:
        phones = g2p.predict(word)
        if phones:
            extra[word] = phones
    extra.update(spelling)
    os.makedirs(os.path.join(workdir, "mfa"), exist_ok=True)
    alignment_dictionary = os.path.join(workdir, "mfa", "english_alignment.dict")
    lexicon.write_alignment_dictionary(mfa_dictionary, extra, alignment_dictionary)
    print("dictionary words missing from MFA dictionary:", len(missing), "-> predicted pronunciations", len(extra) - len(spelling), flush=True)
    merged = {word: list(variants) for word, variants in entries.items()}
    for word, phones in extra.items():
        merged[word] = [(1.0, phones)]
    homographs = set(g2p.g2p.homograph2features)
    lexicon.write_runtime_lexicon(merged, runtime_lexicon_path, variant_words=homographs)
    return alignment_dictionary, extra


def select_and_export(config, dataset_dir, workdir, mfa_dictionary, runtime_lexicon_path, limit=None):
    data = config["data"]
    audio_config = config["preprocessing"]["audio"]
    clips = pd.read_csv(os.path.join(workdir, "analysis", "clips.csv"), keep_default_na=False)
    clips["speaker_key"] = clips["speaker_key"].astype(str)
    for column in ("stoi", "pesq", "si_sdr", "snr_proxy_db", "hf_ratio_db", "clip_fraction", "f0_median", "voiced_fraction"):
        clips[column] = pd.to_numeric(clips[column])
    clips["quality"] = quality_score(clips, data["quality_weights"])
    gates = (clips["clip_fraction"] <= data["max_clip_fraction"]) & (clips["voiced_fraction"] >= data["min_voiced_fraction"])
    pool = clips[gates].copy()
    cutoff = np.percentile(pool["quality"], data["pool_drop_percent"])
    pool = pool[pool["quality"] >= cutoff].copy()
    print(
        "analysed %d clips (%.2f h); after hard gates %d; after dropping the worst %d%% by quality: %d clips (%.2f h)"
        % (len(clips), clips["duration"].sum() / 3600, int(gates.sum()), data["pool_drop_percent"], len(pool), pool["duration"].sum() / 3600),
        flush=True,
    )
    speaker_names = name_speakers(pool)
    pd.DataFrame(
        [{"speaker": name, "metadata": key} for key, name in speaker_names.items()]
    ).to_csv(os.path.join(workdir, "speaker_table.csv"), index=False, encoding="utf-8")

    all_words = sorted({w for words in pool["words"] for w in words.split()})
    dictionary_path, _ = build_dictionaries(all_words, workdir, mfa_dictionary, runtime_lexicon_path)

    pool = pool.reset_index(drop=True)
    if limit:
        pool = pool.iloc[:: max(1, len(pool) // limit)].head(limit)
    by_index = pool.set_index("index")
    manifest = []
    started = time.time()
    for count, (index, audio_bytes) in enumerate(dataio.iter_audio(dataset_dir, data["audio_column"], by_index.index), 1):
        row = by_index.loc[index]
        wav_tts, wav_mfa, gain, duration = dataio.decode_and_prepare(
            audio_bytes, audio_config["sampling_rate"], MFA_RATE, audio_config["target_db"], audio_config["peak_limit"]
        )
        utt_id = "%s_%s" % (row["speaker"], os.path.splitext(row["orig_path"])[0])
        corpus_dir = os.path.join(workdir, "mfa_corpus", row["speaker"])
        dataio.write_wav(os.path.join(corpus_dir, utt_id + ".wav"), wav_mfa, MFA_RATE)
        with open(os.path.join(corpus_dir, utt_id + ".lab"), "w", encoding="utf-8", newline="\n") as f:
            f.write(row["words"] + "\n")
        dataio.write_wav(os.path.join(workdir, "wav22", row["speaker"], utt_id + ".wav"), wav_tts, audio_config["sampling_rate"])
        manifest.append(
            {
                "utt_id": utt_id,
                "speaker": row["speaker"],
                "orig_index": int(index),
                "text": row["words"],
                "words": row["words"],
                "duration": round(float(duration), 4),
                "gain_db": round(float(20 * np.log10(gain)), 3),
                "quality": float(row["quality"]),
                "si_sdr": float(row["si_sdr"]),
                "pesq": float(row["pesq"]),
                "stoi": float(row["stoi"]),
                "f0_median": float(row["f0_median"]),
            }
        )
        if count % 500 == 0:
            print("exported %d clips, %.0f s" % (count, time.time() - started), flush=True)
    frame = pd.DataFrame(manifest)
    frame.to_csv(os.path.join(workdir, "manifest.csv"), index=False, encoding="utf-8")
    print("exported clips: %d (%.2f h), speakers: %d" % (len(frame), frame["duration"].sum() / 3600, frame["speaker"].nunique()), flush=True)
    return frame


def finalize(candidates, config):
    data = config["data"]
    pool = candidates.copy()
    if "log_likelihood" in pool and pool["log_likelihood"].notna().any():
        cutoff = np.percentile(pool["log_likelihood"].dropna(), data["min_alignment_percentile"])
        before = len(pool)
        pool = pool[pool["log_likelihood"].fillna(cutoff) >= cutoff]
        print("alignment likelihood filter: kept %d of %d clips (cutoff %.3f)" % (len(pool), before, cutoff), flush=True)
        likelihood_z = (pool["log_likelihood"] - pool["log_likelihood"].mean()) / (pool["log_likelihood"].std() + 1e-9)
    else:
        likelihood_z = 0.0
    pool = pool.assign(score=pool["quality"] + data["alignment_weight"] * likelihood_z)
    pool = pool.sort_values("score", ascending=False)
    budget = (data["train_hours"] + data["val_hours"]) * 3600
    chosen = pool[pool["trimmed_seconds"].cumsum() <= budget]
    print(
        "selected the best %.2f h of %.2f h aligned clips from %d speakers"
        % (chosen["trimmed_seconds"].sum() / 3600, pool["trimmed_seconds"].sum() / 3600, chosen["speaker"].nunique()),
        flush=True,
    )
    val_fraction = data["val_hours"] / (data["train_hours"] + data["val_hours"])
    train, val = split.split_per_speaker(chosen, val_fraction, data["seed"], data["min_speaker_seconds"])
    shared = set(train["words"]) & set(val["words"])
    train = train[~train["words"].isin(shared)].reset_index(drop=True)
    print("removed %d training clips whose text also occurs in validation" % len(shared), flush=True)
    return train, val
