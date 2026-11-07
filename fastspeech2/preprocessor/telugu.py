import os
import re
import time

import numpy as np
import pandas as pd

from fastspeech2.text import telugu
from fastspeech2.preprocessor import dataio, lexicon, split

LATIN = re.compile(r"[A-Za-z]")


def has_unsupported_characters(text):
    cleaned = re.sub("(?<=[" + telugu.TELUGU_BLOCK + "])o", "", text)
    return bool(LATIN.search(cleaned))


def letter_count(normalized):
    return sum(1 for ch in normalized if telugu.is_telugu_char(ch))


def candidate_rows(dataset_dir, config):
    data = config["data"]
    frame = dataio.read_metadata(dataset_dir, data["audio_column"], ["text", "gender"])
    frame = frame[frame["gender"] == data["gender_value"]].copy()
    frame["normalized"] = frame["text"].map(telugu.normalize)
    frame["words"] = frame["normalized"].map(lambda t: " ".join(w for w in telugu.tokenize(t) if w not in telugu.PUNCTUATION_PAUSE))
    frame["letters"] = frame["normalized"].map(letter_count)
    frame["unsupported"] = frame["text"].map(has_unsupported_characters)
    return frame[(frame["words"].str.len() > 0) & (~frame["unsupported"])].reset_index(drop=True)


def select_and_export(config, dataset_dir, workdir, limit=None):
    data = config["data"]
    audio_config = config["preprocessing"]["audio"]
    rows = candidate_rows(dataset_dir, config)
    if limit:
        rows = rows.iloc[:: max(1, len(rows) // limit)].head(limit)
    by_index = rows.set_index("index")
    print("male rows with usable text:", len(rows), flush=True)

    corpus_dir = os.path.join(workdir, "mfa_corpus", data["speaker"])
    wav_dir = os.path.join(workdir, "wav22", data["speaker"])
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    manifest = []
    dropped = {"duration": 0, "cps": 0, "clipped": 0}
    started = time.time()
    for done, (index, audio_bytes) in enumerate(dataio.iter_audio(dataset_dir, data["audio_column"], by_index.index), 1):
        row = by_index.loc[index]
        wav_tts, wav_mfa, gain, duration = dataio.decode_and_prepare(
            audio_bytes,
            audio_config["sampling_rate"],
            data["mfa_sampling_rate"],
            audio_config["target_db"],
            audio_config["peak_limit"],
        )
        cps = row["letters"] / duration
        if not data["min_duration"] <= duration <= data["max_duration"]:
            dropped["duration"] += 1
            continue
        if not data["chars_per_second"][0] <= cps <= data["chars_per_second"][1]:
            dropped["cps"] += 1
            continue
        if (abs(wav_tts) >= 0.999).mean() > 0.001:
            dropped["clipped"] += 1
            continue
        utt_id = os.path.splitext(row["orig_path"])[0]
        dataio.write_wav(os.path.join(corpus_dir, utt_id + ".wav"), wav_mfa, data["mfa_sampling_rate"])
        with open(os.path.join(corpus_dir, utt_id + ".lab"), "w", encoding="utf-8", newline="\n") as f:
            f.write(row["words"] + "\n")
        dataio.write_wav(os.path.join(wav_dir, utt_id + ".wav"), wav_tts, audio_config["sampling_rate"])
        manifest.append(
            {
                "utt_id": utt_id,
                "speaker": data["speaker"],
                "orig_index": int(index),
                "text": row["normalized"],
                "words": row["words"],
                "duration": round(float(duration), 4),
                "gain_db": round(float(20 * np.log10(gain)), 3),
                "cps": round(float(cps), 3),
            }
        )
        if done % 500 == 0:
            print("exported %d clips, %.0f s" % (len(manifest), time.time() - started), flush=True)

    frame = pd.DataFrame(manifest)
    frame.to_csv(os.path.join(workdir, "manifest.csv"), index=False, encoding="utf-8")
    print("selected clips:", len(frame), "hours:", round(frame["duration"].sum() / 3600, 3), "dropped:", dropped, flush=True)

    entries = telugu.lexicon_entries(frame["text"])
    lexicon_path = os.path.join(workdir, "mfa", "lexicon.txt")
    os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
    lexicon.write_simple_dictionary(entries, lexicon_path)
    print("lexicon words:", len(entries), flush=True)
    return frame


def finalize(candidates, config):
    data = config["data"]
    pool = candidates.copy()
    if "log_likelihood" in pool and pool["log_likelihood"].notna().any():
        cutoff = np.percentile(pool["log_likelihood"].dropna(), data["min_alignment_percentile"])
        before = len(pool)
        pool = pool[pool["log_likelihood"].fillna(cutoff) >= cutoff]
        print("alignment likelihood filter: kept %d of %d clips (cutoff %.3f)" % (len(pool), before, cutoff), flush=True)
    print("pool after filtering: %.3f h" % (pool["trimmed_seconds"].sum() / 3600), flush=True)
    return split.split_groups(pool, data["train_hours"], data["val_hours"], data["seed"])


def session_of(utt_id, sessions):
    index = int(utt_id.rsplit("_", 1)[1])
    for name, ranges in sessions.items():
        for first, last in ranges:
            if first <= index <= last:
                return name
    raise ValueError("no session for " + utt_id)


def assign_speakers(candidates, config):
    sessions = config["data"]["sessions"]
    candidates = candidates.copy()
    candidates["speaker"] = candidates["utt_id"].map(lambda u: session_of(u, sessions))
    print("clips per recording session:", candidates["speaker"].value_counts().to_dict(), flush=True)
    return candidates
