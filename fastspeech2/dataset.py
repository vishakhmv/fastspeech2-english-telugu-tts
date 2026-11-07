import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fastspeech2.text import parse_phone_field, phones_to_sequence
from fastspeech2.utils.tools import pad_1D, pad_2D


class TTSDataset(Dataset):
    def __init__(self, split_file, preprocessed_dir, language):
        self.preprocessed_dir = preprocessed_dir
        with open(os.path.join(preprocessed_dir, "speakers.json"), encoding="utf-8") as f:
            self.speaker_map = json.load(f)
        self.items = []
        with open(os.path.join(preprocessed_dir, split_file), encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                utt_id, speaker, phone_field, raw_text = line.split("|", 3)
                self.items.append(
                    {
                        "id": utt_id,
                        "speaker": speaker,
                        "text": np.array(phones_to_sequence(parse_phone_field(phone_field), language), dtype=np.int64),
                        "raw_text": raw_text,
                        "mel": self._load("mel", speaker, utt_id),
                        "pitch": self._load("pitch", speaker, utt_id),
                        "energy": self._load("energy", speaker, utt_id),
                        "duration": self._load("duration", speaker, utt_id),
                    }
                )

    def _load(self, kind, speaker, utt_id):
        return np.load(os.path.join(self.preprocessed_dir, kind, "{}-{}-{}.npy".format(speaker, kind, utt_id)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return {
            "id": item["id"],
            "speaker": self.speaker_map[item["speaker"]],
            "text": item["text"],
            "raw_text": item["raw_text"],
            "mel": item["mel"],
            "pitch": item["pitch"],
            "energy": item["energy"],
            "duration": item["duration"],
        }

    def text_lengths(self):
        return np.array([len(item["text"]) for item in self.items])

    def frame_hours(self, sampling_rate, hop_length):
        return sum(len(item["mel"]) for item in self.items) * hop_length / sampling_rate / 3600


class BucketBatchSampler:
    def __init__(self, lengths, batch_size, group_size, shuffle, seed=0, drop_last=False):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.group_size = group_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def _batches(self):
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            indices = self.rng.permutation(indices)
        span = self.batch_size * self.group_size
        batches = []
        for start in range(0, len(indices), span):
            group = indices[start : start + span]
            group = group[np.argsort(-self.lengths[group], kind="stable")]
            for offset in range(0, len(group), self.batch_size):
                batch = group[offset : offset + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch.tolist())
        if self.shuffle:
            order = self.rng.permutation(len(batches))
            batches = [batches[i] for i in order]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        full, remainder = divmod(len(self.lengths), self.batch_size)
        if remainder and not self.drop_last:
            full += 1
        return full


def collate_fn(samples):
    texts = [s["text"] for s in samples]
    mels = [s["mel"] for s in samples]
    return {
        "ids": [s["id"] for s in samples],
        "raw_texts": [s["raw_text"] for s in samples],
        "speakers": np.array([s["speaker"] for s in samples], dtype=np.int64),
        "texts": pad_1D(texts),
        "src_lens": np.array([len(t) for t in texts], dtype=np.int64),
        "mels": pad_2D(mels),
        "mel_lens": np.array([len(m) for m in mels], dtype=np.int64),
        "pitches": pad_1D([s["pitch"] for s in samples]),
        "energies": pad_1D([s["energy"] for s in samples]),
        "durations": pad_1D([s["duration"] for s in samples]),
    }


def to_device(batch, device):
    out = dict(batch)
    for key, dtype in (
        ("speakers", torch.long),
        ("texts", torch.long),
        ("src_lens", torch.long),
        ("mels", torch.float32),
        ("mel_lens", torch.long),
        ("pitches", torch.float32),
        ("energies", torch.float32),
        ("durations", torch.long),
    ):
        out[key] = torch.from_numpy(batch[key]).to(device=device, dtype=dtype, non_blocking=True)
    return out


def make_loader(dataset, batch_size, group_size, shuffle, num_workers, seed=0, drop_last=False):
    sampler = BucketBatchSampler(dataset.text_lengths(), batch_size, group_size, shuffle, seed, drop_last)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
