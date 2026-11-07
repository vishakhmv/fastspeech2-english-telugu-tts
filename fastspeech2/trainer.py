import csv
import json
import os
import random
import time

import numpy as np
import torch

from fastspeech2.dataset import TTSDataset, make_loader, to_device
from fastspeech2.model import FastSpeech2, FastSpeech2Loss, ScheduledOptim
from fastspeech2.text import get_symbols

LOSS_NAMES = ["total", "mel", "postnet_mel", "pitch", "energy", "duration"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config, stats, n_speakers, device):
    symbols = get_symbols(config["language"])
    model = FastSpeech2(config["preprocessing"], config["model"], len(symbols), n_speakers, stats)
    return model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_payload(model, config, stats, speakers, step, best=None, optimizer=None, scaler=None):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "config": {k: v for k, v in config.items()},
        "stats": stats,
        "speakers": speakers,
        "symbols": get_symbols(config["language"]),
    }
    if best is not None:
        payload["best"] = best
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def atomic_save(payload, path):
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_csv(path, header, row):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp):
    model.eval()
    sums = np.zeros(len(LOSS_NAMES))
    count = 0
    free_error = 0.0
    free_cells = 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(
                batch["speakers"],
                batch["texts"],
                batch["src_lens"],
                batch["mel_lens"],
                batch["pitches"],
                batch["energies"],
                batch["durations"],
            )
            free = model(
                batch["speakers"],
                batch["texts"],
                batch["src_lens"],
                batch["mel_lens"],
                None,
                None,
                batch["durations"],
            )
        losses = loss_fn(batch, output)
        n = len(batch["ids"])
        sums += np.array([l.item() for l in losses]) * n
        count += n
        valid = ~free[7]
        targets = batch["mels"][:, : valid.shape[1], :]
        error = (free[1].float() - targets).abs().mean(dim=-1)
        free_error += error.masked_select(valid).sum().item()
        free_cells += valid.sum().item()
    model.train()
    return sums / max(count, 1), free_error / max(free_cells, 1)


def train(config, data_dir=None, ckpt_dir=None, max_hours=None, total_steps=None, resume=True, vocoder_path=None):
    language = config["language"]
    train_config = config["train"]
    optimizer_config = train_config["optimizer"]
    step_config = train_config["step"]
    data_dir = data_dir or config["paths"]["preprocessed"]
    ckpt_dir = ckpt_dir or config["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    total_steps = total_steps or step_config["total_step"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(train_config.get("amp", False)) and device.type == "cuda"
    set_seed(config["data"]["seed"])

    with open(os.path.join(data_dir, "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    with open(os.path.join(data_dir, "speakers.json"), encoding="utf-8") as f:
        speakers = json.load(f)

    train_set = TTSDataset("train.txt", data_dir, language)
    val_set = TTSDataset("val.txt", data_dir, language)
    batch_size = optimizer_config["batch_size"]
    train_loader = make_loader(
        train_set, batch_size, train_config["group_size"], True, train_config["num_workers"],
        seed=config["data"]["seed"], drop_last=True,
    )
    val_loader = make_loader(val_set, batch_size, 1, False, 0)

    model = build_model(config, stats, len(speakers), device)
    optimizer = ScheduledOptim(model, train_config, config["model"], 0)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = FastSpeech2Loss().to(device)

    step = 0
    best = {"step": 0, "value": float("inf")}
    last_path = os.path.join(ckpt_dir, "last.pt")
    if resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler"])
        step = checkpoint["step"]
        best = checkpoint.get("best", best)
        optimizer.current_step = step
        print("resumed from step", step, flush=True)

    vocoder = None
    vocoder_path = vocoder_path or config["paths"].get("vocoder")
    if vocoder_path and os.path.exists(vocoder_path):
        from fastspeech2.hifigan import load_generator

        vocoder = load_generator(vocoder_path, device=device)

    print(
        "language %s | train %d utts | val %d utts | params %.2fM | device %s | amp %s | steps %d -> %d"
        % (language, len(train_set), len(val_set), count_parameters(model) / 1e6, device, use_amp, step, total_steps),
        flush=True,
    )

    train_log = os.path.join(ckpt_dir, "train_log.csv")
    val_log = os.path.join(ckpt_dir, "val_log.csv")
    log_step = step_config["log_step"]
    val_step = step_config["val_step"]
    save_step = step_config["save_step"]
    synth_step = step_config["synth_step"]
    keep_every = step_config["keep_every"]
    grad_acc = optimizer_config["grad_acc_step"]
    clip = optimizer_config["grad_clip_thresh"]

    started = time.time()
    window = np.zeros(len(LOSS_NAMES))
    window_count = 0
    micro = 0
    model.train()
    optimizer.zero_grad()
    stop = False

    def save_last():
        atomic_save(checkpoint_payload(model, config, stats, speakers, step, best, optimizer, scaler), last_path)

    while step < total_steps and not stop:
        for batch in train_loader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(
                    batch["speakers"],
                    batch["texts"],
                    batch["src_lens"],
                    batch["mel_lens"],
                    batch["pitches"],
                    batch["energies"],
                    batch["durations"],
                )
            losses = loss_fn(batch, output)
            if not torch.isfinite(losses[0]):
                print("non-finite loss at step", step + 1, "- skipping batch", flush=True)
                optimizer.zero_grad()
                micro = 0
                continue
            scaler.scale(losses[0] / grad_acc).backward()
            micro += 1
            if micro % grad_acc != 0:
                continue
            scaler.unscale_(optimizer._optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.update_learning_rate()
            scaler.step(optimizer._optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1

            window += np.array([l.item() for l in losses])
            window_count += 1

            if step % log_step == 0:
                means = window / window_count
                append_csv(
                    train_log,
                    ["step"] + LOSS_NAMES + ["lr", "seconds"],
                    [step] + ["%.5f" % v for v in means] + ["%.3e" % optimizer.current_lr(), "%.0f" % (time.time() - started)],
                )
                print(
                    "step %d/%d total %.4f mel %.4f post %.4f pitch %.4f energy %.4f dur %.4f lr %.2e | %.1f s"
                    % ((step, total_steps) + tuple(means) + (optimizer.current_lr(), time.time() - started)),
                    flush=True,
                )
                window = np.zeros(len(LOSS_NAMES))
                window_count = 0

            if step % val_step == 0 or step == total_steps:
                values, free_mel = evaluate(model, val_loader, loss_fn, device, use_amp)
                append_csv(val_log, ["step"] + LOSS_NAMES + ["free_mel"], [step] + ["%.5f" % v for v in values] + ["%.5f" % free_mel])
                print(
                    "validation step %d: " % step
                    + " ".join("%s %.4f" % (n, v) for n, v in zip(LOSS_NAMES, values))
                    + " free_mel %.4f" % free_mel,
                    flush=True,
                )
                if free_mel < best["value"]:
                    best = {"step": step, "value": float(free_mel)}
                    atomic_save(checkpoint_payload(model, config, stats, speakers, step, best), os.path.join(ckpt_dir, "best.pt"))

            if step % save_step == 0:
                save_last()
            if keep_every and step % keep_every == 0:
                atomic_save(checkpoint_payload(model, config, stats, speakers, step), os.path.join(ckpt_dir, "step_%d.pt" % step))
            if vocoder is not None and synth_step and step % synth_step == 0:
                from fastspeech2.synthesis import render_validation_samples

                render_validation_samples(model, vocoder, val_set, device, os.path.join(ckpt_dir, "samples", "step_%d" % step), config, stats)
                model.train()

            if max_hours and (time.time() - started) > max_hours * 3600:
                print("time limit reached at step", step, flush=True)
                stop = True
                break
            if step >= total_steps:
                break

    save_last()
    atomic_save(checkpoint_payload(model, config, stats, speakers, step, best), os.path.join(ckpt_dir, "final.pt"))
    print("finished at step", step, "best validation free-running mel loss %.4f at step %d" % (best["value"], best["step"]), flush=True)
    return step
