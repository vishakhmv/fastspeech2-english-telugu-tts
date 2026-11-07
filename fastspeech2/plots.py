import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

TELUGU_FONTS = [
    "Nirmala UI", "Noto Sans Telugu", "Noto Serif Telugu", "Lohit Telugu",
    "Gautami", "Vani", "Pothana2000", "Akshar Unicode", "Telugu Sangam MN",
]


def configure_fonts():
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in TELUGU_FONTS if name in available]
    if chosen:
        plt.rcParams["font.family"] = ["DejaVu Sans"] + chosen
    return chosen


configure_fonts()

LOSS_TITLES = {
    "total": "Total loss",
    "mel": "Mel loss (decoder)",
    "postnet_mel": "Mel loss (after PostNet)",
    "pitch": "Pitch loss",
    "energy": "Energy loss",
    "duration": "Duration loss",
}
PAUSE_COLOR = "#e69f00"
PHONE_COLOR = "#0072b2"
MEASURED_COLOR = "#cc3311"


def _edges_seconds(durations, hop_length, sampling_rate):
    frames = np.concatenate([[0], np.cumsum(durations)])
    return frames * hop_length / sampling_rate


def plot_loss_curves(csv_path, output_path, title, best_step=None):
    frame = pd.read_csv(csv_path).drop_duplicates("step", keep="last").sort_values("step")
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for axis, name in zip(axes.ravel(), LOSS_TITLES):
        axis.plot(frame["step"], frame[name], color=PHONE_COLOR, linewidth=1.4)
        if name == "postnet_mel" and "free_mel" in frame:
            axis.plot(frame["step"], frame["free_mel"], color=PAUSE_COLOR, linewidth=1.4, label="with predicted pitch and energy")
            axis.legend(loc="upper right", fontsize=8)
        axis.set_title(LOSS_TITLES[name])
        axis.set_ylabel("loss")
        axis.grid(alpha=0.3)
        if best_step is not None:
            axis.axvline(best_step, color=MEASURED_COLOR, linestyle="--", linewidth=1, label="best checkpoint (step %d)" % best_step)
    for axis in axes[-1]:
        axis.set_xlabel("training step")
    if best_step is not None:
        axes[0][0].legend(loc="upper right", fontsize=8)
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def plot_mel_comparison(output_path, mel_target, mel_prediction, title):
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for axis, mel, label in zip(axes, (mel_prediction, mel_target), ("predicted", "ground truth")):
        axis.imshow(mel.T, origin="lower", aspect="auto", vmin=-11, vmax=2)
        axis.set_ylabel("mel bin")
        axis.set_title(label, fontsize=10)
    axes[-1].set_xlabel("frame")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=110)
    plt.close(figure)


def plot_variance(
    output_path,
    kind,
    title,
    phones,
    durations,
    mel,
    hop_length,
    sampling_rate,
    values,
    words=None,
    measured=None,
):
    durations = np.asarray(durations)
    edges = _edges_seconds(durations, hop_length, sampling_rate)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    total_seconds = edges[-1]
    is_pause = np.array([p in ("sp", "spn") for p in phones])

    width = float(np.clip(0.24 * len(phones) + 2, 12, 28))
    figure, (mel_axis, axis) = plt.subplots(
        2, 1, figsize=(width, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.08}
    )

    mel_axis.imshow(
        mel.T,
        origin="lower",
        aspect="auto",
        extent=[0, mel.shape[0] * hop_length / sampling_rate, 0, mel.shape[1]],
        vmin=-11,
        vmax=2,
        cmap="viridis",
    )
    for edge in edges:
        mel_axis.axvline(edge, color="white", alpha=0.25, linewidth=0.5)
        axis.axvline(edge, color="gray", alpha=0.25, linewidth=0.5)
    mel_axis.set_ylabel("mel bin")
    mel_axis.set_title(title, fontsize=12)
    if words:
        spans = [((edges[s] + edges[e]) / 2, w) for w, s, e in words]
        top = mel_axis.secondary_xaxis("top")
        top.set_xticks([x for x, _ in spans])
        top.set_xticklabels([w for _, w in spans], rotation=45, ha="left", fontsize=9)
        top.tick_params(length=2)

    colors = [PAUSE_COLOR if p else PHONE_COLOR for p in is_pause]
    if kind == "duration":
        axis.bar(centers, durations * hop_length / sampling_rate * 1000, width=widths, color=colors, edgecolor="white", linewidth=0.6)
        axis.set_ylabel("predicted duration (ms)")
    elif kind == "pitch":
        axis.hlines(values, edges[:-1], edges[1:], colors=colors, linewidth=3)
        axis.plot(centers[~is_pause], np.asarray(values)[~is_pause], color=PHONE_COLOR, linewidth=1, alpha=0.6, label="predicted F0 (per phone)")
        if measured is not None:
            times, f0 = measured
            voiced = f0 > 0
            axis.plot(times[voiced], f0[voiced], color=MEASURED_COLOR, linewidth=1.2, alpha=0.85, label="F0 measured on generated audio")
        axis.set_ylabel("F0 (Hz)")
        axis.legend(loc="upper right", fontsize=8)
    else:
        axis.hlines(values, edges[:-1], edges[1:], colors=colors, linewidth=3)
        axis.plot(centers[~is_pause], np.asarray(values)[~is_pause], color=PHONE_COLOR, linewidth=1, alpha=0.6, label="predicted energy (per phone)")
        if measured is not None:
            times, energy = measured
            axis.plot(times, energy, color=MEASURED_COLOR, linewidth=1.0, alpha=0.8, label="energy measured on generated audio")
        axis.set_ylabel("energy (L2 norm of STFT magnitude)")
        axis.legend(loc="upper right", fontsize=8)
    axis.set_xlim(0, total_seconds)
    axis.set_xticks(centers)
    axis.set_xticklabels(phones, rotation=90, fontsize=7)
    axis.set_xlabel("phones (orange = pause); x position = time in the generated utterance")
    axis.grid(axis="y", alpha=0.3)
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(figure)
