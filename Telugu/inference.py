import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastspeech2.inference_runner import run_inference
from fastspeech2.text import telugu
from fastspeech2.utils.config import load_config

HERE = Path(__file__).resolve().parent


def front_end(text, repair):
    return telugu.text_to_phones_and_words(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--checkpoint", default=str(HERE / "checkpoints" / "final.pt"))
    parser.add_argument("--vocoder", default=None)
    parser.add_argument("--output", default=str(HERE.parent / "Results" / "Telugu"))
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--loss_logs", default=str(HERE / "checkpoints"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config, vocoder=args.vocoder)
    run_inference(
        config,
        args.checkpoint,
        config["paths"]["vocoder"],
        args.output,
        front_end,
        speaker=args.speaker,
        device=args.device,
        loss_log_dir=args.loss_logs,
    )


if __name__ == "__main__":
    main()
