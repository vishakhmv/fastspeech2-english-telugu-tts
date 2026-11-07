import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastspeech2.trainer import train
from fastspeech2.utils.config import load_config

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--vocoder", default=None)
    parser.add_argument("--max_hours", type=float, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, preprocessed=args.data_dir, checkpoints=args.ckpt_dir, vocoder=args.vocoder)
    if args.batch_size:
        config["train"]["optimizer"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    train(
        config,
        max_hours=args.max_hours,
        total_steps=args.total_steps,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
