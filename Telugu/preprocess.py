import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastspeech2.preprocessor import mfa
from fastspeech2.preprocessor import telugu as telugu_data
from fastspeech2.utils.config import load_config

HERE = Path(__file__).resolve().parent


def run_select(config, args):
    telugu_data.select_and_export(config, config["paths"]["dataset"], config["paths"]["workdir"], limit=args.limit)


def run_align(config, args):
    workdir = config["paths"]["workdir"]
    mfa_dir = os.path.join(workdir, "mfa")
    mfa.run_mfa(
        [
            "train",
            os.path.join(workdir, "mfa_corpus"),
            os.path.join(mfa_dir, "lexicon.txt"),
            os.path.join(mfa_dir, "telugu_acoustic.zip"),
            "--output_directory",
            os.path.join(mfa_dir, "textgrids"),
            "--num_jobs",
            str(args.jobs),
            "--single_speaker",
            "--clean",
            "--overwrite",
        ],
        root_dir=os.path.join(workdir, "mfa_root"),
        mfa_env=args.mfa_env,
        log_path=os.path.join(mfa_dir, "train.log"),
    )


def run_features(config, args):
    from fastspeech2.preprocessor import pipeline

    pipeline.build_features(config, language="telugu")


def run_pack(config, args):
    from fastspeech2.preprocessor import pipeline

    pipeline.pack(config, args.pack_output)


STAGES = {"select": run_select, "align": run_align, "features": run_features, "pack": run_pack}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES) + ["all"])
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--mfa_env", default=None)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pack_output", default=None)
    args = parser.parse_args()

    config = load_config(
        args.config,
        dataset=args.dataset,
        workdir=args.workdir,
        preprocessed=os.path.join(args.workdir, "preprocessed") if args.workdir else None,
    )
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for stage in stages:
        STAGES[stage](config, args)


if __name__ == "__main__":
    main()
