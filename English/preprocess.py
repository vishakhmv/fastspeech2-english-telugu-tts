import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastspeech2.preprocessor import english as english_data
from fastspeech2.preprocessor import mfa
from fastspeech2.text import english as english_text
from fastspeech2.utils.config import load_config

HERE = Path(__file__).resolve().parent

ALIGN_CONFIG = "beam: 100\nretry_beam: 400\n"


def run_analyze(config, args):
    english_data.analyze(config, config["paths"]["dataset"], config["paths"]["workdir"], limit=args.limit, threads=args.jobs)


def run_select(config, args):
    english_data.select_and_export(
        config,
        config["paths"]["dataset"],
        config["paths"]["workdir"],
        args.mfa_dictionary or config["paths"]["mfa_dictionary"],
        str(english_text.LEXICON_PATH),
        limit=args.limit,
    )


def run_align(config, args):
    workdir = config["paths"]["workdir"]
    mfa_dir = os.path.join(workdir, "mfa")
    acoustic = args.mfa_acoustic_model or config["paths"]["mfa_acoustic_model"]
    config_path = os.path.join(mfa_dir, "align_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(ALIGN_CONFIG)
    common = ["--num_jobs", str(args.jobs), "--clean", "--overwrite", "--config_path", config_path]
    corpus = os.path.join(workdir, "mfa_corpus")
    dictionary = os.path.join(mfa_dir, "english_alignment.dict")
    output = os.path.join(mfa_dir, "textgrids")
    if args.adapt:
        mfa.run_mfa(
            ["adapt", corpus, dictionary, acoustic, os.path.join(mfa_dir, "english_adapted.zip"), "--output_directory", output] + common,
            root_dir=os.path.join(workdir, "mfa_root"),
            mfa_env=args.mfa_env,
            log_path=os.path.join(mfa_dir, "adapt.log"),
        )
    else:
        mfa.run_mfa(
            ["align", corpus, dictionary, acoustic, output] + common,
            root_dir=os.path.join(workdir, "mfa_root"),
            mfa_env=args.mfa_env,
            log_path=os.path.join(mfa_dir, "align.log"),
        )


def run_features(config, args):
    from fastspeech2.preprocessor import pipeline

    pipeline.build_features(config, language="english")


def run_pack(config, args):
    from fastspeech2.preprocessor import pipeline

    pipeline.pack(config, args.pack_output)


STAGES = {
    "analyze": run_analyze,
    "select": run_select,
    "align": run_align,
    "features": run_features,
    "pack": run_pack,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES) + ["all"])
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--mfa_env", default=None)
    parser.add_argument("--mfa_dictionary", default=None)
    parser.add_argument("--mfa_acoustic_model", default=None)
    parser.add_argument("--adapt", action="store_true")
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
