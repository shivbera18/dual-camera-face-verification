"""Threshold tuning alias — sweeps ArcFace match threshold on LFW pairs."""
from __future__ import annotations

import argparse

from src.training.eval_verification import run as eval_run
from src.utils.config import get_dataset_config, resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=str, default=None)
    args = p.parse_args()
    if args.pairs is None:
        args.pairs = get_dataset_config()["raw"]["lfw_pairs"]
    eval_run(args)


if __name__ == "__main__":
    main()
