#!/usr/bin/env python3
"""Stage the D(R,O)-comparison Barrett candidate set for GenDex Gym A/B."""

from __future__ import annotations

import argparse
from pathlib import Path

from prepare_gendex_isaacgym_matched64x32 import DATASETS, stage_hand


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barrett", type=Path, required=True)
    parser.add_argument("--optimization-steps", type=int, default=800)
    parser.add_argument(
        "--dataset-suffix",
        default="DROAB",
        help="Suffix used to isolate the staged GenDex A/B dataset.",
    )
    args = parser.parse_args()
    if args.optimization_steps <= 0:
        parser.error("--optimization-steps must be positive")
    suffix = args.dataset_suffix.strip().replace("/", "-")
    if not suffix:
        parser.error("--dataset-suffix must not be empty")
    DATASETS["Barrett"] = {
        "dataset": (
            f"ContactDiffusionBarrett-GendexOOD10-FK{args.optimization_steps}-"
            f"Matched64x32Top1-{suffix}"
        ),
        "run": (
            f"ood-barrett-contactdiffusion_newfk{args.optimization_steps}_"
            f"matched64x32_top1_{suffix.lower()}"
        ),
    }
    stage_hand(
        "Barrett",
        args.barrett,
        expected_optimization_steps=args.optimization_steps,
    )


if __name__ == "__main__":
    main()
