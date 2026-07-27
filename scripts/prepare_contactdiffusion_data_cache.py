#!/usr/bin/env python3
"""Build filtered Contact Format offset caches before launching DDP training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from torch.utils.data import ConcatDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import build_loaders, load_config  # noqa: E402


def component_lengths(dataset):
    if isinstance(dataset, ConcatDataset):
        return [len(component) for component in dataset.datasets]
    return [len(dataset)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ContactDiffusion manifest offset caches in one process."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    for split in args.splits:
        print(f"Preparing split={split} ...", flush=True)
        loaders = build_loaders(cfg, split=split, shuffle=False)
        for n, loader in sorted(loaders.items()):
            lengths = component_lengths(loader.dataset)
            print(
                f"split={split} n={n} total={sum(lengths)} "
                f"components={lengths}",
                flush=True,
            )
    print("Manifest offset caches are ready.", flush=True)


if __name__ == "__main__":
    main()
