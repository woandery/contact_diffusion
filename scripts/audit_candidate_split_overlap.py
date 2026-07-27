#!/usr/bin/env python3
"""Check whether benchmark object IDs also occur in the row-wise training split."""

from __future__ import annotations

import argparse
import json
import mmap
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.contact_dataset import ContactFormatDataset, build_contact_format_dataset


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(resolve(args.checkpoint), map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(checkpoint["config"])
    records = json.loads(resolve(args.candidates).read_text(encoding="utf-8"))["records"]
    gripper_n = {record["gripper"]: int(record["n"]) for record in records}

    report = {}
    for gripper, n in gripper_n.items():
        selected = sorted(
            {str(record["object_id"]) for record in records if record["gripper"] == gripper}
        )
        dataset = build_contact_format_dataset(
            root_dir=str(model_cfg.dataset.root_dir),
            dataset_dir=list(model_cfg.dataset.dataset_dirs),
            split="train",
            n=n,
            num_points=int(model_cfg.dataset.num_points),
            contact_field=str(model_cfg.dataset.contact_field),
            load_cmap=False,
            load_qpos=False,
            normalize=bool(model_cfg.dataset.normalize),
            split_fractions=tuple(model_cfg.dataset.split_fractions),
            split_names=tuple(model_cfg.dataset.split_names),
            max_samples=None,
            seed=int(model_cfg.train.seed),
            index_cache_dir=str(resolve(model_cfg.dataset.index_cache_dir)),
            shard_cache_size=int(model_cfg.dataset.shard_cache_size),
            object_pc_asset_keys=tuple(model_cfg.dataset.object_pc_asset_keys),
            success_only=True,
            allowed_grippers=[gripper],
            native_n_filter=True,
        )
        if not isinstance(dataset, ContactFormatDataset):
            raise TypeError(f"Expected ContactFormatDataset, got {type(dataset).__name__}")
        train_last_offset = int(dataset.offsets[-1])
        overlap = []
        with dataset.manifest_path.open("rb") as handle:
            mapped = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
            try:
                for object_id in selected:
                    pattern = f'"object_id": {json.dumps(object_id)}'.encode("utf-8")
                    position = 0
                    found_success = False
                    while True:
                        position = mapped.find(pattern, position)
                        if position < 0 or position > train_last_offset:
                            break
                        line_start = mapped.rfind(b"\n", 0, position) + 1
                        line_end = mapped.find(b"\n", position)
                        if line_end < 0:
                            line_end = len(mapped)
                        row = json.loads(mapped[line_start:line_end])
                        if dataset._row_matches_filters(row):
                            found_success = True
                            break
                        position = line_end + 1
                    if found_success:
                        overlap.append(object_id)
            finally:
                mapped.close()
        report[gripper] = {
            "selected_objects": selected,
            "num_selected": len(selected),
            "objects_also_in_train": overlap,
            "num_objects_also_in_train": len(overlap),
            "object_disjoint": len(overlap) == 0,
        }

    serialized = json.dumps(report, indent=2)
    if args.output:
        output_path = resolve(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
