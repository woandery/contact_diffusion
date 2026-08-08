#!/usr/bin/env python3
"""Stage matched-budget ContactDiffusion top-1 grasps for GenDex Isaac Gym."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prepare_gendex_isaacgym_fk800 import (
    GENDEX,
    HAND_SPECS,
    OBJECT_MAP,
    rotation_6d,
)


DATASETS = {
    "Barrett": {
        "dataset": "ContactDiffusionBarrett-GendexOOD10-FK800-Matched64x32Top1",
        "run": "ood-barrett-contactdiffusion_newfk800_matched64x32_top1",
    },
    "shadow_hand": {
        "dataset": "ContactDiffusionShadowhand-GendexOOD10-FK800-Matched64x32Top1",
        "run": "ood-shadowhand-contactdiffusion_newfk800_matched64x32_top1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barrett", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    return parser.parse_args()


def load_records(
    path: Path,
    expected_hand: str,
    *,
    expected_optimization_steps: int = 800,
) -> dict[tuple[str, int], dict]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if int(payload.get("particles", -1)) != 32:
        raise ValueError(f"{path}: expected particles=32")
    if int(payload.get("optimization_steps", -1)) != expected_optimization_steps:
        raise ValueError(
            f"{path}: expected optimization_steps="
            f"{expected_optimization_steps}"
        )
    if int(payload["selection"].get("samples_per_object", -1)) != 64:
        raise ValueError(f"{path}: expected samples_per_object=64")
    records = {}
    for record in payload["records"]:
        if record["gripper"] != expected_hand:
            raise ValueError(
                f"{path}: expected hand {expected_hand}, got {record['gripper']}"
            )
        key = (str(record["object_id"]), int(record["sample_index"]))
        if key in records:
            raise ValueError(f"{path}: duplicate record {key}")
        candidates = record["fk"]["candidates"]
        if len(candidates) != 1 or int(candidates[0]["rank"]) != 0:
            raise ValueError(f"{path}: {key} must retain exactly rank-0")
        records[key] = record
    expected = {
        (object_id, sample_index)
        for object_id in OBJECT_MAP
        for sample_index in range(64)
    }
    if set(records) != expected:
        raise ValueError(
            f"{path}: record mismatch; missing={len(expected - set(records))}, "
            f"extra={len(set(records) - expected)}"
        )
    return records


def stage_hand(
    hand: str,
    source: Path,
    *,
    expected_optimization_steps: int = 800,
) -> None:
    records = load_records(
        source,
        hand,
        expected_optimization_steps=expected_optimization_steps,
    )
    hand_spec = HAND_SPECS[hand]
    run_spec = DATASETS[hand]
    dataset_dir = GENDEX / "logs_gen" / run_spec["dataset"]
    base_dir = dataset_dir / run_spec["run"] / "align_dist"
    trajectory_dir = base_dir / "tra_dir"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "split_train_validate_objects.json").write_text(
        json.dumps(
            {"train": [], "validate": list(OBJECT_MAP.values())},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    staged = []
    for object_id, gendex_name in OBJECT_MAP.items():
        for sample_index in range(64):
            record = records[(object_id, sample_index)]
            fk = record["fk"]
            if list(fk["joint_names"]) != hand_spec["expected_joints"]:
                raise ValueError(
                    f"{hand}/{object_id}/{sample_index}: unexpected joints"
                )
            candidate = fk["candidates"][0]
            pose = torch.as_tensor(candidate["root_pose"], dtype=torch.float32)
            joints = torch.as_tensor(
                candidate["joint_positions"], dtype=torch.float32
            )
            q = torch.cat(
                (pose[:3, 3], rotation_6d(pose[:3, :3]), joints),
                dim=0,
            )
            destination = trajectory_dir / f"tra-{gendex_name}-{sample_index}.pt"
            torch.save(
                {
                    "q_tra": q.reshape(1, 1, -1),
                    "energy": torch.as_tensor(
                        [float(candidate["optimization_score"])],
                        dtype=torch.float32,
                    ),
                    "object_name": gendex_name,
                    "i_sample": sample_index,
                    "source_method": "ContactDiffusion-new-FK-matched64x32-top1",
                    "source_record": record["record_id"],
                    "candidate_rank": 0,
                    "optimization_score": float(
                        candidate["optimization_score"]
                    ),
                },
                destination,
            )
            staged.append(str(destination.resolve()))

    provenance = {
        "schema": "contactdiffusion-gendex-isaacgym-matched64x32-top1-v1",
        "hand": hand,
        "robot": hand_spec["robot"],
        "dataset": run_spec["dataset"],
        "objects": list(OBJECT_MAP.values()),
        "contact_sets_per_object": 64,
        "particles_per_set": 32,
        "retained_per_set": 1,
        "count_per_object": 64,
        "total": len(staged),
        "optimization_steps": expected_optimization_steps,
        "candidate_policy": (
            "64 independently seeded diffusion contact sets per object; "
            "optimize 32 FK particles per set and retain energy rank-0"
        ),
        "exact_gendex_hand_urdf": True,
        "source": str(source.resolve()),
    }
    (base_dir / "contactdiffusion_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


def main() -> None:
    args = parse_args()
    stage_hand("Barrett", args.barrett)
    stage_hand("shadow_hand", args.shadow)


if __name__ == "__main__":
    main()
