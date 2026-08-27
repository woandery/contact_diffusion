#!/usr/bin/env python3
"""Create a compact MultiDex-filtered selection for low-memory Isaac Gym replay."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCK_ROOT = PROJECT_ROOT.parent
DRO_ROOT = MCK_ROOT / "DRO-Grasp"


def load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def key(q: torch.Tensor, object_name: str) -> tuple[str, bytes]:
    return object_name, q.detach().cpu().contiguous().numpy().tobytes()


def evenly_spaced(count: int, take: int) -> list[int]:
    if count < take:
        return list(range(count))
    if take == 1:
        return [count // 2]
    return [round(index * (count - 1) / (take - 1)) for index in range(take)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-name",
        choices=("barrett", "shadowhand", "ezgripper"),
        default="shadowhand",
    )
    parser.add_argument(
        "--filtered",
        type=Path,
        help="Defaults to MultiDex_filtered/<robot-name>/<robot-name>.pt.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        help="Defaults to MultiDex/<robot-name>/<robot-name>.pt.",
    )
    parser.add_argument(
        "--seen-file",
        type=Path,
        default=PROJECT_ROOT / "configs/gendex_seen48_objects.json",
    )
    parser.add_argument(
        "--object-key",
        default="train",
        help="Object-list key in --seen-file (for example: train or validate).",
    )
    parser.add_argument("--samples-per-object", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.samples_per_object <= 0:
        raise ValueError("--samples-per-object must be positive")
    filtered_path = args.filtered or (
        DRO_ROOT / "data/MultiDex_filtered" / args.robot_name / f"{args.robot_name}.pt"
    )
    raw_path = args.raw or (
        DRO_ROOT / "data/MultiDex" / args.robot_name / f"{args.robot_name}.pt"
    )

    object_splits = json.loads(args.seen_file.read_text(encoding="utf-8"))
    if args.object_key not in object_splits:
        raise KeyError(
            f"object key {args.object_key!r} is missing from {args.seen_file}"
        )
    seen = object_splits[args.object_key]
    raw = load(raw_path.resolve())["metadata"]
    raw_indices: dict[tuple[str, bytes], deque[int]] = defaultdict(deque)
    for source_index, (q, object_name, hand_name) in enumerate(raw):
        if hand_name == args.robot_name:
            raw_indices[key(q, object_name)].append(source_index)
    filtered = load(filtered_path.resolve())["metadata"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    unmatched = 0
    for filtered_index, (q, object_name, hand_name) in enumerate(filtered):
        if hand_name != args.robot_name:
            continue
        matches = raw_indices[key(q, object_name)]
        if not matches:
            unmatched += 1
            continue
        grouped[object_name].append(
            {
                "q_rot6d": q.tolist(),
                "filtered_index": filtered_index,
                "source_index": matches.popleft(),
            }
        )
    if unmatched:
        raise RuntimeError(f"{unmatched} filtered records did not match raw MultiDex")

    objects = []
    for object_name in seen:
        rows = grouped.get(object_name, [])
        ranks = evenly_spaced(len(rows), args.samples_per_object)
        samples = []
        for rank in ranks:
            sample = dict(rows[rank])
            sample["object_rank"] = rank
            samples.append(sample)
        objects.append(
            {
                "object_name": object_name,
                "filtered_samples_available": len(rows),
                "samples": samples,
            }
        )
    payload = {
        "schema": "multidex-filtered-multihand-object-selection-v3",
        "robot_name": args.robot_name,
        "filtered_dataset": str(filtered_path.resolve()),
        "raw_dataset": str(raw_path.resolve()),
        "seen_file": str(args.seen_file.resolve()),
        "object_key": args.object_key,
        "samples_per_object": args.samples_per_object,
        "objects": objects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "objects": len(objects),
                "recordable": sum(len(row["samples"]) == args.samples_per_object for row in objects),
                "partially_recordable": sum(0 < len(row["samples"]) < args.samples_per_object for row in objects),
                "missing": [row["object_name"] for row in objects if not row["samples"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
