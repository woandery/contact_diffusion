#!/usr/bin/env python3
"""Aggregate per-hand manifests for the MultiDex-filtered OOD replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robots", nargs="+", required=True)
    args = parser.parse_args()

    hands = {}
    for robot_name in args.robots:
        manifest_path = args.output_root / robot_name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hands[robot_name] = {
            "manifest": str(manifest_path.resolve()),
            "objects_requested": manifest["objects_requested"],
            "objects_completed": manifest["objects_completed"],
            "videos_valid": manifest["videos_valid"],
            "poses_recorded": manifest["poses_recorded"],
            "local_final_successes": manifest["local_final_successes"],
            "local_strict_six_direction_successes": manifest[
                "local_strict_six_direction_successes"
            ],
        }

    summary = {
        "schema": "multidex-filtered-threehand-ood-recordings-v1",
        "selection_rule": "min(filtered_success_count, 32), no repeats",
        "poses_per_video_max": 8,
        "robots": args.robots,
        "poses_recorded": sum(row["poses_recorded"] for row in hands.values()),
        "videos_valid": sum(row["videos_valid"] for row in hands.values()),
        "local_final_successes": sum(
            row["local_final_successes"] for row in hands.values()
        ),
        "local_strict_six_direction_successes": sum(
            row["local_strict_six_direction_successes"] for row in hands.values()
        ),
        "hands": hands,
    }
    destination = args.output_root / "manifest.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
