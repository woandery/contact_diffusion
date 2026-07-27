#!/usr/bin/env python3
"""Summarize fixed-gripper Isaac friction results with Wilson confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def aggregate(records: list[dict]) -> dict:
    planned = len(records)
    present = [record for record in records if record.get("_result_present")]
    simulated = [record for record in present if record.get("status") == "ok"]
    successes = sum(record.get("success") is True for record in simulated)
    valid_initializations = sum(
        record.get("valid_initialization") is True for record in simulated
    )
    displacements = [
        float(record["displacement_m"]) * 1000.0
        for record in simulated
        if record.get("displacement_m") is not None
    ]
    tip_chamfers = [
        float(record["actual_tip_chamfer_m"]) * 1000.0
        for record in simulated
        if record.get("actual_tip_chamfer_m") is not None
    ]
    return {
        "planned": planned,
        "result_files_present": len(present),
        "simulated": len(simulated),
        "errors": len(present) - len(simulated),
        "missing": planned - len(present),
        "valid_initializations": valid_initializations,
        "successes": successes,
        "success_rate_planned": successes / planned if planned else None,
        "success_rate_simulated": successes / len(simulated) if simulated else None,
        "success_rate_simulated_wilson95": wilson(successes, len(simulated)),
        "median_displacement_mm": statistics.median(displacements) if displacements else None,
        "median_actual_tip_chamfer_mm": statistics.median(tip_chamfers) if tip_chamfers else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        default="outputs/isaacsim_validation/fixed_three_grippers_friction_benchmark",
    )
    args = parser.parse_args()
    benchmark_dir = resolve(args.benchmark_dir)
    manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))

    records = []
    for job in manifest["jobs"]:
        result_path = Path(job["result"])
        record = dict(job)
        record["_result_present"] = result_path.exists()
        if result_path.exists():
            try:
                record.update(json.loads(result_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as error:
                record.update(status="error", error=f"JSONDecodeError: {error}")
        records.append(record)

    by_gripper_mu: dict[tuple[str, float], list[dict]] = defaultdict(list)
    by_gripper: dict[str, list[dict]] = defaultdict(list)
    by_object_mu: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for record in records:
        gripper = str(record["gripper"])
        friction = float(record["friction"])
        object_id = str(record["object_id"])
        by_gripper_mu[(gripper, friction)].append(record)
        by_gripper[gripper].append(record)
        by_object_mu[(gripper, object_id, friction)].append(record)

    rows = []
    for (gripper, friction), group in sorted(by_gripper_mu.items()):
        row = {"gripper": gripper, "friction": friction, **aggregate(group)}
        row["num_objects"] = len({record["object_id"] for record in group})
        row["samples_per_object"] = sorted(
            {
                sum(
                    item["object_id"] == object_id
                    for item in group
                )
                for object_id in {record["object_id"] for record in group}
            }
        )
        rows.append(row)

    object_rows = [
        {
            "gripper": gripper,
            "object_id": object_id,
            "friction": friction,
            **aggregate(group),
        }
        for (gripper, object_id, friction), group in sorted(by_object_mu.items())
    ]
    summary = {
        "checkpoint": manifest.get("checkpoint"),
        "checkpoint_step": manifest.get("checkpoint_step"),
        "frictions": manifest["frictions"],
        "overall": aggregate(records),
        "by_gripper": {
            gripper: aggregate(group) for gripper, group in sorted(by_gripper.items())
        },
        "by_gripper_and_friction": rows,
        "by_object_and_friction": object_rows,
    }
    (benchmark_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (benchmark_dir / "summary_by_gripper_mu.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary["overall"], indent=2))
    for row in rows:
        print(
            f"{row['gripper']:14s} mu={row['friction']:.1f} "
            f"success={row['successes']}/{row['simulated']} "
            f"rate={row['success_rate_simulated']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
