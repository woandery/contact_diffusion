#!/usr/bin/env python3
"""Summarize Barrett old-parameter Gym results against the prior 477/640 run."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


BASELINE_SUCCESSES = 477
BASELINE_TRIALS = 640
BASELINE_BY_OBJECT = {
    "contactdb_apple": 51,
    "contactdb_camera": 49,
    "contactdb_cylinder_medium": 53,
    "contactdb_door_knob": 40,
    "contactdb_rubber_duck": 54,
    "contactdb_water_bottle": 49,
    "ycb_005_tomato_soup_can": 32,
    "ycb_010_potted_meat_can": 41,
    "ycb_016_pear": 55,
    "ycb_055_baseball": 53,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(Path(path) for path in glob.glob(args.results))
    if not paths:
        raise FileNotFoundError(f"No results match {args.results}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    trials = sum(int(item["trials"]) for item in payloads)
    successes = sum(int(item["successes"]) for item in payloads)
    valid_trials = sum(int(item["valid_trials"]) for item in payloads)
    if trials != BASELINE_TRIALS:
        raise ValueError(f"Expected 640 trials, got {trials}")
    rate = successes / trials
    baseline_rate = BASELINE_SUCCESSES / BASELINE_TRIALS
    object_summaries = [
        next(row for row in item["object_summaries"] if int(row["trials"]) > 0)
        for item in payloads
    ]
    per_object = {
        object_summary["object_name"]: {
            "successes": int(item["successes"]),
            "trials": int(item["trials"]),
            "success_rate": float(item["success_rate"]),
            "previous_successes": BASELINE_BY_OBJECT[object_summary["object_name"]],
            "success_change": (
                int(item["successes"])
                - BASELINE_BY_OBJECT[object_summary["object_name"]]
            ),
            "percentage_point_change": 100.0 * (
                float(item["success_rate"])
                - BASELINE_BY_OBJECT[object_summary["object_name"]] / 64.0
            ),
        }
        for item, object_summary in zip(payloads, object_summaries)
    }
    summary = {
        "schema": "barrett-dro-oldparams-isaacgym-comparison-v1",
        "complete": trials == 640 and valid_trials == 640,
        "current_model": {
            "checkpoint_step": 45000,
            "particles": 32,
            "optimization_steps": 400,
            "trials": trials,
            "valid_trials": valid_trials,
            "successes": successes,
            "success_rate": rate,
        },
        "previous_dro_gym": {
            "successes": BASELINE_SUCCESSES,
            "trials": BASELINE_TRIALS,
            "success_rate": baseline_rate,
        },
        "change": {
            "successes": successes - BASELINE_SUCCESSES,
            "percentage_points": 100.0 * (rate - baseline_rate),
            "relative_percent": 100.0 * (rate / baseline_rate - 1.0),
        },
        "per_object": per_object,
        "protocol": payloads[0]["protocol"],
        "result_files": [str(path.resolve()) for path in paths],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    rows = [
        "# Barrett 当前模型：全旧参数 Isaac Gym 复验",
        "",
        f"- 新结果：{successes}/{trials} = {100.0 * rate:.2f}%",
        f"- 原 D(R,O) Gym：{BASELINE_SUCCESSES}/{BASELINE_TRIALS} = {100.0 * baseline_rate:.2f}%",
        f"- 变化：{successes - BASELINE_SUCCESSES:+d} 次成功，{100.0 * (rate - baseline_rate):+.2f} 个百分点，相对 {100.0 * (rate / baseline_rate - 1.0):+.2f}%",
        "",
        "| 物体 | 原 D(R,O) Gym | 全旧参数 Gym | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for name, item in per_object.items():
        rows.append(
            f"| {name} | {item['previous_successes']}/64 "
            f"({100.0 * item['previous_successes'] / 64.0:.2f}%) | "
            f"{item['successes']}/{item['trials']} "
            f"({100.0 * item['success_rate']:.2f}%) | "
            f"{item['success_change']:+d} / "
            f"{item['percentage_point_change']:+.2f} pp |"
        )
    args.output_md.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
