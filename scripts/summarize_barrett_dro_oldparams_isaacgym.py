#!/usr/bin/env python3
"""Summarize Barrett old-parameter Gym results against the prior 477/640 run."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


BASELINE_SUCCESSES = 477
BASELINE_TRIALS = 640


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
        "| 物体 | 全旧参数 Gym |",
        "|---|---:|",
    ]
    for name, item in per_object.items():
        rows.append(
            f"| {name} | {item['successes']}/{item['trials']} "
            f"({100.0 * item['success_rate']:.2f}%) |"
        )
    args.output_md.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
