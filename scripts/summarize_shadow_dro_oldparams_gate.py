#!/usr/bin/env python3
"""Summarize the full-old-parameter D(R,O) Gym+Sim recovery gate."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def load(pattern: str) -> list[dict]:
    rows = []
    for name in sorted(glob.glob(pattern)):
        rows.extend(json.loads(Path(name).read_text(encoding="utf-8"))["results"])
    return rows


def stats(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("valid_simulation") is True]
    successes = sum(row.get("success") is True for row in rows)
    return {
        "successes": successes,
        "trials": len(rows),
        "valid_trials": len(valid),
        "invalid_trials": len(rows) - len(valid),
        "success_rate": successes / len(rows) if rows else 0.0,
        "valid_success_rate": successes / len(valid) if valid else 0.0,
    }


def per_object(rows: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["object_name"])].append(row)
    return {name: stats(selected) for name, selected in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym", required=True)
    parser.add_argument("--sim", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--gym-threshold", type=float, default=0.70)
    parser.add_argument("--sim-threshold", type=float, default=0.90)
    args = parser.parse_args()
    gym_rows, sim_rows = load(args.gym), load(args.sim)
    gym, sim = stats(gym_rows), stats(sim_rows)
    complete = len(gym_rows) == 640 and len(sim_rows) == 640
    recovered = (
        complete
        and gym["success_rate"] >= args.gym_threshold
        and sim["success_rate"] >= args.sim_threshold
    )
    payload = {
        "schema": "shadow-dro-oldparams-recovery-gate-v1",
        "complete": complete,
        "recovered": recovered,
        "decision": "run_single_parameter_ablations" if recovered else "failure_report",
        "thresholds": {"gym": args.gym_threshold, "sim": args.sim_threshold},
        "gym": gym,
        "sim": sim,
        "per_object": {"gym": per_object(gym_rows), "sim": per_object(sim_rows)},
        "fixed_components": [
            "D(R,O) FK candidates",
            "D(R,O) ShadowHand base/extended URDF",
            "exact pose-dependent D(R,O) 25% outer / 15% inner closure",
        ],
        "aligned_old_protocol": {
            "steps_per_second": 60,
            "substeps": 2,
            "closure_steps": 200,
            "direction_seconds": 0.8333333333333334,
            "direction_order": ["+x", "-x", "+y", "-y", "+z", "-z"],
            "success": "every direction segment displacement < 0.02 m",
            "friction": 10.0,
            "object_density_kg_m3": 10000.0,
            "object_linear_damping": 10.0,
            "object_angular_damping": 100.0,
            "joint_stiffness": 400.0,
            "joint_damping": 400.0,
            "solver_position_iterations": 4,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# D(R,O) ShadowHand 全旧参数恢复门槛报告",
        "",
        f"- 完整性：{'640+640 完整' if complete else '结果不完整'}",
        f"- Gym：{gym['successes']}/{gym['trials']} = {100*gym['success_rate']:.2f}%（门槛 {100*args.gym_threshold:.0f}%）",
        f"- Sim：{sim['successes']}/{sim['trials']} = {100*sim['success_rate']:.2f}%（门槛 {100*args.sim_threshold:.0f}%）",
        f"- 决策：{'达到恢复门槛，继续单参数消融' if recovered else '未达到恢复门槛，停止单参数消融并判定整套旧参数不足以恢复'}",
        "",
        "## 逐物体",
        "",
        "| 物体 | Gym | Sim |",
        "|---|---:|---:|",
    ]
    names = sorted(set(payload["per_object"]["gym"]) | set(payload["per_object"]["sim"]))
    for name in names:
        g = payload["per_object"]["gym"].get(name, stats([]))
        s = payload["per_object"]["sim"].get(name, stats([]))
        lines.append(
            f"| {name} | {g['successes']}/{g['trials']} ({100*g['success_rate']:.1f}%) | "
            f"{s['successes']}/{s['trials']} ({100*s['success_rate']:.1f}%) |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
