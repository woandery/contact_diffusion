#!/usr/bin/env python3
"""Summarize MultiDex Barrett and ShadowHand D(R,O) Gym/Sim experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ACTUATORS = {
    "Barrett": {
        "gym": "results/isaacgym_new/results",
        "sim": "results/isaacsim_new",
        "candidate": "candidates/barrett.json",
    },
    "ShadowHand": {
        "gym": "results/isaacgym/results",
        "sim": "results/isaacsim",
        "candidate": "candidates/shadow.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barrett-root", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def rate(successes: int, trials: int) -> float | None:
    return successes / trials if trials else None


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def result_cell(successes: int, trials: int) -> str:
    return f"{successes}/{trials} ({percent(rate(successes, trials))})"


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_gym(directory: Path) -> tuple[dict, dict, list[dict]]:
    files = sorted(directory.glob("*.json"))
    if len(files) != 10:
        raise ValueError(f"{directory}: expected 10 Gym results, got {len(files)}")
    per_object = {}
    protocol = None
    result_rows = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = str(payload["object_name"])
        trials = int(payload["trials"])
        if trials != 64:
            raise ValueError(f"{path}: expected 64 trials, got {trials}")
        per_object[name] = {
            "trials": trials,
            "dro_successes": int(payload["dro_successes"]),
            "strict_successes": int(
                payload["strict_six_direction_successes"]
            ),
        }
        result_rows.extend(payload["results"])
        protocol = protocol or payload["protocol"]
    return per_object, protocol, result_rows


def load_sim(directory: Path) -> tuple[dict, dict]:
    files = sorted(directory.glob("shard*.json"))
    if len(files) != 4:
        raise ValueError(f"{directory}: expected 4 Sim shards, got {len(files)}")
    rows = []
    protocol = None
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["results"])
        protocol = protocol or payload["protocol"]
    if len(rows) != 640:
        raise ValueError(f"{directory}: expected 640 Sim rows, got {len(rows)}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["object_name"])].append(row)
    per_object = {}
    for name, object_rows in grouped.items():
        if len(object_rows) != 64:
            raise ValueError(f"{directory}/{name}: expected 64 rows")
        valid = [row for row in object_rows if row.get("valid_simulation", True)]
        per_object[name] = {
            "trials": len(object_rows),
            "valid_trials": len(valid),
            "invalid_trials": len(object_rows) - len(valid),
            "successes": sum(bool(row["success"]) for row in object_rows),
            "valid_successes": sum(bool(row["success"]) for row in valid),
        }
    return per_object, protocol


def totals(rows: dict[str, dict], keys: tuple[str, ...]) -> dict:
    return {key: sum(int(row[key]) for row in rows.values()) for key in keys}


def summarize_actuator(name: str, root: Path, spec: dict) -> dict:
    gym, gym_protocol, gym_rows = load_gym(root / spec["gym"])
    sim, sim_protocol = load_sim(root / spec["sim"])
    if set(gym) != set(sim):
        raise ValueError(f"{name}: Gym/Sim object sets differ")
    candidate = json.loads((root / spec["candidate"]).read_text(encoding="utf-8"))
    candidate_rows = [
        record["fk"]["candidates"][0] for record in candidate["records"]
    ]
    diagnostics = {
        "candidate_contact_chamfer_median_m": quantile(
            [float(row["contact_chamfer_m"]) for row in candidate_rows], 0.5
        ),
        "candidate_assigned_contact_error_median_m": quantile(
            [float(row["assigned_contact_error_m"]) for row in candidate_rows],
            0.5,
        ),
        "candidate_mean_penetration_median_m": quantile(
            [float(row["mean_penetration_m"]) for row in candidate_rows], 0.5
        ),
        "gym_final_displacement_median_m": quantile(
            [float(row["final_displacement_m"]) for row in gym_rows], 0.5
        ),
        "gym_final_displacement_p90_m": quantile(
            [float(row["final_displacement_m"]) for row in gym_rows], 0.9
        ),
        "gym_maximum_segment_displacement_median_m": quantile(
            [float(row["maximum_segment_displacement_m"]) for row in gym_rows],
            0.5,
        ),
        "gym_outer_to_inner_joint_l2_median_rad": quantile(
            [float(row["outer_to_inner_joint_l2_rad"]) for row in gym_rows],
            0.5,
        ),
    }
    gym_total = totals(gym, ("trials", "dro_successes", "strict_successes"))
    sim_total = totals(
        sim,
        (
            "trials",
            "valid_trials",
            "invalid_trials",
            "successes",
            "valid_successes",
        ),
    )
    objects = {}
    for object_name in sorted(gym):
        g = gym[object_name]
        s = sim[object_name]
        gym_rate = rate(g["dro_successes"], g["trials"])
        sim_raw_rate = rate(s["successes"], s["trials"])
        sim_valid_rate = rate(s["valid_successes"], s["valid_trials"])
        objects[object_name] = {
            "gym": g,
            "sim": s,
            "sim_raw_minus_gym_percentage_points": (
                100.0 * (sim_raw_rate - gym_rate)
            ),
            "sim_valid_minus_gym_percentage_points": (
                None
                if sim_valid_rate is None
                else 100.0 * (sim_valid_rate - gym_rate)
            ),
        }
    gym_rate = rate(gym_total["dro_successes"], gym_total["trials"])
    sim_raw_rate = rate(sim_total["successes"], sim_total["trials"])
    sim_valid_rate = rate(
        sim_total["valid_successes"], sim_total["valid_trials"]
    )
    return {
        "root": str(root.resolve()),
        "candidate": {
            "path": str((root / spec["candidate"]).resolve()),
            "checkpoint": candidate["checkpoint"],
            "checkpoint_step": int(candidate["checkpoint_step"]),
            "records": len(candidate["records"]),
            "particles": int(candidate["particles"]),
            "optimization_steps": int(candidate["optimization_steps"]),
            "diffusion_steps": int(candidate["diffusion_steps"]),
        },
        "gym": {
            **gym_total,
            "dro_success_rate": gym_rate,
            "strict_success_rate": rate(
                gym_total["strict_successes"], gym_total["trials"]
            ),
            "protocol": gym_protocol,
        },
        "sim": {
            **sim_total,
            "raw_success_rate": sim_raw_rate,
            "valid_trial_success_rate": sim_valid_rate,
            "protocol": sim_protocol,
        },
        "sim_raw_minus_gym_percentage_points": 100.0 * (
            sim_raw_rate - gym_rate
        ),
        "sim_valid_minus_gym_percentage_points": 100.0 * (
            sim_valid_rate - gym_rate
        ),
        "diagnostics": diagnostics,
        "objects": objects,
    }


def markdown(report: dict) -> str:
    lines = [
        "# MultiDex ContactDiffusion D(R,O) Gym/Sim 总结",
        "",
        "## 实验范围",
        "",
        "Barrett 与 ShadowHand 在本报告中分别总结；不将两种执行器的成功率作为优劣排名。每种执行器内部比较同一批 640 条候选在 D(R,O) Isaac Gym 和 Isaac Sim 中的表现。",
        "",
        "共同预算：OOD10、每物体 64 个 contact sets、每组 32 个 FK particles、保留 top1、400 个 FK 优化步、50 个 diffusion steps。",
        "",
        "## 总体结果",
        "",
        "| 执行器 | Gym D(R,O) | Gym strict | Sim raw | Sim valid-only | Sim 无效 | Sim raw - Gym |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in report["actuators"].items():
        g, s = data["gym"], data["sim"]
        lines.append(
            f"| {name} | {result_cell(g['dro_successes'], g['trials'])} "
            f"| {result_cell(g['strict_successes'], g['trials'])} "
            f"| {result_cell(s['successes'], s['trials'])} "
            f"| {result_cell(s['valid_successes'], s['valid_trials'])} "
            f"| {s['invalid_trials']} "
            f"| {data['sim_raw_minus_gym_percentage_points']:+.2f} pp |"
        )
    for name, data in report["actuators"].items():
        lines.extend([
            "",
            f"## {name}：Gym 与 Sim",
            "",
            "| OOD 物体 | Gym D(R,O) | Gym strict | Sim raw | Sim valid-only | 无效 | Sim raw - Gym |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for object_name, row in data["objects"].items():
            g, s = row["gym"], row["sim"]
            lines.append(
                f"| {object_name} "
                f"| {result_cell(g['dro_successes'], g['trials'])} "
                f"| {result_cell(g['strict_successes'], g['trials'])} "
                f"| {result_cell(s['successes'], s['trials'])} "
                f"| {result_cell(s['valid_successes'], s['valid_trials'])} "
                f"| {s['invalid_trials']} "
                f"| {row['sim_raw_minus_gym_percentage_points']:+.2f} pp |"
            )
        candidate = data["candidate"]
        diagnostic = data["diagnostics"]
        lines.extend([
            "",
            f"候选：checkpoint step {candidate['checkpoint_step']}，"
            f"{candidate['records']} records，{candidate['particles']} particles，"
            f"{candidate['optimization_steps']} FK steps。",
            "",
            "诊断统计："
            f"候选 contact chamfer 中位数 {1000 * diagnostic['candidate_contact_chamfer_median_m']:.2f} mm，"
            f"assigned contact error 中位数 {1000 * diagnostic['candidate_assigned_contact_error_median_m']:.2f} mm，"
            f"mean penetration 中位数 {1000 * diagnostic['candidate_mean_penetration_median_m']:.2f} mm；"
            f"Gym 最终位移中位数 {diagnostic['gym_final_displacement_median_m']:.3f} m，"
            f"P90 {diagnostic['gym_final_displacement_p90_m']:.3f} m，"
            f"最大方向段位移中位数 {diagnostic['gym_maximum_segment_displacement_median_m']:.3f} m。",
        ])
        if name == "ShadowHand":
            lines.extend([
                "",
                "ShadowHand 的低成功率不能仅由 Sim invalid 解释：排除 131 条无效试验后 Sim 仍为 6.48%。候选的几何损失有限，但 Gym 位移达到米级，说明主要失败发生在物理闭合/扰动阶段；后续应优先核查 D(R,O) ShadowHand controller、根坐标与碰撞初始状态，而不是只继续降低 contact chamfer。",
            ])
    lines.extend([
        "",
        "## 口径说明",
        "",
        "- Gym D(R,O)：依次施加六方向扰动后，最终位移不超过 0.02 m。",
        "- Gym strict：要求每一个方向段的位移都不超过 0.02 m。",
        "- Sim raw：无效仿真计入总试验数并按失败处理。",
        "- Sim valid-only：排除验证器标记为 invalid 的试验，仅用于诊断仿真有效性影响。",
        "- Gym 与 Sim 使用相同候选，但物理引擎实现、资产导入和控制细节仍不同；百分比差值描述 simulator gap，不直接归因于模型。",
        "- 控制轨迹并非完全相同：D(R,O) Gym controller 使用 25% outward / 15% inward 调整；当前 Isaac Sim preparation 对 Barrett 使用 0% / 5%，对 ShadowHand 使用 10% / 20%。因此本报告是完整执行栈差异，而不是纯 PhysX 后端差异。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    roots = {
        "Barrett": args.barrett_root.resolve(),
        "ShadowHand": args.shadow_root.resolve(),
    }
    report = {
        "schema": "multidex-contactdiffusion-dro-gym-sim-summary-v1",
        "comparison_scope": (
            "Within-actuator Isaac Gym versus Isaac Sim; no cross-actuator ranking"
        ),
        "actuators": {
            name: summarize_actuator(name, roots[name], ACTUATORS[name])
            for name in ACTUATORS
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown(report), encoding="utf-8")
    print(args.output_markdown.resolve())
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
