#!/usr/bin/env python3
"""Summarize the aligned native Barrett old-stack Gym/Sim experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def load_many(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    payloads: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append(payload)
        rows.extend(payload.get("results", []))
    if not payloads:
        raise FileNotFoundError("No result JSON files found")
    keys = [(row["object_name"], int(row["source_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate object/source keys in result shards")
    return rows, payloads


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: list[dict], simulator: str) -> dict:
    valid = [row for row in rows if bool(row.get("valid_simulation"))]
    successes = sum(row.get("success") is True for row in rows)
    final_successes = sum(
        row.get("final_success", False) is True
        or (
            simulator == "sim"
            and bool(row.get("valid_simulation"))
            and float(row.get("final_displacement_m", float("inf"))) <= 0.02
        )
        for row in rows
    )
    strict_successes = sum(
        row.get("strict_six_direction_success", False) is True
        or (
            simulator == "sim"
            and bool(row.get("valid_simulation"))
            and float(row.get("maximum_segment_displacement_m", float("inf")))
            <= 0.02
        )
        for row in rows
    )
    output = {
        "trials": len(rows),
        "valid_trials": len(valid),
        "invalid_trials": len(rows) - len(valid),
        "recorded_successes": successes,
        "recorded_success_rate_raw": rate(successes, len(rows)),
        "recorded_success_rate_valid": rate(successes, len(valid)),
        "final_successes": final_successes,
        "strict_successes": strict_successes,
    }
    if simulator == "sim":
        contacts = [
            (row, row.get("physx_contact_after_closure"))
            for row in rows
            if row.get("physx_contact_after_closure") is not None
        ]
        for threshold_mm in (0.0, 1.0, 5.0):
            key = f"penetration_gt_{threshold_mm:g}mm"
            output[key] = sum(
                float(contact.get("maximum_collision_penetration_m", 0.0))
                > threshold_mm / 1000.0
                for _, contact in contacts
            )
            output[f"successful_{key}"] = sum(
                row.get("success") is True
                and float(contact.get("maximum_collision_penetration_m", 0.0))
                > threshold_mm / 1000.0
                for row, contact in contacts
            )
        output["contact_diagnostics_trials"] = len(contacts)
    return output


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    gym_rows, gym_payloads = load_many(
        sorted((root / "results/isaacgym").glob("*.json"))
    )
    sim_rows, sim_payloads = load_many(
        sorted((root / "results/isaacsim").glob("*.json"))
    )
    if len(gym_rows) != 640 or len(sim_rows) != 640:
        raise RuntimeError(
            f"Expected 640 Gym and Sim rows, got {len(gym_rows)} and {len(sim_rows)}"
        )
    gym = summarize(gym_rows, "gym")
    sim = summarize(sim_rows, "sim")
    gym_map = {
        (row["object_name"], int(row["source_index"])): row for row in gym_rows
    }
    sim_map = {
        (row["object_name"], int(row["source_index"])): row for row in sim_rows
    }
    common = sorted(set(gym_map) & set(sim_map))
    paired = {
        "common_trials": len(common),
        "both_success": sum(
            gym_map[key].get("success") is True
            and sim_map[key].get("success") is True
            for key in common
        ),
        "gym_only": sum(
            gym_map[key].get("success") is True
            and sim_map[key].get("success") is not True
            for key in common
        ),
        "sim_only": sum(
            gym_map[key].get("success") is not True
            and sim_map[key].get("success") is True
            for key in common
        ),
        "both_fail": sum(
            gym_map[key].get("success") is not True
            and sim_map[key].get("success") is not True
            for key in common
        ),
    }
    by_object: dict[str, dict] = {}
    object_names = sorted({row["object_name"] for row in gym_rows + sim_rows})
    for object_name in object_names:
        gym_object = [row for row in gym_rows if row["object_name"] == object_name]
        sim_object = [row for row in sim_rows if row["object_name"] == object_name]
        by_object[object_name] = {
            "gym": summarize(gym_object, "gym"),
            "sim": summarize(sim_object, "sim"),
        }
    candidate = json.loads((root / "candidates/barrett.json").read_text())
    output = {
        "schema": "multidex-barrett-oldstack-aligned-gym-sim-v1",
        "candidate": {
            "checkpoint": candidate.get("checkpoint"),
            "checkpoint_step": candidate.get("checkpoint_step"),
            "records": len(candidate.get("records", [])),
            "particles": candidate.get("particles"),
            "optimization_steps": candidate.get("optimization_steps"),
        },
        "gym": gym,
        "sim": sim,
        "paired": paired,
        "protocols": {
            "gym": gym_payloads[0].get("protocol"),
            "sim": sim_payloads[0].get("protocol"),
        },
        "by_object": by_object,
    }
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# MultiDex 45k Barrett 旧候选/旧 URDF/旧协议对齐验证",
        "",
        "同一批 OOD10 640 条候选分别在 Isaac Gym Preview 4 与 Isaac Sim 6.0.1 中验证。",
        "候选使用 dex-urdf `bhand_model.urdf`；协议沿用旧原生手/GenDex 设置，",
        "并将 Gym 可配置的根增益、armature、object mesh 和时序向 Sim 对齐。",
        "",
        "## 总结果",
        "",
        "| 环境 | recorded/strict | raw | valid-only | invalid | final |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Gym | {gym['recorded_successes']}/{gym['trials']} | "
        f"{percent(gym['recorded_success_rate_raw'])} | "
        f"{percent(gym['recorded_success_rate_valid'])} | {gym['invalid_trials']} | "
        f"{gym['final_successes']}/{gym['trials']} |",
        f"| Sim | {sim['recorded_successes']}/{sim['trials']} | "
        f"{percent(sim['recorded_success_rate_raw'])} | "
        f"{percent(sim['recorded_success_rate_valid'])} | {sim['invalid_trials']} | "
        f"{sim['final_successes']}/{sim['trials']} |",
        "",
        "## 配对结果",
        "",
        f"- 两端都成功：{paired['both_success']}；仅 Gym：{paired['gym_only']}；"
        f"仅 Sim：{paired['sim_only']}；两端均失败/无效：{paired['both_fail']}。",
        "",
        "## Sim 闭合后穿透诊断",
        "",
        f"- 已记录接触诊断：{sim.get('contact_diagnostics_trials', 0)}/{sim['trials']}。",
        f"- 最大穿透 >1 mm：{sim.get('penetration_gt_1mm', 0)}，其中判成功 "
        f"{sim.get('successful_penetration_gt_1mm', 0)}。",
        f"- 最大穿透 >5 mm：{sim.get('penetration_gt_5mm', 0)}，其中判成功 "
        f"{sim.get('successful_penetration_gt_5mm', 0)}。",
        "",
        "## 逐物体",
        "",
        "| 物体 | Gym strict | Sim strict | Sim invalid |",
        "|---|---:|---:|---:|",
    ]
    for object_name, item in by_object.items():
        lines.append(
            f"| {object_name} | {item['gym']['recorded_successes']}/64 | "
            f"{item['sim']['recorded_successes']}/64 | {item['sim']['invalid_trials']} |"
        )
    lines.extend(
        [
            "",
            "注意：显式协议对齐不等于 PhysX 数值等价；两端 importer、convex cooking、",
            "contact offset 自动语义和并行执行方式仍可能不同。Sim 穿透统计应与原始成功率",
            "同时报告，不能把深穿透卡住直接解释为稳定抓取。",
            "",
        ]
    )
    (summary_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary_dir / "REPORT.md")


if __name__ == "__main__":
    main()

