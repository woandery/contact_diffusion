#!/usr/bin/env python3
"""Merge streaming SQLite aggregates and report seen-48 PhysX stability."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3


def ratio(a: int | float, b: int | float) -> float | None:
    return a / b if b else None


def scope_rows(databases: list[sqlite3.Connection], hand: str | None = None, object_id: str | None = None):
    clauses = []
    parameters: list[str] = []
    if hand is not None:
        clauses.append("hand=?")
        parameters.append(hand)
    if object_id is not None:
        clauses.append("object_id=?")
        parameters.append(object_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    query = (
        "SELECT repeats,valid_count,final_success_count,strict_success_count,"
        "final_displacement_sum,final_displacement_sumsq,final_displacement_min,"
        "final_displacement_max,maximum_segment_sum FROM trials" + where
    )
    for database in databases:
        yield from database.execute(query, parameters)


def aggregate(databases: list[sqlite3.Connection], repeats: int, hand: str | None = None, object_id: str | None = None) -> dict:
    unique = repeat_trials = valid = final = strict = 0
    always_success = always_fail = mixed = 0
    displacement_sum = displacement_sumsq = maximum_segment_sum = 0.0
    minimum = math.inf
    maximum = -math.inf
    histogram = [0] * (repeats + 1)
    incomplete = 0
    for row in scope_rows(databases, hand, object_id):
        count, valid_count, final_count, strict_count, total, total_sq, low, high, max_sum = row
        unique += 1
        repeat_trials += count
        valid += valid_count
        final += final_count
        strict += strict_count
        displacement_sum += total
        displacement_sumsq += total_sq
        maximum_segment_sum += max_sum
        minimum = min(minimum, low)
        maximum = max(maximum, high)
        if count != repeats:
            incomplete += 1
        if 0 <= final_count <= repeats:
            histogram[final_count] += 1
        if final_count == repeats and count == repeats:
            always_success += 1
        elif final_count == 0 and count == repeats:
            always_fail += 1
        else:
            mixed += 1
    mean = ratio(displacement_sum, repeat_trials)
    variance = (
        max(0.0, displacement_sumsq / repeat_trials - mean * mean)
        if repeat_trials and mean is not None
        else None
    )
    return {
        "unique_grasps": unique,
        "repeat_trials": repeat_trials,
        "expected_repeat_trials": unique * repeats,
        "incomplete_grasps": incomplete,
        "valid_trials": valid,
        "invalid_trials": repeat_trials - valid,
        "final_successes": final,
        "final_success_rate": ratio(final, repeat_trials),
        "strict_successes": strict,
        "strict_success_rate": ratio(strict, repeat_trials),
        "always_success_grasps": always_success,
        "mixed_outcome_grasps": mixed,
        "always_fail_grasps": always_fail,
        "binary_outcome_stability_rate": ratio(always_success + always_fail, unique),
        "final_displacement_mean_m": mean,
        "final_displacement_std_m": math.sqrt(variance) if variance is not None else None,
        "final_displacement_min_m": None if minimum == math.inf else minimum,
        "final_displacement_max_m": None if maximum == -math.inf else maximum,
        "maximum_segment_displacement_mean_m": ratio(maximum_segment_sum, repeat_trials),
        "success_count_histogram": {str(i): value for i, value in enumerate(histogram) if value},
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--object-map", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    objects = list(json.loads(args.object_map.read_text(encoding="utf-8")))
    paths = sorted((root / "streaming").glob("gpu*.sqlite"))
    if not paths:
        raise ValueError("No worker databases found")
    databases = [sqlite3.connect(path) for path in paths]
    overall = aggregate(databases, args.repeats)
    expected_unique = 2 * 48 * 128 * 128
    if not args.allow_incomplete and (
        overall["unique_grasps"] != expected_unique
        or overall["incomplete_grasps"]
        or overall["repeat_trials"] != expected_unique * args.repeats
    ):
        raise ValueError(f"Incomplete aggregate: {overall}")

    hands = {hand: aggregate(databases, args.repeats, hand=hand) for hand in ("barrett", "shadowhand")}
    per_object = {
        f"{hand}/{object_id}": aggregate(databases, args.repeats, hand=hand, object_id=object_id)
        for hand in ("barrett", "shadowhand")
        for object_id in objects
    }
    summary = {
        "schema": "contactdiff-seen48-128x128-physx-streaming-summary-v1",
        "run_root": str(root),
        "worker_databases": len(paths),
        "repeats": args.repeats,
        "invalid_counts_as_failure": True,
        "overall": overall,
        "per_hand": hands,
        "per_object_hand": per_object,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = root / "per_grasp_stability.csv"
    fields = [
        "hand", "object_id", "source_index", "candidate_rank", "repeats",
        "valid_count", "final_success_count", "strict_success_count",
        "final_success_rate", "final_displacement_mean_m",
        "final_displacement_std_m", "final_displacement_min_m",
        "final_displacement_max_m", "maximum_segment_displacement_mean_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for database in databases:
            for row in database.execute(
                "SELECT hand,object_id,source_index,candidate_rank,repeats,valid_count,"
                "final_success_count,strict_success_count,final_displacement_sum,"
                "final_displacement_sumsq,final_displacement_min,final_displacement_max,"
                "maximum_segment_sum FROM trials ORDER BY hand,object_id,source_index,candidate_rank"
            ):
                hand, object_id, source, rank, count, valid, final, strict, total, total_sq, low, high, max_sum = row
                mean = total / count
                std = math.sqrt(max(0.0, total_sq / count - mean * mean))
                writer.writerow(
                    dict(
                        zip(
                            fields,
                            (hand, object_id, source, rank, count, valid, final, strict,
                             final / count, mean, std, low, high, max_sum / count),
                            strict=True,
                        )
                    )
                )

    lines = [
        f"# seen48 双手 128×128×{args.repeats} 全粒子 GPU PhysX 稳定性报告",
        "",
        "- Barrett 与 ShadowHand 均覆盖训练集 48 个 seen 物体。",
        f"- 每物体 128 个 diffusion set；每 set 128 个 FK 粒子；每个抓取姿态重复 {args.repeats} 次相同参数 GPU PhysX。",
        f"- 唯一抓取姿态：{overall['unique_grasps']:,}；总 PhysX trials：{overall['repeat_trials']:,}。",
        "- 无姿态/物理参数扰动；各 repeat 仅使用确定性随机排列改变并行环境邻接次序。",
        "- invalid trial 按失败计入固定分母；原始重复轨迹采用 SQLite 流式聚合，不长期保存。",
        "",
        "## 总体与分手型",
        "",
        "| 范围 | 唯一抓取 | 重复 trials | final 成功率 | invalid | 始终成功 | 结果翻转 | 始终失败 | 二值稳定率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("合计", overall), ("Barrett", hands["barrett"]), ("ShadowHand", hands["shadowhand"])):
        lines.append(
            f"| {label} | {item['unique_grasps']:,} | {item['repeat_trials']:,} | {pct(item['final_success_rate'])} | {item['invalid_trials']:,} | {item['always_success_grasps']:,} | {item['mixed_outcome_grasps']:,} | {item['always_fail_grasps']:,} | {pct(item['binary_outcome_stability_rate'])} |"
        )
    lines.extend(["", "## 逐物体/手型", "", "| 手型/物体 | final 成功率 | invalid | 二值稳定率 |", "|---|---:|---:|---:|"])
    for key, item in per_object.items():
        lines.append(f"| {key} | {pct(item['final_success_rate'])} | {item['invalid_trials']:,} | {pct(item['binary_outcome_stability_rate'])} |")
    lines.extend(["", f"完整逐抓取 {args.repeats} 次聚合统计见 `per_grasp_stability.csv`。", ""])
    (root / "FINAL_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    for database in databases:
        database.close()
    print(json.dumps({"overall": overall, "per_hand": hands}, indent=2))


if __name__ == "__main__":
    main()
