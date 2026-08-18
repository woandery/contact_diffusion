#!/usr/bin/env python3
"""Summarize paired full-particle dynamic-vs-fixed closure PhysX trials."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


RANK_CUTOFFS = (1, 2, 4, 8, 16, 32)
EXPECTED_TRIALS = 40_960


def success(row: dict, field: str) -> bool:
    return bool(row.get("valid_simulation")) and row.get(field) is True


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def percentile(values: list[float], fraction: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = fraction * (len(finite) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    alpha = position - lower
    return finite[lower] * (1.0 - alpha) + finite[upper] * alpha


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if not discordant:
        return 1.0
    tail_end = min(a_only, b_only)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(tail_end + 1)
    ]
    maximum = max(log_terms)
    tail = math.exp(maximum) * sum(
        math.exp(value - maximum) for value in log_terms
    )
    return min(1.0, 2.0 * tail)


def trial_key(hand: str, object_id: str, row: dict) -> tuple[str, str, int, int]:
    return (
        hand,
        object_id,
        int(row["source_index"]),
        int(row.get("candidate_rank", 0)),
    )


def load_condition(root: Path, condition: str) -> tuple[dict[tuple, dict], list[dict]]:
    rows: dict[tuple, dict] = {}
    shards = []
    condition_root = root / "results" / condition
    for result_path in sorted(condition_root.glob("*/*/batch_*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        shard = {
            "path": str(result_path),
            "status": payload.get("status"),
            "rows": len(payload.get("results", [])),
        }
        shards.append(shard)
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete result shard: {result_path}")
        hand = result_path.parent.parent.name
        object_id = result_path.parent.name
        for row in payload.get("results", []):
            key = trial_key(hand, object_id, row)
            if key in rows:
                raise ValueError(f"duplicate {condition} trial: {key}")
            rows[key] = row
    return rows, shards


def scalar(row: dict, name: str) -> float | None:
    value = row.get("closure_telemetry", {}).get(name)
    return float(value) if value is not None and math.isfinite(float(value)) else None


def contact_set_metrics(rows: list[dict], field: str) -> dict:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["hand"], row["object_id"], row["source_index"])].append(row)
    top1 = oracle = particles = 0
    success_at_k = {cutoff: 0 for cutoff in RANK_CUTOFFS}
    for candidates in groups.values():
        ordered = sorted(candidates, key=lambda item: item["candidate_rank"])
        outcomes = [item[field] for item in ordered]
        top1 += bool(outcomes and outcomes[0])
        oracle += any(outcomes)
        particles += sum(outcomes)
        for cutoff in RANK_CUTOFFS:
            success_at_k[cutoff] += any(outcomes[:cutoff])
    return {
        "contact_sets": len(groups),
        "particle_success_rate": ratio(particles, len(rows)),
        "top1_success_rate": ratio(top1, len(groups)),
        "oracle_success_rate": ratio(oracle, len(groups)),
        "success_at_k": {
            str(cutoff): ratio(success_at_k[cutoff], len(groups))
            for cutoff in RANK_CUTOFFS
        },
    }


def telemetry_metrics(rows: list[dict], prefix: str) -> dict:
    contacts = [row[f"{prefix}_first_contact_time_s"] for row in rows]
    contact_times = [value for value in contacts if value is not None]
    thumbs = [row[f"{prefix}_thumb_first_contact_time_s"] for row in rows]
    thumb_times = [value for value in thumbs if value is not None]
    link_counts = Counter(
        row[f"{prefix}_first_contact_link"]
        for row in rows
        if row[f"{prefix}_first_contact_link"] is not None
    )

    def distribution(name: str) -> dict:
        values = [
            row[f"{prefix}_{name}"]
            for row in rows
            if row[f"{prefix}_{name}"] is not None
        ]
        return {
            "count": len(values),
            "mean": mean(values),
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }

    return {
        "first_contact_observed": len(contact_times),
        "first_contact_rate": ratio(len(contact_times), len(rows)),
        "first_contact_time_s": {
            "mean": mean(contact_times),
            "median": percentile(contact_times, 0.5),
            "p95": percentile(contact_times, 0.95),
        },
        "thumb_contact_observed": len(thumb_times),
        "thumb_contact_rate": ratio(len(thumb_times), len(rows)),
        "thumb_first_contact_time_s": {
            "mean": mean(thumb_times),
            "median": percentile(thumb_times, 0.5),
            "p95": percentile(thumb_times, 0.95),
        },
        "first_contact_link_counts": dict(link_counts.most_common()),
        "outer_to_first_contact_displacement_m": distribution(
            "outer_to_first_contact_displacement_m"
        ),
        "first_contact_to_inner_displacement_m": distribution(
            "first_contact_to_inner_displacement_m"
        ),
        "outer_to_inner_displacement_m": distribution(
            "outer_to_inner_displacement_m"
        ),
        "cumulative_net_contact_impulse_ns": distribution(
            "cumulative_net_contact_impulse_ns"
        ),
        "peak_frame_net_contact_impulse_ns": distribution(
            "peak_frame_net_contact_impulse_ns"
        ),
        "maximum_object_linear_speed_mps": distribution(
            "maximum_object_linear_speed_mps"
        ),
        "maximum_object_angular_speed_radps": distribution(
            "maximum_object_angular_speed_radps"
        ),
    }


def aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    output: dict[str, object] = {"paired_trials": count}
    for field in ("final", "strict"):
        a_name = f"a_{field}_success"
        b_name = f"b_{field}_success"
        both = sum(row[a_name] and row[b_name] for row in rows)
        a_only = sum(row[a_name] and not row[b_name] for row in rows)
        b_only = sum(row[b_name] and not row[a_name] for row in rows)
        neither = count - both - a_only - b_only
        output[field] = {
            "a_successes": both + a_only,
            "a_success_rate": ratio(both + a_only, count),
            "b_successes": both + b_only,
            "b_success_rate": ratio(both + b_only, count),
            "absolute_rate_change_b_minus_a": ratio(b_only - a_only, count),
            "both_success": both,
            "a_only_success": a_only,
            "b_only_success": b_only,
            "both_fail": neither,
            "paired_exact_mcnemar_p": exact_mcnemar_p(a_only, b_only),
        }
    output["a_invalid_trials"] = sum(not row["a_valid"] for row in rows)
    output["b_invalid_trials"] = sum(not row["b_valid"] for row in rows)
    output["a_contact_sets"] = {
        field: contact_set_metrics(rows, f"a_{field}_success")
        for field in ("final", "strict")
    }
    output["b_contact_sets"] = {
        field: contact_set_metrics(rows, f"b_{field}_success")
        for field in ("final", "strict")
    }
    output["a_telemetry"] = telemetry_metrics(rows, "a")
    output["b_telemetry"] = telemetry_metrics(rows, "b")
    return output


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def fmt_num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    root = args.run_root.resolve()
    output_json = args.output_json or root / "summary.json"
    output_csv = args.output_csv or root / "paired_trials.csv"
    output_md = args.output_md or root / "final_report.md"
    a_rows, a_shards = load_condition(root, "A_dynamic")
    b_rows, b_shards = load_condition(root, "B_fixed")
    missing_a = set(b_rows) - set(a_rows)
    missing_b = set(a_rows) - set(b_rows)
    if missing_a or missing_b:
        raise ValueError(
            f"unpaired trials: missing A={len(missing_a)} missing B={len(missing_b)}"
        )
    if not args.allow_incomplete and len(a_rows) != EXPECTED_TRIALS:
        raise ValueError(f"paired {len(a_rows)} trials, expected {EXPECTED_TRIALS}")

    paired = []
    scalar_names = (
        "outer_to_first_contact_displacement_m",
        "first_contact_to_inner_displacement_m",
        "outer_to_inner_displacement_m",
        "cumulative_net_contact_impulse_ns",
        "peak_frame_net_contact_impulse_ns",
        "maximum_object_linear_speed_mps",
        "maximum_object_angular_speed_radps",
    )
    for key in sorted(a_rows):
        a = a_rows[key]
        b = b_rows[key]
        row = {
            "hand": key[0],
            "object_id": key[1],
            "source_index": key[2],
            "candidate_rank": key[3],
            "a_valid": bool(a.get("valid_simulation")),
            "b_valid": bool(b.get("valid_simulation")),
            "a_final_success": success(a, "final_success"),
            "b_final_success": success(b, "final_success"),
            "a_strict_success": success(a, "strict_six_direction_success"),
            "b_strict_success": success(b, "strict_six_direction_success"),
        }
        for prefix, source in (("a", a), ("b", b)):
            telemetry = source.get("closure_telemetry", {})
            row[f"{prefix}_first_contact_time_s"] = telemetry.get(
                "first_contact_time_s"
            )
            row[f"{prefix}_thumb_first_contact_time_s"] = telemetry.get(
                "thumb_first_contact_time_s"
            )
            row[f"{prefix}_first_contact_link"] = telemetry.get(
                "first_contact_link"
            )
            for name in scalar_names:
                row[f"{prefix}_{name}"] = scalar(source, name)
        paired.append(row)

    scopes = {"overall": aggregate(paired)}
    for hand in sorted({row["hand"] for row in paired}):
        hand_rows = [row for row in paired if row["hand"] == hand]
        scopes[hand] = aggregate(hand_rows)
        for object_id in sorted({row["object_id"] for row in hand_rows}):
            scopes[f"{hand}/{object_id}"] = aggregate(
                [row for row in hand_rows if row["object_id"] == object_id]
            )

    report = {
        "schema": "contactdiff-physx-closure-ab-all-particles-v1",
        "run_root": str(root),
        "condition_a": "dynamic object from outer through closure",
        "condition_b": (
            "fixed-base object through outer/closure; zero-velocity dynamic "
            "object enabled at inner"
        ),
        "contact_measurement": (
            "per-rigid-body GPU net contact force tensor times dt; not exact "
            "per-contact normal lambda"
        ),
        "invalid_counts_as_failure": True,
        "paired_integrity": {
            "a_trials": len(a_rows),
            "b_trials": len(b_rows),
            "a_shards": len(a_shards),
            "b_shards": len(b_shards),
            "missing_a": len(missing_a),
            "missing_b": len(missing_b),
        },
        "scopes": scopes,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_fields = list(paired[0]) if paired else []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(paired)

    overall = scopes["overall"]
    lines = [
        "# step-50k 全粒子闭合阶段 PhysX A/B",
        "",
        f"- 严格配对姿态：{overall['paired_trials']:,}",
        "- A：物体从 outer 起始即为 dynamic。",
        "- B：outer settle/closure 使用 fixed-base 物体；inner 边界清零速度并切换同位姿 dynamic 物体。",
        "- A/B 的候选、batch 顺序、GPU 分配、inner hold 和六方向参数完全相同。",
        "- invalid trial 按失败计入。",
        "- 接触冲量为 GPU 刚体净接触力 × dt；Preview 4 GPU pipeline 不提供逐接触点精确 normal lambda。",
        "",
        "## 成功结果",
        "",
        "| 范围 | A final | B final | B-A | A-only | B-only | McNemar p | A invalid | B invalid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("overall", "barrett", "shadowhand"):
        item = scopes[name]
        final = item["final"]
        lines.append(
            f"| {name} | {fmt_pct(final['a_success_rate'])} | "
            f"{fmt_pct(final['b_success_rate'])} | "
            f"{fmt_pct(final['absolute_rate_change_b_minus_a'])} | "
            f"{final['a_only_success']:,} | {final['b_only_success']:,} | "
            f"{final['paired_exact_mcnemar_p']:.3g} | "
            f"{item['a_invalid_trials']:,} | {item['b_invalid_trials']:,} |"
        )
    lines += [
        "",
        "## 闭合阶段遥测",
        "",
        "| 范围 | 条件 | 首接触率 | 首接触时间中位数(s) | 拇指接触率 | outer→inner中位数(m) | 累计净接触冲量中位数(N·s) | 最大线速度P95(m/s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("overall", "barrett", "shadowhand"):
        for prefix, label in (("a", "A"), ("b", "B")):
            item = scopes[name][f"{prefix}_telemetry"]
            lines.append(
                f"| {name} | {label} | {fmt_pct(item['first_contact_rate'])} | "
                f"{fmt_num(item['first_contact_time_s']['median'])} | "
                f"{fmt_pct(item['thumb_contact_rate'])} | "
                f"{fmt_num(item['outer_to_inner_displacement_m']['median'], 6)} | "
                f"{fmt_num(item['cumulative_net_contact_impulse_ns']['median'], 6)} | "
                f"{fmt_num(item['maximum_object_linear_speed_mps']['p95'])} |"
            )
    lines += [
        "",
        "## 分物体 final success",
        "",
        "| 手型/物体 | A | B | B-A | A-only | B-only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(key for key in scopes if "/" in key):
        final = scopes[name]["final"]
        lines.append(
            f"| {name} | {fmt_pct(final['a_success_rate'])} | "
            f"{fmt_pct(final['b_success_rate'])} | "
            f"{fmt_pct(final['absolute_rate_change_b_minus_a'])} | "
            f"{final['a_only_success']:,} | {final['b_only_success']:,} |"
        )
    lines += [
        "",
        "逐姿态配对指标见 `paired_trials.csv`，每帧原始遥测见 `telemetry/` 下的 NPZ。",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(overall["final"], indent=2))
    print(output_md.resolve())


if __name__ == "__main__":
    main()
