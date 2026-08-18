#!/usr/bin/env python3
"""Measure between-set success variation and within-set particle stability."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def pstdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ratio(a: int | float, b: int | float) -> float | None:
    return a / b if b else None


def bootstrap_mean_ci(values: list[float], seed: int, draws: int = 5000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(draws)
    )
    return [estimates[math.floor(0.025 * (draws - 1))], estimates[math.ceil(0.975 * (draws - 1))]]


def load_particles(path: Path) -> list[dict]:
    particles = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            repeats = int(row.get("repeat_count") or row.get("repeats") or 0)
            successes = int(row.get("final_success_count") or 0)
            valid = int(row.get("valid_count") or repeats)
            particles.append(
                {
                    "hand": row["hand"],
                    "object_id": row["object_id"],
                    "source_index": int(row["source_index"]),
                    "candidate_rank": int(row["candidate_rank"]),
                    "repeats": repeats,
                    "valid_count": valid,
                    "success_count": successes,
                    "success_rate": successes / repeats if repeats else 0.0,
                }
            )
    if not particles:
        raise ValueError(f"No particle rows in {path}")
    return particles


def build_sets(particles: list[dict], expected_particles: int, high_threshold: float) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for particle in particles:
        grouped[(particle["hand"], particle["object_id"], particle["source_index"])].append(particle)
    output = []
    for (hand, object_id, source_index), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["candidate_rank"])
        if len(rows) != expected_particles:
            raise ValueError(f"{hand}/{object_id}/set{source_index}: {len(rows)} particles != {expected_particles}")
        ranks = [row["candidate_rank"] for row in rows]
        if ranks != list(range(expected_particles)):
            raise ValueError(f"{hand}/{object_id}/set{source_index}: ranks are not 0..{expected_particles - 1}")
        rates = [row["success_rate"] for row in rows]
        successes = sum(row["success_count"] for row in rows)
        repeats = sum(row["repeats"] for row in rows)
        valid = sum(row["valid_count"] for row in rows)
        best = max(rows, key=lambda row: (row["success_rate"], -row["candidate_rank"]))
        stable_count = sum(rate == 1.0 for rate in rates)
        high_count = sum(rate >= high_threshold for rate in rates)
        output.append(
            {
                "hand": hand,
                "object_id": object_id,
                "source_index": source_index,
                "particles": len(rows),
                "repeat_trials": repeats,
                "valid_trials": valid,
                "invalid_trials": repeats - valid,
                "successes": successes,
                "set_success_rate": successes / repeats,
                "particle_rate_mean": statistics.fmean(rates),
                "particle_rate_std": statistics.pstdev(rates),
                "particle_rate_min": min(rates),
                "particle_rate_q25": quantile(rates, 0.25),
                "particle_rate_median": quantile(rates, 0.50),
                "particle_rate_q75": quantile(rates, 0.75),
                "particle_rate_max": max(rates),
                "particle_rate_range": max(rates) - min(rates),
                "stable_success_particles": stable_count,
                "high_stability_particles": high_count,
                "has_stable_success_particle": stable_count > 0,
                "has_high_stability_particle": high_count > 0,
                "best_particle_rank": best["candidate_rank"],
                "best_particle_success_rate": best["success_rate"],
                "rank0_success_rate": rows[0]["success_rate"],
                "best_minus_rank0": best["success_rate"] - rows[0]["success_rate"],
                "high_particle_missed_by_rank0": high_count > 0 and rows[0]["success_rate"] < high_threshold,
            }
        )
    return output


def aggregate_scope(sets: list[dict], particles: list[dict], high_threshold: float, seed: int) -> dict:
    set_keys = {(row["hand"], row["object_id"], row["source_index"]) for row in sets}
    selected_particles = [
        row for row in particles
        if (row["hand"], row["object_id"], row["source_index"]) in set_keys
    ]
    set_rates = [row["set_success_rate"] for row in sets]
    particle_rates = [row["success_rate"] for row in selected_particles]
    grand_mean = statistics.fmean(particle_rates)
    total_ss = sum((rate - grand_mean) ** 2 for rate in particle_rates)
    between_ss = sum(row["particles"] * (row["particle_rate_mean"] - grand_mean) ** 2 for row in sets)
    within_ss = max(0.0, total_ss - between_ss)
    n_sets = len(sets)
    particle_count = len(selected_particles)
    k = sets[0]["particles"] if sets else 0
    ms_between = between_ss / (n_sets - 1) if n_sets > 1 else 0.0
    ms_within = within_ss / (particle_count - n_sets) if particle_count > n_sets else 0.0
    icc = (
        (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        if ms_between + (k - 1) * ms_within
        else 0.0
    )
    stable_particles = sum(row["success_rate"] == 1.0 for row in selected_particles)
    high_particles = sum(row["success_rate"] >= high_threshold for row in selected_particles)
    mixed_particles = sum(0.0 < row["success_rate"] < 1.0 for row in selected_particles)
    set_stable = sum(row["has_stable_success_particle"] for row in sets)
    set_high = sum(row["has_high_stability_particle"] for row in sets)
    rank0_high = sum(row["rank0_success_rate"] >= high_threshold for row in sets)
    missed = sum(row["high_particle_missed_by_rank0"] for row in sets)
    best_rates = [row["best_particle_success_rate"] for row in sets]
    spreads = [row["particle_rate_range"] for row in sets]
    return {
        "sets": n_sets,
        "particles": particle_count,
        "repeats_per_particle": sorted({row["repeats"] for row in selected_particles}),
        "set_success_rate": {
            "mean": mean(set_rates),
            "bootstrap_mean_ci95": bootstrap_mean_ci(set_rates, seed),
            "std": pstdev(set_rates),
            "min": min(set_rates),
            "p05": quantile(set_rates, 0.05),
            "p25": quantile(set_rates, 0.25),
            "median": quantile(set_rates, 0.50),
            "p75": quantile(set_rates, 0.75),
            "p95": quantile(set_rates, 0.95),
            "max": max(set_rates),
            "max_minus_min": max(set_rates) - min(set_rates),
        },
        "variance_decomposition": {
            "total_ss": total_ss,
            "between_set_ss": between_ss,
            "within_set_ss": within_ss,
            "between_set_eta_squared": ratio(between_ss, total_ss),
            "icc_1_particle_success_rate": icc,
            "ms_between": ms_between,
            "ms_within": ms_within,
        },
        "particle_stability": {
            "mean_success_rate": grand_mean,
            "stable_success_particles": stable_particles,
            "stable_success_particle_rate": ratio(stable_particles, particle_count),
            "high_stability_threshold": high_threshold,
            "high_stability_particles": high_particles,
            "high_stability_particle_rate": ratio(high_particles, particle_count),
            "mixed_outcome_particles": mixed_particles,
            "mixed_outcome_particle_rate": ratio(mixed_particles, particle_count),
            "success_count_histogram": dict(sorted(Counter(row["success_count"] for row in selected_particles).items())),
        },
        "within_set_high_stability": {
            "sets_with_stable_success_particle": set_stable,
            "set_rate_with_stable_success_particle": ratio(set_stable, n_sets),
            "sets_with_high_stability_particle": set_high,
            "set_rate_with_high_stability_particle": ratio(set_high, n_sets),
            "rank0_high_stability_sets": rank0_high,
            "rank0_high_stability_set_rate": ratio(rank0_high, n_sets),
            "sets_with_high_particle_missed_by_rank0": missed,
            "miss_rate_among_sets_with_high_particle": ratio(missed, set_high),
            "best_particle_rate_mean": mean(best_rates),
            "best_particle_rate_median": quantile(best_rates, 0.50),
            "best_particle_rate_p05": quantile(best_rates, 0.05),
            "mean_best_minus_rank0": mean([row["best_minus_rank0"] for row in sets]),
            "mean_within_set_particle_rate_range": mean(spreads),
            "median_within_set_particle_rate_range": quantile(spreads, 0.50),
        },
    }


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-particles", type=int, required=True)
    parser.add_argument("--high-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if not 0.0 < args.high_threshold <= 1.0:
        raise ValueError("high threshold must be in (0, 1]")
    particles = load_particles(args.input.resolve())
    sets = build_sets(particles, args.expected_particles, args.high_threshold)
    args.output_root.mkdir(parents=True, exist_ok=True)

    scopes = {"overall": aggregate_scope(sets, particles, args.high_threshold, args.seed)}
    for hand in sorted({row["hand"] for row in sets}):
        hand_sets = [row for row in sets if row["hand"] == hand]
        scopes[hand] = aggregate_scope(hand_sets, particles, args.high_threshold, args.seed + len(scopes))
        for object_id in sorted({row["object_id"] for row in hand_sets}):
            object_sets = [row for row in hand_sets if row["object_id"] == object_id]
            scopes[f"{hand}/{object_id}"] = aggregate_scope(
                object_sets, particles, args.high_threshold, args.seed + len(scopes)
            )

    summary = {
        "schema": "contactdiff-set-particle-stability-analysis-v1",
        "input": str(args.input.resolve()),
        "high_stability_definition": f"particle final success rate >= {args.high_threshold}",
        "stable_success_definition": "particle succeeds in every repeat",
        "invalid_counts_as_failure": True,
        "scopes": scopes,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fields = list(sets[0])
    with (args.output_root / "per_set_stability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sets)

    overall = scopes["overall"]
    lines = [
        "# ContactDiffusion set 间差异与 set 内粒子稳定性报告",
        "",
        f"- 输入：`{args.input.resolve()}`",
        f"- 每个 set 粒子数：{args.expected_particles}",
        f"- 稳定成功：全部 repeat 成功；高稳定成功：成功率 ≥ {100*args.high_threshold:.0f}%",
        "- invalid repeat 按失败计入；set 成功率为该 set 全部粒子、全部 repeat 的成功比例。",
        "",
        "## 1. set 间成功率差异",
        "",
        "| 范围 | sets | set均值 | 标准差 | P05 | 中位数 | P95 | 最低–最高 | set解释方差 η² | 粒子率 ICC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("overall", "barrett", "shadowhand"):
        item = scopes[key]
        dist = item["set_success_rate"]
        variance = item["variance_decomposition"]
        lines.append(
            f"| {key} | {item['sets']:,} | {fmt_pct(dist['mean'])} | {fmt_pct(dist['std'])} | "
            f"{fmt_pct(dist['p05'])} | {fmt_pct(dist['median'])} | {fmt_pct(dist['p95'])} | "
            f"{fmt_pct(dist['min'])}–{fmt_pct(dist['max'])} | {fmt_pct(variance['between_set_eta_squared'])} | "
            f"{variance['icc_1_particle_success_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "η² 表示粒子成功率总差异中可由所属 diffusion set 解释的比例；ICC 越高，说明同 set 粒子越相似、set 间差异越强。",
            "",
            "## 2. set 内是否存在高稳定成功粒子",
            "",
            "| 范围 | 稳定成功粒子 | 高稳定粒子 | 含稳定成功粒子的set | 含高稳定粒子的set | rank-0高稳定set | 有高稳定粒子但rank-0漏选 | 最佳粒子均值 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("overall", "barrett", "shadowhand"):
        item = scopes[key]
        particle = item["particle_stability"]
        within = item["within_set_high_stability"]
        lines.append(
            f"| {key} | {particle['stable_success_particles']:,}（{fmt_pct(particle['stable_success_particle_rate'])}） | "
            f"{particle['high_stability_particles']:,}（{fmt_pct(particle['high_stability_particle_rate'])}） | "
            f"{within['sets_with_stable_success_particle']:,}/{item['sets']:,}（{fmt_pct(within['set_rate_with_stable_success_particle'])}） | "
            f"{within['sets_with_high_stability_particle']:,}/{item['sets']:,}（{fmt_pct(within['set_rate_with_high_stability_particle'])}） | "
            f"{within['rank0_high_stability_sets']:,}（{fmt_pct(within['rank0_high_stability_set_rate'])}） | "
            f"{within['sets_with_high_particle_missed_by_rank0']:,}（{fmt_pct(within['miss_rate_among_sets_with_high_particle'])}） | "
            f"{fmt_pct(within['best_particle_rate_mean'])} |"
        )

    lines.extend(["", "## 3. 分物体结果", "", "| 手型/物体 | set均值 | set标准差 | 含高稳定粒子的set | rank-0漏选率 | 最佳粒子均值 |", "|---|---:|---:|---:|---:|---:|"])
    for key in sorted(scope for scope in scopes if "/" in scope):
        item = scopes[key]
        dist = item["set_success_rate"]
        within = item["within_set_high_stability"]
        lines.append(
            f"| {key} | {fmt_pct(dist['mean'])} | {fmt_pct(dist['std'])} | "
            f"{within['sets_with_high_stability_particle']}/{item['sets']}（{fmt_pct(within['set_rate_with_high_stability_particle'])}） | "
            f"{fmt_pct(within['miss_rate_among_sets_with_high_particle'])} | {fmt_pct(within['best_particle_rate_mean'])} |"
        )

    lines.extend(["", "## 4. 极端 set", ""])
    for hand in ("barrett", "shadowhand"):
        hand_sets = sorted((row for row in sets if row["hand"] == hand), key=lambda row: (row["set_success_rate"], row["object_id"], row["source_index"]))
        lines.extend([f"### {hand}", "", "| 分组 | 物体 | set | set成功率 | 最佳粒子率 | 最佳rank | rank-0率 |", "|---|---|---:|---:|---:|---:|---:|"])
        for label, rows in (("最低", hand_sets[:5]), ("最高", hand_sets[-5:][::-1])):
            for row in rows:
                lines.append(
                    f"| {label} | {row['object_id']} | {row['source_index']} | {fmt_pct(row['set_success_rate'])} | "
                    f"{fmt_pct(row['best_particle_success_rate'])} | {row['best_particle_rank']} | {fmt_pct(row['rank0_success_rate'])} |"
                )
        lines.append("")

    lines.extend(
        [
            "## 解读限制",
            "",
            "- 10-repeat 数据的单粒子成功率分辨率为 10%；“≥90%”仅对应 9/10 或 10/10，不能替代 1000-repeat 的精细概率估计。",
            "- 粒子属于同一 diffusion set，且 repeat 共享同一物理协议；报告衡量的是该协议下的经验稳定性，不代表跨物理参数扰动的鲁棒性。",
            "- 完整逐 set 数据见 `per_set_stability.csv`。",
            "",
        ]
    )
    (args.output_root / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall": overall, "barrett": scopes.get("barrett"), "shadowhand": scopes.get("shadowhand")}, indent=2))


if __name__ == "__main__":
    main()
