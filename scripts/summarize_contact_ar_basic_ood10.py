#!/usr/bin/env python3
"""Summarize AR-contact OOD-10 EAWQ Top-1 Isaac Gym results."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def quantile(ordered: list[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def canonical_hand(value: object) -> str:
    name = str(value).lower()
    if "barrett" in name:
        return "barrett"
    if "shadow" in name:
        return "shadowhand"
    return name


def result_object_name(payload: dict, fallback: str) -> str:
    """Resolve the simulated object across old/new validator schemas."""
    summaries = payload.get("object_summaries", {})
    if isinstance(summaries, dict):
        return str(next(iter(summaries), fallback))
    if isinstance(summaries, list):
        for row in summaries:
            if isinstance(row, dict) and row.get("object_name") == fallback:
                return str(row["object_name"])
        for row in summaries:
            if isinstance(row, dict) and int(row.get("trials", 0)) > 0:
                return str(row.get("object_name", fallback))
    results = payload.get("results", [])
    if results and isinstance(results[0], dict):
        return str(results[0].get("object_name", fallback))
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    result_paths = sorted((run_root / "results").glob("*/*.json"))
    if not result_paths:
        raise FileNotFoundError(f"No result JSONs under {run_root / 'results'}")

    totals = defaultdict(
        lambda: {
            "trials": 0,
            "valid": 0,
            "invalid": 0,
            "final_successes": 0,
            "strict_successes": 0,
            "elapsed_seconds": 0.0,
        }
    )
    per_object = []
    final_outcomes = {}
    protocol_ids = set()
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete result: {path}")
        hand = canonical_hand(payload.get("hand", path.parent.name))
        object_name = result_object_name(payload, path.stem)
        trials = int(payload.get("trials", len(payload.get("results", []))))
        valid = int(payload.get("valid_trials", 0))
        invalid = int(payload.get("invalid_trials", trials - valid))
        final_successes = int(payload.get("final_successes", payload.get("successes", 0)))
        strict_successes = int(payload.get("strict_six_direction_successes", 0))
        elapsed = float(payload.get("elapsed_seconds", 0.0))
        row = {
            "hand": hand,
            "object_name": object_name,
            "trials": trials,
            "valid": valid,
            "invalid": invalid,
            "final_successes": final_successes,
            "strict_successes": strict_successes,
            "final_success_rate": safe_rate(final_successes, trials),
            "strict_success_rate": safe_rate(strict_successes, trials),
            "valid_only_final_success_rate": safe_rate(final_successes, valid),
            "valid_only_strict_success_rate": safe_rate(strict_successes, valid),
            "elapsed_seconds": elapsed,
            "path": str(path),
        }
        per_object.append(row)
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            result_object = str(result.get("object_name", object_name))
            source_index = int(result.get("source_index", -1))
            final_outcomes[(hand, result_object, source_index)] = bool(
                result.get("final_success", False)
            )
        for key in (hand, "all"):
            aggregate = totals[key]
            aggregate["trials"] += trials
            aggregate["valid"] += valid
            aggregate["invalid"] += invalid
            aggregate["final_successes"] += final_successes
            aggregate["strict_successes"] += strict_successes
            aggregate["elapsed_seconds"] += elapsed
        if payload.get("protocol_id"):
            protocol_ids.add(str(payload["protocol_id"]))

    by_hand = {}
    for hand, aggregate in sorted(totals.items()):
        aggregate["final_success_rate"] = safe_rate(
            aggregate["final_successes"], aggregate["trials"]
        )
        aggregate["strict_success_rate"] = safe_rate(
            aggregate["strict_successes"], aggregate["trials"]
        )
        aggregate["invalid_rate"] = safe_rate(
            aggregate["invalid"], aggregate["trials"]
        )
        aggregate["valid_only_final_success_rate"] = safe_rate(
            aggregate["final_successes"], aggregate["valid"]
        )
        aggregate["valid_only_strict_success_rate"] = safe_rate(
            aggregate["strict_successes"], aggregate["valid"]
        )
        by_hand[hand] = dict(aggregate)

    projection = defaultdict(list)
    projection_outcomes = defaultdict(list)
    for path in sorted((run_root / "candidates").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            hand = canonical_hand(record.get("gripper", path.parent.name))
            distances = [
                float(value)
                for value in record.get("source_model_projection_distance_m", [])
            ]
            projection[hand].extend(distances)
            key = (hand, str(record.get("object_id", path.stem)), int(record.get("sample_index", -1)))
            if distances and key in final_outcomes:
                projection_outcomes[hand].append(
                    (sum(distances) / len(distances), final_outcomes[key])
                )
    projection_summary = {}
    for hand, values in sorted(projection.items()):
        ordered = sorted(values)
        projection_summary[hand] = {
            "count": len(ordered),
            "mean_mm": 1000.0 * sum(ordered) / max(len(ordered), 1),
            "median_mm": 1000.0 * ordered[len(ordered) // 2] if ordered else 0.0,
            "p90_mm": 1000.0 * quantile(ordered, 0.90),
            "p95_mm": 1000.0 * quantile(ordered, 0.95),
            "max_mm": 1000.0 * ordered[-1] if ordered else 0.0,
        }

    projection_success_diagnostic = {}
    for hand, pairs in sorted(projection_outcomes.items()):
        xs = [pair[0] for pair in pairs]
        ys = [float(pair[1]) for pair in pairs]
        successes = [x for x, success in pairs if success]
        failures = [x for x, success in pairs if not success]
        mean_x = sum(xs) / max(len(xs), 1)
        mean_y = sum(ys) / max(len(ys), 1)
        covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        variance_x = sum((x - mean_x) ** 2 for x in xs)
        variance_y = sum((y - mean_y) ** 2 for y in ys)
        denominator = math.sqrt(variance_x * variance_y)
        ordered_pairs = sorted(pairs, key=lambda pair: pair[0])
        quartiles = []
        for quartile_index in range(4):
            start = len(ordered_pairs) * quartile_index // 4
            stop = len(ordered_pairs) * (quartile_index + 1) // 4
            bucket = ordered_pairs[start:stop]
            quartiles.append(
                {
                    "sets": len(bucket),
                    "mean_projection_mm": 1000.0
                    * sum(pair[0] for pair in bucket)
                    / max(len(bucket), 1),
                    "final_success_rate": safe_rate(
                        sum(bool(pair[1]) for pair in bucket), len(bucket)
                    ),
                }
            )
        projection_success_diagnostic[hand] = {
            "matched_sets": len(pairs),
            "successful_set_mean_projection_mm": 1000.0
            * sum(successes)
            / max(len(successes), 1),
            "failed_set_mean_projection_mm": 1000.0
            * sum(failures)
            / max(len(failures), 1),
            "point_biserial_correlation_with_final_success": (
                covariance / denominator if denominator else 0.0
            ),
            "quartiles_low_to_high_projection": quartiles,
        }

    output = {
        "schema": "contact-ar-basic-ood10-summary-v1",
        "run_root": str(run_root),
        "protocol_ids": sorted(protocol_ids),
        "result_files": len(result_paths),
        "by_hand": by_hand,
        "raw_ar_to_surface_projection": projection_summary,
        "projection_vs_final_success_diagnostic": projection_success_diagnostic,
        "per_object": sorted(per_object, key=lambda row: (row["hand"], row["object_name"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["by_hand"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
