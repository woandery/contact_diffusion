#!/usr/bin/env python3
"""Summarize all-particle Isaac Gym results by contact set and candidate rank."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


RANK_CUTOFFS = (1, 2, 4, 8, 16, 32)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def trial_success(row: dict, definition: str) -> bool:
    if not bool(row.get("valid_simulation", row.get("valid", True))):
        return False
    if definition == "final":
        return bool(row.get("final_success", row.get("success", False)))
    if definition == "strict":
        return bool(
            row.get("strict_six_direction_success", row.get("strict_success", False))
        )
    raise ValueError(definition)


def aggregate_sets(groups: dict[tuple, list[dict]]) -> dict:
    output: dict[str, object] = {"contact_sets": len(groups)}
    total_trials = sum(len(rows) for rows in groups.values())
    output["trials"] = total_trials
    output["invalid_trials"] = sum(
        not bool(row.get("valid_simulation", row.get("valid", True)))
        for rows in groups.values()
        for row in rows
    )
    for definition in ("final", "strict"):
        top1 = 0
        oracle = 0
        particle_successes = 0
        success_at_k = {cutoff: 0 for cutoff in RANK_CUTOFFS}
        for rows in groups.values():
            ordered = sorted(rows, key=lambda row: int(row["candidate_rank"]))
            outcomes = [trial_success(row, definition) for row in ordered]
            top1 += bool(outcomes and outcomes[0])
            oracle += any(outcomes)
            particle_successes += sum(outcomes)
            for cutoff in RANK_CUTOFFS:
                success_at_k[cutoff] += any(outcomes[:cutoff])
        prefix = f"{definition}_"
        output[prefix + "top1_success_sets"] = top1
        output[prefix + "top1_success_rate"] = ratio(top1, len(groups))
        output[prefix + "oracle_success_sets"] = oracle
        output[prefix + "oracle_success_rate"] = ratio(oracle, len(groups))
        output[prefix + "ranking_miss_sets"] = oracle - top1
        output[prefix + "ranking_miss_rate"] = ratio(oracle - top1, len(groups))
        output[prefix + "no_success_particle_sets"] = len(groups) - oracle
        output[prefix + "no_success_particle_set_rate"] = ratio(
            len(groups) - oracle, len(groups)
        )
        output[prefix + "particle_success_rate"] = ratio(
            particle_successes, total_trials
        )
        output[prefix + "success_at_k"] = {
            str(cutoff): {
                "sets": success_at_k[cutoff],
                "rate": ratio(success_at_k[cutoff], len(groups)),
            }
            for cutoff in RANK_CUTOFFS
        }
    return output


def load_results(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("results", []))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    root = args.run_root.resolve()
    candidates_root = root / "candidates"
    results_root = root / "results"
    if not candidates_root.is_dir() or not results_root.is_dir():
        raise FileNotFoundError("run root must contain candidates/ and results/")

    all_groups: dict[tuple, list[dict]] = defaultdict(list)
    per_object: dict[str, dict] = {}
    duplicates: list[tuple] = []
    seen_trials: set[tuple] = set()
    candidate_files = sorted(candidates_root.glob("*/*.json"))
    if not candidate_files:
        raise ValueError("no candidate files found")
    for candidate_path in candidate_files:
        hand = candidate_path.parent.name
        candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        records = candidate_payload["records"]
        if not records:
            continue
        object_ids = {str(row["object_id"]) for row in records}
        if len(object_ids) != 1:
            raise ValueError(f"{candidate_path}: expected one object")
        object_id = next(iter(object_ids))
        rows = load_results(results_root / hand / object_id)
        for row in rows:
            key = (
                hand,
                object_id,
                int(row["source_index"]),
                int(row.get("candidate_rank", 0)),
            )
            if key in seen_trials:
                duplicates.append(key)
                continue
            seen_trials.add(key)
            set_key = key[:3]
            all_groups[set_key].append(row)
        expected_sets = len(records)
        object_groups = {
            key: value
            for key, value in all_groups.items()
            if key[0] == hand and key[1] == object_id
        }
        expected_trials = expected_sets * int(candidate_payload["particles"])
        actual_trials = sum(len(value) for value in object_groups.values())
        if not args.allow_incomplete and actual_trials != expected_trials:
            raise ValueError(
                f"{hand}/{object_id}: {actual_trials} trials, expected {expected_trials}"
            )
        if not args.allow_incomplete:
            bad = {
                key: len(value)
                for key, value in object_groups.items()
                if len(value) != int(candidate_payload["particles"])
            }
            if bad:
                raise ValueError(f"{hand}/{object_id}: incomplete sets: {bad}")
        per_object[f"{hand}/{object_id}"] = aggregate_sets(object_groups)

    if duplicates:
        raise ValueError(f"duplicate result trials: {duplicates[:10]}")
    per_hand = {
        hand: aggregate_sets(
            {key: value for key, value in all_groups.items() if key[0] == hand}
        )
        for hand in sorted({key[0] for key in all_groups})
    }
    report = {
        "schema": "contactdiff-basic-all-particle-summary-v1",
        "run_root": str(root),
        "incomplete_allowed": bool(args.allow_incomplete),
        "rank_cutoffs": list(RANK_CUTOFFS),
        "overall": aggregate_sets(all_groups),
        "per_hand": per_hand,
        "per_object_hand": per_object,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
