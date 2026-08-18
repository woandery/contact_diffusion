#!/usr/bin/env python3
"""Build prepared manifests containing EAWQ rank-fusion top1 per contact set."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


DISTAL_METRIC = "old_eawq_weighted_mean_residual"
FULL_HAND_METRIC = "full_hand_weighted_mean_residual"
FUSION_NAME = "fusion_old_full_residual"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean(value: str) -> bool:
    return value.lower() == "true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particle-metrics", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-hands", type=int, default=2)
    parser.add_argument("--expected-objects-per-hand", type=int, default=10)
    parser.add_argument("--expected-sets-per-object", type=int, default=64)
    parser.add_argument("--expected-particles-per-set", type=int, default=32)
    return parser.parse_args()


def load_metric_rows(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "hand",
            "object_name",
            "source_index",
            "candidate_rank",
            "selection_feasible",
            DISTAL_METRIC,
            FULL_HAND_METRIC,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"particle metrics missing columns: {sorted(missing)}")
        for raw in reader:
            rows.append(
                {
                    "hand": raw["hand"],
                    "object_name": raw["object_name"],
                    "source_index": int(raw["source_index"]),
                    "candidate_rank": int(raw["candidate_rank"]),
                    "selection_feasible": boolean(raw["selection_feasible"]),
                    DISTAL_METRIC: float(raw[DISTAL_METRIC]),
                    FULL_HAND_METRIC: float(raw[FULL_HAND_METRIC]),
                }
            )
    return rows


def select_top1(rows: list[dict], expected_particles: int) -> dict:
    if len(rows) != expected_particles:
        raise ValueError(
            f"contact set contains {len(rows)} particles, expected {expected_particles}"
        )
    candidate_ranks = [row["candidate_rank"] for row in rows]
    if len(set(candidate_ranks)) != len(candidate_ranks):
        raise ValueError(f"duplicate candidate ranks: {candidate_ranks}")
    denominator = max(len(rows) - 1, 1)
    for metric in (DISTAL_METRIC, FULL_HAND_METRIC):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite {metric} values")
        normalized = (rankdata(values, method="average") - 1.0) / denominator
        for row, value in zip(rows, normalized):
            row[f"rank_{metric}"] = float(value)
    for row in rows:
        row[FUSION_NAME] = float(
            0.5
            * (
                row[f"rank_{DISTAL_METRIC}"]
                + row[f"rank_{FULL_HAND_METRIC}"]
            )
        )
    return min(
        rows,
        key=lambda row: (
            -int(row["selection_feasible"]),
            row[FUSION_NAME],
            row["candidate_rank"],
        ),
    )


def main() -> None:
    args = parse_args()
    metrics_path = args.particle_metrics.resolve()
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve()
    rows = load_metric_rows(metrics_path)

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["hand"], row["object_name"], row["source_index"])].append(
            row
        )
    hands = sorted({key[0] for key in grouped})
    objects_by_hand = {
        hand: sorted({key[1] for key in grouped if key[0] == hand})
        for hand in hands
    }
    if len(hands) != args.expected_hands:
        raise ValueError(f"found {len(hands)} hands, expected {args.expected_hands}")
    for hand, objects in objects_by_hand.items():
        if len(objects) != args.expected_objects_per_hand:
            raise ValueError(
                f"{hand} has {len(objects)} objects, "
                f"expected {args.expected_objects_per_hand}"
            )

    selected = []
    for key in sorted(grouped):
        selected.append(
            select_top1(grouped[key], args.expected_particles_per_set).copy()
        )
    selected_by_object: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selected:
        selected_by_object[(row["hand"], row["object_name"])].append(row)
    for key, object_rows in selected_by_object.items():
        if len(object_rows) != args.expected_sets_per_object:
            raise ValueError(
                f"{key} has {len(object_rows)} contact sets, "
                f"expected {args.expected_sets_per_object}"
            )

    output_prepared = output_root / "prepared"
    prepared_manifest_rows = []
    for (hand, object_name), object_rows in sorted(selected_by_object.items()):
        input_path = prepared_root / hand / f"{object_name}.json"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        matching_objects = [
            group for group in payload["objects"] if group["object_name"] == object_name
        ]
        if len(matching_objects) != 1:
            raise ValueError(
                f"expected one {object_name} object in {input_path}, "
                f"found {len(matching_objects)}"
            )
        source_group = matching_objects[0]
        samples = {
            (int(sample["source_index"]), int(sample["candidate_rank"])): sample
            for sample in source_group["samples"]
        }
        selected_samples = []
        for row in sorted(object_rows, key=lambda item: item["source_index"]):
            sample_key = (row["source_index"], row["candidate_rank"])
            if sample_key not in samples:
                raise ValueError(f"selected sample missing from {input_path}: {sample_key}")
            sample = deepcopy(samples[sample_key])
            sample["eawq_rank_fusion"] = {
                "name": FUSION_NAME,
                "distal_metric": DISTAL_METRIC,
                "full_hand_metric": FULL_HAND_METRIC,
                "distal_normalized_rank": row[f"rank_{DISTAL_METRIC}"],
                "full_hand_normalized_rank": row[f"rank_{FULL_HAND_METRIC}"],
                "equal_weight_fusion_score": row[FUSION_NAME],
                "feasibility_gate": True,
                "selected_feasible": row["selection_feasible"],
            }
            selected_samples.append(sample)
            prepared_manifest_rows.append(
                {
                    **row,
                    "particle_index": sample.get("particle_index"),
                    "sample_seed": sample.get("sample_seed"),
                    "prepared_input": str(input_path),
                }
            )

        output_payload = deepcopy(payload)
        output_payload["method"] = "step50k_eawq_rank_fusion_top1"
        output_payload["parent_prepared"] = str(input_path)
        output_payload["parent_prepared_sha256"] = sha256(input_path)
        output_payload["contact_sets_per_object"] = len(selected_samples)
        output_payload["particles_per_set"] = args.expected_particles_per_set
        output_payload["retained_per_set"] = 1
        output_payload["ranking"] = {
            "name": FUSION_NAME,
            "scope": "within each contact set",
            "distal_metric": DISTAL_METRIC,
            "full_hand_metric": FULL_HAND_METRIC,
            "normalization": "average tie rank mapped to [0, 1], lower is better",
            "fusion": "equal arithmetic mean of normalized ranks",
            "selection": "feasible first, then fusion score, then candidate rank",
            "success_labels_used": False,
            "particle_metrics": str(metrics_path),
            "particle_metrics_sha256": sha256(metrics_path),
        }
        output_payload["objects"] = [
            {
                **deepcopy(source_group),
                "available_for_object": len(selected_samples),
                "samples": selected_samples,
            }
        ]
        output_path = output_prepared / hand / f"{object_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output_payload, indent=2) + "\n", encoding="utf-8"
        )

    selection_dir = output_root / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_csv = selection_dir / "eawq_rank_fusion_top1.csv"
    csv_fields = [
        "hand",
        "object_name",
        "source_index",
        "candidate_rank",
        "particle_index",
        "sample_seed",
        "selection_feasible",
        DISTAL_METRIC,
        FULL_HAND_METRIC,
        f"rank_{DISTAL_METRIC}",
        f"rank_{FULL_HAND_METRIC}",
        FUSION_NAME,
        "prepared_input",
    ]
    with selection_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(prepared_manifest_rows)

    manifest = {
        "schema": "contactdiff-eawq-rank-fusion-top1-v1",
        "particle_metrics": str(metrics_path),
        "particle_metrics_sha256": sha256(metrics_path),
        "prepared_root": str(prepared_root),
        "output_prepared_root": str(output_prepared),
        "ranking": {
            "name": FUSION_NAME,
            "distal_metric": DISTAL_METRIC,
            "full_hand_metric": FULL_HAND_METRIC,
            "success_labels_used": False,
            "feasibility_gate": True,
        },
        "hands": hands,
        "objects_per_hand": args.expected_objects_per_hand,
        "sets_per_object": args.expected_sets_per_object,
        "particles_per_set": args.expected_particles_per_set,
        "input_particles": len(rows),
        "selected_top1": len(selected),
        "selected_feasible": sum(row["selection_feasible"] for row in selected),
        "selected_infeasible_fallback": sum(
            not row["selection_feasible"] for row in selected
        ),
        "selected_candidate_rank_counts": dict(
            sorted(Counter(row["candidate_rank"] for row in selected).items())
        ),
        "selection_csv": str(selection_csv),
        "prepared_files": len(selected_by_object),
    }
    manifest_path = selection_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
