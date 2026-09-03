#!/usr/bin/env python3
"""Compute the frozen EAWQ rank-fusion inputs without simulation labels."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_execution_aware_wrench_auc import (
    FRICTION_GRID,
    GAP_SIGMA_M,
    add_qp_metrics,
    process_candidate_file,
)
from analyze_full_hand_contact_wrench_auc import (
    CONTACT_THRESHOLDS_M,
    MAX_CONTACT_REGIONS,
    SPATIAL_SUPPRESSION_M,
    add_full_hand_qp,
    process_candidate_file as process_full_hand_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qp-device", default="cuda:0")
    parser.add_argument("--qp-batch-size", type=int, default=1024)
    parser.add_argument("--qp-iterations", type=int, default=80)
    parser.add_argument("--expected-hands", type=int, default=2)
    parser.add_argument("--expected-objects-per-hand", type=int, default=1)
    parser.add_argument("--expected-sets-per-object", type=int, default=64)
    parser.add_argument("--expected-particles-per-set", type=int, default=32)
    parser.add_argument(
        "--ranking-only",
        action="store_true",
        help=(
            "Compute only the two residuals used by the frozen rank fusion. "
            "This is exact for Top-K selection and skips unused epsilon/convex-hull diagnostics."
        ),
    )
    return parser.parse_args()


def write_metrics(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_shape(rows: list[dict], args: argparse.Namespace) -> None:
    groups: dict[tuple[str, str, int], int] = {}
    for row in rows:
        key = (row["hand"], row["object_name"], int(row["source_index"]))
        groups[key] = groups.get(key, 0) + 1
    hands = sorted({key[0] for key in groups})
    if len(hands) != args.expected_hands:
        raise ValueError(f"found {len(hands)} hands, expected {args.expected_hands}")
    for hand in hands:
        objects = sorted({key[1] for key in groups if key[0] == hand})
        if len(objects) != args.expected_objects_per_hand:
            raise ValueError(
                f"{hand} has {len(objects)} objects, "
                f"expected {args.expected_objects_per_hand}"
            )
        for object_name in objects:
            object_groups = [
                count
                for key, count in groups.items()
                if key[0] == hand and key[1] == object_name
            ]
            if len(object_groups) != args.expected_sets_per_object:
                raise ValueError(
                    f"{hand}/{object_name} has {len(object_groups)} sets, "
                    f"expected {args.expected_sets_per_object}"
                )
            if set(object_groups) != {args.expected_particles_per_set}:
                raise ValueError(
                    f"{hand}/{object_name} particle counts are "
                    f"{sorted(set(object_groups))}"
                )


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted((run_root / "candidates").glob("*/*.json"))
    if not paths:
        raise FileNotFoundError(f"No candidate files under {run_root / 'candidates'}")

    distal_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                process_candidate_file,
                str(path),
                str(run_root),
                False,
                args.expected_particles_per_set,
                args.ranking_only,
            ): path
            for path in paths
        }
        for future in as_completed(futures):
            distal_rows.extend(future.result())
    distal_rows.sort(
        key=lambda row: (
            row["hand"], row["object_name"], row["source_index"], row["candidate_rank"]
        )
    )
    validate_shape(distal_rows, args)
    for hand in sorted({row["hand"] for row in distal_rows}):
        add_qp_metrics(
            [row for row in distal_rows if row["hand"] == hand],
            args.qp_device,
            args.qp_batch_size,
            args.qp_iterations,
            args.ranking_only,
        )
    distal_path = output_dir / "distal_particle_metrics.csv.gz"
    write_metrics(distal_path, distal_rows)

    full_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                process_full_hand_file,
                str(path),
                str(distal_path),
                args.ranking_only,
            ): path
            for path in paths
        }
        for future in as_completed(futures):
            full_rows.extend(future.result())
    full_rows.sort(
        key=lambda row: (
            row["hand"], row["object_name"], row["source_index"], row["candidate_rank"]
        )
    )
    validate_shape(full_rows, args)
    for hand in sorted({row["hand"] for row in full_rows}):
        add_full_hand_qp(
            [row for row in full_rows if row["hand"] == hand],
            args.qp_device,
            args.qp_batch_size,
            args.qp_iterations,
        )
    output_path = output_dir / "particle_metrics.csv.gz"
    write_metrics(output_path, full_rows)
    manifest = {
        "schema": "contactdiff-eawq-rank-fusion-metrics-v1",
        "labels_used": False,
        "ranking_only": bool(args.ranking_only),
        "ranking_equivalence": (
            "exact: frozen selector consumes only the two retained mean residuals"
            if args.ranking_only
            else "full diagnostics"
        ),
        "candidate_files": [str(path.resolve()) for path in paths],
        "particle_metrics": str(output_path),
        "rows": len(full_rows),
        "hands": args.expected_hands,
        "objects_per_hand": args.expected_objects_per_hand,
        "sets_per_object": args.expected_sets_per_object,
        "particles_per_set": args.expected_particles_per_set,
        "distal_metric": "old_eawq_weighted_mean_residual",
        "full_hand_metric": "full_hand_weighted_mean_residual",
        "metric_parameters": {
            "distal_and_palm": {
                "friction_grid": list(FRICTION_GRID),
                "contact_gap_sigma_m": GAP_SIGMA_M,
                "qp_iterations": args.qp_iterations,
            },
            "full_hand": {
                "contact_band_m": max(CONTACT_THRESHOLDS_M),
                "spatial_suppression_m": SPATIAL_SUPPRESSION_M,
                "max_contact_regions": MAX_CONTACT_REGIONS,
                "friction_coefficient": 0.6,
                "qp_iterations": args.qp_iterations,
            },
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
