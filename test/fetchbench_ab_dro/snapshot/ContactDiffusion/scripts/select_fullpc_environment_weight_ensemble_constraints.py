#!/usr/bin/env python3
"""Select one geometry-feasible candidate per contact set across env weights."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def candidate_key(item: dict) -> tuple:
    constraint_score = float(
        item.get("selection_constraint_score", float("inf"))
    )
    if math.isfinite(constraint_score):
        # Constrained-refiner scores share the same physical normalizers across
        # environment weights.  Preserve that ranking instead of reintroducing
        # the legacy 0.1 mm point-cloud hard gate at ensemble selection time.
        return (
            not bool(item.get("selection_feasible", False)),
            constraint_score,
            float(item.get("optimization_score", float("inf"))),
            int(item.get("source_environment_weight", 0)),
            int(item.get("particle", 0)),
        )
    env = float(item.get("environment_max_violation_m", float("inf")))
    penetration = float(item.get("max_penetration_m", float("inf")))
    contact = float(item.get("contact_chamfer_m", float("inf")))
    env_pass = bool(item.get("environment_filter_pass", env <= 0.0001))
    penetration_pass = penetration <= 0.007
    contact_pass = contact <= 0.020
    feasible = bool(item.get("selection_feasible", False))
    normalized_violation = (
        max(0.0, env - 0.0001) / 0.005
        + max(0.0, penetration - 0.007) / 0.007
        + max(0.0, contact - 0.020) / 0.020
    )
    return (
        not feasible,
        not env_pass,
        not penetration_pass,
        not contact_pass,
        normalized_violation,
        contact + penetration,
        env,
        int(item.get("source_environment_weight", 0)),
        int(item.get("particle", 0)),
    )


def main() -> None:
    args = parse_args()
    payloads = [
        json.loads(path.resolve().read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    if not payloads:
        raise ValueError("No input payloads")
    by_sample = {}
    for path, payload in zip(args.inputs, payloads, strict=True):
        # A single prefilter/refill payload has not yet gone through the
        # environment-energy refiner.  Treat it as weight zero so this same
        # selector can form a geometry-only Top-1 ablation.
        weight = int(payload.get("environment_energy", {}).get("weight", 0))
        for record in payload["records"]:
            sample = int(record["sample_index"])
            entry = by_sample.setdefault(sample, {"record": record, "candidates": []})
            for candidate in record["fk"]["candidates"]:
                candidate = deepcopy(candidate)
                candidate["source_environment_weight"] = weight
                candidate["source_environment_file"] = str(path.resolve())
                entry["candidates"].append(candidate)

    output = deepcopy(payloads[0])
    output["records"] = []
    # This file is a post-refinement Top-1 selection artifact.  Keep the
    # original particle budget in ``particles`` for provenance, but advertise
    # the actually retained prefix so the Isaac Gym preparation step can
    # validate it without pretending that 32 candidates remain per set.
    output.setdefault("selection", {})["top_k"] = 1
    weight_counts = {}
    for sample in sorted(by_sample):
        entry = by_sample[sample]
        ranked = sorted(entry["candidates"], key=candidate_key)
        selected = deepcopy(ranked[0])
        source_weight = int(selected["source_environment_weight"])
        weight_counts[str(source_weight)] = weight_counts.get(str(source_weight), 0) + 1
        selected["rank"] = 0
        record = deepcopy(entry["record"])
        record["fk"]["candidates"] = [selected]
        output["records"].append(record)
    output["environment_weight_ensemble"] = {
        "policy": (
            "normalized_object_contact_environment_constraints"
            if all(
                math.isfinite(
                    float(candidate.get("selection_constraint_score", float("inf")))
                )
                for entry in by_sample.values()
                for candidate in entry["candidates"]
            )
            else "feasible_env_penetration_contact_lexicographic"
        ),
        "source_files": [str(path.resolve()) for path in args.inputs],
        "selected_weight_counts": weight_counts,
        "sets": len(output["records"]),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
