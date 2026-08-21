#!/usr/bin/env python3
"""Audit and summarize paired diffusion-vs-random contact guidance results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


VARIANTS = ("diffusion", "matched_random")


def valid_success(row: dict, definition: str) -> bool:
    if not bool(row.get("valid_simulation", row.get("valid", True))):
        return False
    if definition == "final":
        return bool(row.get("final_success", row.get("success", False)))
    return bool(
        row.get("strict_six_direction_success", row.get("strict_success", False))
    )


def load_json_rows(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("results", []))
    return rows


def load_variant(
    root: Path, variant: str, *, load_results: bool = True
) -> tuple[dict, dict]:
    variant_root = root / variant
    metadata: dict[tuple[str, str, int], dict] = {}
    outcomes: dict[tuple[str, str, int, int], dict] = {}
    for candidate_path in sorted((variant_root / "candidates").glob("*/*.json")):
        hand = candidate_path.parent.name
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        if payload.get("contact_target_ab", {}).get("mode") != variant:
            raise ValueError(f"{candidate_path}: contact target mode mismatch")
        records = payload.get("records", [])
        if not records:
            continue
        object_ids = {str(row["object_id"]) for row in records}
        if len(object_ids) != 1:
            raise ValueError(f"{candidate_path}: expected one object")
        object_id = next(iter(object_ids))
        rank_to_particle: dict[tuple[int, int], int] = {}
        for record in records:
            sample = int(record["sample_index"])
            key = (hand, object_id, sample)
            fk = record["fk"]
            particle_indices = tuple(
                sorted(int(candidate["particle"]) for candidate in fk["candidates"])
            )
            metadata[key] = {
                "sample_seed": int(record["sample_seed"]),
                "source_diffusion_contacts_sha256": record.get(
                    "source_diffusion_contacts_sha256"
                ),
                "target_contacts_sha256": record.get("target_contacts_sha256"),
                "initialization_state_sha256": fk.get(
                    "initialization_state_sha256"
                ),
                "matcher": record.get("matched_random"),
                "particle_indices": particle_indices,
            }
            for candidate in fk["candidates"]:
                rank_to_particle[(sample, int(candidate["rank"]))] = int(
                    candidate["particle"]
                )
        if not load_results:
            continue
        for row in load_json_rows(variant_root / "results" / hand / object_id):
            sample = int(row["source_index"])
            rank = int(row.get("candidate_rank", 0))
            particle = rank_to_particle.get((sample, rank))
            if particle is None:
                raise ValueError(
                    f"{variant}/{hand}/{object_id}: no particle for sample={sample}, rank={rank}"
                )
            key = (hand, object_id, sample, particle)
            if key in outcomes:
                raise ValueError(f"duplicate result {variant} {key}")
            outcomes[key] = row
    return metadata, outcomes


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = int(left_only + right_only)
    if discordant == 0:
        return 1.0
    tail = min(int(left_only), int(right_only))
    return float(binomtest(tail, discordant, p=0.5, alternative="two-sided").pvalue)


def object_bootstrap_ci(
    paired: list[tuple[tuple, bool, bool]],
    *,
    seed: int,
    draws: int,
) -> list[float]:
    by_object: dict[tuple[str, str], list[tuple[bool, bool]]] = defaultdict(list)
    for key, left, right in paired:
        by_object[(key[0], key[1])].append((left, right))
    clusters = sorted(by_object)
    if not clusters:
        return [0.0, 0.0]
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        numerator = 0
        denominator = 0
        for index in selected:
            rows = by_object[clusters[int(index)]]
            numerator += sum(int(left) - int(right) for left, right in rows)
            denominator += len(rows)
        values[draw] = numerator / denominator if denominator else 0.0
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def aggregate(
    keys: list[tuple[str, str, int, int]],
    outcomes_a: dict,
    outcomes_b: dict,
    *,
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> dict:
    report: dict[str, object] = {"paired_particles": len(keys)}
    set_keys = sorted({key[:3] for key in keys})
    particles_by_set: dict[tuple[str, str, int], list[tuple[str, str, int, int]]] = (
        defaultdict(list)
    )
    for key in keys:
        particles_by_set[key[:3]].append(key)
    report["paired_contact_sets"] = len(set_keys)
    for definition in ("final", "strict"):
        paired = [
            (
                key,
                valid_success(outcomes_a[key], definition),
                valid_success(outcomes_b[key], definition),
            )
            for key in keys
        ]
        a_success = sum(left for _, left, _ in paired)
        b_success = sum(right for _, _, right in paired)
        a_only = sum(left and not right for _, left, right in paired)
        b_only = sum(right and not left for _, left, right in paired)
        oracle_a = sum(
            any(
                valid_success(outcomes_a[key], definition)
                for key in particles_by_set[set_key]
            )
            for set_key in set_keys
        )
        oracle_b = sum(
            any(
                valid_success(outcomes_b[key], definition)
                for key in particles_by_set[set_key]
            )
            for set_key in set_keys
        )
        oracle_paired = [
            (
                set_key,
                any(
                    valid_success(outcomes_a[key], definition)
                    for key in particles_by_set[set_key]
                ),
                any(
                    valid_success(outcomes_b[key], definition)
                    for key in particles_by_set[set_key]
                ),
            )
            for set_key in set_keys
        ]
        oracle_a_only = sum(
            left and not right for _, left, right in oracle_paired
        )
        oracle_b_only = sum(
            right and not left for _, left, right in oracle_paired
        )
        report[definition] = {
            "diffusion_particle_successes": a_success,
            "matched_random_particle_successes": b_success,
            "diffusion_particle_success_rate": a_success / len(keys),
            "matched_random_particle_success_rate": b_success / len(keys),
            "diffusion_minus_random_particle_success_pp": 100.0
            * (a_success - b_success)
            / len(keys),
            "particle_diffusion_only": a_only,
            "particle_random_only": b_only,
            "particle_mcnemar_exact_p": exact_mcnemar_p(a_only, b_only),
            "particle_delta_object_cluster_bootstrap95_pp": [
                100.0 * value
                for value in object_bootstrap_ci(
                    paired,
                    seed=bootstrap_seed + (0 if definition == "final" else 1),
                    draws=bootstrap_draws,
                )
            ],
            "diffusion_oracle32_sets": oracle_a,
            "matched_random_oracle32_sets": oracle_b,
            "diffusion_no_success_particle_sets": len(set_keys) - oracle_a,
            "matched_random_no_success_particle_sets": len(set_keys) - oracle_b,
            "diffusion_oracle32_rate": oracle_a / len(set_keys),
            "matched_random_oracle32_rate": oracle_b / len(set_keys),
            "diffusion_minus_random_oracle32_pp": 100.0
            * (oracle_a - oracle_b)
            / len(set_keys),
            "oracle32_diffusion_only_sets": oracle_a_only,
            "oracle32_random_only_sets": oracle_b_only,
            "oracle32_mcnemar_exact_p": exact_mcnemar_p(
                oracle_a_only, oracle_b_only
            ),
            "oracle32_delta_object_cluster_bootstrap95_pp": [
                100.0 * value
                for value in object_bootstrap_ci(
                    oracle_paired,
                    seed=bootstrap_seed + (17 if definition == "final" else 18),
                    draws=bootstrap_draws,
                )
            ],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    metadata_a, outcomes_a = load_variant(
        root, "diffusion", load_results=not args.audit_only
    )
    metadata_b, outcomes_b = load_variant(
        root, "matched_random", load_results=not args.audit_only
    )
    if set(metadata_a) != set(metadata_b):
        raise ValueError("A/B candidate contact-set keys differ")
    if set(outcomes_a) != set(outcomes_b):
        raise ValueError("A/B paired particle result keys differ")

    source_mismatch = []
    seed_mismatch = []
    initialization_mismatch = []
    particle_mismatch = []
    identical_target = []
    for key in sorted(metadata_a):
        left, right = metadata_a[key], metadata_b[key]
        if left["sample_seed"] != right["sample_seed"]:
            seed_mismatch.append(key)
        if left["source_diffusion_contacts_sha256"] != right["source_diffusion_contacts_sha256"]:
            source_mismatch.append(key)
        if left["initialization_state_sha256"] != right["initialization_state_sha256"]:
            initialization_mismatch.append(key)
        if left["particle_indices"] != right["particle_indices"]:
            particle_mismatch.append(key)
        if left["target_contacts_sha256"] == right["target_contacts_sha256"]:
            identical_target.append(key)
    audit = {
        "paired_contact_sets": len(metadata_a),
        "candidate_particles": sum(
            len(row["particle_indices"]) for row in metadata_a.values()
        ),
        "simulated_paired_particles": len(outcomes_a),
        "sample_seed_mismatches": len(seed_mismatch),
        "source_diffusion_contact_mismatches": len(source_mismatch),
        "initialization_state_mismatches": len(initialization_mismatch),
        "candidate_particle_index_mismatches": len(particle_mismatch),
        "identical_ab_target_sets": len(identical_target),
        "all_pairing_checks_pass": not (
            seed_mismatch
            or source_mismatch
            or initialization_mismatch
            or particle_mismatch
            or identical_target
        ),
    }
    if not audit["all_pairing_checks_pass"]:
        raise ValueError(f"A/B pairing audit failed: {audit}")

    if args.audit_only:
        payload = {
            "schema": "contactdiff-contact-set-guidance-ab-audit-v1",
            "run_root": str(root),
            "audit": audit,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(audit, indent=2))
        print(f"Wrote {args.output.resolve()}")
        return

    keys = sorted(outcomes_a)
    if not keys:
        raise ValueError("no paired simulator outcomes found")
    hands = sorted({key[0] for key in keys})
    report = {
        "schema": "contactdiff-contact-set-guidance-ab-summary-v1",
        "run_root": str(root),
        "variants": list(VARIANTS),
        "audit": audit,
        "overall": aggregate(
            keys,
            outcomes_a,
            outcomes_b,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_draws=args.bootstrap_draws,
        ),
        "per_hand": {
            hand: aggregate(
                [key for key in keys if key[0] == hand],
                outcomes_a,
                outcomes_b,
                bootstrap_seed=args.bootstrap_seed + 101 * (index + 1),
                bootstrap_draws=args.bootstrap_draws,
            )
            for index, hand in enumerate(hands)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_md is not None:
        final = report["overall"]["final"]
        strict = report["overall"]["strict"]
        markdown = f"""# Contact-set guidance A/B result

- Pairing audit: `pass`
- Contact sets: {audit['paired_contact_sets']}
- Paired particles: {audit['simulated_paired_particles']}

| Metric | Diffusion | Matched random | Delta |
|---|---:|---:|---:|
| Final particle success | {final['diffusion_particle_success_rate']:.2%} | {final['matched_random_particle_success_rate']:.2%} | {final['diffusion_minus_random_particle_success_pp']:+.2f} pp |
| Strict particle success | {strict['diffusion_particle_success_rate']:.2%} | {strict['matched_random_particle_success_rate']:.2%} | {strict['diffusion_minus_random_particle_success_pp']:+.2f} pp |
| Final Oracle@32 | {final['diffusion_oracle32_rate']:.2%} | {final['matched_random_oracle32_rate']:.2%} | {final['diffusion_minus_random_oracle32_pp']:+.2f} pp |
| Strict Oracle@32 | {strict['diffusion_oracle32_rate']:.2%} | {strict['matched_random_oracle32_rate']:.2%} | {strict['diffusion_minus_random_oracle32_pp']:+.2f} pp |

Final particle paired McNemar exact p: `{final['particle_mcnemar_exact_p']:.6g}`.
Object-cluster bootstrap 95% CI: `{final['particle_delta_object_cluster_bootstrap95_pp']}` pp.
Final Oracle@32 contact-set McNemar exact p: `{final['oracle32_mcnemar_exact_p']:.6g}`.
Oracle@32 object-cluster bootstrap 95% CI: `{final['oracle32_delta_object_cluster_bootstrap95_pp']}` pp.
"""
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
