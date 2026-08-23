#!/usr/bin/env python3
"""Summarize paired FK contact-energy ablations and rank-prefix PhysX success."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


RANK_CUTOFFS = (1, 2, 4, 8, 16, 32)


def valid_success(row: dict, definition: str) -> bool:
    if not bool(row.get("valid_simulation", row.get("valid", True))):
        return False
    if definition == "final":
        return bool(row.get("final_success", row.get("success", False)))
    if definition == "strict":
        return bool(
            row.get("strict_six_direction_success", row.get("strict_success", False))
        )
    raise ValueError(definition)


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = int(left_only + right_only)
    if discordant == 0:
        return 1.0
    return float(
        binomtest(
            min(int(left_only), int(right_only)),
            discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def object_cluster_bootstrap_ci(
    paired: list[tuple[tuple, bool, bool]], *, seed: int, draws: int
) -> list[float]:
    by_object: dict[tuple[str, str], list[tuple[bool, bool]]] = defaultdict(list)
    for key, left, right in paired:
        by_object[(str(key[0]), str(key[1]))].append((left, right))
    clusters = sorted(by_object)
    if not clusters:
        return [0.0, 0.0]
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        numerator = 0
        denominator = 0
        for cluster_index in rng.integers(0, len(clusters), size=len(clusters)):
            rows = by_object[clusters[int(cluster_index)]]
            numerator += sum(int(left) - int(right) for left, right in rows)
            denominator += len(rows)
        values[draw] = numerator / denominator if denominator else 0.0
    return [
        100.0 * float(np.quantile(values, 0.025)),
        100.0 * float(np.quantile(values, 0.975)),
    ]


def describe(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
    }


@dataclass
class Arm:
    name: str
    root: Path
    contact_weight: float
    target_mode: str
    metadata: dict
    candidates: dict
    outcomes: dict


def load_result_rows(directory: Path) -> list[dict]:
    rows: list[dict] = []
    paths = sorted(directory.glob("batch_*.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"{path}: result batch is not complete")
        rows.extend(payload.get("results", []))
    return rows


def load_arm(name: str, root: Path, *, load_results: bool = True) -> Arm:
    root = root.resolve()
    metadata: dict[tuple[str, str, int], dict] = {}
    candidates: dict[tuple[str, str, int, int], dict] = {}
    outcomes: dict[tuple[str, str, int, int], dict] = {}
    contact_weights: set[float] = set()
    target_modes: set[str] = set()
    candidate_paths = sorted((root / "candidates").glob("*/*.json"))
    if not candidate_paths:
        raise ValueError(f"{name}: no candidate files under {root}")
    for candidate_path in candidate_paths:
        hand = candidate_path.parent.name
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        contact_weights.add(float(payload["fk_energy"]["contact_weight"]))
        target_modes.add(str(payload["contact_target_ab"]["mode"]))
        records = payload.get("records", [])
        if not records:
            raise ValueError(f"{candidate_path}: no candidate records")
        object_ids = {str(record["object_id"]) for record in records}
        if len(object_ids) != 1:
            raise ValueError(f"{candidate_path}: expected exactly one object")
        object_id = next(iter(object_ids))
        rank_to_particle: dict[tuple[int, int], int] = {}
        for record in records:
            sample = int(record["sample_index"])
            set_key = (hand, object_id, sample)
            if set_key in metadata:
                raise ValueError(f"{name}: duplicate contact set {set_key}")
            fk = record["fk"]
            particle_indices = tuple(
                sorted(int(candidate["particle"]) for candidate in fk["candidates"])
            )
            metadata[set_key] = {
                "sample_seed": int(record["sample_seed"]),
                "source_diffusion_contacts_sha256": record.get(
                    "source_diffusion_contacts_sha256"
                ),
                "target_contacts_sha256": record.get("target_contacts_sha256"),
                "initialization_state_sha256": fk.get("initialization_state_sha256"),
                "particle_indices": particle_indices,
            }
            for candidate in fk["candidates"]:
                particle = int(candidate["particle"])
                rank = int(candidate["rank"])
                key = (*set_key, particle)
                if key in candidates:
                    raise ValueError(f"{name}: duplicate candidate {key}")
                candidates[key] = {
                    "rank": rank,
                    "contact_chamfer_m": float(candidate["contact_chamfer_m"]),
                    "assigned_contact_error_m": float(
                        candidate["assigned_contact_error_m"]
                    ),
                }
                rank_to_particle[(sample, rank)] = particle
        if not load_results:
            continue
        rows = load_result_rows(root / "results" / hand / object_id)
        for row in rows:
            sample = int(row["source_index"])
            rank = int(row.get("candidate_rank", 0))
            particle = rank_to_particle.get((sample, rank))
            if particle is None:
                raise ValueError(
                    f"{name}/{hand}/{object_id}: no particle for sample={sample}, rank={rank}"
                )
            key = (hand, object_id, sample, particle)
            if key in outcomes:
                raise ValueError(f"{name}: duplicate simulator outcome {key}")
            outcomes[key] = {
                "rank": rank,
                "final": valid_success(row, "final"),
                "strict": valid_success(row, "strict"),
                "valid": bool(row.get("valid_simulation", row.get("valid", True))),
            }
    if len(contact_weights) != 1 or len(target_modes) != 1:
        raise ValueError(
            f"{name}: inconsistent contact weights or target modes: "
            f"{contact_weights}, {target_modes}"
        )
    if load_results and set(candidates) != set(outcomes):
        missing = set(candidates) - set(outcomes)
        extra = set(outcomes) - set(candidates)
        raise ValueError(
            f"{name}: candidate/result key mismatch missing={len(missing)} extra={len(extra)}"
        )
    return Arm(
        name=name,
        root=root,
        contact_weight=next(iter(contact_weights)),
        target_mode=next(iter(target_modes)),
        metadata=metadata,
        candidates=candidates,
        outcomes=outcomes,
    )


def aggregate(arm: Arm, keys: list[tuple]) -> dict:
    set_keys = sorted({key[:3] for key in keys})
    keys_by_set: dict[tuple, list[tuple]] = defaultdict(list)
    for key in keys:
        keys_by_set[key[:3]].append(key)
    output: dict[str, object] = {
        "contact_sets": len(set_keys),
        "particles": len(keys),
        "invalid_particles": sum(not arm.outcomes[key]["valid"] for key in keys),
    }
    contact_chamfer = [arm.candidates[key]["contact_chamfer_m"] for key in keys]
    assigned_error = [
        arm.candidates[key]["assigned_contact_error_m"] for key in keys
    ]
    output["fk_contact_attainment"] = {
        "contact_chamfer_m": describe(contact_chamfer),
        "assigned_contact_error_m": describe(assigned_error),
    }
    for definition in ("final", "strict"):
        particle_successes = sum(arm.outcomes[key][definition] for key in keys)
        success_at_k: dict[str, dict] = {}
        for cutoff in RANK_CUTOFFS:
            successes = 0
            for set_key in set_keys:
                ranked = sorted(
                    keys_by_set[set_key], key=lambda key: arm.outcomes[key]["rank"]
                )
                successes += any(
                    arm.outcomes[key][definition] for key in ranked[:cutoff]
                )
            success_at_k[str(cutoff)] = {
                "sets": successes,
                "rate": successes / len(set_keys),
            }
        success_keys = [key for key in keys if arm.outcomes[key][definition]]
        failed_keys = [key for key in keys if not arm.outcomes[key][definition]]
        output[definition] = {
            "particle_successes": particle_successes,
            "particle_success_rate": particle_successes / len(keys),
            "success_at_k": success_at_k,
            "no_success_particle_sets": len(set_keys)
            - success_at_k[str(RANK_CUTOFFS[-1])]["sets"],
            "contact_chamfer_m_by_success": {
                "success": describe(
                    [arm.candidates[key]["contact_chamfer_m"] for key in success_keys]
                ),
                "failure": describe(
                    [arm.candidates[key]["contact_chamfer_m"] for key in failed_keys]
                ),
            },
        }
    return output


def compare(arm: Arm, reference: Arm, keys: list[tuple], *, seed: int, draws: int) -> dict:
    set_keys = sorted({key[:3] for key in keys})
    keys_by_set: dict[tuple, list[tuple]] = defaultdict(list)
    for key in keys:
        keys_by_set[key[:3]].append(key)
    output: dict[str, object] = {}
    for definition_index, definition in enumerate(("final", "strict")):
        particle_pairs = [
            (
                key,
                bool(arm.outcomes[key][definition]),
                bool(reference.outcomes[key][definition]),
            )
            for key in keys
        ]
        left_only = sum(left and not right for _, left, right in particle_pairs)
        right_only = sum(right and not left for _, left, right in particle_pairs)
        result: dict[str, object] = {
            "particle_delta_pp": 100.0
            * sum(int(left) - int(right) for _, left, right in particle_pairs)
            / len(particle_pairs),
            "particle_arm_only": left_only,
            "particle_reference_only": right_only,
            "particle_mcnemar_exact_p": exact_mcnemar_p(left_only, right_only),
            "particle_delta_object_cluster_bootstrap95_pp": object_cluster_bootstrap_ci(
                particle_pairs, seed=seed + definition_index, draws=draws
            ),
            "success_at_k": {},
        }
        for cutoff in RANK_CUTOFFS:
            set_pairs: list[tuple[tuple, bool, bool]] = []
            for set_key in set_keys:
                arm_ranked = sorted(
                    keys_by_set[set_key], key=lambda key: arm.outcomes[key]["rank"]
                )
                reference_ranked = sorted(
                    keys_by_set[set_key],
                    key=lambda key: reference.outcomes[key]["rank"],
                )
                set_pairs.append(
                    (
                        set_key,
                        any(
                            arm.outcomes[key][definition]
                            for key in arm_ranked[:cutoff]
                        ),
                        any(
                            reference.outcomes[key][definition]
                            for key in reference_ranked[:cutoff]
                        ),
                    )
                )
            arm_only = sum(left and not right for _, left, right in set_pairs)
            reference_only = sum(right and not left for _, left, right in set_pairs)
            result["success_at_k"][str(cutoff)] = {
                "delta_pp": 100.0
                * sum(int(left) - int(right) for _, left, right in set_pairs)
                / len(set_pairs),
                "arm_only_sets": arm_only,
                "reference_only_sets": reference_only,
                "mcnemar_exact_p": exact_mcnemar_p(arm_only, reference_only),
                "delta_object_cluster_bootstrap95_pp": object_cluster_bootstrap_ci(
                    set_pairs,
                    seed=seed + 100 * cutoff + definition_index,
                    draws=draws,
                ),
            }
        output[definition] = result
    return output


def parse_arm(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("--arm must use NAME=PATH")
    return name, Path(path)


def render_markdown(report: dict) -> str:
    names = report["arm_order"]
    lines = [
        "# FK contact-energy guidance diagnostic",
        "",
        f"- Pairing audit: `{'pass' if report['audit']['all_pairing_checks_pass'] else 'fail'}`",
        f"- Contact sets per arm: {report['audit']['contact_sets_per_arm']}",
        f"- Particles per arm: {report['audit']['particles_per_arm']}",
        f"- Reference arm: `{report['reference_arm']}`",
        "- Dataset-GT upper bound: `unavailable` (OOD-10 has no accessible annotated contact set in this platform snapshot)",
        "",
        "## Final success and FK contact attainment",
        "",
        "| Arm | Contact weight | Target | Particle | @1 | @4 | @8 | @16 | @32 | Median Chamfer (m) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        arm = report["arms"][name]
        final = arm["overall"]["final"]
        ks = final["success_at_k"]
        chamfer = arm["overall"]["fk_contact_attainment"]["contact_chamfer_m"]
        lines.append(
            f"| {name} | {arm['contact_weight']:.4g} | {arm['target_mode']} | "
            f"{final['particle_success_rate']:.2%} | {ks['1']['rate']:.2%} | "
            f"{ks['4']['rate']:.2%} | {ks['8']['rate']:.2%} | "
            f"{ks['16']['rate']:.2%} | {ks['32']['rate']:.2%} | "
            f"{chamfer['median']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired final particle comparison against reference",
            "",
            "| Arm | Delta (pp) | McNemar p | Object-cluster bootstrap 95% CI (pp) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in names:
        if name == report["reference_arm"]:
            continue
        comparison = report["comparisons_to_reference"][name]["overall"]["final"]
        lines.append(
            f"| {name} | {comparison['particle_delta_pp']:+.3f} | "
            f"{comparison['particle_mcnemar_exact_p']:.6g} | "
            f"{comparison['particle_delta_object_cluster_bootstrap95_pp']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--reference-arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    if len({name for name, _ in args.arm}) != len(args.arm):
        raise ValueError("duplicate --arm name")
    arms = {
        name: load_arm(name, path, load_results=not args.audit_only)
        for name, path in args.arm
    }
    if args.reference_arm not in arms:
        raise ValueError("--reference-arm must name one of the supplied arms")
    reference = arms[args.reference_arm]
    reference_keys = set(reference.candidates)
    reference_sets = set(reference.metadata)
    audit = {
        "contact_sets_per_arm": len(reference_sets),
        "particles_per_arm": len(reference_keys),
        "candidate_key_mismatches": {},
        "contact_set_key_mismatches": {},
        "sample_seed_mismatches": {},
        "source_diffusion_contact_mismatches": {},
        "initialization_state_mismatches": {},
        "candidate_particle_index_mismatches": {},
        "diffusion_target_contact_mismatches": {},
    }
    for name, arm in arms.items():
        audit["candidate_key_mismatches"][name] = len(
            reference_keys.symmetric_difference(arm.candidates)
        )
        audit["contact_set_key_mismatches"][name] = len(
            reference_sets.symmetric_difference(arm.metadata)
        )
        common_sets = sorted(reference_sets & set(arm.metadata))
        audit["sample_seed_mismatches"][name] = sum(
            reference.metadata[key]["sample_seed"]
            != arm.metadata[key]["sample_seed"]
            for key in common_sets
        )
        audit["source_diffusion_contact_mismatches"][name] = sum(
            reference.metadata[key]["source_diffusion_contacts_sha256"]
            != arm.metadata[key]["source_diffusion_contacts_sha256"]
            for key in common_sets
        )
        audit["initialization_state_mismatches"][name] = sum(
            reference.metadata[key]["initialization_state_sha256"]
            != arm.metadata[key]["initialization_state_sha256"]
            for key in common_sets
        )
        audit["candidate_particle_index_mismatches"][name] = sum(
            reference.metadata[key]["particle_indices"]
            != arm.metadata[key]["particle_indices"]
            for key in common_sets
        )
        if arm.target_mode == "diffusion":
            audit["diffusion_target_contact_mismatches"][name] = sum(
                reference.metadata[key]["source_diffusion_contacts_sha256"]
                != arm.metadata[key]["target_contacts_sha256"]
                for key in common_sets
            )
    audit["all_pairing_checks_pass"] = not any(
        count
        for key, values in audit.items()
        if key not in {"contact_sets_per_arm", "particles_per_arm"}
        and isinstance(values, dict)
        for count in values.values()
    )
    if not audit["all_pairing_checks_pass"]:
        raise ValueError(f"pairing audit failed: {audit}")

    if args.audit_only:
        payload = {
            "schema": "contactdiff-contact-energy-candidate-audit-v1",
            "reference_arm": args.reference_arm,
            "audit": audit,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(audit, indent=2))
        print(f"Wrote {args.output.resolve()}")
        return

    ordered_names = [name for name, _ in args.arm]
    report = {
        "schema": "contactdiff-contact-energy-diagnostic-v1",
        "arm_order": ordered_names,
        "reference_arm": args.reference_arm,
        "rank_cutoffs": list(RANK_CUTOFFS),
        "dataset_gt_upper_bound": {
            "status": "unavailable",
            "reason": (
                "The OOD-10 manifest contains geometry only, the checkpoint's "
                "seen48 train/validation lists exclude OOD-10, and the annotated "
                "training root is absent from this compute-platform snapshot."
            ),
        },
        "audit": audit,
        "arms": {},
        "comparisons_to_reference": {},
    }
    for name in ordered_names:
        arm = arms[name]
        keys = sorted(arm.outcomes)
        hands = sorted({key[0] for key in keys})
        report["arms"][name] = {
            "root": str(arm.root),
            "contact_weight": arm.contact_weight,
            "target_mode": arm.target_mode,
            "overall": aggregate(arm, keys),
            "per_hand": {
                hand: aggregate(arm, [key for key in keys if key[0] == hand])
                for hand in hands
            },
        }
        if name != args.reference_arm:
            report["comparisons_to_reference"][name] = {
                "overall": compare(
                    arm,
                    reference,
                    keys,
                    seed=args.bootstrap_seed + 1000 * (ordered_names.index(name) + 1),
                    draws=args.bootstrap_draws,
                ),
                "per_hand": {
                    hand: compare(
                        arm,
                        reference,
                        [key for key in keys if key[0] == hand],
                        seed=args.bootstrap_seed
                        + 1000 * (ordered_names.index(name) + 1)
                        + 100 * (hand_index + 1),
                        draws=args.bootstrap_draws,
                    )
                    for hand_index, hand in enumerate(hands)
                },
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["audit"], indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
