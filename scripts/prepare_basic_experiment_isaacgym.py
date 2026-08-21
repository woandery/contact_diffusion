#!/usr/bin/env python3
"""Prepare FK candidates for the frozen D(R,O)-aligned Gym protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.basic_experiment_protocol import (  # noqa: E402
    ALL_PARTICLE_PROTOCOL_ID,
    FILTERED50K_EAWQ_V3_PROTOCOL_ID,
    MODEL_145K_ALL_PARTICLE_PROTOCOL_ID,
    O10I20_V2_PROTOCOL_ID,
    PROTOCOL_ID,
    SUPPORTED_PROTOCOL_IDS,
    VIRTUAL_ROOT_JOINTS,
    apply_root_post_transform,
    derive_fk_to_sim_root_post_transform,
    method_specific_closure_targets,
    ordered_joint_array,
)


HAND_NAMES = {
    "franka_panda": ("franka_panda", "franka_panda"),
    "Barrett": ("barrett", "gendex_barrett"),
    "shadow_hand": ("shadowhand", "shadowhand"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gripper", choices=tuple(HAND_NAMES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--allow-runtime-budget",
        action="store_true",
        help=(
            "Allow an explicit small-scale execution protocol to override the "
            "candidate YAML's frozen contact-set/particle budget. The model, "
            "checkpoint, FK configuration and optimization-step checks remain."
        ),
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("top1", "all"),
        default="top1",
        help="Retain only rank 0 (legacy) or every frozen FK particle.",
    )
    parser.add_argument(
        "--execution-protocol",
        type=Path,
        help=(
            "Authoritative execution protocol whose closure profile replaces "
            "legacy closure metadata in the candidate-generation config."
        ),
    )
    parser.add_argument(
        "--closure-outer-fraction",
        type=float,
        help=(
            "Override the configured fraction opened from the FK contact pose; "
            "must be supplied together with --closure-inner-fraction."
        ),
    )
    parser.add_argument(
        "--closure-inner-fraction",
        type=float,
        help=(
            "Override the configured fraction closed from the FK contact pose; "
            "must be supplied together with --closure-outer-fraction."
        ),
    )
    args = parser.parse_args()

    closure_override = (
        args.closure_outer_fraction is not None
        or args.closure_inner_fraction is not None
    )
    if closure_override and (
        args.closure_outer_fraction is None
        or args.closure_inner_fraction is None
    ):
        parser.error(
            "--closure-outer-fraction and --closure-inner-fraction "
            "must be supplied together"
        )
    if closure_override and not (
        0.0 <= args.closure_outer_fraction <= 1.0
        and 0.0 <= args.closure_inner_fraction <= 1.0
    ):
        parser.error("closure fractions must be in [0, 1]")
    if closure_override and args.execution_protocol is not None:
        parser.error(
            "manual closure overrides cannot be combined with "
            "--execution-protocol"
        )

    candidate_path = args.candidates.resolve()
    config_path = args.config.resolve()
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    execution_protocol_path = (
        args.execution_protocol.resolve()
        if args.execution_protocol is not None
        else None
    )
    execution_protocol = (
        yaml.safe_load(execution_protocol_path.read_text(encoding="utf-8"))
        if execution_protocol_path is not None
        else None
    )
    spec = config["grippers"][args.gripper]
    config_digest = sha256(config_path)
    if payload.get("config_sha256") != config_digest:
        raise ValueError(
            "Candidate/config provenance mismatch: "
            f"candidate={payload.get('config_sha256')}, file={config_digest}"
        )
    checkpoint_path = resolve(payload["checkpoint"])
    if not checkpoint_path.is_file():
        checkpoint_path = (
            resolve(config["diffusion"]["run_dir"])
            / "checkpoints"
            / Path(payload["checkpoint"]).name
        ).resolve()
    checkpoint_digest = sha256(checkpoint_path)
    if payload.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError(
            "Candidate/checkpoint provenance mismatch: "
            f"candidate={payload.get('checkpoint_sha256')}, "
            f"file={checkpoint_digest}"
        )
    baseline = config["baseline"]
    source_protocol_id = str(baseline.get("protocol_id", ""))
    if source_protocol_id not in SUPPORTED_PROTOCOL_IDS:
        raise ValueError(
            f"Unsupported candidate protocol ID: {source_protocol_id!r}"
        )
    configured_protocol_id = source_protocol_id
    frozen_counts = {
        "particles": "particles_per_contact_set",
        "optimization_steps": "optimization_steps",
    }
    for candidate_key, baseline_key in frozen_counts.items():
        if (
            int(payload[candidate_key]) != int(baseline[baseline_key])
            and not args.allow_runtime_budget
        ):
            raise ValueError(
                f"{candidate_key}={payload[candidate_key]} does not match "
                f"baseline.{baseline_key}={baseline[baseline_key]}"
            )
    retained_count = (
        int(payload["particles"])
        if args.candidate_mode == "all"
        else int(baseline["retained_per_contact_set"])
    )
    if execution_protocol is not None:
        allowed_execution_ids = (
            {
                ALL_PARTICLE_PROTOCOL_ID,
                FILTERED50K_EAWQ_V3_PROTOCOL_ID,
                MODEL_145K_ALL_PARTICLE_PROTOCOL_ID,
                PROTOCOL_ID,
            }
            if args.candidate_mode == "all"
            else {
                O10I20_V2_PROTOCOL_ID,
                FILTERED50K_EAWQ_V3_PROTOCOL_ID,
                PROTOCOL_ID,
            }
        )
        actual_execution_id = execution_protocol.get("protocol_id")
        runtime_base_id = execution_protocol.get("base_protocol_id")
        runtime_protocol_valid = (
            args.allow_runtime_budget
            and runtime_base_id
            in {FILTERED50K_EAWQ_V3_PROTOCOL_ID, PROTOCOL_ID}
        )
        if actual_execution_id not in allowed_execution_ids and not runtime_protocol_valid:
            raise ValueError(
                "Execution protocol ID mismatch: "
                f"{actual_execution_id!r} not in "
                f"{sorted(allowed_execution_ids)!r}"
            )
        if args.candidate_mode == "all":
            expected_base_id = (
                runtime_base_id
                if runtime_protocol_valid
                else (
                    actual_execution_id
                    if actual_execution_id
                    in {FILTERED50K_EAWQ_V3_PROTOCOL_ID, PROTOCOL_ID}
                    else O10I20_V2_PROTOCOL_ID
                )
            )
            if execution_protocol.get("base_protocol_id") != expected_base_id:
                raise ValueError(
                    "all-particle protocol has an unexpected base protocol"
                )
        generation = execution_protocol.get("generation", {})
        expected_generation = (
            {
                "contact_sets_per_object": int(
                    payload["selection"]["samples_per_object"]
                ),
                "particles_per_contact_set": int(payload["particles"]),
                "optimization_steps": int(payload["optimization_steps"]),
                "retained_per_contact_set": retained_count,
            }
            if args.allow_runtime_budget
            else {
                "contact_sets_per_object": int(
                    baseline["contact_sets_per_object"]
                ),
                "particles_per_contact_set": int(
                    baseline["particles_per_contact_set"]
                ),
                "optimization_steps": int(baseline["optimization_steps"]),
                "retained_per_contact_set": retained_count,
            }
        )
        if generation != expected_generation:
            raise ValueError(
                "Execution protocol generation budget does not match the "
                f"candidate config: {generation} != {expected_generation}"
            )
        checkpoint_spec = execution_protocol.get("model_checkpoint")
        if checkpoint_spec is not None:
            expected_step = int(checkpoint_spec["step"])
            expected_hash = str(checkpoint_spec["sha256"])
            if int(payload["checkpoint_step"]) != expected_step:
                raise ValueError(
                    "Candidate checkpoint step does not match execution "
                    f"protocol: {payload['checkpoint_step']} != {expected_step}"
                )
            if str(payload["checkpoint_sha256"]) != expected_hash:
                raise ValueError(
                    "Candidate checkpoint SHA256 does not match execution "
                    f"protocol: {payload['checkpoint_sha256']} != {expected_hash}"
                )
        configured_protocol_id = str(actual_execution_id)
    if int(payload["selection"]["top_k"]) != retained_count:
        raise ValueError("Candidate Top-K does not match frozen retained count")
    records = sorted(
        (row for row in payload["records"] if row["gripper"] == args.gripper),
        key=lambda row: (str(row["object_id"]), int(row["sample_index"])),
    )
    if not records:
        raise ValueError(f"No {args.gripper} records in {candidate_path}")

    joint_names = list(records[0]["fk"]["joint_names"])
    expected_count = (
        int(payload["selection"]["samples_per_object"])
        if args.allow_runtime_budget
        else int(config["baseline"]["contact_sets_per_object"])
    )
    close_direction = ordered_joint_array(
        spec["close_dir"], joint_names, label=f"{args.gripper}.close_dir"
    )
    closure = (
        execution_protocol["hands"][args.gripper]["closure_adapter"]
        if execution_protocol is not None
        else spec["closure_adapter"]
    )
    outer_fraction = (
        float(args.closure_outer_fraction)
        if closure_override
        else float(closure["outer_fraction"])
    )
    inner_fraction = (
        float(args.closure_inner_fraction)
        if closure_override
        else float(closure["inner_fraction"])
    )
    fk_hand_root = resolve(
        spec.get("urdf_root", config["paths"]["gripper_root"])
    )
    fk_hand_urdf = (fk_hand_root / spec["urdf"]).resolve()
    simulation_hand_urdf = resolve(spec["simulation_urdf"])
    root_post_transform = derive_fk_to_sim_root_post_transform(
        fk_hand_urdf, simulation_hand_urdf
    )
    object_groups = []
    for object_id in sorted({str(row["object_id"]) for row in records}):
        selected = [row for row in records if row["object_id"] == object_id]
        if len(selected) != expected_count and not args.allow_incomplete:
            raise ValueError(
                f"{object_id}: got {len(selected)} records, expected {expected_count}"
            )
        indices = [int(row["sample_index"]) for row in selected]
        if indices != sorted(set(indices)):
            raise ValueError(f"{object_id}: duplicate or unsorted sample indices")
        samples = []
        for row in selected:
            fk = row["fk"]
            if list(fk["joint_names"]) != joint_names:
                raise ValueError(f"{row['record_id']}: joint order changed")
            candidates = sorted(
                fk["candidates"], key=lambda item: int(item["rank"])
            )
            ranks = [int(candidate["rank"]) for candidate in candidates]
            if (
                len(candidates) != retained_count
                or ranks != list(range(retained_count))
            ):
                raise ValueError(
                    f"{row['record_id']}: expected ranks "
                    f"0..{retained_count - 1}"
                )
            for candidate in candidates:
                rank = int(candidate["rank"])
                pose = np.asarray(candidate["root_pose"], dtype=np.float64)
                joints = np.asarray(
                    candidate["joint_positions"], dtype=np.float64
                )
                if pose.shape != (4, 4) or joints.shape != (len(joint_names),):
                    raise ValueError(f"{row['record_id']}: invalid pose/joint shape")
                simulation_pose = apply_root_post_transform(
                    pose, root_post_transform
                )
                q_contact = np.concatenate(
                    (
                        simulation_pose[:3, 3],
                        Rotation.from_matrix(
                            simulation_pose[:3, :3]
                        ).as_euler("XYZ"),
                        joints,
                    )
                )
                outer, inner = method_specific_closure_targets(
                    q_contact,
                    fk["joint_lower"],
                    fk["joint_upper"],
                    close_direction,
                    outer_fraction=outer_fraction,
                    inner_fraction=inner_fraction,
                )
                samples.append(
                    {
                    "source_index": int(row["sample_index"]),
                    "attempt_index": (
                        int(row["sample_index"]) * retained_count + rank
                    ),
                    "candidate_rank": rank,
                    "particle_index": int(candidate["particle"]),
                    "source_file": str(candidate_path),
                    "sample_seed": int(row["sample_seed"]),
                    "contact_target_mode": str(
                        row.get("contact_target_mode", "diffusion")
                    ),
                    "source_diffusion_contacts_object": row.get(
                        "source_diffusion_contacts",
                        fk["target_contacts"],
                    ),
                    "source_diffusion_contacts_sha256": row.get(
                        "source_diffusion_contacts_sha256"
                    ),
                    "target_contacts_sha256": row.get(
                        "target_contacts_sha256"
                    ),
                    "fk_initialization_state_sha256": fk.get(
                        "initialization_state_sha256"
                    ),
                    "best_energy": float(candidate["optimization_score"]),
                    "q_contact_euler": q_contact.tolist(),
                    "outer_q_euler": outer.tolist(),
                    "inner_q_euler": inner.tolist(),
                    "selection_feasible": bool(
                        candidate.get("selection_feasible", False)
                    ),
                    "selection_fallback": bool(
                        candidate.get("selection_fallback", False)
                    ),
                    "target_contacts_object": fk[
                        "target_contacts"
                    ],
                    # Backward-compatible field consumed by existing viewers.
                    "diffusion_target_contacts_object": fk["target_contacts"],
                    "fk_matched_contact_points_object": candidate.get(
                        "matched_contact_points"
                    ),
                    "fk_metrics": {
                        key: candidate.get(key)
                        for key in (
                            "contact_chamfer_m",
                            "assigned_contact_error_m",
                            "mean_penetration_m",
                            "cvar_penetration_m",
                            "hinge_penetration_m",
                            "max_penetration_m",
                            "raw_max_penetration_m",
                            "confidence_weighted_max_penetration_m",
                            "mean_self_collision_m",
                            "cvar_self_collision_m",
                            "max_self_collision_m",
                            "self_collision_pair_fraction",
                            "palm_approach_cosine",
                            "palm_unsigned_distance_m",
                            "graspqp_score",
                        )
                    },
                    }
                )
        object_groups.append(
            {
                "object_name": object_id,
                "object_mesh": selected[0].get("object_mesh"),
                "available_for_object": len(samples),
                "samples": samples,
            }
        )

    protocol_hand, validator_hand = HAND_NAMES[args.gripper]
    hand_urdf = simulation_hand_urdf
    fraction_label = f"o{100.0 * outer_fraction:g}-i{100.0 * inner_fraction:g}"
    fraction_label = fraction_label.replace(".", "p")
    protocol_id = (
        f"{configured_protocol_id}-closure-{fraction_label}-ab-v1"
        if closure_override
        else configured_protocol_id
    )
    output = {
        "schema": "contactdiff-basic-experiment-prepared-v1",
        "protocol_id": protocol_id,
        **(
            {
                "base_protocol_id": (
                    (
                        execution_protocol.get(
                            "base_protocol_id", configured_protocol_id
                        )
                        if execution_protocol is not None
                        else (
                            configured_protocol_id
                            if configured_protocol_id
                            in {
                                FILTERED50K_EAWQ_V3_PROTOCOL_ID,
                                PROTOCOL_ID,
                            }
                            else O10I20_V2_PROTOCOL_ID
                        )
                    )
                    if args.candidate_mode == "all"
                    else configured_protocol_id
                )
            }
            if closure_override or args.candidate_mode == "all"
            else {}
        ),
        "source_candidate_protocol_id": source_protocol_id,
        "execution_protocol_config": (
            str(execution_protocol_path)
            if execution_protocol_path is not None
            else None
        ),
        "execution_protocol_config_sha256": (
            sha256(execution_protocol_path)
            if execution_protocol_path is not None
            else None
        ),
        "method": "contactdiffusion",
        "hand": validator_hand,
        "comparison_hand": protocol_hand,
        "source_method": (
            f"MultiDex-checkpoint-step{int(payload['checkpoint_step'])}"
            "+simplified-six-term-FK"
        ),
        "generation_from_scratch": "all_particle_replay" not in payload,
        "frozen_projected_contacts_reused": "all_particle_replay" in payload,
        "contact_sets_per_object": expected_count,
        "particles_per_set": int(payload["particles"]),
        "retained_per_set": retained_count,
        "checkpoint": payload["checkpoint"],
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_step": int(payload["checkpoint_step"]),
        "optimization_steps": int(payload["optimization_steps"]),
        "diffusion_steps": int(payload["diffusion_steps"]),
        "joint_names": list(VIRTUAL_ROOT_JOINTS) + joint_names,
        "closure_adapter": {
            "name": (
                f"{args.gripper.lower()}_fixed_direction_{fraction_label}_ab"
                if closure_override
                else str(closure["name"])
            ),
            "outer_fraction": outer_fraction,
            "inner_fraction": inner_fraction,
            "direction_source": "frozen close_dir by joint name",
            "root_policy": "hold simulation-aligned FK root pose",
            "is_baseline_override": closure_override,
        },
        "root_pose_adapter": {
            "name": "fk_urdf_to_simulation_urdf_fixed_base_v1",
            "fk_hand_urdf": str(fk_hand_urdf),
            "fk_hand_urdf_sha256": sha256(fk_hand_urdf),
            "simulation_hand_urdf": str(simulation_hand_urdf),
            "post_transform": root_post_transform.tolist(),
            "composition": "T_sim_root = T_fk_root @ T_post",
        },
        "simulation_hand_urdf": str(hand_urdf),
        "simulation_hand_urdf_sha256": sha256(hand_urdf),
        "candidate_file": str(candidate_path),
        "candidate_file_sha256": sha256(candidate_path),
        "config": str(config_path),
        "config_sha256": config_digest,
        "seed_protocol": payload["selection"],
        "contact_target_ab": payload.get("contact_target_ab"),
        "objects": object_groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "hand": args.gripper,
                "objects": len(object_groups),
                "samples": sum(len(group["samples"]) for group in object_groups),
                "fallbacks": sum(
                    sample["selection_fallback"]
                    for group in object_groups
                    for sample in group["samples"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
