#!/usr/bin/env python3
"""Convert matched 64x32 rank-0 candidates to Isaac Sim manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multigripper_fk_gendex_isaacsim_local_800.yaml"
RUN_ROOT = ROOT / "outputs" / "gendex_ood10_matched64x32_isaacsim"
HAND_SPECS = {
    "Barrett": ("gendex_barrett", "barrett.json"),
    "shadow_hand": ("gendex_shadowhand", "shadow.json"),
}
POSE_NAMES = [
    "virtual_joint_x",
    "virtual_joint_y",
    "virtual_joint_z",
    "virtual_joint_roll",
    "virtual_joint_pitch",
    "virtual_joint_yaw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--barrett",
        type=Path,
        default=RUN_ROOT / "candidates" / "barrett.json",
    )
    parser.add_argument(
        "--shadow",
        type=Path,
        default=RUN_ROOT / "candidates" / "shadow.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "prepared",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG,
    )
    parser.add_argument(
        "--barrett-validator-hand",
        choices=("gendex_barrett", "contactdiff_barrett"),
        default="gendex_barrett",
    )
    parser.add_argument(
        "--shadow-validator-hand",
        choices=(
            "gendex_shadowhand",
            "contactdiff_shadowhand",
            "shadowhand",
        ),
        default="gendex_shadowhand",
    )
    parser.add_argument(
        "--hand",
        choices=("all", "barrett", "shadow"),
        default="all",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit reduced object/sample counts for pipeline smoke tests.",
    )
    parser.add_argument(
        "--expected-optimization-steps",
        type=int,
        default=800,
        help="Require this FK step count in the candidate payload.",
    )
    return parser.parse_args()


def ordered(values: dict[str, float], names: list[str]) -> np.ndarray:
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"Missing joint values: {missing}")
    return np.asarray([values[name] for name in names], dtype=np.float64)


def load(
    path: Path,
    hand: str,
    *,
    allow_incomplete: bool,
    expected_optimization_steps: int,
) -> tuple[dict, list[dict]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if int(payload.get("particles", -1)) != 32:
        raise ValueError(f"{path}: expected particles=32")
    if int(payload.get("optimization_steps", -1)) != int(
        expected_optimization_steps
    ):
        raise ValueError(
            f"{path}: expected optimization_steps="
            f"{expected_optimization_steps}"
        )
    if (
        not allow_incomplete
        and int(payload["selection"].get("samples_per_object", -1)) != 64
    ):
        raise ValueError(f"{path}: expected samples_per_object=64")
    records = payload["records"]
    if not allow_incomplete and len(records) != 640:
        raise ValueError(f"{path}: expected 640 records, got {len(records)}")
    if any(record["gripper"] != hand for record in records):
        raise ValueError(f"{path}: mixed or unexpected hands")
    return payload, records


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for hand, (validator_hand, output_name) in HAND_SPECS.items():
        if args.hand == "barrett" and hand != "Barrett":
            continue
        if args.hand == "shadow" and hand != "shadow_hand":
            continue
        if hand == "Barrett":
            validator_hand = args.barrett_validator_hand
        else:
            validator_hand = args.shadow_validator_hand
        source = args.barrett if hand == "Barrett" else args.shadow
        payload, records = load(
            source,
            hand,
            allow_incomplete=args.allow_incomplete,
            expected_optimization_steps=args.expected_optimization_steps,
        )
        spec = config["grippers"][hand]
        first_fk = records[0]["fk"]
        joint_names = list(first_fk["joint_names"])
        close_direction = ordered(spec["close_dir"], joint_names)
        lower = np.asarray(first_fk["joint_lower"], dtype=np.float64)
        upper = np.asarray(first_fk["joint_upper"], dtype=np.float64)
        open_limit = np.where(close_direction > 0.0, lower, upper)
        close_limit = np.where(close_direction > 0.0, upper, lower)
        active = (close_direction != 0.0).astype(np.float64)
        outer_fraction, inner_fraction = (
            (0.0, 0.05) if hand == "Barrett" else (0.10, 0.20)
        )

        by_object: dict[str, list[dict]] = {}
        object_meshes: dict[str, str] = {}
        for record in records:
            candidates = record["fk"]["candidates"]
            if len(candidates) != 1 or int(candidates[0]["rank"]) != 0:
                raise ValueError(
                    f"{record['record_id']}: expected exactly rank-0"
                )
            candidate = candidates[0]
            root = np.asarray(candidate["root_pose"], dtype=np.float64)
            joints = np.asarray(
                candidate["joint_positions"], dtype=np.float64
            )
            euler = Rotation.from_matrix(root[:3, :3]).as_euler("XYZ")
            pose = np.concatenate((root[:3, 3], euler))
            outer_joints = np.clip(
                joints + outer_fraction * (open_limit - joints) * active,
                lower,
                upper,
            )
            inner_joints = np.clip(
                joints + inner_fraction * (close_limit - joints) * active,
                lower,
                upper,
            )
            object_id = str(record["object_id"])
            object_meshes[object_id] = str(record["object_mesh"])
            by_object.setdefault(object_id, []).append(
                {
                    "source_index": int(record["sample_index"]),
                    "attempt_index": int(record["sample_index"]),
                    "candidate_rank": 0,
                    "particle_index": int(candidate["particle"]),
                    "source_file": str(source.resolve()),
                    "best_energy": float(candidate["optimization_score"]),
                    "outer_q_euler": np.concatenate(
                        (pose, outer_joints)
                    ).tolist(),
                    "inner_q_euler": np.concatenate(
                        (pose, inner_joints)
                    ).tolist(),
                    "q_contact_euler": np.concatenate(
                        (pose, joints)
                    ).tolist(),
                    "fk_metrics": {
                        key: candidate.get(key)
                        for key in (
                            "penetration_filter_pass",
                            "envelope_filter_pass",
                            "contact_chamfer_m",
                            "assigned_contact_error_m",
                            "mean_penetration_m",
                            "max_penetration_m",
                            "penetrating_surface_fraction",
                            "mean_self_collision_m",
                            "max_self_collision_m",
                            "envelope_side_cosine",
                            "palm_approach_cosine",
                            "palm_sdf_error_m",
                            "dexgraspnet_dfc_energy",
                            "graspqp_score",
                        )
                    },
                }
            )
        groups = []
        for object_id in payload["selection"]["object_ids"]:
            if args.allow_incomplete and object_id not in by_object:
                continue
            samples = sorted(
                by_object[object_id],
                key=lambda item: int(item["source_index"]),
            )
            if not args.allow_incomplete and len(samples) != 64:
                raise ValueError(
                    f"{hand}/{object_id}: expected 64, got {len(samples)}"
                )
            groups.append(
                {
                    "object_name": object_id,
                    "object_mesh": object_meshes[object_id],
                    "available_for_object": len(samples),
                    "samples": samples,
                }
            )
        output = {
            "schema": "contactdiffusion-gendex-ood10-matched64x32-v1",
            "method": "contactdiffusion",
            "hand": validator_hand,
            "comparison_hand": (
                "barrett" if hand == "Barrett" else "shadowhand"
            ),
            "source_method": "ContactDiffusion-new-FK",
            "generation_from_scratch": True,
            "contact_sets_per_object": 64,
            "particles_per_set": 32,
            "retained_per_set": 1,
            "checkpoint": payload["checkpoint"],
            "checkpoint_step": payload["checkpoint_step"],
            "optimization_steps": payload["optimization_steps"],
            "diffusion_steps": payload["diffusion_steps"],
            "joint_names": POSE_NAMES + joint_names,
            "merge_fixed_joints": True,
            # CEDex/D(R,O)'s shadowhand asset is already the extended URDF
            # with six virtual root joints.  The source GenDex/native URDFs
            # need those joints synthesized by the Isaac Sim adapter.
            "add_virtual_root": validator_hand != "shadowhand",
            "exact_gendex_hand_urdf": validator_hand.startswith("gendex_"),
            "exact_dro_hand_urdf": validator_hand == "shadowhand",
            "objects": groups,
        }
        destination = args.output_dir / output_name
        destination.write_text(
            json.dumps(output, indent=2) + "\n",
            encoding="utf-8",
        )
        print(destination)


if __name__ == "__main__":
    main()
