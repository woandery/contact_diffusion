#!/usr/bin/env python3
"""Calibrate fingertip contact-pad offsets from successful MGG records."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multigripper_fk_isaac.yaml")
    parser.add_argument("--max-samples-per-gripper", type=int, default=256)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def iter_success_samples(shard_dir: Path, limit: int):
    yielded = 0
    for shard_path in sorted(shard_dir.glob("shard_*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            success = np.asarray(shard["success"], dtype=bool)
            valid_rows = np.flatnonzero(success)
            for row in valid_rows:
                mask = np.asarray(shard["finger_valid_mask"][row], dtype=bool)
                yield {
                    "T_grasp": np.asarray(shard["T_grasp"][row], dtype=np.float64),
                    "q": np.asarray(shard["final_dofs"][row], dtype=np.float64),
                    "contacts": np.asarray(shard["contacts"][row], dtype=np.float64)[mask],
                }
                yielded += 1
                if yielded >= limit:
                    return


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def end_effector_alignment(axis: int) -> np.ndarray:
    rotations = {
        1: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        2: [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        3: [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        -1: [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        -2: [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        -3: np.eye(3),
    }
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotations[int(axis)], dtype=np.float64)
    return matrix


def calibrate_one(name: str, spec: dict, config: dict, limit: int) -> dict:
    gripper_root = resolve(config["paths"]["gripper_root"])
    contact_root = resolve(config["paths"]["contact_format_root"])
    urdf_path = gripper_root / spec["urdf"]
    chain = pk.build_chain_from_urdf(urdf_path.read_bytes()).to(dtype=torch.float64)
    tip_links = list(spec["tip_links"])
    frame_names = set(chain.get_frame_names())
    missing = [link for link in tip_links if link not in frame_names]
    if missing:
        raise KeyError(f"{name}: tip links absent from URDF: {missing}")

    records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    used = 0
    for sample in iter_success_samples(contact_root / name / "shards", limit):
        q = sample["q"]
        contacts = sample["contacts"]
        if len(q) != len(chain.get_joint_parameter_names()) or len(contacts) != len(tip_links):
            continue
        fk = chain.forward_kinematics(torch.from_numpy(q)[None])
        tip_matrices = np.stack([fk[link].get_matrix()[0].detach().cpu().numpy() for link in tip_links])
        base_to_object = sample["T_grasp"]
        records.append((base_to_object, tip_matrices, contacts))
        used += 1

    if not records:
        raise RuntimeError(f"{name}: no usable successful records")

    # Contact sets are unordered. Pick one global slot permutation whose
    # inverse-FK offsets are most consistent across all calibration records.
    # This avoids independently swapping similar fingers in each sample.
    ef_alignment = end_effector_alignment(int(spec["ef_axis"]))
    alignments = {
        "identity": np.eye(4, dtype=np.float64),
        "ef_axis": ef_alignment,
        "ef_axis_inverse": np.linalg.inv(ef_alignment),
    }
    # Several EF transforms are self-inverse; do not evaluate duplicates.
    unique_alignments = {}
    for alignment_name, alignment in alignments.items():
        if not any(np.allclose(alignment, previous) for previous in unique_alignments.values()):
            unique_alignments[alignment_name] = alignment

    best_score = float("inf")
    best_permutation = None
    best_values = None
    best_alignment_name = None
    best_alignment = None
    for alignment_name, alignment in unique_alignments.items():
        for permutation in itertools.permutations(range(len(tip_links))):
            values_by_tip: list[list[np.ndarray]] = [[] for _ in tip_links]
            for base_to_object, tip_matrices, contacts in records:
                for tip_index, contact_index in enumerate(permutation):
                    tip_to_object = base_to_object @ alignment @ tip_matrices[tip_index]
                    local = np.linalg.solve(
                        tip_to_object, np.append(contacts[contact_index], 1.0)
                    )[:3]
                    if np.all(np.isfinite(local)) and np.linalg.norm(local) < 0.5:
                        values_by_tip[tip_index].append(local)
            if any(not values for values in values_by_tip):
                continue
            score = 0.0
            for values in values_by_tip:
                array = np.asarray(values)
                median = np.median(array, axis=0)
                score += float(np.median(np.linalg.norm(array - median[None, :], axis=1)))
            if score < best_score:
                best_score = score
                best_permutation = permutation
                best_values = values_by_tip
                best_alignment_name = alignment_name
                best_alignment = alignment
    if best_permutation is None or best_values is None or best_alignment is None:
        raise RuntimeError(f"{name}: unable to determine a contact-slot permutation")

    samples_by_tip = {
        link: best_values[index] for index, link in enumerate(tip_links)
    }
    assignment_errors = []
    for base_to_object, tip_matrices, contacts in records:
        origins = np.stack(
            [(base_to_object @ best_alignment @ matrix)[:3, 3] for matrix in tip_matrices]
        )
        for tip_index, contact_index in enumerate(best_permutation):
            assignment_errors.append(
                float(np.linalg.norm(origins[tip_index] - contacts[contact_index]))
            )

    result = {
        "urdf": str(urdf_path),
        "joint_parameter_names": list(chain.get_joint_parameter_names()),
        "num_records": used,
        "base_alignment": best_alignment_name,
        "base_alignment_matrix": best_alignment.tolist(),
        "tip_to_contact_slot": list(best_permutation),
        "permutation_dispersion_score": best_score,
        "assignment_origin_distance_m": {
            "mean": float(np.mean(assignment_errors)),
            "p95": float(np.percentile(assignment_errors, 95)),
        },
        "tips": {},
    }
    for link, values in samples_by_tip.items():
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            raise RuntimeError(f"{name}/{link}: no valid calibration points")
        median = np.median(array, axis=0)
        residual = np.linalg.norm(array - median[None, :], axis=1)
        cutoff = max(float(np.percentile(residual, 90)), 1e-6)
        inliers = array[residual <= cutoff]
        robust = np.median(inliers, axis=0)
        robust_residual = np.linalg.norm(inliers - robust[None, :], axis=1)
        result["tips"][link] = {
            "offset_xyz": robust.tolist(),
            "num_observations": int(len(array)),
            "num_inliers": int(len(inliers)),
            "residual_mean_m": float(np.mean(robust_residual)),
            "residual_p95_m": float(np.percentile(robust_residual, 95)),
        }
    return result


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_path = resolve(args.output or config["paths"]["tip_offset_calibration"])
    payload = {
        "method": "median inverse-FK contact offset from successful MultiGripperGrasp records",
        "max_samples_per_gripper": int(args.max_samples_per_gripper),
        "grippers": {},
    }
    for name, spec in config["grippers"].items():
        print(f"Calibrating {name} ...", flush=True)
        payload["grippers"][name] = calibrate_one(
            name, spec, config, int(args.max_samples_per_gripper)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
