#!/usr/bin/env python3
"""Locally refine ContactDiffusion FK candidates with scene collision energy.

The target-object terms remain those used by the v4 FK optimizer.  The added
environment term combines a signed tabletop support-plane penalty with an
unsigned point-cloud clearance penalty for shelves and other visible scene
geometry.  This is deliberately a point-cloud proxy, not a watertight SDF.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


WORKSPACE = Path(__file__).resolve().parents[1]
CONTACT_ROOT = WORKSPACE.parent / "ContactDiffusion"
if str(CONTACT_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTACT_ROOT))

from utils.multigripper_fk import (  # noqa: E402
    load_gripper_from_calibration,
    nearest_surface_distance,
    ordered_joint_values,
    rotation_6d_to_matrix,
    self_collision_mean_cvar_energy,
)
from utils.point_cloud_geometry import (  # noqa: E402
    estimate_point_cloud_geometry,
    point_cloud_penetration_energy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--object-pc", type=Path, required=True)
    parser.add_argument("--scene-pc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="First sample_index to refine; default zero preserves the full run.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Optional record cap for a pilot run.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--environment-weight", type=float, default=100.0)
    parser.add_argument("--environment-clearance", type=float, default=0.005)
    parser.add_argument(
        "--selection-max-environment-violation",
        type=float,
        default=0.0001,
        help=(
            "Maximum residual scene-clearance violation for a candidate to be "
            "marked feasible. This is a hard gate in addition to the soft energy."
        ),
    )
    parser.add_argument(
        "--environment-feasibility-mode",
        choices=("hard", "soft"),
        default="hard",
        help=(
            "Use the point-cloud environment threshold as a hard analytic gate "
            "or only as an optimization/ranking signal. PhysX remains the hard "
            "execution-time collision gate in soft mode."
        ),
    )
    parser.add_argument(
        "--selection-rank-mode",
        choices=("environment_first", "normalized_constraints"),
        default="environment_first",
        help="Candidate fallback ordering after refinement.",
    )
    parser.add_argument("--environment-cvar-fraction", type=float, default=0.10)
    parser.add_argument(
        "--environment-closure-sweep-samples",
        type=int,
        default=1,
        help=(
            "Number of hand postures sampled from q_outer to q_inner for the "
            "environment term. One preserves the legacy q_contact-only term."
        ),
    )
    parser.add_argument("--closure-outer-fraction", type=float, default=0.10)
    parser.add_argument("--closure-inner-fraction", type=float, default=0.20)
    parser.add_argument("--scene-voxel-size", type=float, default=0.008)
    parser.add_argument("--scene-crop-margin", type=float, default=0.35)
    parser.add_argument("--max-scene-points", type=int, default=4096)
    parser.add_argument(
        "--environment-safe-initialization",
        action="store_true",
        help=(
            "Before joint/root optimization, translate each complete hand along "
            "object-outward/upward directions and keep the shortest tested offset "
            "with the lowest point-cloud environment violation."
        ),
    )
    parser.add_argument("--safe-init-max-offset", type=float, default=0.12)
    parser.add_argument("--safe-init-step", type=float, default=0.01)
    parser.add_argument(
        "--object-constraint-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of a normalized squared barrier on maximum object "
            "penetration. Zero preserves the original refiner."
        ),
    )
    parser.add_argument(
        "--environment-constraint-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of a normalized squared barrier on maximum environment "
            "violation. Zero preserves the original refiner."
        ),
    )
    parser.add_argument(
        "--contact-constraint-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of a normalized squared barrier on the worst assigned "
            "finger-to-contact error. Zero preserves the original refiner."
        ),
    )
    parser.add_argument("--contact-constraint-limit", type=float, default=0.010)
    parser.add_argument(
        "--require-contact-feasibility",
        action="store_true",
        help="Require every assigned finger contact to satisfy the contact limit.",
    )
    parser.add_argument(
        "--pose-constraint-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of normalized barriers that preserve the configured "
            "envelope side and palm approach direction."
        ),
    )
    parser.add_argument("--selection-min-envelope-cosine", type=float, default=0.5)
    parser.add_argument("--selection-min-approach-cosine", type=float, default=0.8)
    parser.add_argument(
        "--require-cosine-feasibility",
        action="store_true",
        help=(
            "Apply envelope/approach cosines as hard feasibility gates. "
            "The v6 baseline leaves this disabled and treats them as diagnostics."
        ),
    )
    parser.add_argument("--object-constraint-limit", type=float, default=0.007)
    parser.add_argument(
        "--constraint-ramp-start-fraction", type=float, default=0.25
    )
    parser.add_argument(
        "--restore-best-constraint-state",
        action="store_true",
        help=(
            "Restore the lowest-energy state observed while both hard "
            "constraints were satisfied; if none was feasible, restore the "
            "state with the lowest normalized barrier objective."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_xyz(path: Path) -> np.ndarray:
    points = np.asarray(np.load(path.resolve()), dtype=np.float32)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected [N, >=3] point cloud at {path}, got {points.shape}")
    return np.ascontiguousarray(points[:, :3])


def voxel_downsample(points: np.ndarray, voxel: float, maximum: int) -> np.ndarray:
    keys = np.floor(points / float(voxel)).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    sampled = points[np.sort(indices)]
    if len(sampled) > int(maximum):
        select = np.linspace(0, len(sampled) - 1, int(maximum)).round().astype(np.int64)
        sampled = sampled[select]
    return np.ascontiguousarray(sampled, dtype=np.float32)


def prepare_environment(
    scene: np.ndarray,
    object_pc: np.ndarray,
    *,
    crop_margin: float,
    voxel: float,
    maximum: int,
) -> tuple[np.ndarray, dict]:
    lower = object_pc.min(axis=0) - float(crop_margin)
    upper = object_pc.max(axis=0) + float(crop_margin)
    mask = np.all((scene >= lower[None]) & (scene <= upper[None]), axis=1)
    local = scene[mask]
    sampled = voxel_downsample(local, voxel, maximum)

    object_center = object_pc.mean(axis=0)
    radial = np.linalg.norm(scene[:, :2] - object_center[None, :2], axis=1)
    support_band = scene[
        (radial < 0.20)
        & (np.abs(scene[:, 2] - float(object_pc[:, 2].min())) < 0.02)
    ]
    if len(support_band) < 32:
        raise RuntimeError("Could not estimate the local tabletop support plane")
    support_z = float(np.median(support_band[:, 2]))
    tabletop = scene[np.abs(scene[:, 2] - support_z) < 0.002]
    table_xy_min = np.quantile(tabletop[:, :2], 0.01, axis=0)
    table_xy_max = np.quantile(tabletop[:, :2], 0.99, axis=0)
    return sampled, {
        "input_scene_points": int(len(scene)),
        "cropped_scene_points": int(len(local)),
        "optimized_scene_points": int(len(sampled)),
        "voxel_size_m": float(voxel),
        "support_plane_z_m": support_z,
        "table_xy_min": table_xy_min.tolist(),
        "table_xy_max": table_xy_max.tolist(),
    }


def cvar(values: torch.Tensor, fraction: float) -> torch.Tensor:
    count = max(1, int(np.ceil(float(fraction) * values.shape[1])))
    return values.topk(count, dim=1).values.mean(dim=1)


def environment_energy(
    hand_surface: torch.Tensor,
    scene_points: torch.Tensor,
    *,
    clearance: float,
    cvar_fraction: float,
    support_z: float,
    table_xy_min: torch.Tensor,
    table_xy_max: torch.Tensor,
) -> dict[str, torch.Tensor]:
    nearest = torch.cdist(hand_surface, scene_points.unsqueeze(0)).min(dim=2).values
    proximity = F.relu(float(clearance) - nearest)

    # Smooth table footprint gate keeps the signed half-space penalty local to
    # the tabletop: a hand hanging outside the table edge is not a collision.
    transition = 0.01
    x, y, z = hand_surface.unbind(dim=2)
    gate = (
        torch.sigmoid((x - table_xy_min[0]) / transition)
        * torch.sigmoid((table_xy_max[0] - x) / transition)
        * torch.sigmoid((y - table_xy_min[1]) / transition)
        * torch.sigmoid((table_xy_max[1] - y) / transition)
    )
    table_depth = F.relu(float(support_z + clearance) - z) * gate
    violations = torch.cat((proximity, table_depth), dim=1)
    mean = violations.mean(dim=1)
    tail = cvar(violations, cvar_fraction)
    return {
        "energy": mean + tail,
        "mean": mean,
        "cvar": tail,
        "max": violations.max(dim=1).values,
        "proximity_max": proximity.max(dim=1).values,
        "table_max": table_depth.max(dim=1).values,
        "fraction": (violations > 0).float().mean(dim=1),
        "nearest_scene_distance": nearest.min(dim=1).values,
    }


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrix[:, :, 0], matrix[:, :, 1]), dim=1)


def refine_record(
    record: dict,
    gripper,
    config: dict,
    object_pc: torch.Tensor,
    object_normals: torch.Tensor,
    object_confidence: torch.Tensor,
    scene_points: torch.Tensor,
    env_info: dict,
    args: argparse.Namespace,
) -> dict:
    fk = deepcopy(record["fk"])
    candidates = sorted(fk["candidates"], key=lambda item: int(item["rank"]))
    device, dtype = gripper.device, gripper.dtype
    batch = len(candidates)

    root_pose = torch.as_tensor(
        [item["root_pose"] for item in candidates], device=device, dtype=dtype
    )
    raw_pose = root_pose @ torch.linalg.inv(gripper.base_alignment).unsqueeze(0)
    translation = raw_pose[:, :3, 3].clone().requires_grad_(True)
    rotation_6d = matrix_to_rotation_6d(raw_pose[:, :3, :3]).clone().requires_grad_(True)
    joints_initial = torch.as_tensor(
        [item["joint_positions"] for item in candidates], device=device, dtype=dtype
    )
    q_raw = gripper.unconstrain_joints(joints_initial).clone().requires_grad_(True)

    spec = config["grippers"][record["gripper"]]
    opened = torch.as_tensor(
        ordered_joint_values(spec["opened_dofs"], gripper.joint_names, label="opened_dofs"),
        device=device,
        dtype=dtype,
    )
    close_direction = torch.as_tensor(
        ordered_joint_values(spec["close_dir"], gripper.joint_names, label="close_dir"),
        device=device,
        dtype=dtype,
    )
    active_closure = close_direction != 0
    open_limit = torch.where(close_direction > 0, gripper.lower, gripper.upper)
    close_limit = torch.where(close_direction > 0, gripper.upper, gripper.lower)
    target = torch.as_tensor(fk["target_contacts"], device=device, dtype=dtype)
    permutations = torch.as_tensor(
        list(itertools.permutations(range(target.shape[0]))),
        device=device,
        dtype=torch.long,
    )
    permuted = target[permutations]
    object_center = object_pc.mean(dim=0)
    measured_axis = (
        gripper.tip_points_in_aligned_base(opened.unsqueeze(0)).mean(dim=1)[0]
        - gripper.palm_point_in_aligned_base(opened.unsqueeze(0))[0]
    )
    measured_axis = F.normalize(measured_axis, dim=0)
    energy_cfg = config["fk_optimization"]
    contact_geometry_mode = str(fk.get("contact_geometry_mode", "distal_surface"))
    if contact_geometry_mode not in {"tip_point", "distal_surface"}:
        raise ValueError(
            "Environment refiner currently supports tip_point and "
            f"distal_surface contact geometry, got {contact_geometry_mode!r}"
        )
    optimizer = torch.optim.Adam(
        [translation, rotation_6d, q_raw], lr=float(args.learning_rate)
    )
    table_min = torch.as_tensor(env_info["table_xy_min"], device=device, dtype=dtype)
    table_max = torch.as_tensor(env_info["table_xy_max"], device=device, dtype=dtype)

    def terms(constraint_multiplier: float = 1.0):
        joints = gripper.constrain_joints(q_raw)
        tips = gripper.tip_points(joints, translation, rotation_6d)
        if contact_geometry_mode == "distal_surface":
            distal_surface, _ = gripper.tip_link_surface_geometry(
                joints, translation, rotation_6d
            )
            finger_target_distances = torch.linalg.norm(
                distal_surface[:, :, :, None, :]
                - target[None, None, None, :, :],
                dim=4,
            ).min(dim=2).values
            per_finger_by_permutation = finger_target_distances[
                :, None, :, :
            ].expand(-1, permutations.shape[0], -1, -1)
            target_indices = permutations[None, :, :, None].expand(
                batch, -1, -1, 1
            )
            per_finger_by_permutation = torch.gather(
                per_finger_by_permutation, 3, target_indices
            ).squeeze(3)
        else:
            distal_surface = None
            per_finger_by_permutation = torch.linalg.norm(
                tips[:, None, :, :] - permuted[None, :, :, :], dim=3
            )
        permutation_cost = per_finger_by_permutation.mean(dim=2)
        best_permutation_index = permutation_cost.argmin(dim=1)
        batch_index = torch.arange(batch, device=device)
        per_finger_contact = per_finger_by_permutation[
            batch_index, best_permutation_index
        ]
        max_finger_contact = per_finger_contact.max(dim=1).values
        best_permutations = permutations[best_permutation_index]
        matched_targets = target[best_permutations]
        if distal_surface is None:
            matched_contacts = tips
        else:
            finger_index = torch.arange(target.shape[0], device=device)[None, :]
            assigned_surface_distances = torch.linalg.norm(
                distal_surface - matched_targets[:, :, None, :], dim=3
            )
            nearest_surface_index = assigned_surface_distances.argmin(dim=2)
            matched_contacts = distal_surface[
                batch_index[:, None], finger_index, nearest_surface_index
            ]
        contact = 2.0 * permutation_cost[batch_index, best_permutation_index]
        hand_surface = gripper.surface_points(joints, translation, rotation_6d)
        penetration = point_cloud_penetration_energy(
            hand_surface,
            object_pc,
            object_normals,
            object_confidence,
            cvar_fraction=float(energy_cfg.get("penetration_cvar_fraction", 0.1)),
            cvar_weight=float(energy_cfg.get("penetration_cvar_weight", 1.0)),
            depth_mode=str(energy_cfg.get("penetration_depth_mode", "point_to_plane")),
            aggregation=str(energy_cfg.get("penetration_aggregation", "mean_cvar")),
            confidence_mode=str(energy_cfg.get("penetration_confidence_mode", "weighted")),
            gate_metric=str(energy_cfg.get("penetration_gate_metric", "confidence_weighted")),
            hinge_threshold_m=float(energy_cfg.get("penetration_hinge_threshold_m", 0.0)),
            hinge_weight=float(energy_cfg.get("penetration_hinge_weight", 0.0)),
        )
        link_points = gripper.self_collision_link_points(
            joints, int(energy_cfg.get("self_collision_points_per_link", 12))
        )
        self_collision = self_collision_mean_cvar_energy(
            link_points,
            gripper.self_collision_pairs,
            float(energy_cfg.get("self_collision_clearance", 0.002)),
            cvar_fraction=float(energy_cfg.get("self_collision_cvar_fraction", 0.25)),
            cvar_weight=float(energy_cfg.get("self_collision_cvar_weight", 1.0)),
        )
        palm = gripper.palm_points(joints, translation, rotation_6d)
        palm_world_axis = torch.einsum(
            "bij,j->bi", rotation_6d_to_matrix(rotation_6d), measured_axis
        )
        approach_cosine = (
            palm_world_axis * F.normalize(object_center[None] - palm, dim=1)
        ).sum(dim=1)
        root_direction = F.normalize(palm - object_center[None], dim=1)
        envelope_side = root_direction[:, 2]
        approach = (1.0 - approach_cosine) ** 2
        joint = (((joints - opened[None]) / gripper.span) ** 2).mean(dim=1)
        environment_surface = hand_surface
        if int(args.environment_closure_sweep_samples) > 1:
            outer_joints = torch.where(
                active_closure[None],
                joints
                + float(args.closure_outer_fraction)
                * (open_limit[None] - joints),
                joints,
            )
            inner_joints = torch.where(
                active_closure[None],
                joints
                + float(args.closure_inner_fraction)
                * (close_limit[None] - joints),
                joints,
            )
            sweep_alpha = torch.linspace(
                0.0,
                1.0,
                int(args.environment_closure_sweep_samples),
                device=device,
                dtype=dtype,
            )
            sweep_joints = (
                outer_joints[:, None, :]
                + sweep_alpha[None, :, None]
                * (inner_joints - outer_joints)[:, None, :]
            )
            sweep_count = int(args.environment_closure_sweep_samples)
            sweep_surface = gripper.surface_points(
                sweep_joints.reshape(batch * sweep_count, -1),
                translation[:, None, :]
                .expand(-1, sweep_count, -1)
                .reshape(batch * sweep_count, -1),
                rotation_6d[:, None, :]
                .expand(-1, sweep_count, -1)
                .reshape(batch * sweep_count, -1),
            )
            environment_surface = sweep_surface.reshape(
                batch, sweep_count * sweep_surface.shape[1], 3
            )
        environment = environment_energy(
            environment_surface,
            scene_points,
            clearance=float(args.environment_clearance),
            cvar_fraction=float(args.environment_cvar_fraction),
            support_z=float(env_info["support_plane_z_m"]),
            table_xy_min=table_min,
            table_xy_max=table_max,
        )
        base_total = (
            float(energy_cfg.get("contact_weight", 100.0)) * contact
            + float(energy_cfg.get("penetration_weight", 100.0)) * penetration["energy"]
            + float(energy_cfg.get("self_collision_weight", 100.0)) * self_collision["energy"]
            + 2.0 * approach
            + float(energy_cfg.get("joint_regularization", 0.001)) * joint
            + float(args.environment_weight) * environment["energy"]
        )
        object_excess = F.relu(
            penetration["max"] - float(args.object_constraint_limit)
        ) / max(float(args.object_constraint_limit), 1.0e-6)
        environment_excess = F.relu(
            environment["max"]
            - float(args.selection_max_environment_violation)
        ) / max(float(args.environment_clearance), 1.0e-6)
        object_barrier = object_excess.square()
        environment_barrier = environment_excess.square()
        contact_excess = F.relu(
            max_finger_contact - float(args.contact_constraint_limit)
        ) / max(float(args.contact_constraint_limit), 1.0e-6)
        contact_barrier = contact_excess.square()
        envelope_excess = F.relu(
            float(args.selection_min_envelope_cosine) - envelope_side
        ) / max(1.0 + float(args.selection_min_envelope_cosine), 1.0e-6)
        approach_excess = F.relu(
            float(args.selection_min_approach_cosine) - approach_cosine
        ) / max(1.0 + float(args.selection_min_approach_cosine), 1.0e-6)
        pose_barrier = envelope_excess.square() + approach_excess.square()
        total = (
            base_total
            + float(constraint_multiplier)
            * float(args.object_constraint_weight)
            * object_barrier
            + float(constraint_multiplier)
            * float(args.environment_constraint_weight)
            * environment_barrier
            + float(constraint_multiplier)
            * float(args.contact_constraint_weight)
            * contact_barrier
            + float(constraint_multiplier)
            * float(args.pose_constraint_weight)
            * pose_barrier
        )
        return (
            joints, tips, palm, approach_cosine, envelope_side, contact,
            per_finger_contact, max_finger_contact,
            matched_contacts, matched_targets,
            penetration, self_collision, environment, object_barrier,
            environment_barrier, contact_barrier, pose_barrier, total,
        )

    initialization_diagnostics = None
    if args.environment_safe_initialization:
        if float(args.safe_init_step) <= 0 or float(args.safe_init_max_offset) < 0:
            raise ValueError("safe initialization offsets must be non-negative")
        with torch.no_grad():
            original_translation = translation.detach().clone()
            joints = gripper.constrain_joints(q_raw)
            palm = gripper.palm_points(joints, translation, rotation_6d)
            outward = F.normalize(palm - object_center[None], dim=1)
            upward = torch.zeros_like(outward)
            upward[:, 2] = 1.0
            outward_upward = F.normalize(outward + upward, dim=1)
            directions = (outward, upward, outward_upward)

            initial_surface = gripper.surface_points(
                joints, translation, rotation_6d
            )
            initial_environment = environment_energy(
                initial_surface,
                scene_points,
                clearance=float(args.environment_clearance),
                cvar_fraction=float(args.environment_cvar_fraction),
                support_z=float(env_info["support_plane_z_m"]),
                table_xy_min=table_min,
                table_xy_max=table_max,
            )
            best_translation = original_translation.clone()
            best_violation = initial_environment["max"].clone()
            best_offset = torch.zeros(batch, device=device, dtype=dtype)
            best_direction = torch.full(
                (batch,), -1, device=device, dtype=torch.long
            )
            distances = torch.arange(
                float(args.safe_init_step),
                float(args.safe_init_max_offset) + 0.5 * float(args.safe_init_step),
                float(args.safe_init_step),
                device=device,
                dtype=dtype,
            )
            for direction_index, direction in enumerate(directions):
                for distance in distances:
                    proposal = original_translation + distance * direction
                    proposal_surface = gripper.surface_points(
                        joints, proposal, rotation_6d
                    )
                    proposal_environment = environment_energy(
                        proposal_surface,
                        scene_points,
                        clearance=float(args.environment_clearance),
                        cvar_fraction=float(args.environment_cvar_fraction),
                        support_z=float(env_info["support_plane_z_m"]),
                        table_xy_min=table_min,
                        table_xy_max=table_max,
                    )
                    improved = proposal_environment["max"] < best_violation - 1.0e-8
                    best_violation = torch.where(
                        improved, proposal_environment["max"], best_violation
                    )
                    best_translation = torch.where(
                        improved[:, None], proposal, best_translation
                    )
                    best_offset = torch.where(improved, distance, best_offset)
                    best_direction = torch.where(
                        improved,
                        best_direction.new_full((batch,), direction_index),
                        best_direction,
                    )
            translation.copy_(best_translation)
            initialization_diagnostics = {
                "before_max": initial_environment["max"].detach().clone(),
                "after_max": best_violation.detach().clone(),
                "offset": best_offset.detach().clone(),
                "direction": best_direction.detach().clone(),
            }

    best_state = None
    best_feasible = torch.zeros(batch, device=device, dtype=torch.bool)
    best_score = torch.full(
        (batch,), float("inf"), device=device, dtype=dtype
    )

    def remember_state(
        current_total: torch.Tensor,
        max_finger_contact: torch.Tensor,
        envelope_side: torch.Tensor,
        approach_cosine: torch.Tensor,
        penetration: dict[str, torch.Tensor],
        environment: dict[str, torch.Tensor],
    ) -> None:
        nonlocal best_state, best_feasible, best_score
        feasible_now = penetration["max"] <= float(args.object_constraint_limit)
        if args.environment_feasibility_mode == "hard":
            feasible_now = feasible_now & (
                environment["max"]
                <= float(args.selection_max_environment_violation)
            )
        if args.require_contact_feasibility:
            feasible_now = feasible_now & (
                max_finger_contact <= float(args.contact_constraint_limit)
            )
        if args.require_cosine_feasibility:
            feasible_now = (
                feasible_now
                & (envelope_side >= float(args.selection_min_envelope_cosine))
                & (approach_cosine >= float(args.selection_min_approach_cosine))
            )
        object_excess = F.relu(
            penetration["max"] - float(args.object_constraint_limit)
        ) / max(float(args.object_constraint_limit), 1.0e-6)
        environment_excess = F.relu(
            environment["max"]
            - float(args.selection_max_environment_violation)
        ) / max(float(args.environment_clearance), 1.0e-6)
        contact_excess = F.relu(
            max_finger_contact - float(args.contact_constraint_limit)
        ) / max(float(args.contact_constraint_limit), 1.0e-6)
        infeasible_score = object_excess.square() + environment_excess.square()
        if args.require_contact_feasibility:
            infeasible_score = infeasible_score + contact_excess.square()
        envelope_excess = F.relu(
            float(args.selection_min_envelope_cosine) - envelope_side
        ) / max(1.0 + float(args.selection_min_envelope_cosine), 1.0e-6)
        approach_excess = F.relu(
            float(args.selection_min_approach_cosine) - approach_cosine
        ) / max(1.0 + float(args.selection_min_approach_cosine), 1.0e-6)
        if args.require_cosine_feasibility:
            infeasible_score = (
                infeasible_score
                + envelope_excess.square()
                + approach_excess.square()
            )
        score_now = torch.where(feasible_now, current_total, infeasible_score)
        improved = (
            (feasible_now & ~best_feasible)
            | ((feasible_now == best_feasible) & (score_now < best_score))
        )
        if best_state is None:
            best_state = (
                translation.detach().clone(),
                rotation_6d.detach().clone(),
                q_raw.detach().clone(),
            )
        else:
            best_state = tuple(
                torch.where(
                    improved.reshape((-1,) + (1,) * (value.ndim - 1)),
                    current.detach(),
                    value,
                )
                for value, current in zip(
                    best_state, (translation, rotation_6d, q_raw), strict=True
                )
            )
        best_feasible = torch.where(improved, feasible_now, best_feasible)
        best_score = torch.where(improved, score_now.detach(), best_score)

    ramp_start = float(args.constraint_ramp_start_fraction)
    for step in range(int(args.steps)):
        progress = float(step + 1) / max(1, int(args.steps))
        constraint_multiplier = min(
            1.0, max(0.0, (progress - ramp_start) / max(1.0 - ramp_start, 1.0e-6))
        )
        (
            _, _, _, approach_cosine_step, envelope_side_step, _, _,
            max_finger_contact_step, _, _, penetration_step, _, environment_step,
            _, _, _, _, total,
        ) = terms(constraint_multiplier)
        if args.restore_best_constraint_state:
            with torch.no_grad():
                remember_state(
                    total, max_finger_contact_step,
                    envelope_side_step, approach_cosine_step,
                    penetration_step, environment_step,
                )
        optimizer.zero_grad(set_to_none=True)
        total.mean().backward()
        optimizer.step()

    with torch.no_grad():
        (
            joints, tips, palm, approach_cosine, envelope_side, contact,
            per_finger_contact, max_finger_contact,
            matched_contacts, matched_targets,
            penetration, self_collision, environment, object_barrier,
            environment_barrier, contact_barrier, pose_barrier, total,
        ) = terms()
        if args.restore_best_constraint_state:
            remember_state(
                total, max_finger_contact, envelope_side, approach_cosine,
                penetration, environment,
            )
            translation.copy_(best_state[0])
            rotation_6d.copy_(best_state[1])
            q_raw.copy_(best_state[2])
            (
                joints, tips, palm, approach_cosine, envelope_side, contact,
                per_finger_contact, max_finger_contact,
                matched_contacts, matched_targets,
                penetration, self_collision, environment, object_barrier,
                environment_barrier, contact_barrier, pose_barrier, total,
            ) = terms()
        palm_surface = gripper.palm_surface_points(joints, translation, rotation_6d)
        palm_distance, palm_hand, palm_object, palm_index = nearest_surface_distance(
            palm_surface, object_pc
        )
        poses = gripper.root_pose_matrix(translation, rotation_6d)
        environment_filter_pass = (
            environment["max"] <= float(args.selection_max_environment_violation)
        )
        contact_filter_pass = (
            max_finger_contact <= float(args.contact_constraint_limit)
        )
        cosine_filter_pass = (
            (envelope_side >= float(args.selection_min_envelope_cosine))
            & (approach_cosine >= float(args.selection_min_approach_cosine))
        )
        feasible = (
            (penetration["max"] <= float(args.object_constraint_limit))
        )
        if args.environment_feasibility_mode == "hard":
            feasible = feasible & environment_filter_pass
        if args.require_contact_feasibility:
            feasible = feasible & contact_filter_pass
        if args.require_cosine_feasibility:
            feasible = (
                feasible
                & (envelope_side >= float(args.selection_min_envelope_cosine))
                & (approach_cosine >= float(args.selection_min_approach_cosine))
            )
        normalized_constraint_score = (
            penetration["max"] / float(args.object_constraint_limit)
            + max_finger_contact / float(args.contact_constraint_limit)
            + environment["max"] / float(args.environment_clearance)
        )
        if args.selection_rank_mode == "normalized_constraints":
            order = sorted(
                range(batch),
                key=lambda index: (
                    not bool(feasible[index].item()),
                    float(normalized_constraint_score[index].item()),
                    float(total[index].item()),
                    index,
                ),
            )
        else:
            order = sorted(
                range(batch),
                key=lambda index: (
                    not bool(feasible[index].item()),
                    float(environment["max"][index].item()),
                    float(total[index].item()),
                    index,
                ),
            )

    refined = []
    for rank, index in enumerate(order):
        source = deepcopy(candidates[index])
        source.update({
            "rank": rank,
            "source_candidate_rank": int(
                candidates[index].get("source_candidate_rank", candidates[index]["rank"])
            ),
            "selection_feasible": bool(feasible[index].item()),
            "selection_fallback": not bool(feasible[index].item()),
            "selection_constraint_score": float(
                normalized_constraint_score[index].item()
            ),
            "optimization_score": float(total[index].item()),
            "contact_chamfer_m": float(contact[index].item()),
            "assigned_contact_error_m": float(0.5 * contact[index].item()),
            "per_finger_contact_errors_m": per_finger_contact[index].cpu().tolist(),
            "max_finger_contact_error_m": float(
                max_finger_contact[index].item()
            ),
            "all_fingers_within_contact_threshold": bool(
                contact_filter_pass[index].item()
            ),
            "cosine_filter_pass": bool(
                not args.require_cosine_feasibility
                or cosine_filter_pass[index].item()
            ),
            "mean_penetration_m": float(penetration["mean"][index].item()),
            "cvar_penetration_m": float(penetration["cvar"][index].item()),
            "hinge_penetration_m": float(penetration["hinge"][index].item()),
            "max_penetration_m": float(penetration["max"][index].item()),
            "raw_max_penetration_m": float(penetration["raw_max"][index].item()),
            "confidence_weighted_max_penetration_m": float(
                penetration["confidence_weighted_max"][index].item()
            ),
            "penetrating_surface_fraction": float(penetration["fraction"][index].item()),
            "mean_self_collision_m": float(self_collision["mean"][index].item()),
            "cvar_self_collision_m": float(self_collision["cvar"][index].item()),
            "max_self_collision_m": float(self_collision["max"][index].item()),
            "self_collision_pair_fraction": float(self_collision["fraction"][index].item()),
            "envelope_side_cosine": float(envelope_side[index].item()),
            "palm_approach_cosine": float(approach_cosine[index].item()),
            "palm_unsigned_distance_m": float(palm_distance[index].item()),
            "palm_distance_error_m": float((palm_distance[index] - 0.001).abs().item()),
            "environment_mean_violation_m": float(environment["mean"][index].item()),
            "environment_cvar_violation_m": float(environment["cvar"][index].item()),
            "environment_max_violation_m": float(environment["max"][index].item()),
            "environment_min_scene_distance_m": float(
                environment["nearest_scene_distance"][index].item()
            ),
            "environment_table_max_violation_m": float(
                environment["table_max"][index].item()
            ),
            "environment_filter_pass": bool(environment_filter_pass[index].item()),
            "object_constraint_barrier": float(object_barrier[index].item()),
            "environment_constraint_barrier": float(
                environment_barrier[index].item()
            ),
            "contact_constraint_barrier": float(contact_barrier[index].item()),
            "pose_constraint_barrier": float(pose_barrier[index].item()),
            "restored_best_constraint_state": bool(
                args.restore_best_constraint_state
            ),
            "graspqp_recomputed_after_refinement": False,
            "root_pose": poses[index].cpu().tolist(),
            "joint_positions": joints[index].cpu().tolist(),
            "tip_points": tips[index].cpu().tolist(),
            "matched_contact_points": matched_contacts[index].cpu().tolist(),
            "matched_target_object_points": matched_targets[index].cpu().tolist(),
            "palm_position": palm[index].cpu().tolist(),
            "palm_surface_contact": palm_hand[index].cpu().tolist(),
            "palm_object_contact": palm_object[index].cpu().tolist(),
            "palm_object_normal": object_normals[palm_index[index]].cpu().tolist(),
            "palm_object_normal_confidence": float(
                object_confidence[palm_index[index]].item()
            ),
            "penetration_filter_pass": bool(
                (
                    penetration["max"][index]
                    <= float(args.object_constraint_limit)
                ).item()
            ),
            "envelope_filter_pass": bool(
                not args.require_cosine_feasibility
                or (
                    envelope_side[index]
                    >= float(args.selection_min_envelope_cosine)
                ).item()
            ),
        })
        if initialization_diagnostics is not None:
            direction_names = ("object_outward", "world_up", "outward_plus_up")
            direction_index = int(
                initialization_diagnostics["direction"][index].item()
            )
            source["environment_safe_initialization"] = {
                "enabled": True,
                "before_max_violation_m": float(
                    initialization_diagnostics["before_max"][index].item()
                ),
                "after_max_violation_m": float(
                    initialization_diagnostics["after_max"][index].item()
                ),
                "translation_offset_m": float(
                    initialization_diagnostics["offset"][index].item()
                ),
                "direction": (
                    direction_names[direction_index]
                    if direction_index >= 0
                    else "unchanged"
                ),
            }
        refined.append(source)

    fk["candidates"] = refined
    previous_refinement = fk.get("environment_refinement", {})
    cumulative_steps = int(
        previous_refinement.get("cumulative_steps", previous_refinement.get("steps", 0))
    ) + int(args.steps)
    fk["environment_refinement"] = {
        "steps": int(args.steps),
        "cumulative_steps": cumulative_steps,
        "learning_rate": float(args.learning_rate),
        "weight": float(args.environment_weight),
        "clearance_m": float(args.environment_clearance),
        "selection_max_violation_m": float(
            args.selection_max_environment_violation
        ),
        "environment_feasibility_mode": str(args.environment_feasibility_mode),
        "selection_rank_mode": str(args.selection_rank_mode),
        "cvar_fraction": float(args.environment_cvar_fraction),
        "closure_sweep_samples": int(args.environment_closure_sweep_samples),
        "closure_outer_fraction": float(args.closure_outer_fraction),
        "closure_inner_fraction": float(args.closure_inner_fraction),
        "palm_distance_gate_enabled": False,
        "environment_safe_initialization": bool(
            args.environment_safe_initialization
        ),
        "safe_init_max_offset_m": float(args.safe_init_max_offset),
        "safe_init_step_m": float(args.safe_init_step),
        "object_constraint_weight": float(args.object_constraint_weight),
        "environment_constraint_weight": float(
            args.environment_constraint_weight
        ),
        "contact_constraint_weight": float(args.contact_constraint_weight),
        "contact_constraint_limit_m": float(args.contact_constraint_limit),
        "contact_feasibility_required": bool(args.require_contact_feasibility),
        "pose_constraint_weight": float(args.pose_constraint_weight),
        "selection_min_envelope_cosine": float(
            args.selection_min_envelope_cosine
        ),
        "selection_min_approach_cosine": float(args.selection_min_approach_cosine),
        "cosine_feasibility_gates_enabled": bool(
            args.require_cosine_feasibility
        ),
        "object_constraint_limit_m": float(args.object_constraint_limit),
        "constraint_ramp_start_fraction": float(
            args.constraint_ramp_start_fraction
        ),
        "restore_best_constraint_state": bool(
            args.restore_best_constraint_state
        ),
        **env_info,
    }
    updated = deepcopy(record)
    updated["fk"] = fk
    return updated


def main() -> None:
    args = parse_args()
    if (
        args.object_constraint_weight < 0
        or args.environment_constraint_weight < 0
        or args.contact_constraint_weight < 0
        or args.pose_constraint_weight < 0
    ):
        raise ValueError("constraint weights must be non-negative")
    if args.object_constraint_limit <= 0:
        raise ValueError("object constraint limit must be positive")
    if args.contact_constraint_limit <= 0:
        raise ValueError("contact constraint limit must be positive")
    if args.environment_closure_sweep_samples <= 0:
        raise ValueError("environment closure sweep samples must be positive")
    if not 0.0 <= args.closure_outer_fraction <= 1.0:
        raise ValueError("closure outer fraction must be in [0, 1]")
    if not 0.0 <= args.closure_inner_fraction <= 1.0:
        raise ValueError("closure inner fraction must be in [0, 1]")
    if not 0.0 <= args.constraint_ramp_start_fraction < 1.0:
        raise ValueError("constraint ramp start fraction must be in [0, 1)")
    if args.sample_start < 0:
        raise ValueError("sample start must be non-negative")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("max records must be positive")
    for path in (args.candidates, args.config, args.object_pc, args.scene_pc):
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    output_path = args.output.resolve()
    payload = json.loads(args.candidates.resolve().read_text())
    import yaml

    config = yaml.safe_load(args.config.resolve().read_text())
    object_np = load_xyz(args.object_pc)
    scene_np = load_xyz(args.scene_pc)
    scene_local, env_info = prepare_environment(
        scene_np,
        object_np,
        crop_margin=args.scene_crop_margin,
        voxel=args.scene_voxel_size,
        maximum=args.max_scene_points,
    )

    os.chdir(CONTACT_ROOT)
    device = torch.device(args.device)
    object_geometry = estimate_point_cloud_geometry(
        object_np,
        k_neighbors=int(config["fk_optimization"].get("object_normal_k_neighbors", 30)),
    )
    object_pc = torch.as_tensor(object_np, device=device)
    object_normals = torch.as_tensor(object_geometry["normals"], device=device)
    object_confidence = torch.as_tensor(object_geometry["confidence"], device=device)
    scene_points = torch.as_tensor(scene_local, device=device)
    hand_name = str(payload["records"][0]["gripper"])
    calibration = (CONTACT_ROOT / config["paths"]["tip_offset_calibration"]).resolve()
    gripper = load_gripper_from_calibration(
        hand_name, config, calibration, device=device
    )

    output = deepcopy(payload)
    previous_environment = payload.get("environment_energy", {})
    cumulative_steps = int(
        previous_environment.get(
            "cumulative_refinement_steps",
            previous_environment.get("refinement_steps", 0),
        )
    ) + int(args.steps)
    output["environment_energy"] = {
        "type": "signed_table_plane_plus_visible_scene_clearance_mean_cvar",
        "weight": float(args.environment_weight),
        "clearance_m": float(args.environment_clearance),
        "selection_max_violation_m": float(
            args.selection_max_environment_violation
        ),
        "environment_feasibility_mode": str(args.environment_feasibility_mode),
        "selection_rank_mode": str(args.selection_rank_mode),
        "cvar_fraction": float(args.environment_cvar_fraction),
        "closure_sweep_samples": int(args.environment_closure_sweep_samples),
        "closure_outer_fraction": float(args.closure_outer_fraction),
        "closure_inner_fraction": float(args.closure_inner_fraction),
        "refinement_steps": int(args.steps),
        "cumulative_refinement_steps": cumulative_steps,
        "palm_distance_gradient_weight": 0.0,
        "palm_distance_gate_enabled": False,
        "environment_safe_initialization": bool(
            args.environment_safe_initialization
        ),
        "safe_init_max_offset_m": float(args.safe_init_max_offset),
        "safe_init_step_m": float(args.safe_init_step),
        "object_constraint_weight": float(args.object_constraint_weight),
        "environment_constraint_weight": float(
            args.environment_constraint_weight
        ),
        "contact_constraint_weight": float(args.contact_constraint_weight),
        "contact_constraint_limit_m": float(args.contact_constraint_limit),
        "contact_feasibility_required": bool(args.require_contact_feasibility),
        "pose_constraint_weight": float(args.pose_constraint_weight),
        "selection_min_envelope_cosine": float(
            args.selection_min_envelope_cosine
        ),
        "selection_min_approach_cosine": float(args.selection_min_approach_cosine),
        "cosine_feasibility_gates_enabled": bool(
            args.require_cosine_feasibility
        ),
        "object_constraint_limit_m": float(args.object_constraint_limit),
        "constraint_ramp_start_fraction": float(
            args.constraint_ramp_start_fraction
        ),
        "restore_best_constraint_state": bool(
            args.restore_best_constraint_state
        ),
        "paired_source": str(args.candidates.resolve()),
        **env_info,
    }
    selected_records = [
        record
        for record in payload["records"]
        if int(record["sample_index"]) >= int(args.sample_start)
    ]
    if args.max_records is not None:
        selected_records = selected_records[: int(args.max_records)]
    if not selected_records:
        raise ValueError("No candidate records matched the requested pilot slice")
    output["records"] = []
    output["environment_energy"]["sample_start"] = int(args.sample_start)
    output["environment_energy"]["max_records"] = (
        None if args.max_records is None else int(args.max_records)
    )
    for record in selected_records:
        refined = refine_record(
            record,
            gripper,
            config,
            object_pc,
            object_normals,
            object_confidence,
            scene_points,
            env_info,
            args,
        )
        output["records"].append(refined)
        best = refined["fk"]["candidates"][0]
        print(
            f"{hand_name} set={record['sample_index']}: "
            f"env_max={1000 * best['environment_max_violation_m']:.2f} mm, "
            f"object_max_pen={1000 * best['max_penetration_m']:.2f} mm, "
            f"palm={1000 * best['palm_unsigned_distance_m']:.2f} mm, "
            f"feasible={best['selection_feasible']}",
            flush=True,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
