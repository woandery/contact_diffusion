"""Validation-time physical metrics for generated contact sets."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .multi_contact_grasp_quality import (
    PointCloudMultiContactGraspEvaluator,
    estimate_point_cloud_normals,
    match_predicted_contacts_to_point_cloud,
)


def _get(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, default)
    return default if value is None and default is not None else value


@torch.no_grad()
def evaluate_generated_contact_batch(
    model,
    object_pc_model: torch.Tensor,
    object_pc_xyz: torch.Tensor,
    num_contacts: int,
    config: Any,
    *,
    object_normals: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Sample and physically score a subset of one validation batch.

    Continuous predictions are uniquely matched to the current input cloud.
    Ground-truth ``selected_indices`` are deliberately not accepted by this
    function. Invalid projections contribute zero to all-sample force closure
    and epsilon, while valid-only variants are reported separately.
    """
    batch_limit = int(_get(config, "samples_per_batch", 4))
    sample_count = min(int(object_pc_xyz.shape[0]), batch_limit)
    if sample_count <= 0:
        return {}

    model_inputs = object_pc_model[:sample_count]
    xyz_inputs = object_pc_xyz[:sample_count, :, :3]
    normal_inputs = None if object_normals is None else object_normals[:sample_count]
    device = model_inputs.device
    cuda_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        predicted = model.sample(
            object_pc=model_inputs,
            num_contacts=int(num_contacts),
            dc=int(_get(config, "dc", 3)),
            num_steps=int(_get(config, "num_steps", 20)),
            sampler=str(_get(config, "sampler", "ddim")),
            # Matching below performs auditable unique projection and gating.
            project_to_surface=False,
        )

    clouds = xyz_inputs.detach().cpu().numpy()
    predictions = predicted[..., :3].detach().cpu().numpy()
    supplied_normals = (
        None if normal_inputs is None else normal_inputs.detach().cpu().numpy()
    )
    projection_valid = 0
    quality_valid = 0
    force_closures = []
    valid_force_closures = []
    epsilons = []
    valid_epsilons = []
    projection_distances = []
    precomputed_normal_count = 0

    absolute_gate = _get(config, "max_projection_distance", None)
    distance_factor = float(_get(config, "max_projection_distance_factor", 2.0))
    require_precomputed = bool(_get(config, "require_precomputed_normals", False))
    viewpoint = _get(config, "normal_viewpoint", None)
    if viewpoint is not None:
        viewpoint = np.asarray(viewpoint, dtype=np.float64)

    for i in range(sample_count):
        match = match_predicted_contacts_to_point_cloud(
            predictions[i],
            clouds[i],
            max_projection_distance=absolute_gate,
            max_distance_factor=distance_factor,
        )
        if match["projection_distances"].size:
            projection_distances.extend(match["projection_distances"].tolist())
        if not match["valid"]:
            force_closures.append(0.0)
            epsilons.append(0.0)
            continue
        projection_valid += 1

        if supplied_normals is not None:
            normals = supplied_normals[i]
            precomputed_normal_count += 1
        elif require_precomputed:
            force_closures.append(0.0)
            epsilons.append(0.0)
            continue
        else:
            try:
                normals = estimate_point_cloud_normals(
                    clouds[i],
                    k_neighbors=int(_get(config, "normal_k_neighbors", 30)),
                    viewpoint=viewpoint,
                )
            except (TypeError, ValueError, np.linalg.LinAlgError):
                force_closures.append(0.0)
                epsilons.append(0.0)
                continue

        result = PointCloudMultiContactGraspEvaluator.evaluate(
            clouds[i],
            match["predicted_indices"],
            normals,
            friction_coef=float(_get(config, "friction_coef", 0.5)),
            num_cone_faces=int(_get(config, "num_cone_faces", 8)),
            soft_fingers=bool(_get(config, "soft_fingers", True)),
            finger_radius=float(_get(config, "finger_radius", 0.005)),
            torque_scaling=_get(config, "torque_scaling", None),
            center_of_mass=None,
            fixed_total_force_budget=bool(
                _get(config, "fixed_total_force_budget", True)
            ),
        )
        closure = float(result["force_closure"]) if result["valid"] else 0.0
        epsilon = float(result["epsilon"]) if result["valid"] else 0.0
        force_closures.append(closure)
        epsilons.append(epsilon)
        if result["valid"]:
            quality_valid += 1
            valid_force_closures.append(closure)
            valid_epsilons.append(epsilon)

    valid_epsilon_array = np.asarray(valid_epsilons, dtype=np.float64)
    projection_array = np.asarray(projection_distances, dtype=np.float64)
    return {
        "grasp_projection_valid_rate": projection_valid / float(sample_count),
        "grasp_quality_valid_rate": quality_valid / float(sample_count),
        "grasp_force_closure_rate": float(np.mean(force_closures)),
        "grasp_force_closure_rate_valid": (
            float(np.mean(valid_force_closures)) if valid_force_closures else 0.0
        ),
        "grasp_epsilon_mean": float(np.mean(epsilons)),
        "grasp_epsilon_mean_valid": (
            float(np.mean(valid_epsilon_array)) if valid_epsilon_array.size else 0.0
        ),
        "grasp_epsilon_p10_valid": (
            float(np.percentile(valid_epsilon_array, 10))
            if valid_epsilon_array.size
            else 0.0
        ),
        "grasp_projection_distance_mean": (
            float(np.mean(projection_array)) if projection_array.size else 0.0
        ),
        "grasp_projection_distance_max": (
            float(np.max(projection_array)) if projection_array.size else 0.0
        ),
        "grasp_precomputed_normal_rate": precomputed_normal_count
        / float(sample_count),
        "grasp_quality_num_samples": float(sample_count),
    }


__all__ = ["evaluate_generated_contact_batch"]
