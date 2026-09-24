"""Geometry estimates and collision proxies for XYZ-only object point clouds."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


def _as_xyz(points: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        value = points.detach().cpu().numpy()
    else:
        value = np.asarray(points)
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 4:
        raise ValueError("point cloud must have shape [N, 3] with N >= 4")
    if not np.isfinite(value).all():
        raise ValueError("point cloud contains NaN or Inf")
    return value


def load_xyz_point_cloud(path: str | Path) -> np.ndarray:
    """Load every finite XYZ row from an NPY/NPZ point-cloud asset."""

    path = Path(path)
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if not loaded.files:
                raise ValueError(f"Empty NPZ point-cloud asset: {path}")
            key = (
                "object_pc"
                if "object_pc" in loaded.files
                else "points"
                if "points" in loaded.files
                else loaded.files[0]
            )
            points = loaded[key].copy()
        finally:
            loaded.close()
    else:
        points = loaded
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 4:
        raise ValueError(
            f"Expected [N, >=3] point cloud with N >= 4 at {path}, got {points.shape}"
        )
    points = np.ascontiguousarray(points[:, :3])
    if not np.isfinite(points).all():
        raise ValueError(f"Point cloud contains NaN or Inf: {path}")
    return points


def prepare_point_cloud_inputs(
    path: str | Path,
    *,
    model_point_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic model input plus the unabridged geometry cloud."""

    if int(model_point_count) < 1:
        raise ValueError("model_point_count must be positive")
    full_xyz = load_xyz_point_cloud(path)
    model_xyz = full_xyz
    if len(model_xyz) != int(model_point_count):
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(
            len(model_xyz),
            size=int(model_point_count),
            replace=len(model_xyz) < int(model_point_count),
        )
        model_xyz = model_xyz[indices]
    return (
        torch.from_numpy(np.ascontiguousarray(model_xyz)),
        torch.from_numpy(full_xyz.copy()),
    )


def estimate_point_cloud_geometry(
    points: np.ndarray | torch.Tensor,
    *,
    k_neighbors: int = 30,
) -> dict[str, np.ndarray | float | int]:
    """Estimate consistently oriented normals and a per-point confidence.

    Local PCA estimates an unoriented normal.  A deterministic traversal of
    the k-nearest-neighbour graph propagates its sign, after which each graph
    component is oriented predominantly away from the point-cloud centroid.
    This assumes a reasonably complete, orientable surface cloud.  It remains
    a surface proxy rather than a watertight inside/outside guarantee.

    Confidence combines the PCA normal eigengap, neighbourhood normal
    agreement, and sampling-density consistency.  Values are in ``[0, 1]``.
    """

    xyz = _as_xyz(points)
    count = len(xyz)
    k = min(int(k_neighbors), count)
    if k < 4:
        raise ValueError("k_neighbors must provide at least four neighbours")

    distances, neighbors = cKDTree(xyz).query(xyz, k=k)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
        distances = distances[:, None]

    normals = np.zeros_like(xyz)
    eigengap = np.zeros(count, dtype=np.float64)
    local_scale = np.maximum(distances[:, -1], 1.0e-12)
    center = np.mean(xyz, axis=0)

    for index, local_indices in enumerate(neighbors):
        local = xyz[local_indices]
        centered = local - np.mean(local, axis=0)
        covariance = centered.T.dot(centered) / float(len(local))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        if not np.isfinite(eigenvalues).all() or eigenvalues[1] <= 1.0e-15:
            radial = xyz[index] - center
            radial_norm = np.linalg.norm(radial)
            normals[index] = (
                radial / radial_norm
                if radial_norm > 1.0e-12
                else np.array([0.0, 0.0, 1.0])
            )
            eigengap[index] = 0.0
            continue
        normals[index] = eigenvectors[:, 0]
        eigengap[index] = (eigenvalues[1] - eigenvalues[0]) / (
            eigenvalues[1] + 1.0e-12
        )

    # Sign propagation prevents neighbouring PCA normals from arbitrarily
    # flipping.  Highest-confidence seeds make the result deterministic and
    # less sensitive to edges or sparse regions.
    visited = np.zeros(count, dtype=bool)
    component_ids = np.full(count, -1, dtype=np.int64)
    component_count = 0
    seed_order = np.lexsort((np.arange(count), -eigengap))
    for seed in seed_order:
        if visited[seed]:
            continue
        visited[seed] = True
        component_ids[seed] = component_count
        queue: deque[int] = deque([int(seed)])
        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current, 1:]:
                neighbor = int(neighbor)
                if visited[neighbor]:
                    continue
                if np.dot(normals[current], normals[neighbor]) < 0.0:
                    normals[neighbor] *= -1.0
                visited[neighbor] = True
                component_ids[neighbor] = component_count
                queue.append(neighbor)
        component_count += 1

    radial = xyz - center
    for component in range(component_count):
        mask = component_ids == component
        orientation_score = np.sum(
            eigengap[mask] * np.einsum("ij,ij->i", normals[mask], radial[mask])
        )
        if orientation_score < 0.0:
            normals[mask] *= -1.0

    neighbor_normals = normals[neighbors[:, 1:]]
    agreement = np.mean(
        np.abs(np.einsum("ij,ikj->ik", normals, neighbor_normals)), axis=1
    )
    median_scale = float(np.median(local_scale))
    density_ratio = local_scale / max(median_scale, 1.0e-12)
    density_confidence = np.exp(-np.abs(np.log(np.maximum(density_ratio, 1.0e-12))))
    confidence = np.clip(eigengap * agreement * density_confidence, 0.0, 1.0)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(normal_norm, 1.0e-12)

    return {
        "normals": normals.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "local_scale": local_scale.astype(np.float32),
        "k_neighbors": int(k),
        "mean_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
        "median_local_scale": median_scale,
    }


def point_cloud_penetration_energy(
    hand_surface_points: torch.Tensor,
    object_points: torch.Tensor,
    object_normals: torch.Tensor,
    normal_confidence: torch.Tensor,
    *,
    cvar_fraction: float = 0.1,
    cvar_weight: float = 1.0,
    depth_mode: str = "point_to_plane",
    aggregation: str = "mean_cvar",
    confidence_mode: str = "weighted",
    gate_metric: str = "confidence_weighted",
    hinge_threshold_m: float = 0.0,
    hinge_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return a configurable point-cloud penetration proxy.

    The signed point-to-plane proxy is positive outside and negative behind
    the estimated oriented surface.  It is meaningful for complete surface
    clouds with coherent normals, but cannot certify unobserved geometry in a
    partial single-view cloud.
    """

    if hand_surface_points.ndim != 3 or hand_surface_points.shape[-1] != 3:
        raise ValueError("hand_surface_points must have shape [B, H, 3]")
    if object_points.ndim != 2 or object_points.shape[-1] != 3:
        raise ValueError("object_points must have shape [N, 3]")
    if object_normals.shape != object_points.shape:
        raise ValueError("object_normals must match object_points")
    if normal_confidence.shape != object_points.shape[:1]:
        raise ValueError("normal_confidence must have shape [N]")
    if not 0.0 < float(cvar_fraction) <= 1.0:
        raise ValueError("cvar_fraction must be in (0, 1]")
    if float(cvar_weight) < 0.0:
        raise ValueError("cvar_weight must be non-negative")
    if depth_mode not in {"point_to_plane", "signed_euclidean"}:
        raise ValueError(
            "depth_mode must be 'point_to_plane' or 'signed_euclidean'"
        )
    if aggregation not in {"mean", "mean_cvar", "max"}:
        raise ValueError("aggregation must be 'mean', 'mean_cvar', or 'max'")
    if confidence_mode not in {"weighted", "none"}:
        raise ValueError("confidence_mode must be 'weighted' or 'none'")
    if gate_metric not in {"confidence_weighted", "raw"}:
        raise ValueError("gate_metric must be 'confidence_weighted' or 'raw'")
    if float(hinge_threshold_m) < 0.0:
        raise ValueError("hinge_threshold_m must be non-negative")
    if float(hinge_weight) < 0.0:
        raise ValueError("hinge_weight must be non-negative")

    distances = torch.cdist(hand_surface_points, object_points.unsqueeze(0))
    nearest_distance, nearest_index = distances.min(dim=2)
    nearest_points = object_points[nearest_index]
    nearest_normals = torch.nn.functional.normalize(
        object_normals[nearest_index], dim=2
    )
    confidence = normal_confidence[nearest_index].clamp(0.0, 1.0)
    signed_distance = (
        (hand_surface_points - nearest_points) * nearest_normals
    ).sum(dim=2)
    inside = signed_distance < 0.0
    if depth_mode == "point_to_plane":
        depth = torch.relu(-signed_distance)
    else:
        depth = inside.to(nearest_distance.dtype) * nearest_distance
    weighted_depth = confidence * depth
    if confidence_mode == "weighted":
        optimization_depth = weighted_depth
        confidence_sum = confidence.sum(dim=1).clamp_min(1.0e-6)
        mean_depth = optimization_depth.sum(dim=1) / confidence_sum
    else:
        optimization_depth = depth
        mean_depth = depth.mean(dim=1)

    tail_count = max(
        1, int(np.ceil(float(cvar_fraction) * hand_surface_points.shape[1]))
    )
    cvar_depth = optimization_depth.topk(tail_count, dim=1).values.mean(dim=1)
    excess = torch.relu(depth - float(hinge_threshold_m))
    hinge_depth = excess.topk(tail_count, dim=1).values.mean(dim=1)
    if aggregation == "mean":
        base_energy = mean_depth
    elif aggregation == "max":
        base_energy = optimization_depth.max(dim=1).values
    else:
        base_energy = mean_depth + float(cvar_weight) * cvar_depth
    energy = base_energy + float(hinge_weight) * hinge_depth
    raw_max = depth.max(dim=1).values
    confidence_weighted_max = weighted_depth.max(dim=1).values
    selected_max = (
        raw_max if gate_metric == "raw" else confidence_weighted_max
    )

    return {
        "energy": energy,
        "mean": mean_depth,
        "cvar": cvar_depth,
        "hinge": hinge_depth,
        "max": selected_max,
        "raw_max": raw_max,
        "confidence_weighted_max": confidence_weighted_max,
        "fraction": (depth > 0.0).float().mean(dim=1),
        "mean_confidence": confidence.mean(dim=1),
        "nearest_distance": nearest_distance,
    }


__all__ = [
    "estimate_point_cloud_geometry",
    "load_xyz_point_cloud",
    "point_cloud_penetration_energy",
    "prepare_point_cloud_inputs",
]
