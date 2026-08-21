"""Differentiable multi-gripper FK optimization for generated contact sets."""

from __future__ import annotations

import hashlib
import json
import itertools
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Mapping, Sequence

import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn.functional as F

from utils.point_cloud_geometry import (
    estimate_point_cloud_geometry,
    point_cloud_penetration_energy,
)


def ordered_joint_values(
    values: Mapping[str, float] | Sequence[float],
    joint_names: Sequence[str],
    *,
    label: str,
) -> list[float]:
    """Resolve joint values by name while retaining legacy list support."""
    names = list(joint_names)
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        extra = [name for name in values if name not in names]
        if missing or extra:
            raise ValueError(
                f"{label} joint-name mismatch: missing={missing}, extra={extra}"
            )
        return [float(values[name]) for name in names]
    resolved = [float(value) for value in values]
    if len(resolved) != len(names):
        raise ValueError(
            f"{label} contains {len(resolved)} values for {len(names)} joints"
        )
    return resolved


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert Zhou et al. 6D rotations to right-handed 3x3 matrices."""
    first = F.normalize(rotation_6d[..., :3], dim=-1)
    second_raw = rotation_6d[..., 3:]
    second = F.normalize(second_raw - (first * second_raw).sum(-1, keepdim=True) * first, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def random_rotation_6d(count: int, *, device, dtype, generator=None) -> torch.Tensor:
    matrices = torch.randn(count, 3, 3, device=device, dtype=dtype, generator=generator)
    q, _ = torch.linalg.qr(matrices)
    determinant = torch.linalg.det(q)
    q[:, :, 2] *= torch.where(determinant < 0, -1.0, 1.0).unsqueeze(1)
    return torch.cat((q[:, :, 0], q[:, :, 1]), dim=-1)


def kabsch_rotation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return batched column-vector rotations aligning source to target."""
    source_centered = source - source.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    covariance = source_centered.transpose(1, 2) @ target_centered
    left, _, right_h = torch.linalg.svd(covariance)
    correction = torch.ones(
        source.shape[0], 3, device=source.device, dtype=source.dtype
    )
    correction[:, -1] = torch.sign(torch.linalg.det(left @ right_h))
    row_rotation = left @ torch.diag_embed(correction) @ right_h
    return row_rotation.transpose(1, 2)


def cedex_center_facing_rotations(
    root_positions: torch.Tensor,
    object_center: torch.Tensor,
    local_palm_axis: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Adapt CEDex's center-facing pose initialization to column rotations.

    CEDex first points a hand-specific palm axis toward the object center and
    then samples a free roll about that approach direction.  Constructing the
    mapping from orthonormal bases avoids depending on CEDex's row-vector
    rotation convention.
    """
    count = root_positions.shape[0]
    approach = object_center.unsqueeze(0) - root_positions
    approach = F.normalize(approach, dim=1)
    local_axis = F.normalize(local_palm_axis.reshape(3), dim=0)

    def perpendicular_reference(vectors: torch.Tensor) -> torch.Tensor:
        x_axis = torch.zeros_like(vectors)
        x_axis[:, 0] = 1.0
        y_axis = torch.zeros_like(vectors)
        y_axis[:, 1] = 1.0
        use_y = torch.abs((vectors * x_axis).sum(dim=1)) > 0.95
        return torch.where(use_y[:, None], y_axis, x_axis)

    local = local_axis.unsqueeze(0).expand(count, -1)
    local_first = F.normalize(
        torch.cross(perpendicular_reference(local), local, dim=1), dim=1
    )
    local_second = torch.cross(local, local_first, dim=1)
    local_basis = torch.stack((local_first, local_second, local), dim=2)

    world_first = F.normalize(
        torch.cross(perpendicular_reference(approach), approach, dim=1), dim=1
    )
    world_second = torch.cross(approach, world_first, dim=1)
    world_basis = torch.stack((world_first, world_second, approach), dim=2)
    base_rotation = world_basis @ local_basis.transpose(1, 2)

    roll = 2.0 * torch.pi * torch.rand(
        count,
        device=root_positions.device,
        dtype=root_positions.dtype,
        generator=generator,
    )
    cosine = torch.cos(roll)
    sine = torch.sin(roll)
    one_minus_cosine = 1.0 - cosine
    x, y, z = approach.unbind(dim=1)
    roll_rotation = torch.stack(
        (
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ),
        dim=1,
    ).reshape(count, 3, 3)
    return roll_rotation @ base_rotation


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return the URDF fixed-axis roll/pitch/yaw rotation matrix."""
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _parse_vector(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    parsed = np.fromstring(value, sep=" ", dtype=np.float64)
    if parsed.size != len(default):
        raise ValueError(f"Expected {len(default)} values, got {value!r}")
    return parsed


def _resolve_urdf_mesh(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("file://"):
        filename = filename[7:]
    if filename.startswith("package://"):
        filename = filename[len("package://") :]
    candidate = Path(filename)
    if candidate.is_absolute():
        return candidate
    direct = (urdf_path.parent / candidate).resolve()
    if direct.is_file():
        return direct
    # Some exported URDFs use package://<package>/meshes/... while the URDF
    # itself already lives in <package>. Try progressively removing the
    # package prefix without relying on a ROS package index.
    parts = candidate.parts
    for offset in range(1, len(parts)):
        fallback = (urdf_path.parent / Path(*parts[offset:])).resolve()
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(f"URDF mesh does not exist: {filename} (from {urdf_path})")


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                polygon = []
                for field in line.split()[1:]:
                    index = int(field.split("/", 1)[0])
                    polygon.append(index - 1 if index > 0 else len(vertices) + index)
                for offset in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[offset], polygon[offset + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    triangle_count = int.from_bytes(payload[80:84], "little") if len(payload) >= 84 else -1
    if triangle_count >= 0 and len(payload) == 84 + 50 * triangle_count:
        record_dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        )
        records = np.frombuffer(
            payload, dtype=record_dtype, count=triangle_count, offset=84
        )
        vertices = records["vertices"].astype(np.float64).reshape(-1, 3)
    else:
        vertices = np.asarray(
            [
                [float(value) for value in line.strip().split()[1:4]]
                for line in payload.decode("utf-8", errors="ignore").splitlines()
                if line.strip().lower().startswith("vertex ")
            ],
            dtype=np.float64,
        )
    if len(vertices) % 3:
        raise ValueError(f"Invalid STL triangle vertex count in {path}")
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _load_triangle_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        vertices, faces = _load_obj(path)
    elif suffix == ".stl":
        vertices, faces = _load_stl(path)
    else:
        raise ValueError(
            f"Unsupported mesh format {suffix!r} for differentiable collision sampling: {path}"
        )
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh has no triangle geometry: {path}")
    return vertices, faces


def _box_mesh(extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = extents / 2.0
    vertices = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _cylinder_mesh(radius: float, height: float, sections: int = 24) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    ring = np.stack((radius * np.cos(angles), radius * np.sin(angles)), axis=1)
    vertices = np.concatenate(
        (
            np.column_stack((ring, np.full(sections, -height / 2.0))),
            np.column_stack((ring, np.full(sections, height / 2.0))),
            np.asarray([[0.0, 0.0, -height / 2.0], [0.0, 0.0, height / 2.0]]),
        ),
        axis=0,
    )
    bottom_center, top_center = 2 * sections, 2 * sections + 1
    faces = []
    for index in range(sections):
        nxt = (index + 1) % sections
        faces.extend(
            [
                [index, nxt, sections + nxt],
                [index, sections + nxt, sections + index],
                [bottom_center, nxt, index],
                [top_center, sections + index, sections + nxt],
            ]
        )
    return vertices.astype(np.float64), np.asarray(faces, dtype=np.int64)


def _sphere_mesh(radius: float, latitudes: int = 8, longitudes: int = 16) -> tuple[np.ndarray, np.ndarray]:
    vertices = [[0.0, 0.0, radius], [0.0, 0.0, -radius]]
    for latitude in range(1, latitudes):
        phi = np.pi * latitude / latitudes
        for longitude in range(longitudes):
            theta = 2.0 * np.pi * longitude / longitudes
            vertices.append(
                [
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi),
                ]
            )
    faces = []
    first_ring = 2
    for longitude in range(longitudes):
        nxt = (longitude + 1) % longitudes
        faces.append([0, first_ring + longitude, first_ring + nxt])
    for latitude in range(latitudes - 2):
        ring = first_ring + latitude * longitudes
        next_ring = ring + longitudes
        for longitude in range(longitudes):
            nxt = (longitude + 1) % longitudes
            faces.extend(
                [
                    [ring + longitude, next_ring + longitude, next_ring + nxt],
                    [ring + longitude, next_ring + nxt, ring + nxt],
                ]
            )
    last_ring = first_ring + (latitudes - 2) * longitudes
    for longitude in range(longitudes):
        nxt = (longitude + 1) % longitudes
        faces.append([1, last_ring + nxt, last_ring + longitude])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically sample triangle points and corresponding face normals."""
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    if triangles.size == 0:
        if vertices.size == 0:
            raise ValueError("Mesh has no vertices")
        indices = generator.integers(0, len(vertices), size=count)
        points = vertices[indices]
        normals = points - vertices.mean(axis=0, keepdims=True)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
        return points, normals
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    probabilities = double_area / double_area.sum() if double_area.sum() > 0 else None
    face_indices = generator.choice(len(triangles), size=count, replace=True, p=probabilities)
    selected = triangles[face_indices]
    uv = generator.random((count, 2))
    sqrt_u = np.sqrt(uv[:, :1])
    points = (
        (1.0 - sqrt_u) * selected[:, 0]
        + sqrt_u * (1.0 - uv[:, 1:]) * selected[:, 1]
        + sqrt_u * uv[:, 1:] * selected[:, 2]
    )
    selected_cross = cross[face_indices]
    normals = selected_cross / np.linalg.norm(
        selected_cross, axis=1, keepdims=True
    ).clip(min=1e-12)
    return points, normals


def _geometry_mesh(
    geometry: ET.Element, urdf_path: Path
) -> tuple[np.ndarray, np.ndarray] | None:
    mesh_element = geometry.find("mesh")
    if mesh_element is not None:
        filename = mesh_element.get("filename")
        if not filename:
            return None
        vertices, faces = _load_triangle_mesh(_resolve_urdf_mesh(urdf_path, filename))
        scale = _parse_vector(mesh_element.get("scale"), (1.0, 1.0, 1.0))
        return vertices * scale[None, :], faces
    box = geometry.find("box")
    if box is not None:
        return _box_mesh(_parse_vector(box.get("size"), (1.0, 1.0, 1.0)))
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return _cylinder_mesh(
            radius=float(cylinder.get("radius", "0")),
            height=float(cylinder.get("length", "0")),
        )
    sphere = geometry.find("sphere")
    if sphere is not None:
        return _sphere_mesh(radius=float(sphere.get("radius", "0")))
    return None


def load_urdf_surface_points(
    urdf_path: str | Path,
    points_per_link: int,
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sample collision (or fallback visual) geometry in each URDF link frame."""
    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    sampled: dict[str, np.ndarray] = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        geometries = link.findall("collision")
        if not geometries:
            geometries = link.findall("visual")
        geometries = [element for element in geometries if element.find("geometry") is not None]
        if not geometries:
            continue
        link_seed = int(seed) + zlib.crc32(link_name.encode("utf-8"))
        generator = np.random.default_rng(link_seed)
        count_each = max(1, math.ceil(int(points_per_link) / len(geometries)))
        link_points = []
        for element in geometries:
            mesh_data = _geometry_mesh(element.find("geometry"), urdf_path)
            if mesh_data is None:
                continue
            points, _ = _sample_mesh_surface(*mesh_data, count_each, generator)
            origin = element.find("origin")
            xyz = _parse_vector(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = _parse_vector(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            link_points.append(points @ _rpy_matrix(rpy).T + xyz[None, :])
        if link_points:
            points = np.concatenate(link_points, axis=0)
            if len(points) > int(points_per_link):
                indices = np.linspace(0, len(points) - 1, int(points_per_link), dtype=np.int64)
                points = points[indices]
            sampled[link_name] = points.astype(np.float32)
    if not sampled:
        raise ValueError(f"No collision or visual surface geometry found in {urdf_path}")
    return sampled


def gendex_penetration_energy(
    hand_surface_points: torch.Tensor,
    object_surface_points: torch.Tensor,
    object_surface_normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GenDexGrasp nearest-point normal penetration energy.

    A hand point is inside when ``(object_point - hand_point) dot normal > 0``.
    The returned tensors contain mean penetration, max penetration and the
    penetrating surface-point fraction for each particle.
    """
    if object_surface_points.ndim != 2 or object_surface_points.shape[-1] != 3:
        raise ValueError("object_surface_points must have shape [M, 3]")
    if object_surface_normals.shape != object_surface_points.shape:
        raise ValueError("object_surface_normals must match object_surface_points")
    distances = torch.cdist(hand_surface_points, object_surface_points.unsqueeze(0))
    nearest_distance, nearest_index = distances.min(dim=2)
    nearest_points = object_surface_points[nearest_index]
    nearest_normals = object_surface_normals[nearest_index]
    inside = ((nearest_points - hand_surface_points) * nearest_normals).sum(dim=2) > 0
    depth = nearest_distance * inside.to(nearest_distance.dtype)
    return depth.mean(dim=1), depth.max(dim=1).values, inside.float().mean(dim=1)


def self_collision_clearance_energy(
    link_points: list[torch.Tensor],
    collision_pairs: list[tuple[int, int]],
    clearance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize close surfaces on non-adjacent hand links.

    Each item in ``link_points`` has shape ``[B, N, 3]``.  The loss is a
    differentiable surface-clearance proxy: zero when a configured link pair
    is at least ``clearance`` apart and positive as the surfaces approach or
    intersect. Directly connected links and links from the same finger are
    excluded when ``collision_pairs`` is constructed.
    """
    if not link_points:
        raise ValueError("Self-collision energy requires sampled link surfaces")
    batch = link_points[0].shape[0]
    if not collision_pairs:
        zeros = torch.zeros(
            batch, device=link_points[0].device, dtype=link_points[0].dtype
        )
        return zeros, zeros, zeros
    violations = []
    for first, second in collision_pairs:
        minimum_distance = torch.cdist(
            link_points[first], link_points[second]
        ).flatten(1).min(dim=1).values
        violations.append(F.relu(float(clearance) - minimum_distance))
    stacked = torch.stack(violations, dim=1)
    return (
        stacked.mean(dim=1),
        stacked.max(dim=1).values,
        (stacked > 0).float().mean(dim=1),
    )


def self_collision_mean_cvar_energy(
    link_points: list[torch.Tensor],
    collision_pairs: list[tuple[int, int]],
    clearance: float,
    *,
    cvar_fraction: float = 0.25,
    cvar_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combine broad self-clearance violations with their worst CVaR tail."""

    if not 0.0 < float(cvar_fraction) <= 1.0:
        raise ValueError("self-collision cvar_fraction must be in (0, 1]")
    if float(cvar_weight) < 0.0:
        raise ValueError("self-collision cvar_weight must be non-negative")
    if not link_points:
        raise ValueError("Self-collision energy requires sampled link surfaces")
    batch = link_points[0].shape[0]
    if not collision_pairs:
        zeros = torch.zeros(
            batch, device=link_points[0].device, dtype=link_points[0].dtype
        )
        return {
            "energy": zeros,
            "mean": zeros,
            "cvar": zeros,
            "max": zeros,
            "fraction": zeros,
        }
    violations = []
    for first, second in collision_pairs:
        minimum_distance = torch.cdist(
            link_points[first], link_points[second]
        ).flatten(1).min(dim=1).values
        violations.append(F.relu(float(clearance) - minimum_distance))
    stacked = torch.stack(violations, dim=1)
    tail_count = max(
        1, int(math.ceil(float(cvar_fraction) * stacked.shape[1]))
    )
    mean = stacked.mean(dim=1)
    cvar = stacked.topk(tail_count, dim=1).values.mean(dim=1)
    return {
        "energy": mean + float(cvar_weight) * cvar,
        "mean": mean,
        "cvar": cvar,
        "max": stacked.max(dim=1).values,
        "fraction": (stacked > 0).float().mean(dim=1),
    }


def nearest_surface_contact(
    query_points: torch.Tensor,
    object_surface_points: torch.Tensor,
    object_surface_normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closest oriented object-surface sample for each query-point batch.

    The returned distance is an oriented point-SDF approximation: positive
    outside the object and negative inside.  The closest query point is
    selected by Euclidean surface distance, so it remains meaningful near
    curved palm geometry.
    """
    distances = torch.cdist(
        query_points, object_surface_points.unsqueeze(0)
    )
    nearest_distance, nearest_index = distances.min(dim=2)
    palm_index = nearest_distance.argmin(dim=1)
    batch_index = torch.arange(query_points.shape[0], device=query_points.device)
    object_index = nearest_index[batch_index, palm_index]
    hand_contact = query_points[batch_index, palm_index]
    object_contact = object_surface_points[object_index]
    object_normal = F.normalize(object_surface_normals[object_index], dim=1)
    signed_distance = (
        (hand_contact - object_contact) * object_normal
    ).sum(dim=1)
    return signed_distance, hand_contact, object_contact, object_normal


def nearest_surface_distance(
    query_points: torch.Tensor,
    object_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unsigned closest distance between a batched hand surface and XYZ cloud."""

    distances = torch.cdist(query_points, object_points.unsqueeze(0))
    nearest_per_query, object_indices = distances.min(dim=2)
    distance, query_indices = nearest_per_query.min(dim=1)
    batch_indices = torch.arange(query_points.shape[0], device=query_points.device)
    object_index = object_indices[batch_indices, query_indices]
    return (
        distance,
        query_points[batch_indices, query_indices],
        object_points[object_index],
        object_index,
    )


def dexgraspnet_dfc_energy(
    contact_points: torch.Tensor,
    inward_normals: torch.Tensor,
    object_center: torch.Tensor,
    torque_scale: torch.Tensor,
    contact_activation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lightweight DexGraspNet DFC: squared equal-normal resultant wrench."""
    if contact_activation is None:
        contact_activation = torch.ones(
            contact_points.shape[:2],
            device=contact_points.device,
            dtype=contact_points.dtype,
        )
    weighted_normals = inward_normals * contact_activation.unsqueeze(2)
    centered = (
        contact_points - object_center.reshape(1, 1, 3)
    ) / torque_scale.clamp_min(1.0e-6)
    force = weighted_normals.sum(dim=1)
    torque = torch.cross(centered, weighted_normals, dim=2).sum(dim=1)
    contact_count = float(contact_points.shape[1])
    return (
        force.square().sum(dim=1) + torque.square().sum(dim=1)
    ) / (contact_count * contact_count)


def paper_aligned_graspqp_energy(
    contact_points: torch.Tensor,
    outward_normals: torch.Tensor,
    object_center: torch.Tensor,
    *,
    friction_coefficient: float = 0.2,
    cone_edges: int = 4,
    iterations: int = 20,
    max_force_coefficient: float = 50.0,
    torque_weight: float = 5.0,
    svd_gain: float = 0.1,
    normal_confidence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Point-cloud adaptation of the constrained GraspQP energy.

    GraspQP minimizes ``0.5 ||W gamma||^2`` with a strictly positive
    coefficient for every friction-cone ray.  The lower bound is the crucial
    distinction from an activation-gated DFC loss: a hand cannot improve this
    energy by moving away and turning all contacts off.  We unroll projected
    gradient iterations instead of adding qpth as a runtime dependency, so the
    result remains differentiable with respect to FK contact locations.

    The defaults follow the paper/repository: four cone rays, ``mu=0.2``,
    ``1 <= gamma <= 50``, torque weight 5, and SVD gain 0.1.  Point-cloud
    normal confidence may attenuate a ray, but has a 0.5 floor and therefore
    never disables a required contact.
    """
    if contact_points.ndim != 3 or contact_points.shape[-1] != 3:
        raise ValueError("contact_points must have shape [B, C, 3]")
    if outward_normals.shape != contact_points.shape:
        raise ValueError("outward_normals must match contact_points")
    if int(cone_edges) < 3:
        raise ValueError("cone_edges must be at least 3")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < float(friction_coefficient) < 1.0:
        raise ValueError("friction_coefficient must be in (0, 1)")
    if float(max_force_coefficient) <= 1.0:
        raise ValueError("max_force_coefficient must be greater than 1")

    # Object SDF normals point outwards; contact forces on the object point
    # inwards.  Build a robust orthonormal tangent frame for every contact.
    normal = -F.normalize(outward_normals, dim=2)
    reference_x = torch.zeros_like(normal)
    reference_x[..., 0] = 1.0
    reference_y = torch.zeros_like(normal)
    reference_y[..., 1] = 1.0
    reference = torch.where(
        (normal[..., 0].abs() > 0.9).unsqueeze(2),
        reference_y,
        reference_x,
    )
    tangent_one = F.normalize(torch.cross(normal, reference, dim=2), dim=2)
    tangent_two = torch.cross(normal, tangent_one, dim=2)
    angles = torch.arange(
        int(cone_edges), device=normal.device, dtype=normal.dtype
    ) * (2.0 * math.pi / float(cone_edges))
    tangent = (
        torch.cos(angles)[None, None, :, None] * tangent_one[:, :, None, :]
        + torch.sin(angles)[None, None, :, None] * tangent_two[:, :, None, :]
    )
    rays = (
        math.sqrt(1.0 - float(friction_coefficient) ** 2)
        * normal[:, :, None, :]
        + float(friction_coefficient) * tangent
    ) / float(cone_edges)
    if normal_confidence is not None:
        if normal_confidence.shape != contact_points.shape[:2]:
            raise ValueError("normal_confidence must have shape [B, C]")
        confidence_scale = 0.5 + 0.5 * normal_confidence.clamp(0.0, 1.0)
        rays = rays * confidence_scale[:, :, None, None]

    lever = contact_points - object_center.reshape(1, 1, 3)
    torque = torch.cross(
        lever[:, :, None, :].expand_as(rays), rays, dim=3
    ) * float(torque_weight)
    wrench = torch.cat((rays, torque), dim=3).reshape(
        contact_points.shape[0], -1, 6
    ).transpose(1, 2)

    gram = torch.bmm(wrench.transpose(1, 2), wrench)
    coefficients = torch.full(
        (wrench.shape[0], wrench.shape[2]),
        1.5,
        device=wrench.device,
        dtype=wrench.dtype,
    )
    # The Frobenius norm upper-bounds the spectral norm, yielding a stable
    # batch-specific step without an eigendecomposition in every FK step.
    step_size = 0.9 / gram.square().sum(dim=(1, 2)).sqrt().clamp_min(1.0e-6)
    for _ in range(int(iterations)):
        gradient = torch.bmm(gram, coefficients.unsqueeze(2)).squeeze(2)
        coefficients = torch.clamp(
            coefficients - step_size[:, None] * gradient,
            min=1.0,
            max=float(max_force_coefficient),
        )

    resultant = torch.bmm(wrench, coefficients.unsqueeze(2)).squeeze(2)
    residual = 0.5 * resultant.square().sum(dim=1)
    singular_values = torch.linalg.svdvals(wrench)
    log_geometric_mean = torch.log(
        singular_values.clamp_min(1.0e-8)
    ).mean(dim=1)
    svd_geometric_mean = torch.exp(log_geometric_mean)
    energy = 2.0 * (residual + 1.0e-2) * torch.exp(
        -float(svd_gain) * svd_geometric_mean
    )
    return {
        "energy": energy,
        "residual": residual,
        "minimum_singular_value": singular_values[:, -1],
        "svd_geometric_mean": svd_geometric_mean,
        "mean_force_coefficient": coefficients.mean(dim=1),
        "max_force_coefficient": coefficients.max(dim=1).values,
    }


def graspqp_friction_cone_metrics(
    contact_points: torch.Tensor,
    outward_normals: torch.Tensor,
    object_center: torch.Tensor,
    torque_scale: torch.Tensor,
    *,
    friction_coefficient: float = 0.8,
    cone_edges: int = 8,
    iterations: int = 80,
    contact_activation: torch.Tensor | None = None,
    include_torque_disturbances: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GraspQP-style friction-cone QP residual and wrench-rank metrics.

    Each friction cone is linearized into rays.  Six non-negative QPs solve
    min ||W lambda + w_ext||^2 for +/-X, +/-Y and +/-Z unit disturbances; the
    worst residual ranks the candidate.  Isaac's six-direction 2 cm test
    remains authoritative.
    """
    normal = F.normalize(outward_normals, dim=2)
    reference_x = torch.zeros_like(normal)
    reference_x[..., 0] = 1.0
    reference_y = torch.zeros_like(normal)
    reference_y[..., 1] = 1.0
    use_y = normal[..., 0].abs() > 0.9
    reference = torch.where(use_y.unsqueeze(2), reference_y, reference_x)
    tangent_one = F.normalize(torch.cross(normal, reference, dim=2), dim=2)
    tangent_two = torch.cross(normal, tangent_one, dim=2)
    angles = torch.arange(
        int(cone_edges), device=normal.device, dtype=normal.dtype
    ) * (2.0 * math.pi / float(cone_edges))
    tangential = (
        torch.cos(angles)[None, None, :, None] * tangent_one[:, :, None, :]
        + torch.sin(angles)[None, None, :, None] * tangent_two[:, :, None, :]
    )
    rays = F.normalize(
        -normal[:, :, None, :]
        + float(friction_coefficient) * tangential,
        dim=3,
    )
    if contact_activation is not None:
        if contact_activation.shape != contact_points.shape[:2]:
            raise ValueError("contact_activation must have shape [B, C]")
        rays = rays * contact_activation[:, :, None, None]
    lever = (
        contact_points - object_center.reshape(1, 1, 3)
    ) / torque_scale.clamp_min(1.0e-6)
    torque = torch.cross(lever[:, :, None, :].expand_as(rays), rays, dim=3)
    wrench = torch.cat((rays, torque), dim=3).reshape(
        contact_points.shape[0], -1, 6
    ).transpose(1, 2)
    gram = torch.bmm(wrench.transpose(1, 2), wrench)
    disturbance_count = 12 if include_torque_disturbances else 6
    disturbances = torch.zeros(
        (disturbance_count, 6), device=wrench.device, dtype=wrench.dtype
    )
    for axis in range(3):
        disturbances[2 * axis, axis] = 1.0
        disturbances[2 * axis + 1, axis] = -1.0
    if include_torque_disturbances:
        for axis in range(3):
            disturbances[6 + 2 * axis, 3 + axis] = 1.0
            disturbances[6 + 2 * axis + 1, 3 + axis] = -1.0
    coefficients = torch.zeros(
        (wrench.shape[0], disturbance_count, wrench.shape[2]),
        device=wrench.device,
        dtype=wrench.dtype,
    )
    # A conservative batch step size from the Frobenius norm avoids an
    # expensive eigendecomposition during ranking.
    step_size = 0.45 / gram.square().sum(dim=(1, 2)).sqrt().clamp_min(1.0e-6)
    for _ in range(int(iterations)):
        response = torch.einsum("bwr,bdr->bdw", wrench, coefficients)
        wrench_error = response + disturbances.unsqueeze(0)
        gradient = (
            2.0 * torch.einsum("bwr,bdw->bdr", wrench, wrench_error)
            + 1.0e-3 * coefficients
        )
        coefficients = torch.clamp(
            coefficients - step_size[:, None, None] * gradient,
            min=0.0,
        )
    response = torch.einsum("bwr,bdr->bdw", wrench, coefficients)
    directional_residual = (
        response + disturbances.unsqueeze(0)
    ).square().sum(dim=2)
    residual = directional_residual.max(dim=1).values
    singular_values = torch.linalg.svdvals(wrench)
    minimum_singular_value = singular_values[:, -1]
    log_volume = torch.log(singular_values.clamp_min(1.0e-6)).sum(dim=1)
    # This stabilized form follows GraspQP's residual/rank-volume principle.
    score = residual * torch.exp(-log_volume.clamp(min=-12.0, max=12.0))
    return score, residual, minimum_singular_value


def realized_surface_contact_regions(
    region_surface_points: torch.Tensor,
    object_points: torch.Tensor,
    object_normals: torch.Tensor,
    object_normal_confidence: torch.Tensor,
    *,
    gap_sigma_m: float,
) -> dict[str, torch.Tensor]:
    """Extract one execution-aware point-cloud contact per hand region.

    ``region_surface_points`` has shape ``[B, C, S, 3]``.  Each region uses
    its closest hand/object surface pair.  The selected hand point keeps a
    useful piecewise-smooth FK gradient, while the point-cloud normal and its
    PCA confidence define the local friction cone.  The Gaussian gap gate
    prevents distant target contacts from pretending to be load-bearing.
    """
    if region_surface_points.ndim != 4 or region_surface_points.shape[-1] != 3:
        raise ValueError("region_surface_points must have shape [B, C, S, 3]")
    if float(gap_sigma_m) <= 0.0:
        raise ValueError("gap_sigma_m must be positive")
    batch, regions, samples, _ = region_surface_points.shape
    flat = region_surface_points.reshape(batch, regions * samples, 3)
    distances = torch.cdist(flat, object_points.unsqueeze(0))
    nearest_gap, nearest_object_index = distances.min(dim=2)
    nearest_gap = nearest_gap.reshape(batch, regions, samples)
    nearest_object_index = nearest_object_index.reshape(batch, regions, samples)
    gap, surface_index = nearest_gap.min(dim=2)
    batch_index = torch.arange(batch, device=flat.device)[:, None]
    region_index = torch.arange(regions, device=flat.device)[None, :]
    hand_contact = region_surface_points[
        batch_index, region_index, surface_index
    ]
    object_index = nearest_object_index[
        batch_index, region_index, surface_index
    ]
    object_contact = object_points[object_index]
    outward_normal = F.normalize(object_normals[object_index], dim=2)
    confidence = object_normal_confidence[object_index].clamp(0.0, 1.0)
    activation = torch.exp(
        -0.5 * (gap / float(gap_sigma_m)).square()
    ) * (0.25 + 0.75 * confidence)
    return {
        "hand_contact": hand_contact,
        "object_contact": object_contact,
        "outward_normal": outward_normal,
        "normal_confidence": confidence,
        "gap": gap,
        "activation": activation,
    }


def execution_aware_force_closure_energy(
    contact_points: torch.Tensor,
    outward_normals: torch.Tensor,
    contact_activation: torch.Tensor,
    object_center: torch.Tensor,
    torque_scale: torch.Tensor,
    *,
    friction_coefficient: float,
    cone_edges: int,
    qp_iterations: int,
    qp_weight: float,
    dfc_weight: float,
    coverage_weight: float,
    formulation: str = "legacy",
    normal_confidence: torch.Tensor | None = None,
    max_force_coefficient: float = 50.0,
    torque_weight: float = 5.0,
    svd_gain: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Point-cloud FC energy with legacy and paper-aligned formulations."""
    if formulation not in {"legacy", "dexgraspnet", "graspqp"}:
        raise ValueError("formulation must be legacy, dexgraspnet, or graspqp")
    paper_qp = None
    if formulation == "graspqp":
        paper_qp = paper_aligned_graspqp_energy(
            contact_points,
            outward_normals,
            object_center,
            friction_coefficient=friction_coefficient,
            cone_edges=cone_edges,
            iterations=qp_iterations,
            max_force_coefficient=max_force_coefficient,
            torque_weight=torque_weight,
            svd_gain=svd_gain,
            normal_confidence=normal_confidence,
        )
        qp_score = paper_qp["energy"]
        qp_residual = paper_qp["residual"]
        minimum_singular_value = paper_qp["minimum_singular_value"]
    elif formulation == "legacy" and float(qp_weight) > 0.0:
        qp_score, qp_residual, minimum_singular_value = (
            graspqp_friction_cone_metrics(
                contact_points,
                outward_normals,
                object_center,
                torque_scale,
                friction_coefficient=friction_coefficient,
                cone_edges=cone_edges,
                iterations=qp_iterations,
                contact_activation=contact_activation,
                include_torque_disturbances=True,
            )
        )
    else:
        qp_score = torch.zeros(
            contact_points.shape[0],
            device=contact_points.device,
            dtype=contact_points.dtype,
        )
        qp_residual = torch.zeros_like(qp_score)
        minimum_singular_value = torch.zeros_like(qp_score)
    if formulation == "dexgraspnet":
        # Official DexGraspNet uses every selected contact with equal unit
        # force; it has no distance activation that can collapse to zero.
        centered = contact_points - object_center.reshape(1, 1, 3)
        inward = -F.normalize(outward_normals, dim=2)
        force = inward.sum(dim=1)
        torque = torch.cross(centered, inward, dim=2).sum(dim=1)
        dfc = force.square().sum(dim=1) + (
            float(torque_weight) * torque
        ).square().sum(dim=1)
    else:
        dfc = dexgraspnet_dfc_energy(
            contact_points,
            -outward_normals,
            object_center,
            torque_scale,
            contact_activation=(
                contact_activation if formulation == "legacy" else None
            ),
        )
    coverage = (1.0 - contact_activation).square().mean(dim=1)
    if formulation == "graspqp":
        energy = float(qp_weight) * qp_score
    elif formulation == "dexgraspnet":
        energy = float(dfc_weight) * dfc
    else:
        energy = (
            float(qp_weight) * qp_residual
            + float(dfc_weight) * dfc
            + float(coverage_weight) * coverage
        )
    zeros = torch.zeros_like(energy)
    return {
        "energy": energy,
        "qp_score": qp_score,
        "qp_residual": qp_residual,
        "minimum_singular_value": minimum_singular_value,
        "dfc": dfc,
        "coverage": coverage,
        "mean_activation": contact_activation.mean(dim=1),
        "svd_geometric_mean": (
            paper_qp["svd_geometric_mean"] if paper_qp is not None else zeros
        ),
        "mean_force_coefficient": (
            paper_qp["mean_force_coefficient"] if paper_qp is not None else zeros
        ),
        "max_force_coefficient": (
            paper_qp["max_force_coefficient"] if paper_qp is not None else zeros
        ),
    }


def blend_main_and_force_closure_energy(
    main_energy: torch.Tensor,
    force_closure_energy: torch.Tensor,
    *,
    force_closure_weight: float,
    force_closure_ramp: float,
    target_fraction: float | None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Combine losses using a raw weight or a detached target contribution."""
    if target_fraction is None:
        scale = torch.ones((), device=main_energy.device, dtype=main_energy.dtype)
        return (
            main_energy
            + float(force_closure_weight)
            * float(force_closure_ramp)
            * force_closure_energy,
            scale,
            0.0,
        )
    effective_fraction = float(target_fraction) * float(force_closure_ramp)
    scale = (
        main_energy.detach().mean()
        / force_closure_energy.detach().mean().clamp_min(1.0e-8)
    )
    total = (
        (1.0 - effective_fraction) * main_energy
        + effective_fraction * scale * force_closure_energy
    )
    return total, scale, effective_fraction


class DifferentiableGripper:
    def __init__(
        self,
        urdf_path: str | Path,
        tip_links: list[str],
        tip_offsets: dict[str, list[float]],
        base_alignment: list[list[float]],
        surface_points_per_link: int = 0,
        locked_joints: Mapping[str, float] | None = None,
        palm_link: str | None = None,
        palm_surface_link: str | None = None,
        *,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        self.urdf_path = Path(urdf_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.chain = pk.build_chain_from_urdf(self.urdf_path.read_bytes()).to(
            device=self.device, dtype=dtype
        )
        self.tip_links = list(tip_links)
        self.palm_link = palm_link
        self.palm_surface_link = palm_surface_link
        if self.palm_link is not None:
            frame_names = set(self.chain.get_frame_names())
            if self.palm_link not in frame_names:
                raise ValueError(
                    f"Palm link {self.palm_link!r} is not in the URDF chain"
                )
        self.joint_names = list(self.chain.get_joint_parameter_names())
        lower, upper = self.chain.get_joint_limits()
        self.lower = torch.as_tensor(lower, device=self.device, dtype=dtype)
        self.upper = torch.as_tensor(upper, device=self.device, dtype=dtype)
        invalid = ~torch.isfinite(self.lower) | ~torch.isfinite(self.upper)
        self.lower = torch.where(invalid, torch.full_like(self.lower, -torch.pi), self.lower)
        self.upper = torch.where(invalid, torch.full_like(self.upper, torch.pi), self.upper)
        self.span = (self.upper - self.lower).clamp_min(1e-6)
        locked_joints = dict(locked_joints or {})
        unknown_locked = [name for name in locked_joints if name not in self.joint_names]
        if unknown_locked:
            raise ValueError(f"Unknown locked joints: {unknown_locked}")
        self.locked_mask = torch.zeros(
            len(self.joint_names), device=self.device, dtype=dtype
        )
        self.locked_values = torch.zeros_like(self.locked_mask)
        for name, value in locked_joints.items():
            index = self.joint_names.index(name)
            numeric = float(value)
            if numeric < float(self.lower[index]) or numeric > float(self.upper[index]):
                raise ValueError(
                    f"Locked joint {name}={numeric} is outside "
                    f"[{float(self.lower[index])}, {float(self.upper[index])}]"
                )
            self.locked_mask[index] = 1.0
            self.locked_values[index] = numeric
        self.tip_offsets = torch.as_tensor(
            [tip_offsets[link] for link in self.tip_links],
            device=self.device,
            dtype=dtype,
        )
        self.base_alignment = torch.as_tensor(
            base_alignment, device=self.device, dtype=dtype
        )
        sampled_surface = (
            load_urdf_surface_points(self.urdf_path, int(surface_points_per_link))
            if int(surface_points_per_link) > 0
            else {}
        )
        frame_names = set(self.chain.get_frame_names())
        urdf_root = ET.parse(self.urdf_path).getroot()
        joint_by_child = {}
        all_link_names = {
            link.get("name")
            for link in urdf_root.findall("link")
            if link.get("name")
        }
        for joint in urdf_root.findall("joint"):
            child = joint.find("child")
            if child is not None and child.get("link"):
                joint_by_child[child.get("link")] = joint
        root_links = sorted(all_link_names - set(joint_by_child))
        self.root_link = root_links[0] if len(root_links) == 1 else None
        # pytorch_kinematics normally exposes link names as FK dictionary keys,
        # but the Barrett asset exposes intermediate movable-joint links while
        # its geometry sits on fixed descendants. Move points through each
        # fixed URDF origin until reaching a frame returned by FK.
        self.surface_link_names = []
        self.surface_points_local = {}
        self.surface_source_links: dict[str, set[str]] = {}
        self.collapsed_frame_by_urdf_link: dict[str, str] = {}

        def collapsed_frame(link_name: str) -> str:
            """Map a URDF link to the FK frame that owns it after folding."""

            frame_name = link_name
            visited = set()
            while (
                frame_name not in frame_names
                and frame_name != self.root_link
                and frame_name not in visited
            ):
                visited.add(frame_name)
                joint = joint_by_child.get(frame_name)
                if joint is None:
                    break
                parent = joint.find("parent")
                if parent is None or not parent.get("link"):
                    break
                frame_name = parent.get("link")
            return frame_name

        for link_name in all_link_names:
            self.collapsed_frame_by_urdf_link[link_name] = collapsed_frame(
                link_name
            )
        for link_name, points in sampled_surface.items():
            frame_name = link_name
            frame_points = np.asarray(points, dtype=np.float64)
            visited = set()
            while (
                frame_name not in frame_names
                and frame_name != self.root_link
                and frame_name not in visited
            ):
                visited.add(frame_name)
                joint = joint_by_child.get(frame_name)
                if joint is None:
                    break
                origin = joint.find("origin")
                xyz = _parse_vector(
                    origin.get("xyz") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                )
                rpy = _parse_vector(
                    origin.get("rpy") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                )
                frame_points = frame_points @ _rpy_matrix(rpy).T + xyz[None, :]
                parent = joint.find("parent")
                if parent is None or not parent.get("link"):
                    break
                frame_name = parent.get("link")
            if frame_name not in frame_names and frame_name != self.root_link:
                continue
            points_tensor = torch.as_tensor(
                frame_points, device=self.device, dtype=dtype
            )
            if frame_name in self.surface_points_local:
                self.surface_points_local[frame_name] = torch.cat(
                    (self.surface_points_local[frame_name], points_tensor), dim=0
                )
            else:
                self.surface_link_names.append(frame_name)
                self.surface_points_local[frame_name] = points_tensor
            self.surface_source_links.setdefault(frame_name, set()).add(link_name)
        if (
            self.palm_surface_link is not None
            and self.palm_surface_link not in self.surface_points_local
        ):
            raise ValueError(
                f"Palm surface link {self.palm_surface_link!r} has no sampled "
                "collision/visual geometry"
            )
        self.self_collision_pairs = self._build_self_collision_pairs(urdf_root)

    def _frame_matrix(
        self, transforms: Mapping[str, object], link: str, batch: int
    ) -> torch.Tensor:
        if link == self.root_link:
            return torch.eye(
                4, device=self.device, dtype=self.dtype
            ).unsqueeze(0).expand(batch, -1, -1)
        return transforms[link].get_matrix()

    @staticmethod
    def _digit_group(link_name: str) -> str | None:
        lowered = link_name.lower()
        barrett = re.fullmatch(r"bh_finger_([123])\d*_link", lowered)
        if barrett is not None:
            return f"finger_{barrett.group(1)}"
        for finger_index in ("1", "2", "3"):
            if lowered.startswith(f"finger_{finger_index}_"):
                return f"finger_{finger_index}"
        prefix = lowered[:2]
        return prefix if prefix in {"ff", "mf", "rf", "lf", "th"} else None

    @staticmethod
    def _noncolliding_base_link(link_name: str) -> bool:
        """Exclude fixed-world/forearm support geometry from hand collision."""

        return link_name.lower() in {"world", "forearm", "wrist"}

    def _build_self_collision_pairs(
        self, urdf_root: ET.Element
    ) -> list[tuple[int, int]]:
        adjacent_collapsed_frames: set[frozenset[str]] = set()
        for joint in urdf_root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if (
                parent is not None
                and child is not None
                and parent.get("link")
                and child.get("link")
            ):
                parent_frame = self.collapsed_frame_by_urdf_link.get(
                    parent.get("link"), parent.get("link")
                )
                child_frame = self.collapsed_frame_by_urdf_link.get(
                    child.get("link"), child.get("link")
                )
                if parent_frame != child_frame:
                    adjacent_collapsed_frames.add(
                        frozenset((parent_frame, child_frame))
                    )
        pairs = []
        for first in range(len(self.surface_link_names)):
            first_name = self.surface_link_names[first]
            if self._noncolliding_base_link(first_name):
                continue
            first_group = self._digit_group(first_name)
            for second in range(first + 1, len(self.surface_link_names)):
                second_name = self.surface_link_names[second]
                if self._noncolliding_base_link(second_name):
                    continue
                second_group = self._digit_group(second_name)
                # Base/forearm geometry does not self-collide for fixed wrist
                # FK, and connected surfaces naturally meet at their joint.
                if first_group is None and second_group is None:
                    continue
                if frozenset((first_name, second_name)) in adjacent_collapsed_frames:
                    continue
                # Adjacent segments of one digit share joint volume in the
                # URDF. Inter-digit and digit-vs-palm collisions are the
                # relevant invalid Shadow Hand configurations.
                if first_group is not None and first_group == second_group:
                    continue
                pairs.append((first, second))
        self.self_collision_pair_names = [
            (self.surface_link_names[first], self.surface_link_names[second])
            for first, second in pairs
        ]
        return pairs

    def constrain_joints(self, raw: torch.Tensor) -> torch.Tensor:
        joints = self.lower + torch.sigmoid(raw) * self.span
        return (
            joints * (1.0 - self.locked_mask)
            + self.locked_values * self.locked_mask
        )

    def unconstrain_joints(self, joints: torch.Tensor) -> torch.Tensor:
        # Avoid sigmoid saturation when a configured open pose lies exactly at
        # a URDF limit; otherwise the optimizer cannot move that finger closed.
        fraction = ((joints - self.lower) / self.span).clamp(1e-2, 1.0 - 1e-2)
        return torch.logit(fraction)

    def tip_points_in_aligned_base(self, joints: torch.Tensor) -> torch.Tensor:
        transforms = self.chain.forward_kinematics(joints)
        points = []
        for index, link in enumerate(self.tip_links):
            matrix = self._frame_matrix(transforms, link, joints.shape[0])
            offset = self.tip_offsets[index]
            point = torch.einsum("bij,j->bi", matrix[:, :3, :3], offset) + matrix[:, :3, 3]
            aligned = (
                torch.einsum("ij,bj->bi", self.base_alignment[:3, :3], point)
                + self.base_alignment[:3, 3]
            )
            points.append(aligned)
        return torch.stack(points, dim=1)

    def tip_points(
        self,
        joints: torch.Tensor,
        translation: torch.Tensor,
        rotation_6d: torch.Tensor,
    ) -> torch.Tensor:
        local = self.tip_points_in_aligned_base(joints)
        rotation = rotation_6d_to_matrix(rotation_6d)
        return torch.einsum("bij,bnj->bni", rotation, local) + translation[:, None, :]

    def palm_point_in_aligned_base(self, joints: torch.Tensor) -> torch.Tensor:
        if self.palm_link is None:
            return torch.zeros(
                joints.shape[0], 3, device=self.device, dtype=self.dtype
            )
        transforms = self.chain.forward_kinematics(joints)
        point = transforms[self.palm_link].get_matrix()[:, :3, 3]
        return (
            torch.einsum("ij,bj->bi", self.base_alignment[:3, :3], point)
            + self.base_alignment[None, :3, 3]
        )

    def palm_points(
        self,
        joints: torch.Tensor,
        translation: torch.Tensor,
        rotation_6d: torch.Tensor,
    ) -> torch.Tensor:
        local = self.palm_point_in_aligned_base(joints)
        rotation = rotation_6d_to_matrix(rotation_6d)
        return torch.einsum("bij,bj->bi", rotation, local) + translation

    def surface_points(
        self,
        joints: torch.Tensor,
        translation: torch.Tensor,
        rotation_6d: torch.Tensor,
    ) -> torch.Tensor:
        if not self.surface_link_names:
            raise RuntimeError("Gripper surface points were not configured")
        transforms = self.chain.forward_kinematics(joints)
        points = []
        for link in self.surface_link_names:
            matrix = transforms[link].get_matrix()
            local = self.surface_points_local[link]
            point = torch.einsum("bij,nj->bni", matrix[:, :3, :3], local)
            point = point + matrix[:, None, :3, 3]
            aligned = torch.einsum(
                "ij,bnj->bni", self.base_alignment[:3, :3], point
            ) + self.base_alignment[None, None, :3, 3]
            points.append(aligned)
        aligned_surface = torch.cat(points, dim=1)
        rotation = rotation_6d_to_matrix(rotation_6d)
        return (
            torch.einsum("bij,bnj->bni", rotation, aligned_surface)
            + translation[:, None, :]
        )

    def link_surface_points(
        self,
        link: str,
        joints: torch.Tensor,
        translation: torch.Tensor,
        rotation_6d: torch.Tensor,
    ) -> torch.Tensor:
        """Return one link's sampled physical surface as [batch, point, xyz]."""
        if link not in self.surface_points_local:
            raise RuntimeError(
                f"No sampled collision/visual surface is available for {link}"
            )
        transforms = self.chain.forward_kinematics(joints)
        matrix = self._frame_matrix(transforms, link, joints.shape[0])
        local = self.surface_points_local[link]
        point = torch.einsum("bij,nj->bni", matrix[:, :3, :3], local)
        point = point + matrix[:, None, :3, 3]
        aligned = torch.einsum(
            "ij,bnj->bni", self.base_alignment[:3, :3], point
        ) + self.base_alignment[None, None, :3, 3]
        rotation = rotation_6d_to_matrix(rotation_6d)
        return (
            torch.einsum("bij,bnj->bni", rotation, aligned)
            + translation[:, None, :]
        )

    def palm_surface_points(
        self,
        joints: torch.Tensor,
        translation: torch.Tensor,
        rotation_6d: torch.Tensor,
    ) -> torch.Tensor:
        """Return the configured physical palm surface."""
        if self.palm_surface_link is None:
            raise RuntimeError("No palm_surface_link was configured")
        return self.link_surface_points(
            self.palm_surface_link, joints, translation, rotation_6d
        )

    def tip_link_surface_points(
        self,
        joints: torch.Tensor,
        translation: torch.Tensor,
        rotation_6d: torch.Tensor,
    ) -> torch.Tensor:
        """Return sampled distal-link surfaces as [batch, finger, point, xyz]."""
        transforms = self.chain.forward_kinematics(joints)
        points = []
        for link in self.tip_links:
            if link not in self.surface_points_local:
                raise RuntimeError(
                    f"No sampled collision/visual surface is available for {link}"
                )
            matrix = self._frame_matrix(transforms, link, joints.shape[0])
            local = self.surface_points_local[link]
            point = torch.einsum("bij,nj->bni", matrix[:, :3, :3], local)
            point = point + matrix[:, None, :3, 3]
            aligned = torch.einsum(
                "ij,bnj->bni", self.base_alignment[:3, :3], point
            ) + self.base_alignment[None, None, :3, 3]
            points.append(aligned)
        aligned_surface = torch.stack(points, dim=1)
        rotation = rotation_6d_to_matrix(rotation_6d)
        return (
            torch.einsum("bij,bfnj->bfni", rotation, aligned_surface)
            + translation[:, None, None, :]
        )

    def self_collision_link_points(
        self, joints: torch.Tensor, points_per_link: int = 12
    ) -> list[torch.Tensor]:
        if not self.surface_link_names:
            raise RuntimeError("Gripper surface points were not configured")
        transforms = self.chain.forward_kinematics(joints)
        points = []
        for link in self.surface_link_names:
            matrix = self._frame_matrix(transforms, link, joints.shape[0])
            local = self.surface_points_local[link]
            count = min(max(1, int(points_per_link)), local.shape[0])
            if local.shape[0] > count:
                indices = torch.linspace(
                    0,
                    local.shape[0] - 1,
                    count,
                    device=self.device,
                ).round().long()
                local = local[indices]
            point = torch.einsum("bij,nj->bni", matrix[:, :3, :3], local)
            point = point + matrix[:, None, :3, 3]
            aligned = torch.einsum(
                "ij,bnj->bni", self.base_alignment[:3, :3], point
            ) + self.base_alignment[None, None, :3, 3]
            points.append(aligned)
        return points

    def root_pose_matrix(
        self, translation: torch.Tensor, rotation_6d: torch.Tensor
    ) -> torch.Tensor:
        batch = translation.shape[0]
        base = torch.eye(4, device=self.device, dtype=self.dtype).repeat(batch, 1, 1)
        base[:, :3, :3] = rotation_6d_to_matrix(rotation_6d)
        base[:, :3, 3] = translation
        return base @ self.base_alignment.unsqueeze(0)


def optimize_gripper_to_contacts(
    gripper: DifferentiableGripper,
    target_contacts: torch.Tensor,
    object_pc: torch.Tensor,
    initial_joints: torch.Tensor,
    *,
    particles: int = 64,
    steps: int = 300,
    learning_rate: float = 5e-3,
    contact_weight: float = 1.0,
    penetration_weight: float = 0.0,
    penetration_cvar_fraction: float = 0.1,
    penetration_cvar_weight: float = 1.0,
    penetration_depth_mode: str = "point_to_plane",
    penetration_aggregation: str = "mean_cvar",
    penetration_confidence_mode: str = "weighted",
    penetration_gate_metric: str = "confidence_weighted",
    penetration_hinge_threshold_m: float = 0.0,
    penetration_hinge_weight: float = 0.0,
    object_normal_k_neighbors: int = 30,
    self_collision_weight: float = 0.0,
    self_collision_cvar_fraction: float = 0.25,
    self_collision_cvar_weight: float = 1.0,
    self_collision_clearance: float = 0.002,
    self_collision_points_per_link: int = 12,
    joint_regularization: float = 1e-3,
    seed: int = 42,
    top_k: int = 8,
    object_normals: torch.Tensor | None = None,
    object_normal_confidence: torch.Tensor | None = None,
    initialization_mode: str = "kabsch",
    cedex_local_palm_axis: torch.Tensor | None = None,
    close_direction: torch.Tensor | None = None,
    cedex_joint_init_fraction: float = 0.5,
    cedex_cleanup_steps: int = 0,
    cedex_cleanup_learning_rate: float = 1e-3,
    cedex_contact_guard_m: float = 0.015,
    selection_max_penetration_m: float | None = None,
    envelope_approach_weight: float = 0.0,
    contact_assignment_mode: str = "chamfer",
    preferred_root_direction: torch.Tensor | None = None,
    assignment_temperature_m: float = 0.005,
    contact_geometry_mode: str = "tip_point",
    selection_min_envelope_cosine: float | None = None,
    selection_min_approach_cosine: float | None = None,
    palm_distance_weight: float = 0.0,
    palm_target_distance_m: float = 0.001,
    selection_max_palm_distance_m: float | None = None,
    graspqp_friction_coefficient: float = 0.8,
    graspqp_cone_edges: int = 8,
    graspqp_iterations: int = 80,
    force_closure_weight: float = 0.0,
    force_closure_start_fraction: float = 0.75,
    force_closure_gap_sigma_m: float = 0.01,
    force_closure_qp_iterations: int = 20,
    force_closure_qp_weight: float = 1.0,
    force_closure_dfc_weight: float = 0.1,
    force_closure_coverage_weight: float = 0.25,
    force_closure_target_fraction: float | None = None,
    force_closure_ramp_mode: str = "linear",
    force_closure_formulation: str = "legacy",
    force_closure_include_palm: bool = True,
    force_closure_friction_coefficient: float | None = None,
    force_closure_cone_edges: int | None = None,
    force_closure_max_force_coefficient: float = 50.0,
    force_closure_torque_weight: float = 5.0,
    force_closure_svd_gain: float = 0.1,
    selection_rank_mode: str = "optimization",
    initialization_contacts: torch.Tensor | None = None,
) -> dict:
    """Fit a gripper to contacts using the frozen base energy plus optional FC.

    The optimized objective contains contact fit, confidence-aware point-cloud
    penetration, self collision, palm approach, joint prior, and unsigned palm
    distance.  When enabled, execution-aware force closure is a seventh term;
    final candidate ranking remains independently controlled by
    ``selection_rank_mode``.
    """
    device, dtype = gripper.device, gripper.dtype
    # clone() converts tensors produced under torch.inference_mode() back into
    # regular tensors that autograd may safely retain for backward.
    target_contacts = target_contacts.to(device=device, dtype=dtype).detach().clone()
    initialization_contacts = (
        target_contacts
        if initialization_contacts is None
        else initialization_contacts.to(device=device, dtype=dtype).detach().clone()
    )
    object_pc = object_pc.to(device=device, dtype=dtype).detach().clone()
    if object_pc.ndim != 2 or object_pc.shape[-1] != 3 or object_pc.shape[0] < 4:
        raise ValueError("object_pc must have shape [N, 3] with N >= 4")
    geometry_diagnostics: dict[str, object]
    if object_normals is None or object_normal_confidence is None:
        estimated_geometry = estimate_point_cloud_geometry(
            object_pc, k_neighbors=int(object_normal_k_neighbors)
        )
        object_normals = torch.as_tensor(
            estimated_geometry["normals"], device=device, dtype=dtype
        )
        object_normal_confidence = torch.as_tensor(
            estimated_geometry["confidence"], device=device, dtype=dtype
        )
        geometry_diagnostics = {
            key: value
            for key, value in estimated_geometry.items()
            if key not in {"normals", "confidence", "local_scale"}
        }
        geometry_diagnostics["source"] = "estimated_from_xyz"
    else:
        object_normals = object_normals.to(
            device=device, dtype=dtype
        ).detach().clone()
        object_normal_confidence = object_normal_confidence.to(
            device=device, dtype=dtype
        ).detach().clone()
        geometry_diagnostics = {
            "source": "precomputed_from_xyz",
            "k_neighbors": int(object_normal_k_neighbors),
            "mean_confidence": float(object_normal_confidence.mean().item()),
            "median_confidence": float(object_normal_confidence.median().item()),
        }
    if object_normals.shape != object_pc.shape:
        raise ValueError("object_normals must match object_pc")
    if object_normal_confidence.shape != object_pc.shape[:1]:
        raise ValueError("object_normal_confidence must have shape [N]")
    object_surface_points = object_pc
    object_surface_normals = F.normalize(object_normals, dim=1)
    object_normal_confidence = object_normal_confidence.clamp(0.0, 1.0)
    has_collision_geometry = True
    resolved_palm_distance_weight = float(palm_distance_weight)
    initial_joints = initial_joints.to(device=device, dtype=dtype).detach().clone()
    if target_contacts.shape != (len(gripper.tip_links), 3):
        raise ValueError(
            f"Expected contacts {(len(gripper.tip_links), 3)}, got {tuple(target_contacts.shape)}"
        )
    if initialization_contacts.shape != target_contacts.shape:
        raise ValueError(
            "initialization_contacts must have the same shape as target_contacts"
        )
    if initial_joints.numel() != len(gripper.joint_names):
        raise ValueError(
            f"Expected {len(gripper.joint_names)} initial joints, got {initial_joints.numel()}"
        )

    if initialization_mode not in {"kabsch", "cedex", "enveloping"}:
        raise ValueError(
            "initialization_mode must be 'kabsch', 'cedex', or 'enveloping'"
        )
    if not 0.0 <= float(cedex_joint_init_fraction) <= 1.0:
        raise ValueError("cedex_joint_init_fraction must be in [0, 1]")
    if int(cedex_cleanup_steps) != 0:
        raise ValueError(
            "CEDex cleanup is disabled by the six-term XYZ-only base energy; "
            "set cedex_cleanup_steps=0"
        )
    if float(cedex_cleanup_learning_rate) <= 0:
        raise ValueError("cedex_cleanup_learning_rate must be positive")
    if float(cedex_contact_guard_m) < 0:
        raise ValueError("cedex_contact_guard_m must be non-negative")
    if (
        selection_max_penetration_m is not None
        and float(selection_max_penetration_m) < 0
    ):
        raise ValueError("selection_max_penetration_m must be non-negative")
    if not 0.0 < float(penetration_cvar_fraction) <= 1.0:
        raise ValueError("penetration_cvar_fraction must be in (0, 1]")
    if float(penetration_cvar_weight) < 0.0:
        raise ValueError("penetration_cvar_weight must be non-negative")
    if penetration_depth_mode not in {"point_to_plane", "signed_euclidean"}:
        raise ValueError("unsupported penetration_depth_mode")
    if penetration_aggregation not in {"mean", "mean_cvar", "max"}:
        raise ValueError("unsupported penetration_aggregation")
    if penetration_confidence_mode not in {"weighted", "none"}:
        raise ValueError("unsupported penetration_confidence_mode")
    if penetration_gate_metric not in {"confidence_weighted", "raw"}:
        raise ValueError("unsupported penetration_gate_metric")
    if float(penetration_hinge_threshold_m) < 0.0:
        raise ValueError("penetration_hinge_threshold_m must be non-negative")
    if float(penetration_hinge_weight) < 0.0:
        raise ValueError("penetration_hinge_weight must be non-negative")
    if not 0.0 < float(self_collision_cvar_fraction) <= 1.0:
        raise ValueError("self_collision_cvar_fraction must be in (0, 1]")
    if float(self_collision_cvar_weight) < 0.0:
        raise ValueError("self_collision_cvar_weight must be non-negative")
    if float(envelope_approach_weight) < 0:
        raise ValueError("envelope_approach_weight must be non-negative")
    if contact_assignment_mode not in {
        "chamfer",
        "permutation",
        "soft_permutation",
    }:
        raise ValueError(
            "contact_assignment_mode must be 'chamfer', 'permutation', "
            "or 'soft_permutation'"
        )
    if float(assignment_temperature_m) <= 0:
        raise ValueError("assignment_temperature_m must be positive")
    if contact_geometry_mode not in {"tip_point", "distal_surface"}:
        raise ValueError(
            "contact_geometry_mode must be 'tip_point' or 'distal_surface'"
        )
    if (
        contact_geometry_mode == "distal_surface"
        and contact_assignment_mode != "soft_permutation"
    ):
        raise ValueError(
            "distal_surface currently requires soft_permutation assignment"
        )
    if selection_rank_mode not in {"optimization", "graspqp"}:
        raise ValueError(
            "selection_rank_mode must be 'optimization' or 'graspqp'"
        )
    if resolved_palm_distance_weight < 0:
        raise ValueError("palm_distance_weight must be non-negative")
    if float(palm_target_distance_m) < 0:
        raise ValueError("palm_target_distance_m must be non-negative")
    if selection_max_palm_distance_m is not None and float(
        selection_max_palm_distance_m
    ) < 0:
        raise ValueError("selection_max_palm_distance_m must be non-negative")
    if not 0.0 < float(graspqp_friction_coefficient):
        raise ValueError("graspqp_friction_coefficient must be positive")
    if int(graspqp_cone_edges) < 3 or int(graspqp_iterations) < 1:
        raise ValueError("GraspQP requires >=3 cone edges and >=1 iteration")
    if float(force_closure_weight) < 0.0:
        raise ValueError("force_closure_weight must be non-negative")
    if not 0.0 <= float(force_closure_start_fraction) < 1.0:
        raise ValueError("force_closure_start_fraction must be in [0, 1)")
    if float(force_closure_gap_sigma_m) <= 0.0:
        raise ValueError("force_closure_gap_sigma_m must be positive")
    if int(force_closure_qp_iterations) < 1:
        raise ValueError("force_closure_qp_iterations must be positive")
    if float(force_closure_qp_weight) < 0.0:
        raise ValueError("force_closure_qp_weight must be non-negative")
    if float(force_closure_dfc_weight) < 0.0:
        raise ValueError("force_closure_dfc_weight must be non-negative")
    if float(force_closure_coverage_weight) < 0.0:
        raise ValueError("force_closure_coverage_weight must be non-negative")
    if force_closure_target_fraction is not None and not (
        0.0 < float(force_closure_target_fraction) < 1.0
    ):
        raise ValueError("force_closure_target_fraction must be in (0, 1)")
    if force_closure_ramp_mode not in {"linear", "constant"}:
        raise ValueError("force_closure_ramp_mode must be linear or constant")
    if force_closure_formulation not in {"legacy", "dexgraspnet", "graspqp"}:
        raise ValueError(
            "force_closure_formulation must be legacy, dexgraspnet, or graspqp"
        )
    if float(force_closure_max_force_coefficient) <= 1.0:
        raise ValueError("force_closure_max_force_coefficient must be > 1")
    if float(force_closure_torque_weight) <= 0.0:
        raise ValueError("force_closure_torque_weight must be positive")
    if float(force_closure_svd_gain) < 0.0:
        raise ValueError("force_closure_svd_gain must be non-negative")
    resolved_force_closure_friction = float(
        graspqp_friction_coefficient
        if force_closure_friction_coefficient is None
        else force_closure_friction_coefficient
    )
    resolved_force_closure_cone_edges = int(
        graspqp_cone_edges
        if force_closure_cone_edges is None
        else force_closure_cone_edges
    )
    if not 0.0 < resolved_force_closure_friction < 1.0:
        raise ValueError("force_closure_friction_coefficient must be in (0, 1)")
    if resolved_force_closure_cone_edges < 3:
        raise ValueError("force_closure_cone_edges must be at least 3")
    force_closure_enabled = (
        float(force_closure_weight) > 0.0
        or force_closure_target_fraction is not None
    )
    if force_closure_enabled and contact_geometry_mode != "distal_surface":
        raise ValueError("force closure requires distal_surface contact geometry")
    palm_constraints_enabled = (
        resolved_palm_distance_weight > 0.0
        or selection_max_palm_distance_m is not None
        or selection_rank_mode == "graspqp"
        or force_closure_enabled
    )
    if palm_constraints_enabled:
        if gripper.palm_surface_link is None:
            raise ValueError(
                "Palm distance/GraspQP requires palm_surface_link"
            )
        if contact_assignment_mode != "soft_permutation":
            raise ValueError(
                "Palm distance/GraspQP requires soft_permutation contacts"
            )
    for name, value in (
        ("selection_min_envelope_cosine", selection_min_envelope_cosine),
        ("selection_min_approach_cosine", selection_min_approach_cosine),
    ):
        if value is not None and not -1.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [-1, 1]")

    generator = torch.Generator(device=device).manual_seed(int(seed))
    q0 = initial_joints.clamp(gripper.lower + 1e-5, gripper.upper - 1e-5)
    object_center = object_pc.mean(dim=0)

    def contact_patch_frame(contacts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        center = contacts.mean(dim=0)
        patch_vector = center - object_center
        patch_norm = torch.linalg.norm(patch_vector)
        # Sparse ContactDiffusion points normally form a localized patch.  In
        # the rare nearly balanced case, use the smallest-variance set axis and
        # orient it toward the contact farthest from the object center.
        if float(patch_norm) < 1.0e-5:
            centered = contacts - center
            _, _, right_h = torch.linalg.svd(centered, full_matrices=False)
            patch_vector = right_h[-1]
            farthest = contacts[
                torch.linalg.norm(contacts - object_center, dim=1).argmax()
            ] - object_center
            patch_vector = patch_vector * torch.where(
                torch.dot(patch_vector, farthest) < 0,
                patch_vector.new_tensor(-1.0),
                patch_vector.new_tensor(1.0),
            )
        return center, F.normalize(patch_vector, dim=0)

    contact_center, contact_patch_direction = contact_patch_frame(target_contacts)
    initialization_contact_center, initialization_patch_direction = (
        contact_patch_frame(initialization_contacts)
    )
    if preferred_root_direction is None:
        desired_root_direction = -contact_patch_direction
        initialization_root_direction = -initialization_patch_direction
    else:
        desired_root_direction = F.normalize(
            preferred_root_direction.to(
                device=device, dtype=dtype
            ).reshape(3),
            dim=0,
        )
        initialization_root_direction = desired_root_direction
    bbox_diagonal = torch.linalg.norm(
        object_pc.max(dim=0).values - object_pc.min(dim=0).values
    )
    torque_scale = 0.5 * bbox_diagonal
    if has_collision_geometry:
        target_surface_distance = torch.cdist(
            target_contacts.unsqueeze(0), object_surface_points.unsqueeze(0)
        ).squeeze(0)
        target_surface_index = target_surface_distance.argmin(dim=1)
        target_object_contacts = object_surface_points[target_surface_index]
        target_object_normals = F.normalize(
            object_surface_normals[target_surface_index], dim=1
        )
    kabsch_particle_count = 0
    with torch.no_grad():
        if initialization_mode in {"cedex", "enveloping"}:
            if cedex_local_palm_axis is None or close_direction is None:
                raise ValueError(
                    "Center-facing initialization requires cedex_local_palm_axis "
                    "and close_direction"
                )
            close_direction = close_direction.to(
                device=device, dtype=dtype
            ).reshape(-1)
            if close_direction.numel() != q0.numel():
                raise ValueError("close_direction does not match joint count")
            random_fraction = float(cedex_joint_init_fraction) * torch.rand(
                particles,
                q0.numel(),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            close_limit = torch.where(
                close_direction > 0,
                gripper.upper,
                torch.where(close_direction < 0, gripper.lower, gripper.lower),
            )
            joint_init = q0.unsqueeze(0) + random_fraction * (
                close_limit.unsqueeze(0) - q0.unsqueeze(0)
            )
            # For joints without a configured closing direction, retain
            # CEDex's lower-half joint-range sampling.
            neutral = close_direction == 0
            if bool(neutral.any()):
                lower_half = gripper.lower.unsqueeze(0) + random_fraction * (
                    gripper.upper - gripper.lower
                ).unsqueeze(0)
                joint_init[:, neutral] = lower_half[:, neutral]
            joint_init = torch.where(
                gripper.locked_mask.bool().unsqueeze(0),
                gripper.locked_values.unsqueeze(0),
                joint_init,
            )
            q_raw_init = gripper.unconstrain_joints(joint_init)

            if initialization_mode == "cedex":
                # Match CEDex exactly in spirit: sample roots in a cube centered
                # on the object, orient a hand-specific palm axis at the center,
                # then add a uniformly random roll about that approach axis.
                object_radius = torch.linalg.norm(
                    object_pc - object_center, dim=1
                ).max()
                cube_size = 1.5 * object_radius
                translation_init = object_center.unsqueeze(0) + (
                    torch.rand(
                        particles,
                        3,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                    - 0.5
                ) * cube_size
            else:
                # Contact positions alone do not determine which side of a
                # locally planar patch should contain the palm.  Kabsch can
                # therefore fit the same fingertip points with the palm below
                # them, producing a non-enveloping "three fingertips under an
                # apple" solution.  Place the palm on the hemisphere opposite
                # the generated contact patch before matching the tip centroid.
                initial_local = gripper.tip_points_in_aligned_base(joint_init)
                initial_palm_local = gripper.palm_point_in_aligned_base(
                    joint_init
                )
                mean_tip_reach = torch.linalg.norm(
                    initial_local.mean(dim=1) - initial_palm_local, dim=1
                ).clamp_min(0.05)
                root_distance = mean_tip_reach
                translation_init = (
                    object_center.unsqueeze(0)
                    + initialization_root_direction.unsqueeze(0)
                    * root_distance.unsqueeze(1)
                )
            rotation = cedex_center_facing_rotations(
                translation_init,
                object_center,
                cedex_local_palm_axis.to(device=device, dtype=dtype),
                generator=generator,
            )
            rotation_init = torch.cat(
                (rotation[:, :, 0], rotation[:, :, 1]), dim=-1
            )
            if initialization_mode == "enveloping":
                # Preserve the selected approach hemisphere while aligning the
                # average fingertip with the sparse target-contact centroid.
                rotated_tip_center = torch.einsum(
                    "bij,bj->bi", rotation, initial_local.mean(dim=1)
                )
                translation_init = (
                    initialization_contact_center.unsqueeze(0) - rotated_tip_center
                )
                # Keep half of the particles on the original Kabsch manifold.
                # These preserve low-error poses when Kabsch already chose the
                # desired overhead hemisphere, while the other half supplies
                # the missing cross-hemisphere hypotheses for bottom contacts.
                kabsch_particle_count = particles // 2
                if kabsch_particle_count:
                    kabsch_q_raw = gripper.unconstrain_joints(q0).repeat(
                        kabsch_particle_count, 1
                    )
                    kabsch_q_raw = kabsch_q_raw + 0.35 * torch.randn(
                        kabsch_q_raw.shape,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                    broad_kabsch = kabsch_particle_count // 2
                    if broad_kabsch:
                        broad_fraction = 0.02 + 0.96 * torch.rand(
                            broad_kabsch,
                            kabsch_q_raw.shape[1],
                            device=device,
                            dtype=dtype,
                            generator=generator,
                        )
                        kabsch_q_raw[-broad_kabsch:] = torch.logit(
                            broad_fraction
                        )
                    q_raw_init[:kabsch_particle_count] = kabsch_q_raw
                    kabsch_local = gripper.tip_points_in_aligned_base(
                        gripper.constrain_joints(kabsch_q_raw)
                    )
                    kabsch_targets = torch.stack(
                        [
                            initialization_contacts[
                                torch.randperm(
                                    initialization_contacts.shape[0],
                                    device=device,
                                    generator=generator,
                                )
                            ]
                            for _ in range(kabsch_particle_count)
                        ],
                        dim=0,
                    )
                    kabsch_rotation_init = kabsch_rotation(
                        kabsch_local, kabsch_targets
                    )
                    rotation_init[:kabsch_particle_count] = torch.cat(
                        (
                            kabsch_rotation_init[:, :, 0],
                            kabsch_rotation_init[:, :, 1],
                        ),
                        dim=-1,
                    )
                    rotated_kabsch_center = torch.einsum(
                        "bij,bj->bi",
                        kabsch_rotation_init,
                        kabsch_local.mean(dim=1),
                    )
                    translation_init[:kabsch_particle_count] = (
                        kabsch_targets.mean(dim=1) - rotated_kabsch_center
                    )
                # Recompute the center-facing orientation after centroid
                # alignment and repeat once; this removes most approach error
                # without sacrificing the free roll hypotheses.
                aligned_palm_position = (
                    torch.einsum(
                        "bij,bj->bi", rotation, initial_palm_local
                    )
                    + translation_init
                )
                rotation = cedex_center_facing_rotations(
                    aligned_palm_position,
                    object_center,
                    cedex_local_palm_axis.to(device=device, dtype=dtype),
                    generator=generator,
                )
                rotation_init = torch.cat(
                    (rotation[:, :, 0], rotation[:, :, 1]), dim=-1
                )
                rotated_tip_center = torch.einsum(
                    "bij,bj->bi", rotation, initial_local.mean(dim=1)
                )
                translation_init = (
                    initialization_contact_center.unsqueeze(0) - rotated_tip_center
                )
        else:
            q_raw_init = gripper.unconstrain_joints(q0).repeat(particles, 1)
            q_raw_init = q_raw_init + 0.35 * torch.randn(
                q_raw_init.shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            # Preserve open-pose hypotheses while giving half the particles
            # broad joint-space coverage.
            broad_count = particles // 2
            if broad_count:
                broad_fraction = 0.02 + 0.96 * torch.rand(
                    broad_count,
                    q_raw_init.shape[1],
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                q_raw_init[-broad_count:] = torch.logit(broad_fraction)
            initial_local = gripper.tip_points_in_aligned_base(
                gripper.constrain_joints(q_raw_init)
            )
            target_permutations = torch.stack(
                [
                    initialization_contacts[
                        torch.randperm(
                            initialization_contacts.shape[0],
                            device=device,
                            generator=generator,
                        )
                    ]
                    for _ in range(particles)
                ],
                dim=0,
            )
            rotation = kabsch_rotation(initial_local, target_permutations)
            rotation_init = torch.cat(
                (rotation[:, :, 0], rotation[:, :, 1]), dim=-1
            )
            rotated_center = torch.einsum(
                "bij,bj->bi", rotation, initial_local.mean(dim=1)
            )
            translation_init = (
                target_permutations.mean(dim=1) - rotated_center
            )

    initial_state_digest = hashlib.sha256()
    for initial_tensor in (translation_init, rotation_init, q_raw_init):
        initial_state_digest.update(
            initial_tensor.detach().cpu().contiguous().numpy().tobytes()
        )
    initialization_state_sha256 = initial_state_digest.hexdigest()

    translation = torch.nn.Parameter(translation_init)
    rotation_6d = torch.nn.Parameter(rotation_init)
    q_raw = torch.nn.Parameter(q_raw_init)
    optimizer = torch.optim.Adam([translation, rotation_6d, q_raw], lr=float(learning_rate))
    target_batch = target_contacts.unsqueeze(0).expand(particles, -1, -1)
    palm_axis = (
        None
        if cedex_local_palm_axis is None
        else F.normalize(
            cedex_local_palm_axis.to(device=device, dtype=dtype).reshape(3),
            dim=0,
        )
    )
    assigned_target_batch = None
    permuted_targets = None
    if contact_assignment_mode in {"permutation", "soft_permutation"}:
        # n is only 2, 3, or 5 in ContactDiffusion, so exhaustive assignment
        # has at most 120 permutations.  Select one assignment per FK particle
        # once, then retain it throughout optimization.  Unlike unordered
        # Chamfer, this prevents several fingertips collapsing onto one target.
        permutations = torch.as_tensor(
            list(itertools.permutations(range(target_contacts.shape[0]))),
            device=device,
            dtype=torch.long,
        )
        permuted_targets = target_contacts[permutations]
        with torch.no_grad():
            initial_tips = gripper.tip_points(
                gripper.constrain_joints(q_raw),
                translation,
                rotation_6d,
            )
            assignment_cost = torch.linalg.norm(
                initial_tips[:, None, :, :]
                - permuted_targets[None, :, :, :],
                dim=3,
            ).mean(dim=2)
            if contact_assignment_mode == "permutation":
                assigned_target_batch = permuted_targets[
                    assignment_cost.argmin(dim=1)
                ]

    force_closure_start_step = int(
        math.floor(float(force_closure_start_fraction) * int(steps))
    )
    for step_index in range(int(steps)):
        joints = gripper.constrain_joints(q_raw)
        tips = gripper.tip_points(joints, translation, rotation_6d)
        distances = torch.cdist(tips, target_batch)
        chamfer_contact_loss = (
            distances.min(dim=2).values.mean(dim=1)
            + distances.min(dim=1).values.mean(dim=1)
        )
        matched_finger_points = tips
        matched_finger_object_points = None
        matched_finger_object_normals = None
        if contact_assignment_mode == "soft_permutation":
            if contact_geometry_mode == "distal_surface":
                distal_surfaces = gripper.tip_link_surface_points(
                    joints, translation, rotation_6d
                )
                all_surface_distances = torch.linalg.norm(
                    distal_surfaces[:, :, :, None, :]
                    - target_contacts[None, None, None, :, :],
                    dim=4,
                )
                finger_target_distances, nearest_surface_indices = (
                    all_surface_distances.min(dim=2)
                )
                expanded_costs = finger_target_distances[:, None, :, :].expand(
                    -1, permutations.shape[0], -1, -1
                )
                target_indices = permutations[None, :, :, None].expand(
                    particles, -1, -1, 1
                )
                permutation_costs = torch.gather(
                    expanded_costs, 3, target_indices
                ).squeeze(3).mean(dim=2)
            else:
                permutation_costs = torch.linalg.norm(
                    tips[:, None, :, :] - permuted_targets[None, :, :, :],
                    dim=3,
                ).mean(dim=2)
            best_permutations = permutations[permutation_costs.argmin(dim=1)]
            batch_indices = torch.arange(particles, device=device)[:, None]
            finger_indices = torch.arange(
                target_contacts.shape[0], device=device
            )[None, :]
            if contact_geometry_mode == "distal_surface":
                surface_indices = nearest_surface_indices[
                    batch_indices, finger_indices, best_permutations
                ]
                matched_finger_points = distal_surfaces[
                    batch_indices, finger_indices, surface_indices
                ]
            matched_finger_object_points = target_object_contacts[
                best_permutations
            ]
            matched_finger_object_normals = target_object_normals[
                best_permutations
            ]
            temperature = float(assignment_temperature_m)
            contact_loss = (
                2.0 * (
                    -temperature
                * torch.logsumexp(-permutation_costs / temperature, dim=1)
                + temperature * math.log(permutation_costs.shape[1])
                )
            )
            if kabsch_particle_count and contact_geometry_mode == "tip_point":
                contact_loss[:kabsch_particle_count] = (
                    chamfer_contact_loss[:kabsch_particle_count]
                )
        elif assigned_target_batch is not None:
            contact_loss = torch.linalg.norm(
                tips - assigned_target_batch, dim=2
            ).mean(dim=1)
        else:
            contact_loss = chamfer_contact_loss
        joint_loss = (((joints - q0) / gripper.span) ** 2).mean(dim=1)
        palm_positions = gripper.palm_points(
            joints, translation, rotation_6d
        )
        root_direction = F.normalize(
            palm_positions - object_center.unsqueeze(0), dim=1
        )
        envelope_side_cosine = torch.einsum(
            "bi,i->b", root_direction, desired_root_direction
        )
        if palm_axis is not None:
            palm_world = torch.einsum(
                "bij,j->bi", rotation_6d_to_matrix(rotation_6d), palm_axis
            )
            approach_direction = F.normalize(
                object_center.unsqueeze(0) - palm_positions, dim=1
            )
            envelope_approach_cosine = (
                palm_world * approach_direction
            ).sum(dim=1)
            envelope_approach_loss = (
                1.0 - envelope_approach_cosine
            ) ** 2
        else:
            envelope_approach_loss = torch.zeros_like(contact_loss)
        if palm_constraints_enabled:
            palm_surface = gripper.palm_surface_points(
                joints, translation, rotation_6d
            )
            (
                palm_unsigned_distance,
                palm_hand_contact,
                palm_object_contact,
                palm_object_index,
            ) = nearest_surface_distance(
                palm_surface, object_pc
            )
            palm_distance_loss = (
                palm_unsigned_distance - float(palm_target_distance_m)
            ).abs()
        else:
            palm_unsigned_distance = torch.zeros_like(contact_loss)
            palm_distance_loss = torch.zeros_like(contact_loss)
            palm_hand_contact = palm_positions
            palm_object_contact = palm_positions
            palm_object_index = torch.zeros_like(contact_loss, dtype=torch.long)
        if float(penetration_weight) > 0:
            hand_surface = gripper.surface_points(
                joints, translation, rotation_6d
            )
            penetration_terms = point_cloud_penetration_energy(
                hand_surface,
                object_pc,
                object_surface_normals,
                object_normal_confidence,
                cvar_fraction=penetration_cvar_fraction,
                cvar_weight=penetration_cvar_weight,
                depth_mode=penetration_depth_mode,
                aggregation=penetration_aggregation,
                confidence_mode=penetration_confidence_mode,
                gate_metric=penetration_gate_metric,
                hinge_threshold_m=penetration_hinge_threshold_m,
                hinge_weight=penetration_hinge_weight,
            )
        else:
            penetration_terms = {
                key: torch.zeros_like(contact_loss)
                for key in (
                    "energy", "mean", "cvar", "hinge", "max", "raw_max",
                    "confidence_weighted_max", "fraction", "mean_confidence",
                )
            }
        if float(self_collision_weight) > 0:
            self_link_points = gripper.self_collision_link_points(
                joints, self_collision_points_per_link
            )
            self_collision_terms = self_collision_mean_cvar_energy(
                self_link_points,
                gripper.self_collision_pairs,
                self_collision_clearance,
                cvar_fraction=self_collision_cvar_fraction,
                cvar_weight=self_collision_cvar_weight,
            )
        else:
            self_collision_terms = {
                key: torch.zeros_like(contact_loss)
                for key in ("energy", "mean", "cvar", "max", "fraction")
            }
        if force_closure_enabled and step_index >= force_closure_start_step:
            realized_fingers = realized_surface_contact_regions(
                distal_surfaces,
                object_pc,
                object_surface_normals,
                object_normal_confidence,
                gap_sigma_m=force_closure_gap_sigma_m,
            )
            if force_closure_include_palm:
                realized_palm = realized_surface_contact_regions(
                    palm_surface[:, None, :, :],
                    object_pc,
                    object_surface_normals,
                    object_normal_confidence,
                    gap_sigma_m=force_closure_gap_sigma_m,
                )
                realized_contacts = {
                    key: torch.cat(
                        (realized_fingers[key], realized_palm[key]), dim=1
                    )
                    for key in realized_fingers
                }
            else:
                realized_contacts = realized_fingers
            force_closure_terms = execution_aware_force_closure_energy(
                realized_contacts["hand_contact"],
                realized_contacts["outward_normal"],
                realized_contacts["activation"],
                object_center,
                torque_scale,
                friction_coefficient=resolved_force_closure_friction,
                cone_edges=resolved_force_closure_cone_edges,
                qp_iterations=force_closure_qp_iterations,
                qp_weight=force_closure_qp_weight,
                dfc_weight=force_closure_dfc_weight,
                coverage_weight=force_closure_coverage_weight,
                formulation=force_closure_formulation,
                normal_confidence=realized_contacts["normal_confidence"],
                max_force_coefficient=force_closure_max_force_coefficient,
                torque_weight=force_closure_torque_weight,
                svd_gain=force_closure_svd_gain,
            )
            force_closure_ramp = (
                1.0
                if force_closure_ramp_mode == "constant"
                else float(step_index - force_closure_start_step + 1)
                / float(max(int(steps) - force_closure_start_step, 1))
            )
        else:
            force_closure_terms = {
                key: torch.zeros_like(contact_loss)
                for key in (
                    "energy",
                    "qp_score",
                    "qp_residual",
                    "minimum_singular_value",
                    "dfc",
                    "coverage",
                    "mean_activation",
                    "svd_geometric_mean",
                    "mean_force_coefficient",
                    "max_force_coefficient",
                )
            }
            force_closure_ramp = 0.0
        main_energy = (
            float(contact_weight) * contact_loss
            + float(penetration_weight) * penetration_terms["energy"]
            + float(self_collision_weight) * self_collision_terms["energy"]
            + float(envelope_approach_weight) * envelope_approach_loss
            + float(joint_regularization) * joint_loss
            + resolved_palm_distance_weight * palm_distance_loss
        )
        total, _, _ = blend_main_and_force_closure_energy(
            main_energy,
            force_closure_terms["energy"],
            force_closure_weight=force_closure_weight,
            force_closure_ramp=force_closure_ramp,
            target_fraction=force_closure_target_fraction,
        )
        optimizer.zero_grad(set_to_none=True)
        total.mean().backward()
        optimizer.step()

    with torch.no_grad():
        joints = gripper.constrain_joints(q_raw)
        tips = gripper.tip_points(joints, translation, rotation_6d)
        distances = torch.cdist(tips, target_batch)
        contact_losses = (
            distances.min(dim=2).values.mean(dim=1)
            + distances.min(dim=1).values.mean(dim=1)
        )
        matched_contact_points = tips
        matched_finger_object_points = None
        matched_finger_object_normals = None
        if permuted_targets is not None:
            if contact_geometry_mode == "distal_surface":
                distal_surfaces = gripper.tip_link_surface_points(
                    joints, translation, rotation_6d
                )
                all_surface_distances = torch.linalg.norm(
                    distal_surfaces[:, :, :, None, :]
                    - target_contacts[None, None, None, :, :],
                    dim=4,
                )
                finger_target_distances, nearest_surface_indices = (
                    all_surface_distances.min(dim=2)
                )
                expanded_costs = finger_target_distances[:, None, :, :].expand(
                    -1, permutations.shape[0], -1, -1
                )
                target_indices = permutations[None, :, :, None].expand(
                    particles, -1, -1, 1
                )
                final_permutation_costs = torch.gather(
                    expanded_costs, 3, target_indices
                ).squeeze(3).mean(dim=2)
                best_permutations = permutations[
                    final_permutation_costs.argmin(dim=1)
                ]
                batch_indices = torch.arange(particles, device=device)[:, None]
                finger_indices = torch.arange(
                    target_contacts.shape[0], device=device
                )[None, :]
                surface_indices = nearest_surface_indices[
                    batch_indices, finger_indices, best_permutations
                ]
                matched_contact_points = distal_surfaces[
                    batch_indices, finger_indices, surface_indices
                ]
                matched_distances = torch.cdist(
                    matched_contact_points, target_batch
                )
                contact_losses = (
                    matched_distances.min(dim=2).values.mean(dim=1)
                    + matched_distances.min(dim=1).values.mean(dim=1)
                )
            else:
                final_permutation_costs = torch.linalg.norm(
                    tips[:, None, :, :] - permuted_targets[None, :, :, :],
                    dim=3,
                ).mean(dim=2)
                best_permutations = permutations[
                    final_permutation_costs.argmin(dim=1)
                ]
            matched_finger_object_points = target_object_contacts[
                best_permutations
            ]
            matched_finger_object_normals = target_object_normals[
                best_permutations
            ]
            assigned_contact_losses = final_permutation_costs.min(dim=1).values
            fit_contact_losses = 2.0 * assigned_contact_losses
            if kabsch_particle_count and contact_geometry_mode == "tip_point":
                fit_contact_losses[:kabsch_particle_count] = (
                    contact_losses[:kabsch_particle_count]
                )
        else:
            assigned_contact_losses = contact_losses
            fit_contact_losses = contact_losses
        joint_losses = (((joints - q0) / gripper.span) ** 2).mean(dim=1)
        palm_positions = gripper.palm_points(
            joints, translation, rotation_6d
        )
        root_directions = F.normalize(
            palm_positions - object_center.unsqueeze(0), dim=1
        )
        envelope_side_cosines = torch.einsum(
            "bi,i->b", root_directions, desired_root_direction
        )
        contact_patch_side_cosines = torch.einsum(
            "bi,i->b", root_directions, contact_patch_direction
        )
        if palm_axis is not None:
            palm_world = torch.einsum(
                "bij,j->bi", rotation_6d_to_matrix(rotation_6d), palm_axis
            )
            approach_directions = F.normalize(
                object_center.unsqueeze(0) - palm_positions, dim=1
            )
            envelope_approach_cosines = (
                palm_world * approach_directions
            ).sum(dim=1)
            envelope_approach_losses = (
                1.0 - envelope_approach_cosines
            ) ** 2
        else:
            envelope_approach_cosines = torch.zeros_like(contact_losses)
            envelope_approach_losses = torch.zeros_like(contact_losses)
        if palm_constraints_enabled:
            palm_surface = gripper.palm_surface_points(
                joints, translation, rotation_6d
            )
            (
                palm_unsigned_distances,
                palm_hand_contacts,
                palm_object_contacts,
                palm_object_indices,
            ) = nearest_surface_distance(
                palm_surface, object_pc
            )
            palm_distance_losses = (
                palm_unsigned_distances - float(palm_target_distance_m)
            ).abs()
            palm_object_normals = object_surface_normals[palm_object_indices]
            palm_normal_confidences = object_normal_confidence[palm_object_indices]
        else:
            palm_unsigned_distances = torch.zeros_like(contact_losses)
            palm_distance_losses = torch.zeros_like(contact_losses)
            palm_hand_contacts = palm_positions
            palm_object_contacts = palm_positions
            palm_object_normals = torch.zeros_like(palm_positions)
            palm_normal_confidences = torch.zeros_like(contact_losses)
        if selection_rank_mode == "graspqp":
            all_object_contacts = torch.cat(
                (matched_finger_object_points, palm_object_contacts[:, None, :]),
                dim=1,
            )
            all_object_normals = torch.cat(
                (matched_finger_object_normals, palm_object_normals[:, None, :]),
                dim=1,
            )
            (
                graspqp_scores,
                graspqp_residuals,
                graspqp_minimum_singular_values,
            ) = graspqp_friction_cone_metrics(
                all_object_contacts,
                all_object_normals,
                object_center,
                torque_scale,
                friction_coefficient=resolved_force_closure_friction,
                cone_edges=resolved_force_closure_cone_edges,
                iterations=graspqp_iterations,
            )
        else:
            graspqp_scores = torch.zeros_like(contact_losses)
            graspqp_residuals = torch.zeros_like(contact_losses)
            graspqp_minimum_singular_values = torch.zeros_like(contact_losses)
        if float(penetration_weight) > 0:
            hand_surface = gripper.surface_points(joints, translation, rotation_6d)
            penetration_terms = point_cloud_penetration_energy(
                hand_surface,
                object_pc,
                object_surface_normals,
                object_normal_confidence,
                cvar_fraction=penetration_cvar_fraction,
                cvar_weight=penetration_cvar_weight,
                depth_mode=penetration_depth_mode,
                aggregation=penetration_aggregation,
                confidence_mode=penetration_confidence_mode,
                gate_metric=penetration_gate_metric,
                hinge_threshold_m=penetration_hinge_threshold_m,
                hinge_weight=penetration_hinge_weight,
            )
        else:
            penetration_terms = {
                key: torch.zeros_like(contact_losses)
                for key in (
                    "energy", "mean", "cvar", "hinge", "max", "raw_max",
                    "confidence_weighted_max", "fraction", "mean_confidence",
                )
            }
        penetration_losses = penetration_terms["mean"]
        penetration_cvar_losses = penetration_terms["cvar"]
        penetration_hinge_losses = penetration_terms["hinge"]
        max_penetrations = penetration_terms["max"]
        raw_max_penetrations = penetration_terms["raw_max"]
        confidence_weighted_max_penetrations = penetration_terms[
            "confidence_weighted_max"
        ]
        penetration_fractions = penetration_terms["fraction"]
        penetration_confidences = penetration_terms["mean_confidence"]
        if gripper.surface_link_names and float(self_collision_weight) > 0:
            self_link_points = gripper.self_collision_link_points(
                joints, self_collision_points_per_link
            )
            self_collision_terms = self_collision_mean_cvar_energy(
                self_link_points,
                gripper.self_collision_pairs,
                self_collision_clearance,
                cvar_fraction=self_collision_cvar_fraction,
                cvar_weight=self_collision_cvar_weight,
            )
        else:
            self_collision_terms = {
                key: torch.zeros_like(contact_losses)
                for key in ("energy", "mean", "cvar", "max", "fraction")
            }
        if force_closure_enabled:
            realized_fingers = realized_surface_contact_regions(
                distal_surfaces,
                object_pc,
                object_surface_normals,
                object_normal_confidence,
                gap_sigma_m=force_closure_gap_sigma_m,
            )
            if force_closure_include_palm:
                realized_palm = realized_surface_contact_regions(
                    palm_surface[:, None, :, :],
                    object_pc,
                    object_surface_normals,
                    object_normal_confidence,
                    gap_sigma_m=force_closure_gap_sigma_m,
                )
                realized_contacts = {
                    key: torch.cat(
                        (realized_fingers[key], realized_palm[key]), dim=1
                    )
                    for key in realized_fingers
                }
            else:
                realized_contacts = realized_fingers
            force_closure_terms = execution_aware_force_closure_energy(
                realized_contacts["hand_contact"],
                realized_contacts["outward_normal"],
                realized_contacts["activation"],
                object_center,
                torque_scale,
                friction_coefficient=graspqp_friction_coefficient,
                cone_edges=graspqp_cone_edges,
                qp_iterations=force_closure_qp_iterations,
                qp_weight=force_closure_qp_weight,
                dfc_weight=force_closure_dfc_weight,
                coverage_weight=force_closure_coverage_weight,
                formulation=force_closure_formulation,
                normal_confidence=realized_contacts["normal_confidence"],
                max_force_coefficient=force_closure_max_force_coefficient,
                torque_weight=force_closure_torque_weight,
                svd_gain=force_closure_svd_gain,
            )
            realized_gaps = realized_contacts["gap"]
            realized_confidences = realized_contacts["normal_confidence"]
        else:
            force_closure_terms = {
                key: torch.zeros_like(contact_losses)
                for key in (
                    "energy",
                    "qp_score",
                    "qp_residual",
                    "minimum_singular_value",
                    "dfc",
                    "coverage",
                    "mean_activation",
                    "svd_geometric_mean",
                    "mean_force_coefficient",
                    "max_force_coefficient",
                )
            }
            realized_gaps = torch.zeros(
                (
                    particles,
                    len(gripper.tip_links) + int(force_closure_include_palm),
                ),
                device=device,
                dtype=dtype,
            )
            realized_confidences = torch.zeros_like(realized_gaps)
        self_collision_losses = self_collision_terms["mean"]
        self_collision_cvar_losses = self_collision_terms["cvar"]
        max_self_collisions = self_collision_terms["max"]
        self_collision_pair_fractions = self_collision_terms["fraction"]
        main_losses = (
            float(contact_weight) * fit_contact_losses
            + float(penetration_weight) * penetration_terms["energy"]
            + float(self_collision_weight) * self_collision_terms["energy"]
            + float(envelope_approach_weight) * envelope_approach_losses
            + float(joint_regularization) * joint_losses
            + resolved_palm_distance_weight * palm_distance_losses
        )
        losses, final_force_closure_scale, final_force_closure_fraction = (
            blend_main_and_force_closure_energy(
                main_losses,
                force_closure_terms["energy"],
                force_closure_weight=force_closure_weight,
                force_closure_ramp=1.0,
                target_fraction=force_closure_target_fraction,
            )
        )
        count = min(int(top_k), particles)
        rank_scores = (
            graspqp_scores
            if selection_rank_mode == "graspqp"
            else losses
        )
        if (
            selection_max_penetration_m is not None
            or selection_min_envelope_cosine is not None
            or selection_min_approach_cosine is not None
            or selection_max_palm_distance_m is not None
        ):
            feasible = torch.ones_like(max_penetrations, dtype=torch.bool)
            violation = torch.zeros_like(max_penetrations)
            if selection_max_penetration_m is not None:
                feasible &= max_penetrations <= float(
                    selection_max_penetration_m
                )
                violation += F.relu(
                    max_penetrations
                    - float(selection_max_penetration_m)
                )
            if selection_min_envelope_cosine is not None:
                feasible &= envelope_side_cosines >= float(
                    selection_min_envelope_cosine
                )
                violation += F.relu(
                    float(selection_min_envelope_cosine)
                    - envelope_side_cosines
                )
            if selection_min_approach_cosine is not None:
                feasible &= envelope_approach_cosines >= float(
                    selection_min_approach_cosine
                )
                violation += F.relu(
                    float(selection_min_approach_cosine)
                    - envelope_approach_cosines
                )
            if selection_max_palm_distance_m is not None:
                palm_distance_error = palm_distance_losses
                feasible &= palm_distance_error <= float(
                    selection_max_palm_distance_m
                )
                violation += F.relu(
                    palm_distance_error
                    - float(selection_max_palm_distance_m)
                )
            # Mirror CEDex's optional 5 mm validate_depth filter while keeping
            # generation total: feasible particles retain final-energy order;
            # if none exist, fall back to the least-penetrating particles.
            selection_key = torch.where(
                feasible,
                rank_scores + 1.0e-6 * losses,
                losses.new_tensor(1.0e6)
                + 1.0e6 * violation
                + 1.0e-3 * rank_scores,
            )
            # Stable sorting makes exact ties deterministic: particle index is
            # the final tie-breaker because particles are stored in index order.
            best = torch.argsort(selection_key, stable=True)[:count]
        else:
            feasible = torch.ones_like(max_penetrations, dtype=torch.bool)
            best = torch.argsort(rank_scores, stable=True)[:count]
        selection_feasible_particles = int(feasible.sum().item())
        selection_fallback_used = selection_feasible_particles < count
        poses = gripper.root_pose_matrix(translation, rotation_6d)

    candidates = []
    for rank, particle_index in enumerate(best.tolist()):
        candidates.append(
            {
                "rank": rank,
                "particle": particle_index,
                "selection_feasible": bool(feasible[particle_index].item()),
                "selection_fallback": bool(
                    not feasible[particle_index].item()
                ),
                "optimization_score": float(losses[particle_index].item()),
                "penetration_filter_pass": bool(
                    True
                    if selection_max_penetration_m is None
                    else (
                        max_penetrations[particle_index]
                        <= float(selection_max_penetration_m)
                    ).item()
                ),
                "envelope_filter_pass": bool(
                    feasible[particle_index].item()
                ),
                "contact_chamfer_m": float(contact_losses[particle_index].item()),
                "assigned_contact_error_m": float(
                    assigned_contact_losses[particle_index].item()
                ),
                "mean_penetration_m": float(penetration_losses[particle_index].item()),
                "cvar_penetration_m": float(
                    penetration_cvar_losses[particle_index].item()
                ),
                "hinge_penetration_m": float(
                    penetration_hinge_losses[particle_index].item()
                ),
                "max_penetration_m": float(max_penetrations[particle_index].item()),
                "raw_max_penetration_m": float(
                    raw_max_penetrations[particle_index].item()
                ),
                "confidence_weighted_max_penetration_m": float(
                    confidence_weighted_max_penetrations[particle_index].item()
                ),
                "penetration_normal_confidence": float(
                    penetration_confidences[particle_index].item()
                ),
                "penetrating_surface_fraction": float(
                    penetration_fractions[particle_index].item()
                ),
                "mean_self_collision_m": float(
                    self_collision_losses[particle_index].item()
                ),
                "cvar_self_collision_m": float(
                    self_collision_cvar_losses[particle_index].item()
                ),
                "max_self_collision_m": float(
                    max_self_collisions[particle_index].item()
                ),
                "self_collision_pair_fraction": float(
                    self_collision_pair_fractions[particle_index].item()
                ),
                "envelope_side_cosine": float(
                    envelope_side_cosines[particle_index].item()
                ),
                "contact_patch_side_cosine": float(
                    contact_patch_side_cosines[particle_index].item()
                ),
                "palm_approach_cosine": float(
                    envelope_approach_cosines[particle_index].item()
                ),
                "palm_unsigned_distance_m": float(
                    palm_unsigned_distances[particle_index].item()
                ),
                "palm_distance_error_m": float(
                    palm_distance_losses[particle_index].item()
                ),
                "graspqp_score": float(
                    graspqp_scores[particle_index].item()
                ),
                "graspqp_wrench_residual": float(
                    graspqp_residuals[particle_index].item()
                ),
                "graspqp_min_singular_value": float(
                    graspqp_minimum_singular_values[particle_index].item()
                ),
                "force_closure_energy": float(
                    force_closure_terms["energy"][particle_index].item()
                ),
                "force_closure_normalization_scale": float(
                    final_force_closure_scale.item()
                ),
                "force_closure_effective_fraction": float(
                    final_force_closure_fraction
                ),
                "force_closure_qp_score": float(
                    force_closure_terms["qp_score"][particle_index].item()
                ),
                "force_closure_worst_wrench_residual": float(
                    force_closure_terms["qp_residual"][particle_index].item()
                ),
                "force_closure_min_singular_value": float(
                    force_closure_terms["minimum_singular_value"][particle_index].item()
                ),
                "force_closure_dfc": float(
                    force_closure_terms["dfc"][particle_index].item()
                ),
                "force_closure_contact_coverage": float(
                    force_closure_terms["coverage"][particle_index].item()
                ),
                "force_closure_mean_activation": float(
                    force_closure_terms["mean_activation"][particle_index].item()
                ),
                "force_closure_svd_geometric_mean": float(
                    force_closure_terms["svd_geometric_mean"][particle_index].item()
                ),
                "force_closure_mean_force_coefficient": float(
                    force_closure_terms["mean_force_coefficient"][particle_index].item()
                ),
                "force_closure_max_force_coefficient": float(
                    force_closure_terms["max_force_coefficient"][particle_index].item()
                ),
                "force_closure_contact_gaps_m": realized_gaps[
                    particle_index
                ].detach().cpu().tolist(),
                "force_closure_normal_confidences": realized_confidences[
                    particle_index
                ].detach().cpu().tolist(),
                "palm_position": palm_positions[
                    particle_index
                ].detach().cpu().tolist(),
                "palm_surface_contact": palm_hand_contacts[
                    particle_index
                ].detach().cpu().tolist(),
                "palm_object_contact": palm_object_contacts[
                    particle_index
                ].detach().cpu().tolist(),
                "palm_object_normal": palm_object_normals[
                    particle_index
                ].detach().cpu().tolist(),
                "palm_object_normal_confidence": float(
                    palm_normal_confidences[particle_index].item()
                ),
                "root_pose": poses[particle_index].detach().cpu().tolist(),
                "joint_positions": joints[particle_index].detach().cpu().tolist(),
                "tip_points": tips[particle_index].detach().cpu().tolist(),
                "matched_contact_points": matched_contact_points[
                    particle_index
                ].detach().cpu().tolist(),
            }
        )
    return {
        "energy_model": (
            "wc*contact + wp*(pc_penetration_mean + lambda_p*CVaR) + "
            "ws*(self_mean + lambda_s*CVaR) + wa*approach + wq*joint + "
            "wpd*palm_unsigned_distance + wfc*execution_aware_FC"
        ),
        "object_geometry": geometry_diagnostics,
        "initialization_mode": initialization_mode,
        "cedex_joint_init_fraction": float(cedex_joint_init_fraction),
        "cedex_cleanup_steps": int(cedex_cleanup_steps),
        "cedex_cleanup_learning_rate": float(
            cedex_cleanup_learning_rate
        ),
        "cedex_contact_guard_m": float(cedex_contact_guard_m),
        "selection_max_penetration_m": (
            None
            if selection_max_penetration_m is None
            else float(selection_max_penetration_m)
        ),
        "penetration_cvar_fraction": float(penetration_cvar_fraction),
        "penetration_cvar_weight": float(penetration_cvar_weight),
        "penetration_depth_mode": penetration_depth_mode,
        "penetration_aggregation": penetration_aggregation,
        "penetration_confidence_mode": penetration_confidence_mode,
        "penetration_gate_metric": penetration_gate_metric,
        "penetration_hinge_threshold_m": float(penetration_hinge_threshold_m),
        "penetration_hinge_weight": float(penetration_hinge_weight),
        "self_collision_cvar_fraction": float(self_collision_cvar_fraction),
        "self_collision_cvar_weight": float(self_collision_cvar_weight),
        "self_collision_pair_count": len(gripper.self_collision_pairs),
        "self_collision_pair_names": [
            list(pair) for pair in gripper.self_collision_pair_names
        ],
        "envelope_approach_weight": float(envelope_approach_weight),
        "contact_assignment_mode": contact_assignment_mode,
        "assignment_temperature_m": float(assignment_temperature_m),
        "contact_geometry_mode": contact_geometry_mode,
        "selection_min_envelope_cosine": (
            None
            if selection_min_envelope_cosine is None
            else float(selection_min_envelope_cosine)
        ),
        "selection_min_approach_cosine": (
            None
            if selection_min_approach_cosine is None
            else float(selection_min_approach_cosine)
        ),
        "palm_surface_link": gripper.palm_surface_link,
        "palm_distance_weight": float(resolved_palm_distance_weight),
        "palm_target_distance_m": float(palm_target_distance_m),
        "selection_max_palm_distance_m": (
            None
            if selection_max_palm_distance_m is None
            else float(selection_max_palm_distance_m)
        ),
        "selection_rank_mode": selection_rank_mode,
        "selection_tie_break": "stable_particle_index",
        "selection_feasible_particles": selection_feasible_particles,
        "selection_fallback_used": bool(selection_fallback_used),
        "graspqp_friction_coefficient": float(
            graspqp_friction_coefficient
        ),
        "graspqp_cone_edges": int(graspqp_cone_edges),
        "graspqp_iterations": int(graspqp_iterations),
        "force_closure_weight": float(force_closure_weight),
        "force_closure_start_fraction": float(force_closure_start_fraction),
        "force_closure_gap_sigma_m": float(force_closure_gap_sigma_m),
        "force_closure_qp_iterations": int(force_closure_qp_iterations),
        "force_closure_qp_weight": float(force_closure_qp_weight),
        "force_closure_dfc_weight": float(force_closure_dfc_weight),
        "force_closure_coverage_weight": float(force_closure_coverage_weight),
        "force_closure_target_fraction": (
            None
            if force_closure_target_fraction is None
            else float(force_closure_target_fraction)
        ),
        "force_closure_ramp_mode": force_closure_ramp_mode,
        "force_closure_formulation": force_closure_formulation,
        "force_closure_contact_regions": (
            "distal_links_plus_palm"
            if force_closure_include_palm
            else "distal_links"
        ),
        "force_closure_max_force_coefficient": float(
            force_closure_max_force_coefficient
        ),
        "force_closure_friction_coefficient": resolved_force_closure_friction,
        "force_closure_cone_edges": resolved_force_closure_cone_edges,
        "force_closure_torque_weight": float(force_closure_torque_weight),
        "force_closure_svd_gain": float(force_closure_svd_gain),
        "force_closure_disturbances": (
            "positive-span zero-wrench QP"
            if force_closure_formulation == "graspqp"
            else "unit_+/-Fx,Fy,Fz,Tx,Ty,Tz"
        ),
        "contact_patch_direction": contact_patch_direction.detach().cpu().tolist(),
        "preferred_root_direction": desired_root_direction.detach().cpu().tolist(),
        "initialization_contacts": initialization_contacts.detach().cpu().tolist(),
        "initialization_contact_patch_direction": (
            initialization_patch_direction.detach().cpu().tolist()
        ),
        "initialization_state_sha256": initialization_state_sha256,
        "initialization_contacts_equal_target": bool(
            torch.equal(initialization_contacts, target_contacts)
        ),
        "joint_names": gripper.joint_names,
        "joint_lower": gripper.lower.detach().cpu().tolist(),
        "joint_upper": gripper.upper.detach().cpu().tolist(),
        "target_contacts": target_contacts.detach().cpu().tolist(),
        "candidates": candidates,
    }


def load_gripper_from_calibration(
    gripper_name: str,
    config: dict,
    calibration_path: str | Path,
    *,
    device: str | torch.device,
) -> DifferentiableGripper:
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    spec = config["grippers"][gripper_name]
    calibrated = calibration["grippers"][gripper_name]
    if "tip_offsets" in spec:
        tip_offsets = {
            link: [float(value) for value in offset]
            for link, offset in spec["tip_offsets"].items()
        }
    else:
        tip_offsets = {
            link: value["offset_xyz"]
            for link, value in calibrated["tips"].items()
        }
    base_alignment = spec.get(
        "base_alignment_matrix", calibrated["base_alignment_matrix"]
    )
    gripper_root = Path(
        spec.get("urdf_root", config["paths"]["gripper_root"])
    )
    return DifferentiableGripper(
        gripper_root / spec["urdf"],
        list(spec["tip_links"]),
        tip_offsets,
        base_alignment,
        surface_points_per_link=int(
            config.get("fk_optimization", {}).get("surface_points_per_link", 0)
        ),
        locked_joints=spec.get("locked_joints"),
        palm_link=spec.get("palm_link"),
        palm_surface_link=spec.get("palm_surface_link"),
        device=device,
    )
