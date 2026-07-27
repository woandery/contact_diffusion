"""Differentiable multi-gripper FK optimization for generated contact sets."""

from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn.functional as F


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


def sample_object_surface(
    mesh_path: str | Path,
    count: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample object surface points and consistently outward face normals."""
    vertices, faces = _load_triangle_mesh(Path(mesh_path))
    triangles = vertices[faces]
    signed_volume = np.einsum(
        "ij,ij->i",
        triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2]),
    ).sum() / 6.0
    generator = np.random.default_rng(int(seed))
    points, normals = _sample_mesh_surface(vertices, faces, int(count), generator)
    # A center-to-surface radial test is wrong for concave objects (for
    # example, the inside wall of a bowl). Closed dataset meshes already have
    # consistent winding, so use signed volume to correct only global winding.
    if signed_volume < 0:
        normals *= -1.0
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    return points.astype(np.float32), normals.astype(np.float32)


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


class DifferentiableGripper:
    def __init__(
        self,
        urdf_path: str | Path,
        tip_links: list[str],
        tip_offsets: dict[str, list[float]],
        base_alignment: list[list[float]],
        surface_points_per_link: int = 0,
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
        self.joint_names = list(self.chain.get_joint_parameter_names())
        lower, upper = self.chain.get_joint_limits()
        self.lower = torch.as_tensor(lower, device=self.device, dtype=dtype)
        self.upper = torch.as_tensor(upper, device=self.device, dtype=dtype)
        invalid = ~torch.isfinite(self.lower) | ~torch.isfinite(self.upper)
        self.lower = torch.where(invalid, torch.full_like(self.lower, -torch.pi), self.lower)
        self.upper = torch.where(invalid, torch.full_like(self.upper, torch.pi), self.upper)
        self.span = (self.upper - self.lower).clamp_min(1e-6)
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
        for joint in urdf_root.findall("joint"):
            child = joint.find("child")
            if child is not None and child.get("link"):
                joint_by_child[child.get("link")] = joint
        # pytorch_kinematics normally exposes link names as FK dictionary keys,
        # but the Barrett asset exposes intermediate movable-joint links while
        # its geometry sits on fixed descendants. Move points through each
        # fixed URDF origin until reaching a frame returned by FK.
        self.surface_link_names = []
        self.surface_points_local = {}
        for link_name, points in sampled_surface.items():
            frame_name = link_name
            frame_points = np.asarray(points, dtype=np.float64)
            visited = set()
            while frame_name not in frame_names and frame_name not in visited:
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
            if frame_name not in frame_names:
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

    def constrain_joints(self, raw: torch.Tensor) -> torch.Tensor:
        return self.lower + torch.sigmoid(raw) * self.span

    def unconstrain_joints(self, joints: torch.Tensor) -> torch.Tensor:
        # Avoid sigmoid saturation when a configured open pose lies exactly at
        # a URDF limit; otherwise the optimizer cannot move that finger closed.
        fraction = ((joints - self.lower) / self.span).clamp(1e-2, 1.0 - 1e-2)
        return torch.logit(fraction)

    def tip_points_in_aligned_base(self, joints: torch.Tensor) -> torch.Tensor:
        transforms = self.chain.forward_kinematics(joints)
        points = []
        for index, link in enumerate(self.tip_links):
            matrix = transforms[link].get_matrix()
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
    max_penetration_weight: float = 0.0,
    translation_regularization: float = 0.1,
    joint_regularization: float = 1e-3,
    seed: int = 42,
    top_k: int = 8,
    object_surface_points: torch.Tensor | None = None,
    object_surface_normals: torch.Tensor | None = None,
) -> dict:
    """Fit URDF fingertip pad points to one unordered target contact set."""
    device, dtype = gripper.device, gripper.dtype
    # clone() converts tensors produced under torch.inference_mode() back into
    # regular tensors that autograd may safely retain for backward.
    target_contacts = target_contacts.to(device=device, dtype=dtype).detach().clone()
    object_pc = object_pc.to(device=device, dtype=dtype).detach().clone()
    has_collision_geometry = (
        object_surface_points is not None and object_surface_normals is not None
    )
    if (object_surface_points is None) != (object_surface_normals is None):
        raise ValueError("Object surface points and normals must be provided together")
    if (
        float(penetration_weight) > 0 or float(max_penetration_weight) > 0
    ) and not has_collision_geometry:
        raise ValueError("Penetration loss requires object surface points and normals")
    if has_collision_geometry:
        object_surface_points = (
            object_surface_points.to(device=device, dtype=dtype).detach().clone()
        )
        object_surface_normals = (
            object_surface_normals.to(device=device, dtype=dtype).detach().clone()
        )
    initial_joints = initial_joints.to(device=device, dtype=dtype).detach().clone()
    if target_contacts.shape != (len(gripper.tip_links), 3):
        raise ValueError(
            f"Expected contacts {(len(gripper.tip_links), 3)}, got {tuple(target_contacts.shape)}"
        )
    if initial_joints.numel() != len(gripper.joint_names):
        raise ValueError(
            f"Expected {len(gripper.joint_names)} initial joints, got {initial_joints.numel()}"
        )

    generator = torch.Generator(device=device).manual_seed(int(seed))
    q0 = initial_joints.clamp(gripper.lower + 1e-5, gripper.upper - 1e-5)
    q_raw_init = gripper.unconstrain_joints(q0).repeat(particles, 1)
    q_raw_init = q_raw_init + 0.35 * torch.randn(
        q_raw_init.shape, device=device, dtype=dtype, generator=generator
    )
    # Preserve open-pose hypotheses while giving half the particles broad
    # joint-space coverage.  This is important for grippers whose open pose is
    # exactly on a limit (for example both Franka finger joints at 0.04 m).
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
    with torch.no_grad():
        initial_local = gripper.tip_points_in_aligned_base(gripper.constrain_joints(q_raw_init))
        # Initialize each particle by rigidly aligning its current tips to a
        # random target permutation.  Random rotations make two-finger fits
        # converge slowly because their useful basin occupies little of SO(3).
        target_permutations = torch.stack(
            [
                target_contacts[
                    torch.randperm(
                        target_contacts.shape[0], device=device, generator=generator
                    )
                ]
                for _ in range(particles)
            ],
            dim=0,
        )
        rotation = kabsch_rotation(initial_local, target_permutations)
        rotation_init = torch.cat((rotation[:, :, 0], rotation[:, :, 1]), dim=-1)
        rotated_center = torch.einsum("bij,bj->bi", rotation, initial_local.mean(dim=1))
        translation_init = target_permutations.mean(dim=1) - rotated_center
        object_center = object_pc.mean(dim=0)
        bbox_diagonal = torch.linalg.norm(object_pc.max(dim=0).values - object_pc.min(dim=0).values)

    translation = torch.nn.Parameter(translation_init)
    rotation_6d = torch.nn.Parameter(rotation_init)
    q_raw = torch.nn.Parameter(q_raw_init)
    optimizer = torch.optim.Adam([translation, rotation_6d, q_raw], lr=float(learning_rate))
    target_batch = target_contacts.unsqueeze(0).expand(particles, -1, -1)

    for _ in range(int(steps)):
        joints = gripper.constrain_joints(q_raw)
        tips = gripper.tip_points(joints, translation, rotation_6d)
        distances = torch.cdist(tips, target_batch)
        contact_loss = distances.min(dim=2).values.mean(dim=1) + distances.min(dim=1).values.mean(dim=1)
        joint_loss = (((joints - q0) / gripper.span) ** 2).mean(dim=1)
        translation_distance = torch.linalg.norm(translation - object_center, dim=1)
        translation_loss = F.relu(translation_distance - bbox_diagonal) ** 2
        if float(penetration_weight) > 0 or float(max_penetration_weight) > 0:
            hand_surface = gripper.surface_points(
                joints, translation, rotation_6d
            )
            penetration_loss, max_penetration_loss, _ = gendex_penetration_energy(
                hand_surface, object_surface_points, object_surface_normals
            )
        else:
            penetration_loss = torch.zeros_like(contact_loss)
            max_penetration_loss = torch.zeros_like(contact_loss)
        total = (
            float(contact_weight) * contact_loss
            + float(penetration_weight) * penetration_loss
            + float(max_penetration_weight) * max_penetration_loss
            + float(joint_regularization) * joint_loss
            + float(translation_regularization) * translation_loss
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
        joint_losses = (((joints - q0) / gripper.span) ** 2).mean(dim=1)
        translation_distance = torch.linalg.norm(translation - object_center, dim=1)
        translation_losses = F.relu(translation_distance - bbox_diagonal) ** 2
        if has_collision_geometry:
            hand_surface = gripper.surface_points(joints, translation, rotation_6d)
            penetration_losses, max_penetrations, penetration_fractions = (
                gendex_penetration_energy(
                    hand_surface, object_surface_points, object_surface_normals
                )
            )
        else:
            penetration_losses = torch.zeros_like(contact_losses)
            max_penetrations = torch.zeros_like(contact_losses)
            penetration_fractions = torch.zeros_like(contact_losses)
        losses = (
            float(contact_weight) * contact_losses
            + float(penetration_weight) * penetration_losses
            + float(max_penetration_weight) * max_penetrations
            + float(joint_regularization) * joint_losses
            + float(translation_regularization) * translation_losses
        )
        count = min(int(top_k), particles)
        best = torch.topk(losses, k=count, largest=False).indices
        poses = gripper.root_pose_matrix(translation, rotation_6d)

    candidates = []
    for rank, particle_index in enumerate(best.tolist()):
        candidates.append(
            {
                "rank": rank,
                "particle": particle_index,
                "optimization_score": float(losses[particle_index].item()),
                "contact_chamfer_m": float(contact_losses[particle_index].item()),
                "mean_penetration_m": float(penetration_losses[particle_index].item()),
                "max_penetration_m": float(max_penetrations[particle_index].item()),
                "penetrating_surface_fraction": float(
                    penetration_fractions[particle_index].item()
                ),
                "root_pose": poses[particle_index].detach().cpu().tolist(),
                "joint_positions": joints[particle_index].detach().cpu().tolist(),
                "tip_points": tips[particle_index].detach().cpu().tolist(),
            }
        )
    return {
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
    gripper_root = Path(config["paths"]["gripper_root"])
    return DifferentiableGripper(
        gripper_root / spec["urdf"],
        list(spec["tip_links"]),
        {link: value["offset_xyz"] for link, value in calibrated["tips"].items()},
        calibrated["base_alignment_matrix"],
        surface_points_per_link=int(
            config.get("fk_optimization", {}).get("surface_points_per_link", 0)
        ),
        device=device,
    )
