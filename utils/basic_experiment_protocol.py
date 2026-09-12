"""Frozen protocol helpers for the ContactDiffusion D(R,O)-aligned baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


LEGACY_PROTOCOL_ID = "contactdiff-multidex45k-fk64x32x400-dro-gym-v1"
O10I20_V2_PROTOCOL_ID = (
    "contactdiff-multidex45k-fk64x32x400-dro-gym-o10i20-v2"
)
FILTERED50K_EAWQ_V3_PROTOCOL_ID = (
    "contactdiff-multidex-filtered50k-fk64x32x400-"
    "eawq-rankfusion-dro-gym-o10i20-v3"
)
PROTOCOL_ID = (
    "contactdiff-multidex-filtered50k-fk64x32x400-"
    "eawq-rankfusion-dro-gym-o10i20-palm0-v4"
)
PARTIAL_AR64K_V5_PROTOCOL_ID = (
    "contactdiff-partial-ar-k128-step64k-fk32x32x400-"
    "eawq-rankfusion-dro-gym-o10i20-palm0-v5"
)
MIXED_FULL_PARTIAL_AR56K_V6_PROTOCOL_ID = (
    "contactdiff-mixed-full-partial-ar-k128-step56k-fk32x32x400-"
    "eawq-rankfusion-dro-gym-o10i20-palm0-v6"
)
ALL_PARTICLE_PROTOCOL_ID = (
    "contactdiff-multidex45k-fk64x32x400-all32-dro-gym-o10i20-v1"
)
FETCHBENCH_AR32K_V7_PROTOCOL_ID = (
    "contactdiff-fetchbench-real60-synth20-full20-ar-k128-step32k-fk32x32x400-"
    "eawq-rankfusion-dro-gym-o10i20-palm0-v7"
)
MODEL_145K_ALL_PARTICLE_PROTOCOL_ID = (
    "contactdiff-multidex145k-fk64x32x400-all32-dro-gym-o10i20-v1"
)
SUPPORTED_PROTOCOL_IDS = frozenset(
    (
        LEGACY_PROTOCOL_ID,
        O10I20_V2_PROTOCOL_ID,
        FILTERED50K_EAWQ_V3_PROTOCOL_ID,
        PROTOCOL_ID,
        PARTIAL_AR64K_V5_PROTOCOL_ID,
        MIXED_FULL_PARTIAL_AR56K_V6_PROTOCOL_ID,
        FETCHBENCH_AR32K_V7_PROTOCOL_ID,
    )
)
UNIFIED_CLOSURE_OUTER_FRACTION = 0.10
UNIFIED_CLOSURE_INNER_FRACTION = 0.20
VIRTUAL_ROOT_JOINTS = (
    "virtual_joint_x",
    "virtual_joint_y",
    "virtual_joint_z",
    "virtual_joint_roll",
    "virtual_joint_pitch",
    "virtual_joint_yaw",
)
DRO_DIRECTIONS = (
    ("+x", (1.0, 0.0, 0.0)),
    ("+y", (0.0, 1.0, 0.0)),
    ("+z", (0.0, 0.0, 1.0)),
    ("-x", (-1.0, 0.0, 0.0)),
    ("-y", (0.0, -1.0, 0.0)),
    ("-z", (0.0, 0.0, -1.0)),
)


def _urdf_origin_transform(joint: ET.Element) -> np.ndarray:
    """Return a URDF joint origin as a homogeneous transform."""

    origin = joint.find("origin")
    xyz = np.zeros(3, dtype=np.float64)
    rpy = np.zeros(3, dtype=np.float64)
    if origin is not None:
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError(f"invalid URDF joint origin on {joint.attrib.get('name')}")
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)))
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)))
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = xyz
    return transform


def derive_fk_to_sim_root_post_transform(
    fk_urdf: str | Path, simulation_urdf: str | Path
) -> np.ndarray:
    """Align an FK root pose with a virtual-root simulation URDF.

    FK candidates apply their optimized root transform outside the FK URDF.
    A simulation URDF instead applies that transform through virtual joints.
    If the two URDFs use different fixed transforms immediately before the
    first physical hand link, the optimized root must absorb that difference.
    """

    fk_root = ET.parse(fk_urdf).getroot()
    sim_root = ET.parse(simulation_urdf).getroot()

    def root_link(robot: ET.Element) -> str:
        links = {link.attrib["name"] for link in robot.findall("link")}
        children = {
            child.attrib["link"]
            for joint in robot.findall("joint")
            for child in joint.findall("child")
        }
        roots = sorted(links - children)
        if len(roots) != 1:
            raise ValueError(f"expected one URDF root link, got {roots}")
        return roots[0]

    fk_root_name = root_link(fk_root)
    fk_root_joints = [
        joint
        for joint in fk_root.findall("joint")
        if joint.find("parent") is not None
        and joint.find("parent").attrib["link"] == fk_root_name
    ]
    fixed_fk_root_joints = [
        joint for joint in fk_root_joints if joint.attrib.get("type") == "fixed"
    ]
    if len(fixed_fk_root_joints) == 1:
        fk_anchor_joint = fixed_fk_root_joints[0]
        anchor_link = fk_anchor_joint.find("child").attrib["link"]
        fk_anchor = _urdf_origin_transform(fk_anchor_joint)
    elif not fixed_fk_root_joints:
        # Some hands, including Barrett, use the physical palm/base itself as
        # the FK tree root and therefore have no separate fixed root joint.
        anchor_link = fk_root_name
        fk_anchor = np.eye(4, dtype=np.float64)
    else:
        raise ValueError("FK URDF has ambiguous fixed joints below its root link")

    sim_anchor_joints = [
        joint
        for joint in sim_root.findall("joint")
        if joint.find("child") is not None
        and joint.find("child").attrib["link"] == anchor_link
    ]
    if (
        len(sim_anchor_joints) != 1
        or sim_anchor_joints[0].attrib.get("type") != "fixed"
    ):
        raise ValueError(
            f"simulation URDF must have one fixed joint into {anchor_link}"
        )
    sim_anchor = _urdf_origin_transform(sim_anchor_joints[0])
    correction = fk_anchor @ np.linalg.inv(sim_anchor)
    if not np.isfinite(correction).all():
        raise ValueError("FK-to-simulation root correction is not finite")
    return correction


def apply_root_post_transform(
    root_pose: Sequence[Sequence[float]],
    post_transform: Sequence[Sequence[float]],
) -> np.ndarray:
    """Compose a local-frame correction onto an optimized root pose."""

    pose = np.asarray(root_pose, dtype=np.float64)
    correction = np.asarray(post_transform, dtype=np.float64)
    if pose.shape != (4, 4) or correction.shape != (4, 4):
        raise ValueError("root_pose and post_transform must both be 4x4")
    if not np.isfinite(pose).all() or not np.isfinite(correction).all():
        raise ValueError("root transform contains NaN or Inf")
    return pose @ correction


def transform_object_points(
    points: Sequence[Sequence[float]],
    position: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """Transform object-local points by an Isaac Gym rigid-body pose."""

    local = np.asarray(points, dtype=np.float64)
    translation = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if local.ndim != 2 or local.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if translation.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("position/quaternion must have shape [3]/[4]")
    if not (
        np.isfinite(local).all()
        and np.isfinite(translation).all()
        and np.isfinite(quaternion).all()
    ):
        raise ValueError("object transform contains NaN or Inf")
    norm = np.linalg.norm(quaternion)
    if norm <= 1.0e-12:
        raise ValueError("object quaternion has zero norm")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return local @ rotation.T + translation


def project_world_points_to_camera(
    points: Sequence[Sequence[float]],
    *,
    eye: Sequence[float],
    target: Sequence[float],
    width: int,
    height: int,
    horizontal_fov_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into the fixed look-at camera used by Gym videos."""

    world = np.asarray(points, dtype=np.float64)
    camera_eye = np.asarray(eye, dtype=np.float64)
    camera_target = np.asarray(target, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if camera_eye.shape != (3,) or camera_target.shape != (3,):
        raise ValueError("camera eye and target must have shape [3]")
    if width <= 0 or height <= 0 or not 0.0 < horizontal_fov_degrees < 180.0:
        raise ValueError("invalid camera intrinsics")
    forward = camera_target - camera_eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = world - camera_eye
    depth = relative @ forward
    focal = width / (
        2.0 * np.tan(np.deg2rad(horizontal_fov_degrees) / 2.0)
    )
    safe_depth = np.maximum(depth, 1.0e-9)
    pixels = np.stack(
        (
            width / 2.0 + focal * (relative @ right) / safe_depth,
            height / 2.0 - focal * (relative @ up) / safe_depth,
        ),
        axis=1,
    )
    return pixels, depth


def seed_for(
    base_seed: int,
    *,
    hand_index: int,
    object_index: int,
    sample_index: int,
) -> int:
    """Return the frozen, collision-resistant per-sample seed."""

    return (
        int(base_seed)
        + 1_000_003 * int(hand_index)
        + 1_009 * int(object_index)
        + int(sample_index)
    )


def ordered_joint_array(
    values: Mapping[str, float] | Sequence[float],
    joint_names: Sequence[str],
    *,
    label: str,
) -> np.ndarray:
    names = list(joint_names)
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        extra = [name for name in values if name not in names]
        if missing or extra:
            raise ValueError(
                f"{label} joint mismatch: missing={missing}, extra={extra}"
            )
        result = np.asarray([values[name] for name in names], dtype=np.float64)
    else:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (len(names),):
            raise ValueError(
                f"{label} has shape {result.shape}, expected {(len(names),)}"
            )
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return result


def method_specific_closure_targets(
    q_contact: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    close_direction: Sequence[float],
    *,
    outer_fraction: float,
    inner_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build method-native outer/inner targets around an FK contact pose.

    The first six values are the fixed virtual-root pose. Hand joints with a
    zero close direction are held at their FK value. All other hand joints are
    moved toward the configured open/close limit by the frozen fractions.
    """

    contact = np.asarray(q_contact, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    direction = np.asarray(close_direction, dtype=np.float64)
    if contact.ndim != 1 or contact.size != low.size + len(VIRTUAL_ROOT_JOINTS):
        raise ValueError("q_contact must contain six root values plus hand joints")
    if low.shape != high.shape or low.shape != direction.shape:
        raise ValueError("joint limits and close_direction must have equal shape")
    if np.any(high <= low):
        raise ValueError("every joint upper limit must exceed its lower limit")
    if not 0.0 <= float(outer_fraction) <= 1.0:
        raise ValueError("outer_fraction must be in [0, 1]")
    if not 0.0 <= float(inner_fraction) <= 1.0:
        raise ValueError("inner_fraction must be in [0, 1]")
    joints = contact[len(VIRTUAL_ROOT_JOINTS) :]
    if np.any(joints < low - 1.0e-7) or np.any(joints > high + 1.0e-7):
        raise ValueError("q_contact contains a joint outside its limits")
    active = direction != 0.0
    open_limit = np.where(direction > 0.0, low, high)
    close_limit = np.where(direction > 0.0, high, low)
    outer_joints = np.where(
        active, joints + float(outer_fraction) * (open_limit - joints), joints
    )
    inner_joints = np.where(
        active, joints + float(inner_fraction) * (close_limit - joints), joints
    )
    root = contact[: len(VIRTUAL_ROOT_JOINTS)]
    return (
        np.concatenate((root, np.clip(outer_joints, low, high))),
        np.concatenate((root, np.clip(inner_joints, low, high))),
    )


def dro_displacement_metrics(
    closure_position: Sequence[float],
    direction_endpoints: Sequence[Sequence[float]],
    *,
    threshold_m: float = 0.02,
) -> dict[str, object]:
    """Compute D(R,O) final and strict diagnostic displacement metrics."""

    start = np.asarray(closure_position, dtype=np.float64)
    endpoints = np.asarray(direction_endpoints, dtype=np.float64)
    if start.shape != (3,) or endpoints.shape != (len(DRO_DIRECTIONS), 3):
        raise ValueError("expected one closure position and six XYZ endpoints")
    if not np.isfinite(start).all() or not np.isfinite(endpoints).all():
        raise ValueError("positions contain NaN or Inf")
    previous = np.vstack((start[None, :], endpoints[:-1]))
    segments = np.linalg.norm(endpoints - previous, axis=1)
    cumulative = np.linalg.norm(endpoints - start[None, :], axis=1)
    return {
        "segment_displacements_m": segments.tolist(),
        "cumulative_displacements_m": cumulative.tolist(),
        "final_displacement_m": float(cumulative[-1]),
        "dro_success": bool(cumulative[-1] <= float(threshold_m)),
        "strict_six_direction_success": bool(
            np.all(segments <= float(threshold_m))
        ),
    }
