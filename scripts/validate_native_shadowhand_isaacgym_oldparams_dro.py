#!/usr/bin/env python3
"""Validate native-hand grasps with configurable old or D(R,O) Gym profiles."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from isaacgym import gymapi, gymtorch, gymutil
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.basic_experiment_protocol import (
    FILTERED50K_EAWQ_V3_PROTOCOL_ID,
    LEGACY_PROTOCOL_ID,
    O10I20_V2_PROTOCOL_ID,
    PROTOCOL_ID,
    SUPPORTED_PROTOCOL_IDS,
    UNIFIED_CLOSURE_INNER_FRACTION,
    UNIFIED_CLOSURE_OUTER_FRACTION,
    project_world_points_to_camera,
    transform_object_points,
)


DIRECTIONS_GENDEX = (
    ("+x", (1.0, 0.0, 0.0)),
    ("-x", (-1.0, 0.0, 0.0)),
    ("+y", (0.0, 1.0, 0.0)),
    ("-y", (0.0, -1.0, 0.0)),
    ("+z", (0.0, 0.0, 1.0)),
    ("-z", (0.0, 0.0, -1.0)),
)
DIRECTIONS_CEDEX = (
    ("+x", (1.0, 0.0, 0.0)),
    ("+y", (0.0, 1.0, 0.0)),
    ("+z", (0.0, 0.0, 1.0)),
    ("-x", (-1.0, 0.0, 0.0)),
    ("-y", (0.0, -1.0, 0.0)),
    ("-z", (0.0, 0.0, -1.0)),
)
OBJECT_MAP = {
    "contactdb_apple": ("contactdb", "apple"),
    "contactdb_camera": ("contactdb", "camera"),
    "contactdb_cylinder_medium": ("contactdb", "cylinder_medium"),
    "contactdb_door_knob": ("contactdb", "door_knob"),
    "contactdb_rubber_duck": ("contactdb", "rubber_duck"),
    "contactdb_water_bottle": ("contactdb", "water_bottle"),
    "ycb_055_baseball": ("ycb", "baseball"),
    "ycb_016_pear": ("ycb", "pear"),
    "ycb_010_potted_meat_can": ("ycb", "potted_meat_can"),
    "ycb_005_tomato_soup_can": ("ycb", "tomato_soup_can"),
}
VIRTUAL_ROOT_JOINTS = (
    "virtual_joint_x",
    "virtual_joint_y",
    "virtual_joint_z",
    "virtual_joint_roll",
    "virtual_joint_pitch",
    "virtual_joint_yaw",
)
BASIC_PROTOCOL_ID = PROTOCOL_ID
CONTACT_COLORS_RGB = (
    (1.0, 0.16, 0.16),
    (0.16, 0.90, 0.16),
    (1.0, 0.90, 0.12),
    (0.90, 0.16, 0.90),
    (0.16, 0.86, 0.90),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_basic_protocol(
    prepared: dict, args: argparse.Namespace
) -> None:
    """Fail fast if a frozen-baseline manifest is run with drifted Gym args."""

    base_protocol_id = prepared.get(
        "base_protocol_id", prepared.get("protocol_id")
    )
    if base_protocol_id not in SUPPORTED_PROTOCOL_IDS:
        return
    expected = {
        "asset_profile": "dro",
        "object_source": "dro",
        "native_hand_urdf_is_extended": True,
        "steps_per_second": 100,
        "substeps": 2,
        "closure_steps": 100,
        "closure_trajectory": "step",
        "inner_hold_steps": 0,
        "direction_seconds": 1.0,
        "direction_order": "cedex",
        "success_mode": "final",
        "threshold": 0.02,
        "acceleration": 0.5,
        "robot_friction": 3.0,
        "object_friction": 3.0,
        "object_density": 500.0,
        "joint_stiffness": 1000.0,
        "joint_damping": 200.0,
        "pregrasp_open_fraction": 0.0,
        "closure_overdrive_fraction": 0.0,
        "outer_settle_steps": 0,
        "virtual_root_stiffness": 1000.0,
        "virtual_root_damping": 200.0,
        "solver_position_iterations": 8,
        "solver_velocity_iterations": 0,
        "contact_offset": 0.01,
        "rest_offset": 0.0,
        "no_ground": True,
    }
    if args.closure_ab_experiment:
        # The full-particle closure A/B deliberately adds the same post-inner
        # hold to both arms.  All other frozen execution parameters remain
        # protected by this drift check.
        expected["inner_hold_steps"] = args.inner_hold_steps
    drift = {}
    for name, wanted in expected.items():
        actual = getattr(args, name)
        equal = (
            math.isclose(float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-9)
            if isinstance(wanted, float)
            else actual == wanted
        )
        if not equal:
            drift[name] = {"expected": wanted, "actual": actual}
    if drift:
        raise ValueError(f"Frozen basic protocol argument drift: {drift}")
    if base_protocol_id in {
        O10I20_V2_PROTOCOL_ID,
        FILTERED50K_EAWQ_V3_PROTOCOL_ID,
        PROTOCOL_ID,
    }:
        closure = prepared.get("closure_adapter", {})
        expected_closure = {
            "outer_fraction": UNIFIED_CLOSURE_OUTER_FRACTION,
            "inner_fraction": UNIFIED_CLOSURE_INNER_FRACTION,
        }
        closure_drift = {
            name: {"expected": wanted, "actual": closure.get(name)}
            for name, wanted in expected_closure.items()
            if not math.isclose(
                float(closure.get(name, float("nan"))),
                wanted,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        }
        if closure_drift:
            raise ValueError(
                f"Unified O10/I20 closure drift: {closure_drift}"
            )
        execution_protocol_path = prepared.get(
            "execution_protocol_config"
        )
        execution_protocol_hash = prepared.get(
            "execution_protocol_config_sha256"
        )
        if not execution_protocol_path or not execution_protocol_hash:
            raise ValueError(
                "Unified O10/I20 protocol requires execution-config provenance"
            )
        if (
            sha256(Path(execution_protocol_path).resolve())
            != execution_protocol_hash
        ):
            raise ValueError("Execution protocol config SHA256 mismatch")
    elif base_protocol_id != LEGACY_PROTOCOL_ID:
        raise ValueError(f"Unsupported base protocol: {base_protocol_id}")
    actual_urdf = (args.native_hand_root / args.native_hand_urdf).resolve()
    prepared_urdf = Path(prepared["simulation_hand_urdf"]).resolve()
    if actual_urdf != prepared_urdf:
        raise ValueError(
            f"Frozen hand URDF path mismatch: {actual_urdf} != {prepared_urdf}"
        )
    actual_hash = sha256(actual_urdf)
    if actual_hash != prepared["simulation_hand_urdf_sha256"]:
        raise ValueError("Frozen hand URDF SHA256 mismatch")
    root_adapter = prepared.get("root_pose_adapter")
    if (
        not isinstance(root_adapter, dict)
        or root_adapter.get("name")
        != "fk_urdf_to_simulation_urdf_fixed_base_v1"
    ):
        raise ValueError(
            "Frozen basic protocol requires the FK-to-simulation root adapter"
        )
    fk_urdf = Path(root_adapter["fk_hand_urdf"]).resolve()
    if sha256(fk_urdf) != root_adapter.get("fk_hand_urdf_sha256"):
        raise ValueError("Frozen FK hand URDF SHA256 mismatch")
    post_transform = np.asarray(root_adapter.get("post_transform"), dtype=float)
    if post_transform.shape != (4, 4) or not np.isfinite(post_transform).all():
        raise ValueError("Frozen root post-transform must be a finite 4x4 matrix")
    if not np.allclose(post_transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("Frozen root post-transform has an invalid last row")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gendex-root", type=Path, required=True)
    parser.add_argument("--native-hand-root", type=Path, required=True)
    parser.add_argument(
        "--native-hand-urdf", default="shadow_hand_right_glb.urdf"
    )
    parser.add_argument(
        "--native-hand-urdf-is-extended",
        action="store_true",
        help=(
            "Load an existing six-virtual-joint D(R,O) extended URDF as-is "
            "instead of generating a movable wrapper around a base URDF."
        ),
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--cpu-physics",
        action="store_true",
        help=(
            "Run PhysX and tensor control on CPU while retaining the GPU "
            "graphics device for native Isaac Gym video capture."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-object", action="append")
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Zero-based offset into each prepared object's candidate list.",
    )
    parser.add_argument("--max-samples-per-object", type=int)
    parser.add_argument(
        "--envs-per-row",
        type=int,
        help=(
            "Override the Isaac Gym environment grid width. By default it is "
            "ceil(sqrt(active env count)). Intended for controlled layout audits."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the native Isaac Gym GUI while the validation runs.",
    )
    parser.add_argument(
        "--viewer-show-diffusion-contacts",
        action="store_true",
        help=(
            "Draw object-attached, non-physical wireframe spheres at the "
            "diffusion target contacts in the native GUI. Requires --viewer."
        ),
    )
    parser.add_argument(
        "--viewer-contact-radius-m",
        type=float,
        default=0.006,
        help="Radius in metres of GUI diffusion-contact markers.",
    )
    parser.add_argument(
        "--viewer-no-sync",
        action="store_true",
        help="Do not pace GUI rendering to simulation time.",
    )
    parser.add_argument(
        "--viewer-free-camera",
        action="store_true",
        help=(
            "Start with Isaac Gym's free mouse camera instead of the default "
            "object-coordinate tracking camera. Press C to toggle at runtime."
        ),
    )
    parser.add_argument(
        "--viewer-focus-sample",
        type=int,
        default=0,
        help="Zero-based selected-sample index tracked by the GUI camera.",
    )
    parser.add_argument(
        "--asset-profile",
        choices=("old", "dro"),
        default="old",
        help=(
            "Use the existing recomputed-inertia/VHACD import profile or the "
            "minimal original D(R,O) hand import profile."
        ),
    )
    parser.add_argument(
        "--object-vhacd",
        action="store_true",
        help=(
            "Enable runtime VHACD for prepared raw object meshes. This is a "
            "documented compatibility substitute when the original D(R,O) "
            "COACD object asset is unavailable locally."
        ),
    )
    parser.add_argument(
        "--object-predecomposed",
        action="store_true",
        help=(
            "Load an already convex-decomposed object collision mesh without "
            "running Isaac Gym VHACD. This is independent of the hand asset "
            "profile so parameter-only comparisons can preserve the local hand."
        ),
    )
    parser.add_argument(
        "--virtual-root-stiffness",
        type=float,
        default=400.0,
        help="Position-drive stiffness for the six generated root joints.",
    )
    parser.add_argument(
        "--virtual-root-damping",
        type=float,
        default=400.0,
        help="Position-drive damping for the six generated root joints.",
    )
    parser.add_argument("--steps-per-second", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--closure-steps", type=int, default=200)
    parser.add_argument(
        "--closure-trajectory",
        choices=("step", "linear", "smoothstep"),
        default="step",
        help=(
            "Command the final inner target immediately (step), interpolate "
            "linearly, or use zero-end-velocity cubic smoothstep interpolation."
        ),
    )
    parser.add_argument(
        "--inner-hold-steps",
        type=int,
        default=0,
        help="Hold the final inner target before starting force evaluation.",
    )
    parser.add_argument(
        "--closure-object-mode",
        choices=("dynamic", "fixed_until_inner"),
        default="dynamic",
        help=(
            "Keep the object dynamic for the entire closure, or collide with "
            "an equivalent fixed-base object during outer settle/closure and "
            "atomically switch to a zero-velocity dynamic actor at inner."
        ),
    )
    parser.add_argument(
        "--closure-ab-experiment",
        action="store_true",
        help=(
            "Mark this as the controlled full-particle closure A/B extension; "
            "permits an equal post-inner hold in both arms of a frozen protocol."
        ),
    )
    parser.add_argument(
        "--closure-telemetry-dir",
        type=Path,
        help=(
            "Write one compressed dense per-frame closure telemetry NPZ for "
            "this object/sample window. Also adds compact closure summaries "
            "to each JSON result row."
        ),
    )
    parser.add_argument(
        "--contact-impulse-epsilon",
        type=float,
        default=1.0e-9,
        help="Minimum absolute PhysX normal impulse counted as object-hand contact.",
    )
    parser.add_argument("--direction-seconds", type=float, default=5.0 / 6.0)
    parser.add_argument("--direction-order", choices=("gendex", "cedex"), default="gendex")
    parser.add_argument(
        "--max-directions",
        type=int,
        help=(
            "Run only this many leading disturbance directions for a short "
            "physics prescreen. Omit for the formal six-direction evaluation."
        ),
    )
    parser.add_argument("--success-mode", choices=("final", "per_direction"), default="per_direction")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--acceleration", type=float, default=0.5)
    parser.add_argument("--robot-friction", type=float, default=10.0)
    parser.add_argument("--object-friction", type=float, default=10.0)
    parser.add_argument("--object-density", type=float, default=10000.0)
    parser.add_argument(
        "--hand-density",
        type=float,
        help=(
            "Hand import density. If omitted, preserve legacy behavior by "
            "using --object-density for both assets."
        ),
    )
    parser.add_argument("--object-linear-damping", type=float, default=10.0)
    parser.add_argument("--object-angular-damping", type=float, default=100.0)
    parser.add_argument("--joint-stiffness", type=float, default=400.0)
    parser.add_argument("--joint-damping", type=float, default=400.0)
    parser.add_argument(
        "--finger-joint-max-effort",
        type=float,
        help=(
            "Override the imported maximum effort for non-virtual hand DOFs. "
            "This is required for Barrett URDFs whose finger limits use effort=0."
        ),
    )
    parser.add_argument(
        "--finger-joint-velocity",
        type=float,
        help="Override velocity only for non-virtual finger DOFs.",
    )
    parser.add_argument("--joint-armature", type=float, default=0.01)
    parser.add_argument("--joint-velocity", type=float, default=0.8)
    parser.add_argument(
        "--pregrasp-open-fraction",
        type=float,
        default=0.0,
        help=(
            "Move each closing finger DOF from the prepared outer pose toward "
            "its open joint limit by this fraction before closure."
        ),
    )
    parser.add_argument(
        "--closure-overdrive-fraction",
        type=float,
        default=0.0,
        help=(
            "Move each prepared inner finger target farther toward its closing "
            "joint limit by this fraction."
        ),
    )
    parser.add_argument("--outer-settle-steps", type=int, default=3)
    parser.add_argument("--solver-position-iterations", type=int, default=4)
    parser.add_argument("--solver-velocity-iterations", type=int, default=0)
    parser.add_argument("--contact-offset", type=float, default=0.01)
    parser.add_argument("--rest-offset", type=float, default=0.0)
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument(
        "--object-source",
        choices=("gendex", "prepared", "dro"),
        default="gendex",
        help=(
            "Load GenDex objects, build a one-link prepared raw-mesh URDF, "
            "or load original D(R,O) COACD object assets."
        ),
    )
    parser.add_argument(
        "--dro-object-root",
        type=Path,
        help="Root containing D(R,O) contactdb/ycb COACD object directories.",
    )
    parser.add_argument(
        "--prepared-object-mesh-root",
        type=Path,
        help=(
            "Resolve prepared object meshes as ROOT/object_name.obj instead of "
            "using paths embedded in a manifest downloaded from another host."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Record one native Isaac Gym RGB MP4 for every selected trial.",
    )
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument(
        "--video-show-diffusion-contacts",
        action="store_true",
        help=(
            "Overlay numbered, object-attached projections of the original "
            "diffusion target contacts on every recorded RGB frame. The "
            "markers remain visible through occluding hand links."
        ),
    )
    parser.add_argument(
        "--video-contact-radius-px",
        type=int,
        default=7,
        help="Radius of diffusion-contact overlay rings in output pixels.",
    )
    parser.add_argument(
        "--video-hand-label",
        help="Hand/protocol label rendered into recorded video frames.",
    )
    parser.add_argument(
        "--video-hand-color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="Override hand visual color during video capture.",
    )
    parser.add_argument(
        "--video-object-color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="Override object visual color during video capture.",
    )
    parser.add_argument(
        "--video-stride",
        type=int,
        default=4,
        help="Capture one frame every N physics steps, plus phase boundaries.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Record rigid-body trajectories for model-based offline rendering.",
    )
    parser.add_argument("--state-stride", type=int, default=4)
    args = parser.parse_args()
    if args.viewer_show_diffusion_contacts and not args.viewer:
        parser.error("--viewer-show-diffusion-contacts requires --viewer")
    if args.viewer_contact_radius_m <= 0.0:
        parser.error("--viewer-contact-radius-m must be positive")
    if args.viewer_focus_sample < 0:
        parser.error("--viewer-focus-sample must be non-negative")
    if args.contact_impulse_epsilon < 0.0:
        parser.error("--contact-impulse-epsilon must be non-negative")
    if args.closure_object_mode != "dynamic" and not args.closure_ab_experiment:
        parser.error("fixed_until_inner requires --closure-ab-experiment")
    return args


def select_sample_window(samples: list[dict], args: argparse.Namespace) -> list[dict]:
    start = max(0, int(args.sample_start))
    stop = len(samples)
    if args.max_samples_per_object is not None:
        stop = start + max(0, int(args.max_samples_per_object))
    return samples[start:stop]


def sample_artifact_stem(object_name: str, sample: dict) -> str:
    return (
        f"{object_name}_{int(sample['source_index']):03d}"
        f"_rank{int(sample.get('candidate_rank', 0)):02d}"
    )


def euler_xyz_quaternion(euler: list[float]) -> gymapi.Quat:
    roll, pitch, yaw = (float(value) for value in euler)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return gymapi.Quat(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def make_sim(gym, args: argparse.Namespace):
    params = gymapi.SimParams()
    params.dt = 1.0 / args.steps_per_second
    params.substeps = args.substeps
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    params.num_client_threads = 0
    params.physx.solver_type = 1
    params.physx.num_position_iterations = args.solver_position_iterations
    params.physx.num_velocity_iterations = args.solver_velocity_iterations
    params.physx.contact_offset = args.contact_offset
    params.physx.rest_offset = args.rest_offset
    params.physx.num_threads = 4 if args.cpu_physics else 0
    params.physx.use_gpu = not args.cpu_physics
    params.physx.num_subscenes = 0
    params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    if args.closure_telemetry_dir is not None:
        # Preserve every substep contact so the per-frame sum below represents
        # the complete normal impulse delivered during one control frame.
        params.physx.contact_collection = (
            gymapi.ContactCollection.CC_ALL_SUBSTEPS
        )
    params.use_gpu_pipeline = not args.cpu_physics
    graphics_device = (
        args.device_id if args.video_dir is not None or args.viewer else -1
    )
    sim = gym.create_sim(
        args.device_id, graphics_device, gymapi.SIM_PHYSX, params
    )
    if sim is None:
        backend = "CPU" if args.cpu_physics else "GPU"
        raise RuntimeError(
            f"Isaac Gym failed to create the {backend} PhysX simulation"
        )
    if not args.no_ground:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane.distance = 1.0
        plane.static_friction = 0.1
        plane.dynamic_friction = 0.1
        gym.add_ground(sim, plane)
    return sim


def hand_asset_options(args: argparse.Namespace) -> gymapi.AssetOptions:
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    options.disable_gravity = True
    if args.asset_profile == "dro":
        # Match D(R,O) validation/isaac_validator.py.  D(R,O) only fixes the
        # base, disables gravity and collapses fixed hand joints here.
        options.collapse_fixed_joints = True
    else:
        options.density = (
            args.hand_density
            if args.hand_density is not None
            else args.object_density
        )
        options.flip_visual_attachments = False
        if args.joint_armature >= 0.0:
            options.armature = args.joint_armature
        options.use_mesh_materials = True
        options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        options.override_com = True
        options.override_inertia = True
        options.vhacd_enabled = True
        options.vhacd_params = gymapi.VhacdParams()
        options.vhacd_params.resolution = 1_000_000
    return options


def object_asset_options(
    args: argparse.Namespace, *, fixed_base: bool = False
) -> gymapi.AssetOptions:
    options = gymapi.AssetOptions()
    options.fix_base_link = fixed_base
    options.density = args.object_density
    options.override_com = True
    options.override_inertia = True
    if args.asset_profile == "dro":
        # The original D(R,O) asset is already COACD decomposed and therefore
        # does not request VHACD.  Local raw-mesh replays may opt into VHACD
        # explicitly with --object-vhacd.
        options.vhacd_enabled = args.object_vhacd
        if args.object_vhacd:
            options.vhacd_params = gymapi.VhacdParams()
            options.vhacd_params.resolution = 1_000_000
    else:
        if args.object_linear_damping >= 0.0:
            options.linear_damping = args.object_linear_damping
        if args.object_angular_damping >= 0.0:
            options.angular_damping = args.object_angular_damping
        options.disable_gravity = True
        options.use_mesh_materials = True
        options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        options.vhacd_enabled = not args.object_predecomposed
        if options.vhacd_enabled:
            options.vhacd_params = gymapi.VhacdParams()
            options.vhacd_params.resolution = 1_000_000
    return options


def automatic_thumb_links(hand_body_names: list[str]) -> set[str]:
    """Return the anatomical/opposing digit links for supported hands."""

    shadow = {name for name in hand_body_names if name.lower().startswith("th")}
    if shadow:
        return shadow
    barrett = {
        name
        for name in hand_body_names
        if name.lower().startswith("bh_finger_3")
    }
    if barrett:
        return barrett
    return {name for name in hand_body_names if "thumb" in name.lower()}


def inertial_link(name: str) -> ET.Element:
    link = ET.Element("link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.001"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "1e-6",
            "ixy": "0",
            "ixz": "0",
            "iyy": "1e-6",
            "iyz": "0",
            "izz": "1e-6",
        },
    )
    return link


def prepared_object_urdf(mesh_path: Path, cache_dir: Path, object_name: str) -> Path:
    """Create a deterministic one-link URDF using the exact prepared mesh."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = object_name.replace("+", "_").replace("/", "_")
    output = cache_dir / f"{safe_name}.urdf"
    robot = ET.Element("robot", {"name": safe_name})
    link = ET.SubElement(robot, "link", {"name": "object"})
    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            {"filename": str(mesh_path.resolve()), "scale": "1 1 1"},
        )
    temporary = output.with_suffix(".urdf.tmp")
    ET.ElementTree(robot).write(
        temporary, encoding="utf-8", xml_declaration=True
    )
    temporary.replace(output)
    return output


def ensure_movable_native_urdf(
    hand_root: Path, source_name: str
) -> Path:
    source = (hand_root / source_name).resolve()
    output = hand_root / f"movable_{source.stem}.urdf"
    lock_path = hand_root / f".{output.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if output.is_file() and output.stat().st_mtime >= source.stat().st_mtime:
        lock.close()
        return output
    tree = ET.parse(source)
    robot = tree.getroot()
    original_links = {link.attrib["name"] for link in robot.findall("link")}
    original_children = {
        child.attrib["link"]
        for joint in robot.findall("joint")
        if (child := joint.find("child")) is not None
    }
    original_roots = sorted(original_links - original_children)
    if len(original_roots) != 1:
        raise RuntimeError(
            f"Expected one root link in {source}, got {original_roots}"
        )
    original_root = original_roots[0]
    links = (
        "contactdiff_virtual_anchor",
        "contactdiff_virtual_link_x",
        "contactdiff_virtual_link_y",
        "contactdiff_virtual_link_z",
        "contactdiff_virtual_link_roll",
        "contactdiff_virtual_link_pitch",
        "contactdiff_virtual_link_yaw",
    )
    for name in links:
        robot.insert(0, inertial_link(name))
    chain = (
        (
            "virtual_joint_x",
            "prismatic",
            links[0],
            links[1],
            "1 0 0",
            "-10",
            "10",
            "300",
            "2",
        ),
        (
            "virtual_joint_y",
            "prismatic",
            links[1],
            links[2],
            "0 1 0",
            "-10",
            "10",
            "300",
            "2",
        ),
        (
            "virtual_joint_z",
            "prismatic",
            links[2],
            links[3],
            "0 0 1",
            "-10",
            "10",
            "300",
            "2",
        ),
        (
            "virtual_joint_roll",
            "revolute",
            links[3],
            links[4],
            "1 0 0",
            "-6.283185",
            "6.283185",
            "100",
            "100",
        ),
        (
            "virtual_joint_pitch",
            "revolute",
            links[4],
            links[5],
            "0 1 0",
            "-6.283185",
            "6.283185",
            "100",
            "100",
        ),
        (
            "virtual_joint_yaw",
            "revolute",
            links[5],
            links[6],
            "0 0 1",
            "-6.283185",
            "6.283185",
            "100",
            "100",
        ),
    )
    for name, kind, parent, child, axis, lower, upper, effort, velocity in chain:
        joint = ET.Element("joint", {"name": name, "type": kind})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(joint, "axis", {"xyz": axis})
        ET.SubElement(
            joint,
            "limit",
            {
                "lower": lower,
                "upper": upper,
                "effort": effort,
                "velocity": velocity,
            },
        )
        robot.insert(len(links), joint)
    fixed = ET.Element(
        "joint", {"name": "contactdiff_virtual_robot", "type": "fixed"}
    )
    ET.SubElement(fixed, "parent", {"link": links[-1]})
    ET.SubElement(fixed, "child", {"link": original_root})
    ET.SubElement(fixed, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    robot.insert(len(links) + len(chain), fixed)
    temporary = output.with_suffix(".urdf.tmp")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(output)
    lock.close()
    return output


def validate_object(
    gym,
    args: argparse.Namespace,
    object_group: dict,
) -> list[dict]:
    object_name = str(object_group["object_name"])
    dataset, short_name = OBJECT_MAP[object_name]
    samples = select_sample_window(list(object_group["samples"]), args)
    count = len(samples)
    if not count:
        return []

    sim = make_sim(gym, args)
    viewer = None
    try:
        movable_urdf = (
            (args.native_hand_root / args.native_hand_urdf).resolve()
            if args.native_hand_urdf_is_extended
            else ensure_movable_native_urdf(
                args.native_hand_root.resolve(), args.native_hand_urdf
            )
        )
        if not movable_urdf.is_file():
            raise FileNotFoundError(movable_urdf)
        hand_asset = gym.load_asset(
            sim,
            str(movable_urdf.parent),
            movable_urdf.name,
            hand_asset_options(args),
        )
        if args.object_source == "prepared":
            object_mesh_path = (
                args.prepared_object_mesh_root / f"{object_name}.obj"
                if args.prepared_object_mesh_root is not None
                else Path(object_group["object_mesh"])
            ).resolve()
            if not object_mesh_path.is_file():
                raise FileNotFoundError(object_mesh_path)
            object_urdf = prepared_object_urdf(
                object_mesh_path,
                args.output.resolve().parent / "prepared_object_urdfs",
                object_name,
            )
            object_root = object_urdf.parent
            object_file = object_urdf.name
        elif args.object_source == "dro":
            if args.dro_object_root is None:
                raise ValueError("--dro-object-root is required for --object-source dro")
            object_mesh_path = Path(object_group["object_mesh"]).resolve()
            object_root = args.dro_object_root.resolve()
            object_file = (
                f"{dataset}/{short_name}/"
                "coacd_decomposed_object_one_link.urdf"
            )
            if not (object_root / object_file).is_file():
                raise FileNotFoundError(object_root / object_file)
        else:
            object_mesh_path = (
                args.gendex_root
                / "data"
                / "object"
                / dataset
                / short_name
                / f"{short_name}.stl"
            )
            object_root = (args.gendex_root / "data").resolve()
            object_file = f"object/{dataset}/{short_name}/{short_name}.urdf"
        object_asset = gym.load_asset(
            sim,
            str(object_root),
            object_file,
            object_asset_options(args, fixed_base=False),
        )
        closure_object_asset = (
            gym.load_asset(
                sim,
                str(object_root),
                object_file,
                object_asset_options(args, fixed_base=True),
            )
            if args.closure_object_mode == "fixed_until_inner"
            else object_asset
        )
        hand_dof_names = list(gym.get_asset_dof_names(hand_asset))
        source_joint_names = list(object_group["_joint_names"])
        if set(hand_dof_names) != set(source_joint_names):
            raise RuntimeError(
                f"{object_name}: hand DOF mismatch; gym={hand_dof_names}, "
                f"prepared={source_joint_names}"
            )
        source_index = {name: index for index, name in enumerate(source_joint_names)}
        reorder = np.asarray(
            [source_index[name] for name in hand_dof_names], dtype=np.int64
        )
        dof_count = len(hand_dof_names)
        dof_props = gym.get_asset_dof_properties(hand_asset)
        dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"][:].fill(args.joint_stiffness)
        if args.joint_velocity >= 0.0:
            dof_props["velocity"][:].fill(args.joint_velocity)
        dof_props["damping"][:].fill(args.joint_damping)
        finger_dof_indices = [
            index
            for index, name in enumerate(hand_dof_names)
            if name not in VIRTUAL_ROOT_JOINTS and name not in {"WRJ1", "WRJ2"}
        ]
        if args.finger_joint_max_effort is not None:
            if args.finger_joint_max_effort <= 0.0:
                raise ValueError("--finger-joint-max-effort must be positive")
            dof_props["effort"][finger_dof_indices] = (
                args.finger_joint_max_effort
            )
        if args.finger_joint_velocity is not None:
            if args.finger_joint_velocity <= 0.0:
                raise ValueError("--finger-joint-velocity must be positive")
            dof_props["velocity"][finger_dof_indices] = (
                args.finger_joint_velocity
            )
        if not 0.0 <= args.pregrasp_open_fraction <= 1.0:
            raise ValueError("--pregrasp-open-fraction must be in [0, 1]")
        if not 0.0 <= args.closure_overdrive_fraction <= 1.0:
            raise ValueError("--closure-overdrive-fraction must be in [0, 1]")
        if args.outer_settle_steps < 0:
            raise ValueError("--outer-settle-steps must be non-negative")
        if args.closure_steps < 1:
            raise ValueError("--closure-steps must be positive")
        if args.inner_hold_steps < 0:
            raise ValueError("--inner-hold-steps must be non-negative")
        dof_lower = np.asarray(dof_props["lower"], dtype=np.float32)
        dof_upper = np.asarray(dof_props["upper"], dtype=np.float32)
        missing_virtual = [
            name for name in VIRTUAL_ROOT_JOINTS if name not in hand_dof_names
        ]
        if missing_virtual:
            raise RuntimeError(
                f"{object_name}: missing virtual root DOFs {missing_virtual}"
            )
        for name in VIRTUAL_ROOT_JOINTS:
            index = hand_dof_names.index(name)
            dof_props["stiffness"][index] = args.virtual_root_stiffness
            dof_props["damping"][index] = args.virtual_root_damping

        lower = gymapi.Vec3(-1.0, -1.0, -1.0)
        upper = gymapi.Vec3(1.0, 1.0, 1.0)
        per_row = (
            int(args.envs_per_row)
            if args.envs_per_row is not None
            else max(1, math.ceil(math.sqrt(count)))
        )
        if per_row < 1:
            raise ValueError("--envs-per-row must be positive")
        envs = []
        camera_handles = []
        object_actor_indices = []
        object_body_indices = []
        object_handles = []
        closure_object_actor_indices = []
        closure_object_body_indices = []
        closure_object_handles = []
        hand_body_indices_by_env = []
        recorded_body_indices = []
        diffusion_contacts = []
        outer_targets = []
        inner_targets = []
        for env_index, sample in enumerate(samples):
            env = gym.create_env(sim, lower, upper, per_row)
            envs.append(env)
            outer = np.asarray(sample["outer_q_euler"], dtype=np.float32)
            inner = np.asarray(sample["inner_q_euler"], dtype=np.float32)
            hand_pose = gymapi.Transform()
            hand_actor = gym.create_actor(
                env, hand_asset, hand_pose, "native_hand", env_index, 0, 0
            )
            gym.set_actor_dof_properties(env, hand_actor, dof_props)
            hand_shapes = gym.get_actor_rigid_shape_properties(env, hand_actor)
            for shape in hand_shapes:
                shape.friction = args.robot_friction
                shape.restitution = 0.0
            gym.set_actor_rigid_shape_properties(env, hand_actor, hand_shapes)
            if args.video_hand_color is not None:
                hand_color = gymapi.Vec3(*args.video_hand_color)
                for body_index in range(
                    gym.get_actor_rigid_body_count(env, hand_actor)
                ):
                    gym.set_rigid_body_color(
                        env,
                        hand_actor,
                        body_index,
                        gymapi.MESH_VISUAL,
                        hand_color,
                    )
            outer_target = outer[reorder].copy()
            inner_target = inner[reorder].copy()
            closing_delta = inner_target - outer_target
            closing_mask = np.abs(closing_delta) > 1.0e-7
            open_limit = np.where(closing_delta > 0.0, dof_lower, dof_upper)
            close_limit = np.where(closing_delta > 0.0, dof_upper, dof_lower)
            outer_target[closing_mask] += args.pregrasp_open_fraction * (
                open_limit[closing_mask] - outer_target[closing_mask]
            )
            inner_target[closing_mask] += args.closure_overdrive_fraction * (
                close_limit[closing_mask] - inner_target[closing_mask]
            )
            dof_state = np.zeros(dof_count, dtype=gymapi.DofState.dtype)
            dof_state["pos"] = outer_target
            gym.set_actor_dof_states(env, hand_actor, dof_state, gymapi.STATE_ALL)
            hand_body_names = list(gym.get_asset_rigid_body_names(hand_asset))
            hand_body_indices = [
                gym.get_actor_rigid_body_index(
                    env, hand_actor, body_index, gymapi.DOMAIN_SIM
                )
                for body_index in range(len(hand_body_names))
            ]
            hand_body_indices_by_env.append(hand_body_indices)

            object_pose = gymapi.Transform()
            closure_object_actor = None
            if args.closure_object_mode == "fixed_until_inner":
                closure_object_actor = gym.create_actor(
                    env,
                    closure_object_asset,
                    object_pose,
                    "object_fixed_during_closure",
                    env_index,
                    0,
                    0,
                )
                closure_object_shapes = gym.get_actor_rigid_shape_properties(
                    env, closure_object_actor
                )
                for shape in closure_object_shapes:
                    shape.friction = args.object_friction
                    shape.restitution = 0.0
                gym.set_actor_rigid_shape_properties(
                    env, closure_object_actor, closure_object_shapes
                )
            dynamic_object_pose = gymapi.Transform()
            if closure_object_actor is not None:
                dynamic_object_pose.p = gymapi.Vec3(0.0, 0.0, -10.0)
            object_actor = gym.create_actor(
                env,
                object_asset,
                dynamic_object_pose,
                "object",
                env_index,
                0,
                0,
            )
            object_handles.append(object_actor)
            object_shapes = gym.get_actor_rigid_shape_properties(env, object_actor)
            for shape in object_shapes:
                shape.friction = args.object_friction
                shape.restitution = 0.0
            gym.set_actor_rigid_shape_properties(env, object_actor, object_shapes)
            if args.video_object_color is not None:
                object_color = gymapi.Vec3(*args.video_object_color)
                gym.set_rigid_body_color(
                    env,
                    object_actor,
                    0,
                    gymapi.MESH_VISUAL,
                    object_color,
                )
            object_actor_indices.append(
                gym.get_actor_index(env, object_actor, gymapi.DOMAIN_SIM)
            )
            object_body_name = gym.get_asset_rigid_body_names(object_asset)[0]
            object_body_indices.append(
                gym.find_actor_rigid_body_index(
                    env, object_actor, object_body_name, gymapi.DOMAIN_SIM
                )
            )
            if closure_object_actor is None:
                closure_object_actor = object_actor
                closure_object_body_name = object_body_name
            else:
                closure_object_body_name = gym.get_asset_rigid_body_names(
                    closure_object_asset
                )[0]
            closure_object_handles.append(closure_object_actor)
            closure_object_actor_indices.append(
                gym.get_actor_index(
                    env, closure_object_actor, gymapi.DOMAIN_SIM
                )
            )
            closure_object_body_indices.append(
                gym.find_actor_rigid_body_index(
                    env,
                    closure_object_actor,
                    closure_object_body_name,
                    gymapi.DOMAIN_SIM,
                )
            )
            recorded_body_indices.append(
                hand_body_indices + [closure_object_body_indices[-1]]
            )
            contacts = np.asarray(
                sample.get("diffusion_target_contacts_object", []),
                dtype=np.float64,
            )
            if (
                args.video_show_diffusion_contacts
                or args.viewer_show_diffusion_contacts
            ) and (
                contacts.ndim != 2 or contacts.shape[1:] != (3,)
            ):
                raise ValueError(
                    f"{object_name}:{sample['source_index']} is missing "
                    "diffusion_target_contacts_object"
                )
            diffusion_contacts.append(contacts)
            outer_targets.append(outer_target)
            inner_targets.append(inner_target)
            if args.video_dir is not None:
                camera_properties = gymapi.CameraProperties()
                camera_properties.width = args.video_width
                camera_properties.height = args.video_height
                camera_properties.horizontal_fov = 55.0
                camera = gym.create_camera_sensor(env, camera_properties)
                gym.set_camera_location(
                    camera,
                    env,
                    gymapi.Vec3(0.32, 0.30, 0.23),
                    gymapi.Vec3(0.0, 0.0, 0.0),
                )
                camera_handles.append(camera)

        gym.prepare_sim(sim)
        if args.viewer:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                raise RuntimeError("Isaac Gym failed to create the GUI viewer")
            gym.viewer_camera_look_at(
                viewer,
                envs[0],
                gymapi.Vec3(0.32, 0.30, 0.23),
                gymapi.Vec3(0.0, 0.0, 0.0),
            )
            viewer_actions = (
                (gymapi.KEY_C, "camera_toggle_lock"),
                (gymapi.KEY_A, "camera_orbit_left"),
                (gymapi.KEY_D, "camera_orbit_right"),
                (gymapi.KEY_W, "camera_orbit_up"),
                (gymapi.KEY_S, "camera_orbit_down"),
                (gymapi.KEY_Q, "camera_zoom_in"),
                (gymapi.KEY_E, "camera_zoom_out"),
                (gymapi.KEY_0, "camera_view_oblique"),
                (gymapi.KEY_1, "camera_view_pos_x"),
                (gymapi.KEY_2, "camera_view_neg_x"),
                (gymapi.KEY_3, "camera_view_pos_y"),
                (gymapi.KEY_4, "camera_view_neg_y"),
                (gymapi.KEY_5, "camera_view_top"),
                (gymapi.KEY_LEFT_BRACKET, "camera_previous_sample"),
                (gymapi.KEY_RIGHT_BRACKET, "camera_next_sample"),
            )
            for key, action in viewer_actions:
                gym.subscribe_viewer_keyboard_event(viewer, key, action)
            print(
                "Object camera: WASD orbit, Q/E zoom, 0 oblique, "
                "1..5 axis views, [/] sample, C lock/free.",
                flush=True,
            )
        contact_geometries = [
            gymutil.WireframeSphereGeometry(
                args.viewer_contact_radius_m,
                12,
                12,
                color=color,
            )
            for color in CONTACT_COLORS_RGB
        ]
        video_writers = []
        video_frame_counts = [0] * count
        if args.video_dir is not None:
            import cv2

            args.video_dir.mkdir(parents=True, exist_ok=True)
            for sample in samples:
                video_path = args.video_dir / (
                    f"{sample_artifact_stem(object_name, sample)}.mp4"
                )
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.video_fps,
                    (args.video_width, args.video_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create {video_path}")
                video_writers.append(writer)

        def capture_videos(phase: str, force: bool = False) -> None:
            if not video_writers:
                return
            if not force and capture_videos.physics_step % args.video_stride:
                return
            import cv2

            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            object_states = None
            if args.video_show_diffusion_contacts:
                gym.refresh_actor_root_state_tensor(sim)
                object_states = (
                    root[object_actor_indices_t].detach().cpu().numpy().copy()
                )
            contact_colors = tuple(
                tuple(round(255 * channel) for channel in color[::-1])
                for color in CONTACT_COLORS_RGB
            )
            for index, (env, camera, writer, sample) in enumerate(
                zip(envs, camera_handles, video_writers, samples)
            ):
                rgba = gym.get_camera_image(
                    sim, env, camera, gymapi.IMAGE_COLOR
                )
                rgba = np.asarray(rgba).reshape(
                    args.video_height, args.video_width, 4
                )
                frame = np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])
                if object_states is not None:
                    state = object_states[index]
                    world_contacts = transform_object_points(
                        diffusion_contacts[index], state[:3], state[3:7]
                    )
                    contact_pixels, contact_depth = project_world_points_to_camera(
                        world_contacts,
                        eye=(0.32, 0.30, 0.23),
                        target=(0.0, 0.0, 0.0),
                        width=args.video_width,
                        height=args.video_height,
                        horizontal_fov_degrees=55.0,
                    )
                    for contact_index, (pixel, depth) in enumerate(
                        zip(contact_pixels, contact_depth)
                    ):
                        if depth <= 0.0 or not np.isfinite(pixel).all():
                            continue
                        center = tuple(np.rint(pixel).astype(int))
                        if not (
                            -args.video_contact_radius_px <= center[0]
                            < args.video_width + args.video_contact_radius_px
                            and -args.video_contact_radius_px <= center[1]
                            < args.video_height + args.video_contact_radius_px
                        ):
                            continue
                        color = contact_colors[
                            contact_index % len(contact_colors)
                        ]
                        cv2.circle(
                            frame,
                            center,
                            args.video_contact_radius_px + 2,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            frame,
                            center,
                            args.video_contact_radius_px,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            str(contact_index + 1),
                            (center[0] + 8, center[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                category = str(sample.get("visualization_category", "selected"))
                lines = (
                    (
                        "Isaac Gym Preview 4 | "
                        + (
                            args.video_hand_label
                            or Path(args.native_hand_urdf).stem
                        )
                    ),
                    f"{object_name} | source={int(sample['source_index'])}",
                    f"{category} | phase={phase}",
                    *(
                        ("rings 1..N: diffusion targets (X-ray overlay)",)
                        if args.video_show_diffusion_contacts
                        else ()
                    ),
                )
                for row, label in enumerate(lines):
                    position = (14, 25 + 24 * row)
                    cv2.putText(
                        frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 0, 0), 3, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (245, 245, 245), 1, cv2.LINE_AA,
                    )
                writer.write(frame)
                video_frame_counts[index] += 1

        capture_videos.physics_step = 0
        device = torch.device(
            "cpu" if args.cpu_physics else f"cuda:{args.device_id}"
        )
        root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        rigid = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
        net_contact_force = gymtorch.wrap_tensor(
            gym.acquire_net_contact_force_tensor(sim)
        )
        object_actor_indices_t = torch.as_tensor(
            object_actor_indices, device=device, dtype=torch.long
        )
        object_body_indices_t = torch.as_tensor(
            object_body_indices, device=device, dtype=torch.long
        )
        closure_object_actor_indices_t = torch.as_tensor(
            closure_object_actor_indices, device=device, dtype=torch.long
        )
        closure_object_body_indices_t = torch.as_tensor(
            closure_object_body_indices, device=device, dtype=torch.long
        )
        hand_body_indices_t = torch.as_tensor(
            np.asarray(hand_body_indices_by_env, dtype=np.int64),
            device=device,
            dtype=torch.long,
        )
        object_actor_indices_i32 = torch.as_tensor(
            object_actor_indices, device=device, dtype=torch.int32
        )
        closure_object_actor_indices_i32 = torch.as_tensor(
            closure_object_actor_indices, device=device, dtype=torch.int32
        )
        thumb_links = automatic_thumb_links(hand_body_names)
        camera_default_offset = np.asarray(
            [0.32, 0.30, 0.23], dtype=np.float64
        )
        camera_default_distance = float(np.linalg.norm(camera_default_offset))
        camera_state = {
            "locked": not args.viewer_free_camera,
            "focus_index": min(args.viewer_focus_sample, count - 1),
            "yaw": math.atan2(
                camera_default_offset[1], camera_default_offset[0]
            ),
            "pitch": math.asin(
                camera_default_offset[2] / camera_default_distance
            ),
            "distance": camera_default_distance,
        }

        def update_object_camera(object_states: np.ndarray) -> None:
            if viewer is None:
                return
            angle_step = math.radians(10.0)
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0.0:
                    continue
                action = event.action
                if action == "camera_toggle_lock":
                    camera_state["locked"] = not camera_state["locked"]
                    mode = "object coordinates" if camera_state["locked"] else "free mouse"
                    print(f"Viewer camera mode: {mode}", flush=True)
                elif action == "camera_orbit_left":
                    camera_state["yaw"] += angle_step
                elif action == "camera_orbit_right":
                    camera_state["yaw"] -= angle_step
                elif action == "camera_orbit_up":
                    camera_state["pitch"] += angle_step
                elif action == "camera_orbit_down":
                    camera_state["pitch"] -= angle_step
                elif action == "camera_zoom_in":
                    camera_state["distance"] *= 0.85
                elif action == "camera_zoom_out":
                    camera_state["distance"] *= 1.0 / 0.85
                elif action == "camera_view_oblique":
                    camera_state["yaw"] = math.atan2(0.30, 0.32)
                    camera_state["pitch"] = math.asin(
                        0.23 / camera_default_distance
                    )
                elif action == "camera_view_pos_x":
                    camera_state["yaw"], camera_state["pitch"] = 0.0, 0.0
                elif action == "camera_view_neg_x":
                    camera_state["yaw"], camera_state["pitch"] = math.pi, 0.0
                elif action == "camera_view_pos_y":
                    camera_state["yaw"], camera_state["pitch"] = 0.5 * math.pi, 0.0
                elif action == "camera_view_neg_y":
                    camera_state["yaw"], camera_state["pitch"] = -0.5 * math.pi, 0.0
                elif action == "camera_view_top":
                    camera_state["yaw"] = 0.0
                    camera_state["pitch"] = math.radians(85.0)
                elif action == "camera_previous_sample":
                    camera_state["focus_index"] = (
                        camera_state["focus_index"] - 1
                    ) % count
                elif action == "camera_next_sample":
                    camera_state["focus_index"] = (
                        camera_state["focus_index"] + 1
                    ) % count
            camera_state["pitch"] = float(
                np.clip(
                    camera_state["pitch"],
                    math.radians(-85.0),
                    math.radians(85.0),
                )
            )
            camera_state["distance"] = float(
                np.clip(camera_state["distance"], 0.04, 3.0)
            )
            if not camera_state["locked"]:
                return
            focus_index = int(camera_state["focus_index"])
            state = object_states[focus_index]
            cos_pitch = math.cos(camera_state["pitch"])
            local_eye = np.asarray(
                [[
                    camera_state["distance"]
                    * cos_pitch
                    * math.cos(camera_state["yaw"]),
                    camera_state["distance"]
                    * cos_pitch
                    * math.sin(camera_state["yaw"]),
                    camera_state["distance"]
                    * math.sin(camera_state["pitch"]),
                ]],
                dtype=np.float64,
            )
            eye = transform_object_points(
                local_eye, state[:3], state[3:7]
            )[0]
            target = state[:3]
            gym.viewer_camera_look_at(
                viewer,
                envs[focus_index],
                gymapi.Vec3(*[float(value) for value in eye]),
                gymapi.Vec3(*[float(value) for value in target]),
            )

        def render_viewer() -> None:
            nonlocal viewer
            if viewer is None:
                return
            if gym.query_viewer_has_closed(viewer):
                gym.destroy_viewer(viewer)
                viewer = None
                print(
                    "Isaac Gym viewer closed; continuing validation headlessly.",
                    flush=True,
                )
                return
            gym.step_graphics(sim)
            gym.clear_lines(viewer)
            gym.refresh_actor_root_state_tensor(sim)
            object_states = (
                root[object_actor_indices_t].detach().cpu().numpy().copy()
            )
            update_object_camera(object_states)
            if args.viewer_show_diffusion_contacts:
                for env, contacts, state in zip(
                    envs, diffusion_contacts, object_states
                ):
                    world_contacts = transform_object_points(
                        contacts, state[:3], state[3:7]
                    )
                    for contact_index, contact in enumerate(world_contacts):
                        marker_pose = gymapi.Transform()
                        marker_pose.p = gymapi.Vec3(
                            float(contact[0]),
                            float(contact[1]),
                            float(contact[2]),
                        )
                        gymutil.draw_lines(
                            contact_geometries[
                                contact_index % len(contact_geometries)
                            ],
                            gym,
                            viewer,
                            env,
                            marker_pose,
                        )
            gym.draw_viewer(viewer, sim, False)
            if not args.viewer_no_sync:
                gym.sync_frame_time(sim)

        state_frames = [[] for _ in range(count)]
        state_phases = [[] for _ in range(count)]
        state_steps = [[] for _ in range(count)]

        def capture_states(phase: str, force: bool = False) -> None:
            if args.state_dir is None:
                return
            if not force and capture_states.physics_step % args.state_stride:
                return
            gym.refresh_rigid_body_state_tensor(sim)
            for index, indices in enumerate(recorded_body_indices):
                state_frames[index].append(
                    rigid[indices].detach().cpu().numpy().copy()
                )
                state_phases[index].append(phase)
                state_steps[index].append(capture_states.physics_step)

        capture_states.physics_step = 0
        telemetry_enabled = args.closure_telemetry_dir is not None
        telemetry_phases: list[str] = []
        telemetry_steps: list[int] = []
        telemetry_positions: list[np.ndarray] = []
        telemetry_linear_velocities: list[np.ndarray] = []
        telemetry_angular_velocities: list[np.ndarray] = []
        telemetry_net_impulse_vectors: list[np.ndarray] = []
        telemetry_net_impulse_magnitudes: list[np.ndarray] = []
        telemetry_max_hand_link_impulses: list[np.ndarray] = []
        telemetry_contact_counts: list[np.ndarray] = []
        first_contact_steps = np.full(count, -1, dtype=np.int32)
        thumb_first_contact_steps = np.full(count, -1, dtype=np.int32)
        first_contact_links: list[str | None] = [None] * count
        first_contact_links_all: list[list[str]] = [[] for _ in range(count)]
        first_contact_positions = np.full((count, 3), np.nan, dtype=np.float32)
        contact_link_impulse_totals: list[dict[str, float]] = [
            {} for _ in range(count)
        ]

        def capture_closure_telemetry(
            phase: str, *, dynamic_object: bool = False
        ) -> None:
            if not telemetry_enabled:
                return
            gym.refresh_actor_root_state_tensor(sim)
            indices = (
                object_actor_indices_t
                if dynamic_object
                else closure_object_actor_indices_t
            )
            states = root[indices]
            positions = states[:, :3].detach().cpu().numpy().copy()
            linear_velocity = states[:, 7:10].detach().cpu().numpy().copy()
            angular_velocity = states[:, 10:13].detach().cpu().numpy().copy()
            gym.refresh_net_contact_force_tensor(sim)
            object_body_tensor = (
                object_body_indices_t
                if dynamic_object
                else closure_object_body_indices_t
            )
            dt = 1.0 / float(args.steps_per_second)
            object_impulse_vector_t = net_contact_force[object_body_tensor] * dt
            object_impulse_magnitude_t = torch.linalg.norm(
                object_impulse_vector_t, dim=1
            )
            hand_link_impulse_t = torch.linalg.norm(
                net_contact_force[hand_body_indices_t] * dt, dim=2
            )
            object_impulse_vector = (
                object_impulse_vector_t.detach().cpu().numpy().copy()
            )
            object_impulse_magnitude = (
                object_impulse_magnitude_t.detach().cpu().numpy().copy()
            )
            hand_link_impulse = (
                hand_link_impulse_t.detach().cpu().numpy().copy()
            )
            active = (
                hand_link_impulse > args.contact_impulse_epsilon
            ) & (
                object_impulse_magnitude[:, None]
                > args.contact_impulse_epsilon
            )
            contact_count = active.sum(axis=1).astype(np.int32)
            max_hand_link_impulse = hand_link_impulse.max(axis=1)
            frame_links: list[dict[str, float]] = [
                {
                    hand_body_names[body_index]: float(
                        hand_link_impulse[env_index, body_index]
                    )
                    for body_index in np.flatnonzero(active[env_index])
                }
                for env_index in range(count)
            ]
            for env_index, links in enumerate(frame_links):
                for hand_link, impulse in links.items():
                    contact_link_impulse_totals[env_index][hand_link] = (
                        contact_link_impulse_totals[env_index].get(
                            hand_link, 0.0
                        )
                        + impulse
                    )
            step = capture_states.physics_step
            for index, links in enumerate(frame_links):
                if links and first_contact_steps[index] < 0:
                    first_contact_steps[index] = step
                    first_contact_positions[index] = positions[index]
                    ordered = sorted(
                        links, key=lambda name: (-links[name], name)
                    )
                    first_contact_links[index] = ordered[0]
                    first_contact_links_all[index] = ordered
                if (
                    thumb_first_contact_steps[index] < 0
                    and any(name in thumb_links for name in links)
                ):
                    thumb_first_contact_steps[index] = step
            telemetry_phases.append(phase)
            telemetry_steps.append(step)
            telemetry_positions.append(positions)
            telemetry_linear_velocities.append(linear_velocity)
            telemetry_angular_velocities.append(angular_velocity)
            telemetry_net_impulse_vectors.append(object_impulse_vector)
            telemetry_net_impulse_magnitudes.append(object_impulse_magnitude)
            telemetry_max_hand_link_impulses.append(max_hand_link_impulse)
            telemetry_contact_counts.append(contact_count)

        dof_targets = torch.as_tensor(
            np.stack(outer_targets), device=device, dtype=torch.float32
        ).reshape(-1)
        inner_targets_tensor = torch.as_tensor(
            np.stack(inner_targets), device=device, dtype=torch.float32
        ).reshape(-1)
        gym.set_dof_position_target_tensor(
            sim, gymtorch.unwrap_tensor(dof_targets)
        )
        render_viewer()
        capture_videos("initial", force=True)
        capture_states("initial", force=True)
        gym.refresh_actor_root_state_tensor(sim)
        outer_object_positions = (
            root[closure_object_actor_indices_t, :3]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        for _ in range(args.outer_settle_steps):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            render_viewer()
            capture_videos.physics_step += 1
            capture_states.physics_step += 1
            capture_closure_telemetry("outer_settle")
        capture_videos("outer_settle", force=True)
        capture_states("outer_settle", force=True)
        for closure_step in range(args.closure_steps):
            if args.closure_trajectory == "step":
                alpha = 1.0
            else:
                alpha = float(closure_step + 1) / float(args.closure_steps)
                if args.closure_trajectory == "smoothstep":
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            closure_targets = dof_targets + alpha * (
                inner_targets_tensor - dof_targets
            )
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(closure_targets)
            )
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            render_viewer()
            capture_videos.physics_step += 1
            capture_videos("closure")
            capture_states.physics_step += 1
            capture_states("closure")
            capture_closure_telemetry("closure")
        capture_videos("closure_end", force=True)
        capture_states("closure_end", force=True)
        gym.refresh_actor_root_state_tensor(sim)
        inner_object_positions = (
            root[closure_object_actor_indices_t, :3]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        if args.closure_object_mode == "fixed_until_inner":
            # Swap the active copies using the supported GPU root-state tensor:
            # dynamic moves from the parking pose to the fixed pose with zero
            # velocity, while the fixed copy is parked far below the hand.
            root[object_actor_indices_t, :7] = root[
                closure_object_actor_indices_t, :7
            ]
            root[object_actor_indices_t, 7:13] = 0.0
            root[closure_object_actor_indices_t, 0:2] = root[
                object_actor_indices_t, 0:2
            ]
            root[closure_object_actor_indices_t, 2] = -10.0
            root[closure_object_actor_indices_t, 7:13] = 0.0
            swapped_actor_indices_i32 = torch.cat(
                (
                    object_actor_indices_i32,
                    closure_object_actor_indices_i32,
                )
            )
            if not gym.set_actor_root_state_tensor_indexed(
                sim,
                gymtorch.unwrap_tensor(root),
                gymtorch.unwrap_tensor(swapped_actor_indices_i32),
                int(swapped_actor_indices_i32.numel()),
            ):
                raise RuntimeError("Failed to swap fixed/dynamic object states")
        for _ in range(args.inner_hold_steps):
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(inner_targets_tensor)
            )
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            render_viewer()
            capture_videos.physics_step += 1
            capture_videos("inner_hold")
            capture_states.physics_step += 1
            capture_states("inner_hold")
            capture_closure_telemetry("inner_hold", dynamic_object=True)
        capture_videos("inner_hold_end", force=True)
        capture_states("inner_hold_end", force=True)

        outer_to_inner_vectors = inner_object_positions - outer_object_positions
        outer_to_contact_vectors = first_contact_positions - outer_object_positions
        contact_to_inner_vectors = inner_object_positions - first_contact_positions
        telemetry_path = None
        if telemetry_enabled:
            position_frames = np.stack(telemetry_positions, axis=0)
            linear_velocity_frames = np.stack(
                telemetry_linear_velocities, axis=0
            )
            angular_velocity_frames = np.stack(
                telemetry_angular_velocities, axis=0
            )
            net_impulse_vector_frames = np.stack(
                telemetry_net_impulse_vectors, axis=0
            )
            net_impulse_magnitude_frames = np.stack(
                telemetry_net_impulse_magnitudes, axis=0
            )
            max_hand_link_impulse_frames = np.stack(
                telemetry_max_hand_link_impulses, axis=0
            )
            contact_count_frames = np.stack(telemetry_contact_counts, axis=0)
            cumulative_net_contact_impulse = (
                net_impulse_magnitude_frames.sum(axis=0)
            )
            peak_frame_net_contact_impulse = (
                net_impulse_magnitude_frames.max(axis=0)
            )
            max_hand_link_net_contact_impulse = (
                max_hand_link_impulse_frames.max(axis=0)
            )
            maximum_linear_speed = np.linalg.norm(
                linear_velocity_frames, axis=2
            ).max(axis=0)
            maximum_angular_speed = np.linalg.norm(
                angular_velocity_frames, axis=2
            ).max(axis=0)
            step_to_phase = dict(zip(telemetry_steps, telemetry_phases))
            args.closure_telemetry_dir.mkdir(parents=True, exist_ok=True)
            window_start = max(0, int(args.sample_start))
            window_end = window_start + count - 1
            telemetry_path = args.closure_telemetry_dir / (
                f"{object_name}_batch_{window_start:04d}_{window_end:04d}.npz"
            )
            np.savez_compressed(
                telemetry_path,
                schema=np.asarray("contactdiff-physx-closure-telemetry-v2"),
                contact_measurement=np.asarray(
                    "per-rigid-body net contact force tensor times dt; "
                    "not per-contact normal lambda"
                ),
                closure_object_mode=np.asarray(args.closure_object_mode),
                steps_per_second=np.asarray(
                    args.steps_per_second, dtype=np.int32
                ),
                physics_step=np.asarray(telemetry_steps, dtype=np.int32),
                time_s=np.asarray(telemetry_steps, dtype=np.float64)
                / float(args.steps_per_second),
                phase=np.asarray(telemetry_phases),
                source_index=np.asarray(
                    [int(sample["source_index"]) for sample in samples],
                    dtype=np.int64,
                ),
                candidate_rank=np.asarray(
                    [int(sample["candidate_rank"]) for sample in samples],
                    dtype=np.int32,
                ),
                object_position_m=position_frames,
                object_linear_velocity_mps=linear_velocity_frames,
                object_angular_velocity_radps=angular_velocity_frames,
                object_net_contact_impulse_vector_ns=net_impulse_vector_frames,
                object_net_contact_impulse_magnitude_ns=(
                    net_impulse_magnitude_frames
                ),
                max_hand_link_net_contact_impulse_ns=(
                    max_hand_link_impulse_frames
                ),
                active_hand_link_count=contact_count_frames,
                first_contact_step=first_contact_steps,
                thumb_first_contact_step=thumb_first_contact_steps,
                first_contact_link=np.asarray(
                    [name or "" for name in first_contact_links]
                ),
                thumb_link_names=np.asarray(sorted(thumb_links)),
            )
        else:
            cumulative_net_contact_impulse = np.zeros(count, dtype=np.float32)
            peak_frame_net_contact_impulse = np.zeros(count, dtype=np.float32)
            max_hand_link_net_contact_impulse = np.zeros(
                count, dtype=np.float32
            )
            maximum_linear_speed = np.zeros(count, dtype=np.float32)
            maximum_angular_speed = np.zeros(count, dtype=np.float32)
            step_to_phase = {}

        object_properties = gym.get_actor_rigid_body_properties(
            envs[0], object_handles[0]
        )
        object_mass = float(sum(prop.mass for prop in object_properties))
        if not math.isfinite(object_mass) or object_mass <= 0.0:
            raise RuntimeError(
                f"{object_name}: invalid imported object mass {object_mass}"
            )
        force_magnitude = object_mass * args.acceleration
        trajectory = [[] for _ in range(count)]
        zero_forces = torch.zeros(
            rigid.shape[0], 3, device=device, dtype=torch.float32
        )
        zero_torques = torch.zeros_like(zero_forces)
        directions = (
            DIRECTIONS_CEDEX
            if args.direction_order == "cedex"
            else DIRECTIONS_GENDEX
        )
        if args.max_directions is not None:
            if not 1 <= int(args.max_directions) <= len(directions):
                raise ValueError("--max-directions must be within [1, 6]")
            directions = directions[: int(args.max_directions)]
        direction_steps = max(
            1, round(args.steps_per_second * args.direction_seconds)
        )
        gym.refresh_actor_root_state_tensor(sim)
        overall_start = root[object_actor_indices_t, :3].clone()
        strict_success = torch.ones(count, device=device, dtype=torch.bool)
        for direction_name, direction in directions:
            gym.refresh_actor_root_state_tensor(sim)
            direction_start = root[object_actor_indices_t, :3].clone()
            forces = zero_forces.clone()
            direction_tensor = torch.tensor(
                direction, device=device, dtype=torch.float32
            )
            forces[object_body_indices_t] = force_magnitude * direction_tensor
            for _ in range(direction_steps):
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(inner_targets_tensor)
                )
                gym.apply_rigid_body_force_tensors(
                    sim,
                    gymtorch.unwrap_tensor(forces),
                    gymtorch.unwrap_tensor(zero_torques),
                    gymapi.ENV_SPACE,
                )
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                render_viewer()
                capture_videos.physics_step += 1
                capture_videos(f"force_{direction_name}")
                capture_states.physics_step += 1
                capture_states(f"force_{direction_name}")
            capture_videos(f"force_{direction_name}_end", force=True)
            capture_states(f"force_{direction_name}_end", force=True)
            gym.refresh_actor_root_state_tensor(sim)
            end = root[object_actor_indices_t, :3].clone()
            displacement = torch.linalg.norm(end - direction_start, dim=1)
            strict_success &= displacement <= args.threshold
            for index in range(count):
                trajectory[index].append(
                    {
                        "direction": direction_name,
                        "segment_displacement_m": float(displacement[index].item()),
                    }
                )

        gym.refresh_actor_root_state_tensor(sim)
        overall_end = root[object_actor_indices_t, :3].clone()
        final_displacement = torch.linalg.norm(overall_end - overall_start, dim=1)
        final_success = final_displacement <= args.threshold
        results = []
        for index, sample in enumerate(samples):
            finite = all(
                math.isfinite(point["segment_displacement_m"])
                for point in trajectory[index]
            ) and math.isfinite(float(final_displacement[index].item()))
            bounded = bool(
                torch.max(torch.abs(overall_start[index])).item() < 10.0
                and torch.max(torch.abs(overall_end[index])).item() < 10.0
                and final_displacement[index].item() < 10.0
            )
            valid = finite and bounded
            selected_success = (
                final_success[index]
                if args.success_mode == "final"
                else strict_success[index]
            )
            results.append(
                {
                    "object_name": object_name,
                    "source_index": int(sample["source_index"]),
                    "candidate_rank": int(sample["candidate_rank"]),
                    "valid_simulation": valid,
                    "success": bool(selected_success.item()) if valid else None,
                    "final_success": bool(final_success[index].item()) if valid else None,
                    "strict_six_direction_success": (
                        bool(strict_success[index].item()) if valid else None
                    ),
                    "start_position_m": overall_start[index].tolist(),
                    "end_position_m": overall_end[index].tolist(),
                    "final_displacement_m": float(final_displacement[index].item()),
                    "trajectory": trajectory[index],
                    "maximum_segment_displacement_m": max(
                        point["segment_displacement_m"]
                        for point in trajectory[index]
                    ),
                    "closure_telemetry": ({
                        "schema": "contactdiff-physx-closure-summary-v2",
                        "contact_measurement": (
                            "per-rigid-body net contact force tensor times dt; "
                            "not per-contact normal lambda"
                        ),
                        "object_mode": args.closure_object_mode,
                        "frame_count": len(telemetry_steps),
                        "first_contact_observed": bool(
                            first_contact_steps[index] >= 0
                        ),
                        "first_contact_physics_step": (
                            int(first_contact_steps[index])
                            if first_contact_steps[index] >= 0
                            else None
                        ),
                        "first_contact_time_s": (
                            float(first_contact_steps[index])
                            / float(args.steps_per_second)
                            if first_contact_steps[index] >= 0
                            else None
                        ),
                        "first_contact_phase": (
                            step_to_phase.get(int(first_contact_steps[index]))
                            if first_contact_steps[index] >= 0
                            else None
                        ),
                        "first_contact_link": first_contact_links[index],
                        "first_contact_links_same_frame": (
                            first_contact_links_all[index]
                        ),
                        "thumb_link_names": sorted(thumb_links),
                        "thumb_first_contact_observed": bool(
                            thumb_first_contact_steps[index] >= 0
                        ),
                        "thumb_first_contact_physics_step": (
                            int(thumb_first_contact_steps[index])
                            if thumb_first_contact_steps[index] >= 0
                            else None
                        ),
                        "thumb_first_contact_time_s": (
                            float(thumb_first_contact_steps[index])
                            / float(args.steps_per_second)
                            if thumb_first_contact_steps[index] >= 0
                            else None
                        ),
                        "outer_object_position_m": (
                            outer_object_positions[index].tolist()
                        ),
                        "first_contact_object_position_m": (
                            first_contact_positions[index].tolist()
                            if first_contact_steps[index] >= 0
                            else None
                        ),
                        "inner_object_position_m": (
                            inner_object_positions[index].tolist()
                        ),
                        "outer_to_first_contact_vector_m": (
                            outer_to_contact_vectors[index].tolist()
                            if first_contact_steps[index] >= 0
                            else None
                        ),
                        "outer_to_first_contact_displacement_m": (
                            float(np.linalg.norm(outer_to_contact_vectors[index]))
                            if first_contact_steps[index] >= 0
                            else None
                        ),
                        "first_contact_to_inner_vector_m": (
                            contact_to_inner_vectors[index].tolist()
                            if first_contact_steps[index] >= 0
                            else None
                        ),
                        "first_contact_to_inner_displacement_m": (
                            float(np.linalg.norm(contact_to_inner_vectors[index]))
                            if first_contact_steps[index] >= 0
                            else None
                        ),
                        "outer_to_inner_vector_m": (
                            outer_to_inner_vectors[index].tolist()
                        ),
                        "outer_to_inner_displacement_m": float(
                            np.linalg.norm(outer_to_inner_vectors[index])
                        ),
                        "cumulative_net_contact_impulse_ns": float(
                            cumulative_net_contact_impulse[index]
                        ),
                        "peak_frame_net_contact_impulse_ns": float(
                            peak_frame_net_contact_impulse[index]
                        ),
                        "max_hand_link_net_contact_impulse_ns": float(
                            max_hand_link_net_contact_impulse[index]
                        ),
                        "maximum_object_linear_speed_mps": float(
                            maximum_linear_speed[index]
                        ),
                        "maximum_object_angular_speed_radps": float(
                            maximum_angular_speed[index]
                        ),
                        "per_link_cumulative_net_contact_impulse_ns": dict(
                            sorted(contact_link_impulse_totals[index].items())
                        ),
                        "dense_npz": (
                            str(telemetry_path.resolve())
                            if telemetry_path is not None
                            else None
                        ),
                    } if telemetry_enabled else None),
                    "force_magnitude_n": force_magnitude,
                    "imported_object_mass_kg": object_mass,
                    "video": (
                        str(
                            (
                                args.video_dir
                                / f"{sample_artifact_stem(object_name, sample)}.mp4"
                            ).resolve()
                        )
                        if args.video_dir is not None
                        else None
                    ),
                    "video_frames": video_frame_counts[index],
                }
            )
        for writer in video_writers:
            writer.release()
        if args.state_dir is not None:
            args.state_dir.mkdir(parents=True, exist_ok=True)
            body_names = list(gym.get_asset_rigid_body_names(hand_asset)) + [
                "object"
            ]
            for index, sample in enumerate(samples):
                state_path = args.state_dir / (
                    f"{sample_artifact_stem(object_name, sample)}.npz"
                )
                np.savez_compressed(
                    state_path,
                    rigid_body_states=np.stack(state_frames[index]),
                    phase=np.asarray(state_phases[index]),
                    physics_step=np.asarray(state_steps[index], dtype=np.int64),
                    body_names=np.asarray(body_names),
                    object_mesh=np.asarray(str(object_mesh_path.resolve())),
                    hand_urdf=np.asarray(str(movable_urdf.resolve())),
                    object_name=np.asarray(object_name),
                    source_index=np.asarray(int(sample["source_index"])),
                    visualization_category=np.asarray(
                        str(sample.get("visualization_category", "selected"))
                    ),
                )
        return results
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


def summarize(report: dict) -> None:
    results = report["results"]
    valid = [row for row in results if row["valid_simulation"]]
    report["trials"] = len(results)
    report["valid_trials"] = len(valid)
    report["invalid_trials"] = len(results) - len(valid)
    report["successes"] = sum(row["success"] is True for row in results)
    report["final_successes"] = sum(
        row.get("final_success") is True for row in results
    )
    report["strict_six_direction_successes"] = sum(
        row.get("strict_six_direction_success") is True for row in results
    )
    report["success_rate"] = (
        report["successes"] / report["trials"] if report["trials"] else 0.0
    )
    report["valid_trial_success_rate"] = (
        report["successes"] / report["valid_trials"]
        if report["valid_trials"]
        else 0.0
    )
    report["object_summaries"] = []
    for object_name in OBJECT_MAP:
        rows = [row for row in results if row["object_name"] == object_name]
        valid_rows = [row for row in rows if row["valid_simulation"]]
        successes = sum(row["success"] is True for row in rows)
        final_successes = sum(row.get("final_success") is True for row in rows)
        strict_successes = sum(
            row.get("strict_six_direction_success") is True for row in rows
        )
        report["object_summaries"].append(
            {
                "object_name": object_name,
                "successes": successes,
                "final_successes": final_successes,
                "strict_six_direction_successes": strict_successes,
                "trials": len(rows),
                "valid_trials": len(valid_rows),
                "invalid_trials": len(rows) - len(valid_rows),
                "success_rate": successes / len(rows) if rows else 0.0,
            }
        )


def main() -> None:
    args = parse_args()
    prepared = json.loads(args.prepared.resolve().read_text(encoding="utf-8"))
    validate_frozen_basic_protocol(prepared, args)
    if prepared["hand"] not in {
        "franka_panda",
        "contactdiff_shadowhand",
        "shadowhand",
        "barrett",
        "gendex_barrett",
        "contactdiff_barrett",
    }:
        raise ValueError(
            "Prepared manifest must use a supported Panda, ShadowHand, or Barrett hand"
        )
    groups = list(prepared["objects"])
    prepared_joint_names = list(prepared["joint_names"])
    for group in groups:
        group["_joint_names"] = prepared_joint_names
    if args.only_object:
        selected = set(args.only_object)
        groups = [group for group in groups if group["object_name"] in selected]
    expected_trials = sum(
        len(select_sample_window(list(group["samples"]), args))
        for group in groups
    )
    previous_results = []
    if args.resume and args.output.is_file():
        previous_results = json.loads(
            args.output.read_text(encoding="utf-8")
        ).get("results", [])
    completed_objects = {
        row["object_name"]
        for row in previous_results
        if sum(
            existing["object_name"] == row["object_name"]
            for existing in previous_results
        )
        == next(
            (
                len(select_sample_window(list(group["samples"]), args))
                for group in groups
                if group["object_name"] == row["object_name"]
            ),
            -1,
        )
    }
    report = {
        "schema": "contactdiff-dro-native-hand-oldparams-isaacgym-v1",
        "protocol_id": prepared.get("protocol_id"),
        "base_protocol_id": prepared.get("base_protocol_id"),
        "source_candidate_protocol_id": prepared.get(
            "source_candidate_protocol_id"
        ),
        "execution_protocol_config": prepared.get(
            "execution_protocol_config"
        ),
        "execution_protocol_config_sha256": prepared.get(
            "execution_protocol_config_sha256"
        ),
        "hand": prepared["hand"],
        "simulator": (
            "Isaac Gym Preview 4 CPU PhysX"
            if args.cpu_physics
            else "Isaac Gym Preview 4 GPU PhysX"
        ),
        "prepared": str(args.prepared.resolve()),
        "sample_window": {
            "start": max(0, int(args.sample_start)),
            "maximum_count_per_object": args.max_samples_per_object,
        },
        "native_hand_urdf": str(
            (args.native_hand_root / args.native_hand_urdf).resolve()
        ),
        "protocol": {
            "closure_adapter": prepared.get("closure_adapter"),
            "asset_profile": args.asset_profile,
            "object_vhacd": args.object_vhacd,
            "object_predecomposed": args.object_predecomposed,
            "object_collision": (
                "prepared raw mesh + runtime VHACD"
                if args.asset_profile == "old" and not args.object_predecomposed
                else "predecomposed collision mesh"
                if args.object_predecomposed
                else "D(R,O) object source"
            ),
            "native_hand_urdf_is_extended": args.native_hand_urdf_is_extended,
            "physics_backend": (
                "CPU PhysX compatibility replay"
                if args.cpu_physics
                else "GPU PhysX"
            ),
            "tensor_pipeline": "CPU" if args.cpu_physics else "GPU",
            "dt": 1.0 / args.steps_per_second,
            "steps_per_second": args.steps_per_second,
            "substeps": args.substeps,
            "solver_type": "TGS",
            "solver_position_iterations": args.solver_position_iterations,
            "solver_velocity_iterations": args.solver_velocity_iterations,
            "contact_offset_m": args.contact_offset,
            "rest_offset_m": args.rest_offset,
            "gravity_mps2": 0.0,
            "robot_friction": args.robot_friction,
            "object_friction": args.object_friction,
            "object_density_kg_m3": args.object_density,
            "hand_density_kg_m3": (
                args.hand_density
                if args.hand_density is not None
                else args.object_density
            ),
            "object_linear_damping": (
                args.object_linear_damping
                if args.object_linear_damping >= 0.0
                else "inherit importer default"
            ),
            "object_angular_damping": (
                args.object_angular_damping
                if args.object_angular_damping >= 0.0
                else "inherit importer default"
            ),
            "joint_stiffness": args.joint_stiffness,
            "joint_damping": args.joint_damping,
            "finger_joint_max_effort": (
                args.finger_joint_max_effort
                if args.finger_joint_max_effort is not None
                else "inherit URDF/importer"
            ),
            "finger_joint_velocity": (
                args.finger_joint_velocity
                if args.finger_joint_velocity is not None
                else "inherit URDF/importer"
            ),
            "joint_armature": (
                args.joint_armature
                if args.joint_armature >= 0.0
                else "inherit importer default"
            ),
            "pregrasp_open_fraction": args.pregrasp_open_fraction,
            "closure_overdrive_fraction": args.closure_overdrive_fraction,
            "outer_settle_steps": args.outer_settle_steps,
            "virtual_root_joint_names": list(VIRTUAL_ROOT_JOINTS),
            "virtual_root_stiffness": args.virtual_root_stiffness,
            "virtual_root_damping": args.virtual_root_damping,
            "joint_velocity": (
                args.joint_velocity
                if args.joint_velocity >= 0.0
                else "inherit URDF/importer"
            ),
            "closure_steps": args.closure_steps,
            "closure_trajectory": args.closure_trajectory,
            "inner_hold_steps": args.inner_hold_steps,
            "closure_ab_experiment": args.closure_ab_experiment,
            "closure_object_mode": args.closure_object_mode,
            "closure_object_transition": (
                "fixed-base collision actor parked at inner via GPU root-state "
                "tensor; zero-velocity dynamic actor moved to the identical pose"
                if args.closure_object_mode == "fixed_until_inner"
                else "dynamic from outer through force evaluation"
            ),
            "closure_telemetry": {
                "enabled": args.closure_telemetry_dir is not None,
                "dense_format": "compressed NPZ, frame-major",
                "contact_impulse_source": (
                    "GPU per-rigid-body net contact force tensor times dt"
                ),
                "exact_per_contact_normal_lambda_available": False,
                "contact_impulse_epsilon_ns": args.contact_impulse_epsilon,
                "thumb_link_detection": (
                    "ShadowHand th*; Barrett bh_finger_3*"
                ),
            },
            "direction_seconds": args.direction_seconds,
            "direction_order": args.direction_order,
            "max_directions": args.max_directions,
            "partial_disturbance_prescreen": args.max_directions is not None,
            "directions": [
                name
                for name, _ in (
                    DIRECTIONS_CEDEX
                    if args.direction_order == "cedex"
                    else DIRECTIONS_GENDEX
                )[: args.max_directions]
            ],
            "steps_per_direction": max(
                1, round(args.steps_per_second * args.direction_seconds)
            ),
            "acceleration_mps2": args.acceleration,
            "force_magnitude": "Isaac imported actor mass * acceleration",
            "success_mode": args.success_mode,
            "success_threshold_m": args.threshold,
            "ground_enabled": not args.no_ground,
            "environment_grid": {
                "envs_per_row_override": args.envs_per_row,
                "default": "ceil(sqrt(active env count))",
                "purpose": "controlled PhysX batch-layout audit",
            },
            "viewer_enabled": args.viewer,
            "viewer_show_diffusion_contacts": bool(
                args.viewer and args.viewer_show_diffusion_contacts
            ),
            "viewer_contact_radius_m": (
                args.viewer_contact_radius_m
                if args.viewer_show_diffusion_contacts
                else None
            ),
            "viewer_camera_mode_initial": (
                "free" if args.viewer_free_camera else "object_coordinates"
            ),
            "viewer_focus_sample": args.viewer_focus_sample,
            "video_show_diffusion_contacts": bool(
                args.video_dir is not None
                and args.video_show_diffusion_contacts
            ),
            "video_contact_radius_px": (
                args.video_contact_radius_px
                if args.video_show_diffusion_contacts
                else None
            ),
            "object_source": args.object_source,
            "prepared_object_mesh_root": (
                str(args.prepared_object_mesh_root.resolve())
                if args.prepared_object_mesh_root is not None
                else None
            ),
            "dro_object_root": (
                str(args.dro_object_root.resolve())
                if args.dro_object_root is not None
                else None
            ),
        },
        "expected_trials": expected_trials,
        "results": previous_results,
        "status": "running",
    }
    started = time.monotonic()
    gym = gymapi.acquire_gym()
    for group in groups:
        object_name = group["object_name"]
        if object_name in completed_objects:
            print(f"{object_name}: resume-skip", flush=True)
            continue
        object_results = validate_object(gym, args, group)
        report["results"].extend(object_results)
        summarize(report)
        report["elapsed_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"{object_name}: {sum(row['success'] is True for row in object_results)}"
            f"/{len(object_results)}; total={len(report['results'])}/"
            f"{expected_trials}",
            flush=True,
        )
    summarize(report)
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = (
        "complete" if len(report["results"]) == expected_trials else "partial"
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"complete: {report['successes']}/{report['trials']} "
        f"({100.0 * report['success_rate']:.2f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
