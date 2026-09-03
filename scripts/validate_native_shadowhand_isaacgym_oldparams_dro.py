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
from scipy.spatial import cKDTree
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
from utils.contact_aware_closure import shadowhand_finger_groups


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
SHADOW_SURFACE_SYNC_DOF_NAMES = {
    "FFJ3", "FFJ2", "FFJ1",
    "MFJ3", "MFJ2", "MFJ1",
    "RFJ3", "RFJ2", "RFJ1",
    "LFJ5", "LFJ3", "LFJ2", "LFJ1",
    "THJ4", "THJ2", "THJ1",
}
HAND_OBJECT_COLLISION_DISABLE_BIT = 1 << 29


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
    if args.contact_aware_closure:
        # This explicit experimental branch changes only the closure controller
        # and its solver support. Generation, selection, disturbance and success
        # criteria remain frozen and comparable.
        expected["closure_trajectory"] = args.closure_trajectory
        expected["solver_velocity_iterations"] = args.solver_velocity_iterations
    if args.surface_sync_closure or args.experimental_low_contact_offset:
        # Explicit experimental controller: only the collision envelope is
        # changed from the frozen O10/I20 execution profile.  The generated
        # pose, object dynamics, friction and disturbance test stay identical.
        expected["contact_offset"] = 0.0005
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
    parser.add_argument(
        "--cpu-tensor-pipeline",
        action="store_true",
        help=(
            "Keep PhysX on GPU but expose simulation tensors and exact rigid "
            "contact-pair queries through the CPU pipeline."
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
        "--viewer-camera-follow-mode",
        choices=("translation", "pose"),
        default="pose",
        help=(
            "Follow only object translation for a stable world-oriented view, "
            "or inherit object rotation as in the legacy object-frame camera."
        ),
    )
    parser.add_argument(
        "--viewer-show-reference-grid",
        action="store_true",
        help="Draw a non-physical world-frame floor grid for visual reference.",
    )
    parser.add_argument(
        "--viewer-final-hold-seconds",
        type=float,
        default=0.0,
        help=(
            "Keep rendering the final state for this many wall-clock seconds "
            "without advancing physics."
        ),
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
        "--contact-aware-closure",
        action="store_true",
        help=(
            "Experimental ShadowHand controller: while the object is fixed, "
            "freeze each digit independently after confirmed rigid-body contact."
        ),
    )
    parser.add_argument(
        "--surface-sync-closure",
        action="store_true",
        help=(
            "Experimental mesh-surface controller that derives an open start from "
            "q_contact and joint limits, stops each calibrated pad 1 mm above the object, "
            "then advances all five digits simultaneously until pad contact. "
            "Prepared outer/inner joint targets are not used."
        ),
    )
    parser.add_argument(
        "--surface-fixed-approach",
        action="store_true",
        help=(
            "Stage the surface-sync 1 mm waiting pose against a fixed copy of "
            "the object, then atomically release a zero-velocity dynamic copy "
            "before synchronized closure."
        ),
    )
    parser.add_argument(
        "--surface-normal-target-closure",
        action="store_true",
        help=(
            "Track each assigned contact point plus its 1 mm outward normal "
            "during staging, then close along the assigned inward normal with "
            "a damped least-squares distal-pad Jacobian controller."
        ),
    )
    parser.add_argument(
        "--surface-normal-approach-speed-m-s", type=float, default=0.02
    )
    parser.add_argument(
        "--surface-normal-close-speed-m-s", type=float, default=0.005
    )
    parser.add_argument(
        "--surface-normal-position-tolerance-m", type=float, default=0.0015
    )
    parser.add_argument(
        "--surface-normal-damping", type=float, default=0.01
    )
    parser.add_argument(
        "--surface-ready-clearance-m",
        type=float,
        default=0.001,
        help="Pad-to-sphere clearance at which a digit waits for the others.",
    )
    parser.add_argument(
        "--surface-sync-pad-samples",
        type=Path,
        default=Path(
            "outputs/shadowhand_pad_pointcloud/"
            "shadowhand_pad_samples_local.json"
        ),
        help=(
            "Calibrated pad-only samples expressed in each distal link frame."
        ),
    )
    parser.add_argument(
        "--surface-sync-mesh-samples",
        type=int,
        default=100000,
        help=(
            "Deterministic collision-mesh surface samples used by the arbitrary-"
            "object nearest-surface query."
        ),
    )
    parser.add_argument(
        "--surface-ready-penetration-tolerance-m",
        type=float,
        default=0.001,
        help="Largest pad overshoot still accepted as a valid ready state.",
    )
    parser.add_argument(
        "--surface-ready-braking-margin-m",
        type=float,
        default=0.0005,
        help=(
            "Begin braking this far outside the desired waiting clearance to "
            "compensate position-servo tracking lag."
        ),
    )
    parser.add_argument(
        "--surface-sync-approach-steps",
        type=int,
        default=500,
        help="Maximum independent pad-approach control frames.",
    )
    parser.add_argument(
        "--surface-sync-start-open-rad",
        type=float,
        default=10.0,
        help=(
            "Derive the pregrasp by subtracting this angle from each active "
            "q_contact flexion DOF, clamped at its URDF open limit. The default "
            "therefore starts every active flexion DOF at its open limit."
        ),
    )
    parser.add_argument(
        "--surface-sync-joint-speed-rad-s",
        type=float,
        default=0.25,
        help="Commanded closing speed for active flexion joints.",
    )
    parser.add_argument(
        "--surface-sync-preload-rad",
        type=float,
        default=0.01,
        help="Additional closing command after confirmed distal contact.",
    )
    parser.add_argument(
        "--surface-sync-max-closure-displacement-m",
        type=float,
        default=0.005,
        help="Closure displacement above which the ball is classified as pushed.",
    )
    parser.add_argument(
        "--contact-stop-force-threshold",
        type=float,
        default=0.1,
        help="Per-digit net rigid-body contact-force threshold in newtons.",
    )
    parser.add_argument(
        "--contact-confirm-steps",
        type=int,
        default=2,
        help="Consecutive contact frames required before freezing a digit.",
    )
    parser.add_argument(
        "--contact-preload-fraction",
        type=float,
        default=0.01,
        help="Extra fraction of the outer-to-inner digit motion after contact.",
    )
    parser.add_argument(
        "--pre-release-hold-steps",
        type=int,
        default=0,
        help="Hold contact-aware targets against the fixed object before release.",
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
    parser.add_argument(
        "--only-direction",
        choices=("+x", "-x", "+y", "-y", "+z", "-z"),
        help=(
            "Run exactly one named world-frame disturbance direction. This is "
            "an explicit directional diagnostic, not the formal six-direction "
            "evaluation."
        ),
    )
    parser.add_argument("--success-mode", choices=("final", "per_direction"), default="per_direction")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--acceleration", type=float, default=0.5)
    parser.add_argument("--robot-friction", type=float, default=10.0)
    parser.add_argument("--object-friction", type=float, default=10.0)
    parser.add_argument(
        "--disable-hand-object-collision-body-prefix",
        action="append",
        default=[],
        help=(
            "Disable collisions between object shapes and hand rigid bodies "
            "whose names begin with this prefix. May be repeated."
        ),
    )
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
    parser.add_argument(
        "--experimental-low-contact-offset",
        action="store_true",
        help=(
            "Permit the explicit 0.5 mm PhysX contact envelope in a standard-"
            "closure control arm, so it can be compared with surface-sync "
            "without collision-envelope drift."
        ),
    )
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
    parser.add_argument(
        "--video-focus-sample",
        type=int,
        help=(
            "Record only this zero-based slot in the selected sample window "
            "while still simulating every environment. This preserves the "
            "full fixed-layout PhysX batch for exact visual replays."
        ),
    )
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument(
        "--video-camera-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.32, 0.30, 0.23),
        help="World-space camera position used by recorded RGB videos.",
    )
    parser.add_argument(
        "--video-camera-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="World-space camera look-at target used by recorded RGB videos.",
    )
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
        "--video-background-color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help=(
            "Replace camera pixels with no rendered geometry by this RGB "
            "color. This is a rendering-only aid for videos recorded without ground."
        ),
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
    parser.add_argument(
        "--state-focus-sample",
        type=int,
        help=(
            "Record rigid-body states only for this zero-based slot while "
            "still simulating the complete selected environment window."
        ),
    )
    parser.add_argument("--state-stride", type=int, default=4)
    args = parser.parse_args()
    if args.viewer_show_diffusion_contacts and not args.viewer:
        parser.error("--viewer-show-diffusion-contacts requires --viewer")
    if args.viewer_contact_radius_m <= 0.0:
        parser.error("--viewer-contact-radius-m must be positive")
    if args.viewer_focus_sample < 0:
        parser.error("--viewer-focus-sample must be non-negative")
    if args.viewer_final_hold_seconds < 0.0:
        parser.error("--viewer-final-hold-seconds must be non-negative")
    if args.video_focus_sample is not None and args.video_focus_sample < 0:
        parser.error("--video-focus-sample must be non-negative")
    for name, color in (
        ("video-hand-color", args.video_hand_color),
        ("video-object-color", args.video_object_color),
        ("video-background-color", args.video_background_color),
    ):
        if color is not None and any(not 0.0 <= value <= 1.0 for value in color):
            parser.error(f"--{name} components must be in [0, 1]")
    if args.state_focus_sample is not None and args.state_focus_sample < 0:
        parser.error("--state-focus-sample must be non-negative")
    if args.contact_impulse_epsilon < 0.0:
        parser.error("--contact-impulse-epsilon must be non-negative")
    if args.closure_object_mode != "dynamic" and not args.closure_ab_experiment:
        parser.error("fixed_until_inner requires --closure-ab-experiment")
    if args.contact_aware_closure:
        if args.closure_object_mode != "fixed_until_inner":
            parser.error(
                "--contact-aware-closure requires "
                "--closure-object-mode fixed_until_inner"
            )
        if not args.closure_ab_experiment:
            parser.error("--contact-aware-closure requires --closure-ab-experiment")
        if args.closure_trajectory == "step":
            parser.error("--contact-aware-closure requires linear or smoothstep closure")
    if args.surface_sync_closure:
        if args.contact_aware_closure:
            parser.error(
                "--surface-sync-closure and --contact-aware-closure are exclusive"
            )
        if args.closure_object_mode != "dynamic":
            parser.error("--surface-sync-closure requires a dynamic object")
        if args.surface_ready_clearance_m <= 0.0:
            parser.error("--surface-ready-clearance-m must be positive")
        if args.surface_ready_penetration_tolerance_m < 0.0:
            parser.error(
                "--surface-ready-penetration-tolerance-m must be non-negative"
            )
        if args.surface_ready_braking_margin_m < 0.0:
            parser.error("--surface-ready-braking-margin-m must be non-negative")
        if args.surface_sync_approach_steps < 1:
            parser.error("--surface-sync-approach-steps must be positive")
        if args.surface_sync_mesh_samples < 1000:
            parser.error("--surface-sync-mesh-samples must be at least 1000")
        if args.surface_sync_start_open_rad <= 0.0:
            parser.error("--surface-sync-start-open-rad must be positive")
        if args.surface_sync_joint_speed_rad_s <= 0.0:
            parser.error("--surface-sync-joint-speed-rad-s must be positive")
        if args.surface_sync_preload_rad < 0.0:
            parser.error("--surface-sync-preload-rad must be non-negative")
        if args.surface_sync_max_closure_displacement_m <= 0.0:
            parser.error(
                "--surface-sync-max-closure-displacement-m must be positive"
            )
    if args.surface_fixed_approach and not args.surface_sync_closure:
        parser.error("--surface-fixed-approach requires --surface-sync-closure")
    if args.surface_normal_target_closure:
        if not args.surface_sync_closure:
            parser.error(
                "--surface-normal-target-closure requires --surface-sync-closure"
            )
        if not args.surface_fixed_approach:
            parser.error(
                "--surface-normal-target-closure requires --surface-fixed-approach"
            )
    if args.surface_normal_approach_speed_m_s <= 0.0:
        parser.error("--surface-normal-approach-speed-m-s must be positive")
    if args.surface_normal_close_speed_m_s <= 0.0:
        parser.error("--surface-normal-close-speed-m-s must be positive")
    if args.surface_normal_position_tolerance_m <= 0.0:
        parser.error("--surface-normal-position-tolerance-m must be positive")
    if args.surface_normal_damping <= 0.0:
        parser.error("--surface-normal-damping must be positive")
    if args.contact_stop_force_threshold <= 0.0:
        parser.error("--contact-stop-force-threshold must be positive")
    if args.contact_confirm_steps < 1:
        parser.error("--contact-confirm-steps must be positive")
    if not 0.0 <= args.contact_preload_fraction <= 1.0:
        parser.error("--contact-preload-fraction must be in [0, 1]")
    if args.pre_release_hold_steps < 0:
        parser.error("--pre-release-hold-steps must be non-negative")
    if args.only_direction is not None and args.max_directions is not None:
        parser.error("--only-direction cannot be combined with --max-directions")
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
    if (
        args.closure_telemetry_dir is not None
        or args.contact_aware_closure
        or args.surface_sync_closure
    ):
        # Preserve every substep contact so the per-frame sum below represents
        # the complete normal impulse delivered during one control frame.
        params.physx.contact_collection = (
            gymapi.ContactCollection.CC_ALL_SUBSTEPS
        )
    params.use_gpu_pipeline = not (
        args.cpu_physics or args.cpu_tensor_pipeline
    )
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
    if args.viewer or args.video_dir is not None:
        # Rendering-only fill lights. They do not modify contact dynamics.
        gym.set_light_parameters(
            sim,
            0,
            gymapi.Vec3(1.0, 0.98, 0.95),
            gymapi.Vec3(0.55, 0.58, 0.62),
            gymapi.Vec3(-1.0, -1.0, -2.0),
        )
        gym.set_light_parameters(
            sim,
            1,
            gymapi.Vec3(0.55, 0.60, 0.70),
            gymapi.Vec3(0.25, 0.28, 0.32),
            gymapi.Vec3(1.0, 0.5, 1.0),
        )
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
    if args.video_focus_sample is not None and args.video_focus_sample >= count:
        raise ValueError(
            f"--video-focus-sample={args.video_focus_sample} is outside the "
            f"selected sample window of length {count}"
        )
    if args.state_focus_sample is not None and args.state_focus_sample >= count:
        raise ValueError(
            f"--state-focus-sample={args.state_focus_sample} is outside the "
            f"selected sample window of length {count}"
        )
    video_indices = (
        set(range(count))
        if args.video_dir is not None and args.video_focus_sample is None
        else (
            {int(args.video_focus_sample)}
            if args.video_dir is not None
            else set()
        )
    )
    state_indices = (
        set(range(count))
        if args.state_dir is not None and args.state_focus_sample is None
        else (
            {int(args.state_focus_sample)}
            if args.state_dir is not None
            else set()
        )
    )

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
        use_fixed_approach_actor = (
            args.closure_object_mode == "fixed_until_inner"
            or (args.surface_sync_closure and args.surface_fixed_approach)
        )
        closure_object_asset = (
            gym.load_asset(
                sim,
                str(object_root),
                object_file,
                object_asset_options(args, fixed_base=True),
            )
            if use_fixed_approach_actor
            else object_asset
        )
        surface_mesh_tree = None
        surface_mesh_points = None
        surface_mesh_normals = None
        if args.surface_sync_closure:
            collision_mesh_path = (
                object_root / dataset / short_name / "coacd_allinone.obj"
            ).resolve()
            if not collision_mesh_path.is_file():
                raise FileNotFoundError(collision_mesh_path)
            collision_geometry = trimesh.load(
                collision_mesh_path, force="mesh", process=False
            )
            surface_mesh_points, sampled_faces = trimesh.sample.sample_surface(
                collision_geometry,
                int(args.surface_sync_mesh_samples),
                seed=20260826,
            )
            surface_mesh_points = np.asarray(
                surface_mesh_points, dtype=np.float64
            )
            surface_mesh_normals = np.asarray(
                collision_geometry.face_normals[sampled_faces],
                dtype=np.float64,
            )
            surface_mesh_tree = cKDTree(surface_mesh_points)
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
        camera_handles = {}
        object_actor_indices = []
        object_body_indices = []
        object_handles = []
        closure_object_actor_indices = []
        closure_object_body_indices = []
        closure_object_handles = []
        hand_body_indices_by_env = []
        recorded_body_indices = []
        diffusion_contacts = []
        sphere_radii = []
        outer_targets = []
        inner_targets = []
        contact_targets = []
        assigned_target_points = []
        assigned_target_normals = []
        for env_index, sample in enumerate(samples):
            env = gym.create_env(sim, lower, upper, per_row)
            envs.append(env)
            if args.surface_sync_closure:
                contact = np.asarray(
                    sample["q_contact_euler"], dtype=np.float32
                )
                outer = contact
                inner = contact
            else:
                outer = np.asarray(sample["outer_q_euler"], dtype=np.float32)
                inner = np.asarray(sample["inner_q_euler"], dtype=np.float32)
            hand_pose = gymapi.Transform()
            hand_actor = gym.create_actor(
                env,
                hand_asset,
                hand_pose,
                "native_hand",
                env_index,
                1
                if args.contact_aware_closure or args.surface_sync_closure
                else 0,
                0,
            )
            gym.set_actor_dof_properties(env, hand_actor, dof_props)
            hand_shapes = gym.get_actor_rigid_shape_properties(env, hand_actor)
            for shape in hand_shapes:
                shape.friction = args.robot_friction
                shape.restitution = 0.0
            disabled_body_prefixes = tuple(
                prefix.lower()
                for prefix in args.disable_hand_object_collision_body_prefix
            )
            if disabled_body_prefixes:
                hand_body_names_for_filter = list(
                    gym.get_asset_rigid_body_names(hand_asset)
                )
                shape_ranges = gym.get_actor_rigid_body_shape_indices(
                    env, hand_actor
                )
                for body_index, body_name in enumerate(
                    hand_body_names_for_filter
                ):
                    if not body_name.lower().startswith(disabled_body_prefixes):
                        continue
                    shape_range = shape_ranges[body_index]
                    for shape_index in range(
                        int(shape_range.start),
                        int(shape_range.start + shape_range.count),
                    ):
                        hand_shapes[shape_index].filter |= (
                            HAND_OBJECT_COLLISION_DISABLE_BIT
                        )
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
            contact_target = contact[reorder].copy() if args.surface_sync_closure else None
            if args.surface_sync_closure:
                # Preserve the optimized root pose and ab/adduction joints, but
                # derive the flexion start entirely from the URDF open limits.
                # This branch intentionally does not consume prepared O10/I20.
                for dof_index, dof_name in enumerate(hand_dof_names):
                    if dof_name in SHADOW_SURFACE_SYNC_DOF_NAMES:
                        outer_target[dof_index] = max(
                            float(dof_lower[dof_index]),
                            float(outer_target[dof_index])
                            - float(args.surface_sync_start_open_rad),
                        )
                        inner_target[dof_index] = dof_upper[dof_index]
            else:
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
            if use_fixed_approach_actor:
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
                    if disabled_body_prefixes:
                        shape.filter |= HAND_OBJECT_COLLISION_DISABLE_BIT
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
                if disabled_body_prefixes:
                    shape.filter |= HAND_OBJECT_COLLISION_DISABLE_BIT
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
            contact_radii = np.linalg.norm(contacts, axis=1)
            if args.surface_sync_closure and (
                len(contact_radii) != 5
                or not np.isfinite(contact_radii).all()
                or float(np.median(contact_radii)) <= 0.0
            ):
                raise ValueError(
                    f"{object_name}:{sample['source_index']} has invalid "
                    "baseball surface contacts"
                )
            sphere_radii.append(float(np.median(contact_radii)))
            outer_targets.append(outer_target)
            inner_targets.append(inner_target)
            if args.surface_sync_closure:
                contact_targets.append(contact_target)
            if args.surface_normal_target_closure:
                target_points = np.asarray(
                    sample.get("fk_matched_target_points_object"),
                    dtype=np.float32,
                )
                target_normals = np.asarray(
                    sample.get("fk_matched_target_normals_object"),
                    dtype=np.float32,
                )
                if target_points.shape != (5, 3) or target_normals.shape != (5, 3):
                    raise ValueError(
                        f"{object_name}:{sample['source_index']} lacks five "
                        "assigned target points/normals"
                    )
                target_norm = np.linalg.norm(target_normals, axis=1, keepdims=True)
                if not np.isfinite(target_normals).all() or np.any(target_norm <= 1e-8):
                    raise ValueError(
                        f"{object_name}:{sample['source_index']} has invalid "
                        "assigned target normals"
                    )
                assigned_target_points.append(target_points)
                assigned_target_normals.append(target_normals / target_norm)
            if env_index in video_indices:
                camera_properties = gymapi.CameraProperties()
                camera_properties.width = args.video_width
                camera_properties.height = args.video_height
                camera_properties.horizontal_fov = 55.0
                camera = gym.create_camera_sensor(env, camera_properties)
                gym.set_camera_location(
                    camera,
                    env,
                    gymapi.Vec3(*args.video_camera_eye),
                    gymapi.Vec3(*args.video_camera_target),
                )
                camera_handles[env_index] = camera

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
                f"Tracking camera ({args.viewer_camera_follow_mode} follow): "
                "WASD orbit, Q/E zoom, 0 oblique, "
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
        grid_vertices = None
        grid_colors = None
        if args.viewer_show_reference_grid:
            grid_lines = []
            for coordinate in np.linspace(-0.30, 0.30, 13):
                grid_lines.append(
                    [[-0.30, float(coordinate), -0.12],
                     [0.30, float(coordinate), -0.12]]
                )
                grid_lines.append(
                    [[float(coordinate), -0.30, -0.12],
                     [float(coordinate), 0.30, -0.12]]
                )
            grid_vertices = np.asarray(
                grid_lines, dtype=np.float32
            ).reshape(-1, 3)
            grid_colors = np.full(
                (len(grid_lines), 3), (0.32, 0.38, 0.46), dtype=np.float32
            )
        video_writers = {}
        video_frame_counts = [0] * count
        if args.video_dir is not None:
            import cv2

            args.video_dir.mkdir(parents=True, exist_ok=True)
            for index in sorted(video_indices):
                sample = samples[index]
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
                video_writers[index] = writer

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
            for index in sorted(video_writers):
                env = envs[index]
                camera = camera_handles[index]
                writer = video_writers[index]
                sample = samples[index]
                rgba = gym.get_camera_image(
                    sim, env, camera, gymapi.IMAGE_COLOR
                )
                rgba = np.asarray(rgba).reshape(
                    args.video_height, args.video_width, 4
                )
                frame = np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])
                if args.video_background_color is not None:
                    depth = gym.get_camera_image(
                        sim, env, camera, gymapi.IMAGE_DEPTH
                    )
                    depth = np.asarray(depth).reshape(
                        args.video_height, args.video_width
                    )
                    background = ~np.isfinite(depth) | (depth < -1.0e4)
                    background_bgr = np.rint(
                        255.0 * np.asarray(
                            args.video_background_color[::-1], dtype=np.float32
                        )
                    ).astype(np.uint8)
                    frame[background] = background_bgr
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
            "cpu"
            if args.cpu_physics or args.cpu_tensor_pipeline
            else f"cuda:{args.device_id}"
        )
        root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        rigid = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
        dof_state_tensor = gymtorch.wrap_tensor(
            gym.acquire_dof_state_tensor(sim)
        )
        net_contact_force = gymtorch.wrap_tensor(
            gym.acquire_net_contact_force_tensor(sim)
        )
        hand_jacobian = (
            gymtorch.wrap_tensor(
                gym.acquire_jacobian_tensor(sim, "native_hand")
            )
            if args.surface_normal_target_closure
            else None
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
                    mode = (
                        f"object {args.viewer_camera_follow_mode} follow"
                        if camera_state["locked"]
                        else "free mouse"
                    )
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
            if args.viewer_camera_follow_mode == "pose":
                eye = transform_object_points(
                    local_eye, state[:3], state[3:7]
                )[0]
            else:
                eye = state[:3] + local_eye[0]
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
            if grid_vertices is not None and grid_colors is not None:
                focus_index = int(camera_state["focus_index"])
                gym.add_lines(
                    viewer,
                    envs[focus_index],
                    int(grid_colors.shape[0]),
                    grid_vertices,
                    grid_colors,
                )
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
            for index in sorted(state_indices):
                indices = recorded_body_indices[index]
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
        contact_targets_tensor = (
            torch.as_tensor(
                np.stack(contact_targets), device=device, dtype=torch.float32
            )
            if args.surface_sync_closure
            else None
        )
        assigned_target_points_t = (
            torch.as_tensor(
                np.stack(assigned_target_points),
                device=device,
                dtype=torch.float32,
            )
            if args.surface_normal_target_closure
            else None
        )
        assigned_target_normals_t = (
            torch.as_tensor(
                np.stack(assigned_target_normals),
                device=device,
                dtype=torch.float32,
            )
            if args.surface_normal_target_closure
            else None
        )
        evaluation_targets_tensor = inner_targets_tensor.clone()
        contact_groups: list[dict[str, object]] = []
        contact_steps_tensor = None
        peak_group_force_tensor = None
        if args.contact_aware_closure or args.surface_sync_closure:
            contact_groups = shadowhand_finger_groups(
                hand_dof_names, hand_body_names
            )
            contact_steps_tensor = torch.full(
                (count, len(contact_groups)),
                -1,
                device=device,
                dtype=torch.int32,
            )
            peak_group_force_tensor = torch.zeros(
                (count, len(contact_groups)),
                device=device,
                dtype=torch.float32,
            )
        surface_ready_steps_tensor = None
        surface_pad_clearance_tensor = None
        surface_initial_pad_clearance_tensor = None
        surface_ready_pad_clearance_tensor = None
        surface_min_pad_clearance_tensor = None
        surface_nonpad_collision_tensor = None
        surface_severe_overshoot_tensor = None
        surface_approach_pushed_tensor = None
        surface_sync_pushed_tensor = None
        surface_all_ready_tensor = None
        surface_approach_displacement_tensor = None
        surface_sync_displacement_tensor = None
        if args.surface_sync_closure:
            missing_active = sorted(
                SHADOW_SURFACE_SYNC_DOF_NAMES - set(hand_dof_names)
            )
            if missing_active:
                raise RuntimeError(
                    f"Surface-sync hand is missing DOFs: {missing_active}"
                )
            body_name_to_index = {
                name.lower(): index
                for index, name in enumerate(hand_body_names)
            }
            pad_local_body_indices = []
            for group in contact_groups:
                distal_name = f"{str(group['name']).lower()}distal"
                if distal_name not in body_name_to_index:
                    raise RuntimeError(
                        f"Surface-sync hand is missing body {distal_name}"
                    )
                pad_local_body_indices.append(
                    body_name_to_index[distal_name]
                )
            pad_local_body_indices_t = torch.as_tensor(
                pad_local_body_indices, device=device, dtype=torch.long
            )
            pad_body_indices_t = hand_body_indices_t[
                :, pad_local_body_indices_t
            ]
            pad_samples_path = args.surface_sync_pad_samples.resolve()
            if not pad_samples_path.is_file():
                raise FileNotFoundError(pad_samples_path)
            pad_samples_payload = json.loads(pad_samples_path.read_text())
            if (
                pad_samples_payload.get("schema")
                != "contactdiff-shadowhand-pad-samples-local-v1"
            ):
                raise ValueError(
                    f"Unsupported pad calibration: {pad_samples_path}"
                )
            pad_sample_lists = [
                pad_samples_payload["fingers"][
                    f"{str(group['name']).lower()}distal"
                ]["region_points_local_m"]
                for group in contact_groups
            ]
            max_pad_samples = max(len(points) for points in pad_sample_lists)
            padded_pad_samples = np.zeros(
                (len(contact_groups), max_pad_samples, 3), dtype=np.float32
            )
            valid_pad_samples = np.zeros(
                (len(contact_groups), max_pad_samples), dtype=bool
            )
            for finger_index, points in enumerate(pad_sample_lists):
                padded_pad_samples[finger_index, : len(points)] = points
                valid_pad_samples[finger_index, : len(points)] = True
            pad_samples_t = torch.as_tensor(
                padded_pad_samples,
                device=device,
                dtype=torch.float32,
            )
            valid_pad_samples_t = torch.as_tensor(
                valid_pad_samples, device=device, dtype=torch.bool
            )
            if (
                pad_samples_t.ndim != 3
                or pad_samples_t.shape[0] != len(contact_groups)
                or pad_samples_t.shape[2] != 3
            ):
                raise ValueError(
                    f"Invalid local pad sample array in {pad_samples_path}"
                )
            surface_ready_steps_tensor = torch.full(
                (count, len(contact_groups)),
                -1,
                device=device,
                dtype=torch.int32,
            )
            surface_pad_clearance_tensor = torch.full(
                (count, len(contact_groups)),
                float("inf"),
                device=device,
                dtype=torch.float32,
            )
            surface_initial_pad_clearance_tensor = torch.full_like(
                surface_pad_clearance_tensor, float("nan")
            )
            surface_ready_pad_clearance_tensor = torch.full_like(
                surface_pad_clearance_tensor, float("nan")
            )
            surface_min_pad_clearance_tensor = torch.full_like(
                surface_pad_clearance_tensor, float("inf")
            )
            surface_nonpad_collision_tensor = torch.zeros(
                (count, len(contact_groups)), device=device, dtype=torch.bool
            )
            surface_severe_overshoot_tensor = torch.zeros(
                (count, len(contact_groups)), device=device, dtype=torch.bool
            )
            surface_approach_pushed_tensor = torch.zeros(
                count, device=device, dtype=torch.bool
            )
            surface_sync_pushed_tensor = torch.zeros(
                count, device=device, dtype=torch.bool
            )
            surface_all_ready_tensor = torch.zeros(
                count, device=device, dtype=torch.bool
            )
            surface_approach_displacement_tensor = torch.zeros(
                count, device=device, dtype=torch.float32
            )
            surface_sync_displacement_tensor = torch.zeros(
                count, device=device, dtype=torch.float32
            )

            def surface_pad_world_points() -> torch.Tensor:
                gym.refresh_rigid_body_state_tensor(sim)
                pad_states = rigid[pad_body_indices_t]
                pad_positions = pad_states[:, :, :3]
                quaternion = pad_states[:, :, 3:7]
                vector = pad_samples_t.unsqueeze(0).expand(count, -1, -1, -1)
                xyz = quaternion[:, :, None, :3].expand_as(vector)
                qw = quaternion[:, :, None, 3:4]
                twice_cross = 2.0 * torch.cross(xyz, vector, dim=3)
                rotated = (
                    vector
                    + qw * twice_cross
                    + torch.cross(xyz, twice_cross, dim=3)
                )
                return pad_positions[:, :, None, :] + rotated

            def object_assigned_targets_world(
                actor_indices: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                gym.refresh_actor_root_state_tensor(sim)
                states = root[actor_indices]
                quaternion = states[:, None, 3:7]
                xyz = quaternion[:, :, :3]
                qw = quaternion[:, :, 3:4]

                def rotate(vector: torch.Tensor) -> torch.Tensor:
                    twice_cross = 2.0 * torch.cross(
                        xyz.expand_as(vector), vector, dim=2
                    )
                    return (
                        vector
                        + qw * twice_cross
                        + torch.cross(
                            xyz.expand_as(vector), twice_cross, dim=2
                        )
                    )

                points = (
                    states[:, None, :3] + rotate(assigned_target_points_t)
                )
                normals = torch.nn.functional.normalize(
                    rotate(assigned_target_normals_t), dim=2
                )
                return points, normals

            pad_jacobian_indices_t = None
            if args.surface_normal_target_closure:
                if hand_jacobian is None or hand_jacobian.ndim != 4:
                    raise RuntimeError("Isaac Gym did not expose the hand Jacobian")
                jacobian_body_offset = (
                    len(hand_body_names) - int(hand_jacobian.shape[1])
                )
                jacobian_indices = [
                    index - jacobian_body_offset
                    for index in pad_local_body_indices
                ]
                if min(jacobian_indices) < 0 or max(jacobian_indices) >= int(
                    hand_jacobian.shape[1]
                ):
                    raise RuntimeError(
                        "Distal body indices do not match the Isaac Jacobian"
                    )
                pad_jacobian_indices_t = torch.as_tensor(
                    jacobian_indices, device=device, dtype=torch.long
                )

            def selected_pad_points(
                pad_points: torch.Tensor,
                desired_points: torch.Tensor,
                locked_indices: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if locked_indices is None:
                    distances = torch.linalg.norm(
                        pad_points - desired_points[:, :, None, :], dim=3
                    )
                    distances = distances.masked_fill(
                        ~valid_pad_samples_t.unsqueeze(0), float("inf")
                    )
                    indices = distances.argmin(dim=2)
                else:
                    indices = locked_indices
                gather_index = indices[:, :, None, None].expand(-1, -1, 1, 3)
                points = torch.gather(pad_points, 2, gather_index).squeeze(2)
                return points, indices

            def apply_pad_cartesian_step(
                command: torch.Tensor,
                moving: torch.Tensor,
                desired_delta: torch.Tensor,
                control_points: torch.Tensor,
            ) -> None:
                gym.refresh_jacobian_tensors(sim)
                gym.refresh_rigid_body_state_tensor(sim)
                pad_origins = rigid[pad_body_indices_t, :3]
                damping_sq = float(args.surface_normal_damping) ** 2
                identity = torch.eye(3, device=device, dtype=torch.float32)
                max_joint_step = (
                    float(args.surface_sync_joint_speed_rad_s)
                    / float(args.steps_per_second)
                )
                for group_index, dof_indices in enumerate(
                    active_dof_indices_by_group
                ):
                    rows = torch.nonzero(
                        moving[:, group_index], as_tuple=False
                    ).squeeze(1)
                    if not bool(rows.numel()):
                        continue
                    body_jacobian = hand_jacobian[
                        rows, pad_jacobian_indices_t[group_index]
                    ]
                    offset = (
                        control_points[rows, group_index]
                        - pad_origins[rows, group_index]
                    )
                    angular_axes = body_jacobian[:, 3:6, :].transpose(1, 2)
                    point_linear = (
                        body_jacobian[:, :3, :]
                        + torch.cross(
                            angular_axes,
                            offset[:, None, :].expand_as(angular_axes),
                            dim=2,
                        ).transpose(1, 2)
                    )
                    active_jacobian = point_linear[:, :, dof_indices]
                    system = (
                        active_jacobian @ active_jacobian.transpose(1, 2)
                        + damping_sq * identity[None, :, :]
                    )
                    cartesian = desired_delta[rows, group_index]
                    solution = torch.linalg.solve(system, cartesian[:, :, None])
                    delta = (
                        active_jacobian.transpose(1, 2) @ solution
                    ).squeeze(2)
                    delta = delta.clamp(-max_joint_step, max_joint_step)
                    updated = command[
                        rows[:, None], dof_indices[None, :]
                    ] + delta
                    command[
                        rows[:, None], dof_indices[None, :]
                    ] = torch.maximum(
                        torch.minimum(
                            updated,
                            torch.as_tensor(
                                dof_upper[dof_indices.cpu().numpy()],
                                device=device,
                                dtype=torch.float32,
                            )[None, :],
                        ),
                        torch.as_tensor(
                            dof_lower[dof_indices.cpu().numpy()],
                            device=device,
                            dtype=torch.float32,
                        )[None, :],
                    )

            def surface_pad_clearances(
                surface_body_indices: torch.Tensor | None = None,
            ) -> torch.Tensor:
                pad_points = surface_pad_world_points()
                selected_object_body_indices = (
                    object_body_indices_t
                    if surface_body_indices is None
                    else surface_body_indices
                )
                object_centers = rigid[selected_object_body_indices, :3]
                world_radial = (
                    pad_points - object_centers[:, None, None, :]
                )
                object_quaternion = rigid[
                    selected_object_body_indices, 3:7
                ]
                inverse_xyz = -object_quaternion[:, None, None, :3].expand_as(
                    world_radial
                )
                inverse_w = object_quaternion[:, None, None, 3:4]
                twice_cross_local = 2.0 * torch.cross(
                    inverse_xyz, world_radial, dim=3
                )
                local_radial = (
                    world_radial
                    + inverse_w * twice_cross_local
                    + torch.cross(
                        inverse_xyz, twice_cross_local, dim=3
                    )
                )
                local_points = local_radial.detach().cpu().numpy().reshape(-1, 3)
                sampled_distance, nearest = surface_mesh_tree.query(
                    local_points,
                    k=1,
                )
                nearest_delta = local_points - surface_mesh_points[nearest]
                nearest_normal = surface_mesh_normals[nearest]
                outward_sign = np.where(
                    np.einsum("ij,ij->i", nearest_delta, nearest_normal) >= 0.0,
                    1.0,
                    -1.0,
                )
                signed_distance_t = torch.as_tensor(
                    (sampled_distance * outward_sign).reshape(
                        count, len(contact_groups), pad_samples_t.shape[1]
                    ),
                    device=device,
                    dtype=torch.float32,
                )
                clearance = signed_distance_t
                clearance = clearance.masked_fill(
                    ~valid_pad_samples_t.unsqueeze(0), float("inf")
                )
                return clearance.amin(dim=2)
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
        if args.contact_aware_closure:
            outer_matrix = dof_targets.view(count, dof_count)
            inner_matrix = inner_targets_tensor.view(count, dof_count)
            evaluation_matrix = evaluation_targets_tensor.view(count, dof_count)
            closing_delta_matrix = inner_matrix - outer_matrix
            stopped_groups = torch.zeros(
                (count, len(contact_groups)),
                device=device,
                dtype=torch.bool,
            )
            contact_streak = torch.zeros(
                (count, len(contact_groups)),
                device=device,
                dtype=torch.int32,
            )
        if args.surface_sync_closure:
            command_matrix = dof_targets.view(count, dof_count).clone()
            close_limit_matrix = inner_targets_tensor.view(count, dof_count)
            active_dof_indices_by_group = []
            for group in contact_groups:
                active_dof_indices_by_group.append(
                    torch.as_tensor(
                        (
                            list(group["dof_indices"])
                            if args.surface_normal_target_closure
                            else [
                                index
                                for index in group["dof_indices"]
                                if hand_dof_names[index]
                                in SHADOW_SURFACE_SYNC_DOF_NAMES
                            ]
                        ),
                        device=device,
                        dtype=torch.long,
                    )
                )
            joint_step = (
                float(args.surface_sync_joint_speed_rad_s)
                / float(args.steps_per_second)
            )
            approach_start_positions = torch.as_tensor(
                outer_object_positions, device=device, dtype=torch.float32
            )
            ready_mask = torch.zeros(
                (count, len(contact_groups)), device=device, dtype=torch.bool
            )
            normal_selected_pad_indices = torch.zeros(
                (count, len(contact_groups)), device=device, dtype=torch.long
            )
            approach_aborted = torch.zeros(
                count, device=device, dtype=torch.bool
            )
            for approach_step in range(args.surface_sync_approach_steps):
                if args.surface_normal_target_closure:
                    target_points_world, target_normals_world = (
                        object_assigned_targets_world(
                            closure_object_actor_indices_t
                        )
                    )
                    wait_points = (
                        target_points_world
                        + args.surface_ready_clearance_m
                        * target_normals_world
                    )
                    pad_points = surface_pad_world_points()
                    control_points, current_indices = selected_pad_points(
                        pad_points, wait_points
                    )
                    current_indices = torch.where(
                        ready_mask,
                        normal_selected_pad_indices,
                        current_indices,
                    )
                    control_points, _ = selected_pad_points(
                        pad_points, wait_points, current_indices
                    )
                    error = wait_points - control_points
                    error_norm = torch.linalg.norm(error, dim=2, keepdim=True)
                    cartesian_step = (
                        float(args.surface_normal_approach_speed_m_s)
                        / float(args.steps_per_second)
                    )
                    desired_delta = error * torch.clamp(
                        cartesian_step / error_norm.clamp_min(1.0e-9),
                        max=1.0,
                    )
                    apply_pad_cartesian_step(
                        command_matrix,
                        ~ready_mask & ~approach_aborted[:, None],
                        desired_delta,
                        control_points,
                    )
                else:
                    for group_index, dof_indices in enumerate(
                        active_dof_indices_by_group
                    ):
                        moving = ~ready_mask[:, group_index] & ~approach_aborted
                        if bool(moving.any().item()):
                            rows = torch.nonzero(
                                moving, as_tuple=False
                            ).squeeze(1)
                            current = command_matrix[
                                rows[:, None], dof_indices[None, :]
                            ]
                            upper = close_limit_matrix[
                                rows[:, None], dof_indices[None, :]
                            ]
                            command_matrix[
                                rows[:, None], dof_indices[None, :]
                            ] = torch.minimum(current + joint_step, upper)
                gym.set_dof_position_target_tensor(
                    sim,
                    gymtorch.unwrap_tensor(command_matrix.reshape(-1)),
                )
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                if args.surface_normal_target_closure:
                    target_points_world, target_normals_world = (
                        object_assigned_targets_world(
                            closure_object_actor_indices_t
                        )
                    )
                    wait_points = (
                        target_points_world
                        + args.surface_ready_clearance_m
                        * target_normals_world
                    )
                    pad_points = surface_pad_world_points()
                    control_points, current_indices = selected_pad_points(
                        pad_points, wait_points
                    )
                    current_indices = torch.where(
                        ready_mask,
                        normal_selected_pad_indices,
                        current_indices,
                    )
                    control_points, _ = selected_pad_points(
                        pad_points, wait_points, current_indices
                    )
                    position_error = torch.linalg.norm(
                        wait_points - control_points, dim=2
                    )
                    clearances = (
                        (control_points - target_points_world)
                        * target_normals_world
                    ).sum(dim=2)
                    valid_band = (
                        position_error
                        <= args.surface_normal_position_tolerance_m
                    ) & (
                        clearances
                        >= -args.surface_ready_penetration_tolerance_m
                    )
                else:
                    clearances = surface_pad_clearances()
                    valid_band = (
                        clearances
                        <= (
                            args.surface_ready_clearance_m
                            + args.surface_ready_braking_margin_m
                        )
                    ) & (
                        clearances
                        >= -args.surface_ready_penetration_tolerance_m
                    )
                if approach_step == 0:
                    surface_initial_pad_clearance_tensor.copy_(clearances)
                surface_pad_clearance_tensor.copy_(clearances)
                surface_min_pad_clearance_tensor.copy_(
                    torch.minimum(surface_min_pad_clearance_tensor, clearances)
                )
                newly_ready = valid_band & ~ready_mask
                ready_mask |= valid_band
                if args.surface_normal_target_closure:
                    normal_selected_pad_indices[newly_ready] = current_indices[
                        newly_ready
                    ]
                surface_ready_steps_tensor[newly_ready] = approach_step + 1
                surface_ready_pad_clearance_tensor[newly_ready] = clearances[
                    newly_ready
                ]
                # The command is intentionally one speed increment ahead of
                # the measured joint.  Snap a newly waiting digit's target to
                # its measured angle so the high-gain drive does not continue
                # pulling it through the waiting band after the flag is set.
                gym.refresh_dof_state_tensor(sim)
                measured_dof_position = dof_state_tensor.view(
                    count, dof_count, 2
                )[:, :, 0]
                for group_index, dof_indices in enumerate(
                    active_dof_indices_by_group
                ):
                    braking = newly_ready[:, group_index]
                    if bool(braking.any().item()):
                        rows = torch.nonzero(
                            braking, as_tuple=False
                        ).squeeze(1)
                        command_matrix[
                            rows[:, None], dof_indices[None, :]
                        ] = measured_dof_position[
                            rows[:, None], dof_indices[None, :]
                        ]

                gym.refresh_net_contact_force_tensor(sim)
                hand_force_norm = torch.linalg.norm(
                    net_contact_force[hand_body_indices_t], dim=2
                )
                approach_object_body_indices_t = (
                    closure_object_body_indices_t
                    if args.surface_fixed_approach
                    else object_body_indices_t
                )
                object_force_norm = torch.linalg.norm(
                    net_contact_force[approach_object_body_indices_t], dim=1
                )
                for group_index, group in enumerate(contact_groups):
                    group_force = hand_force_norm[
                        :, group["body_indices"]
                    ].amax(dim=1)
                    peak_group_force_tensor[:, group_index] = torch.maximum(
                        peak_group_force_tensor[:, group_index], group_force
                    )
                    distal_force = hand_force_norm[
                        :, pad_local_body_indices[group_index]
                    ]
                    surface_nonpad_collision_tensor[:, group_index] |= (
                        distal_force >= args.contact_stop_force_threshold
                    ) & (
                        clearances[:, group_index]
                        > args.surface_ready_clearance_m
                    ) & (
                        object_force_norm
                        >= args.contact_stop_force_threshold
                    )
                    other_body_indices = [
                        body_index
                        for body_index in group["body_indices"]
                        if body_index != pad_local_body_indices[group_index]
                    ]
                    if other_body_indices:
                        nonpad_force = hand_force_norm[
                            :, other_body_indices
                        ].amax(dim=1)
                        surface_nonpad_collision_tensor[:, group_index] |= (
                            nonpad_force >= args.contact_stop_force_threshold
                        ) & (
                            distal_force < args.contact_stop_force_threshold
                        ) & (
                            object_force_norm
                            >= args.contact_stop_force_threshold
                        )
                gym.refresh_actor_root_state_tensor(sim)
                approach_actor_indices_t = (
                    closure_object_actor_indices_t
                    if args.surface_fixed_approach
                    else object_actor_indices_t
                )
                approach_displacement = torch.linalg.norm(
                    root[approach_actor_indices_t, :3]
                    - approach_start_positions,
                    dim=1,
                )
                surface_approach_pushed_tensor |= (
                    approach_displacement
                    > args.surface_sync_max_closure_displacement_m
                )
                severe_overshoot_by_pad = (
                    clearances
                    < -args.surface_ready_penetration_tolerance_m
                ) & ~ready_mask
                if args.surface_normal_target_closure:
                    severe_overshoot_by_pad &= (
                        position_error
                        <= 3.0 * args.surface_normal_position_tolerance_m
                    )
                surface_severe_overshoot_tensor |= severe_overshoot_by_pad
                severe_overshoot = severe_overshoot_by_pad.any(dim=1)
                approach_aborted |= (
                    surface_nonpad_collision_tensor.any(dim=1)
                    | surface_approach_pushed_tensor
                    | severe_overshoot
                )
                render_viewer()
                capture_videos.physics_step += 1
                capture_videos("surface_approach")
                capture_states.physics_step += 1
                capture_states("surface_approach")
                capture_closure_telemetry(
                    "surface_approach",
                    dynamic_object=not args.surface_fixed_approach,
                )
                if bool((ready_mask.all(dim=1) | approach_aborted).all().item()):
                    break

            surface_all_ready_tensor.copy_(
                ready_mask.all(dim=1) & ~approach_aborted
            )
            gym.refresh_actor_root_state_tensor(sim)
            surface_approach_displacement_tensor.copy_(
                torch.linalg.norm(
                    root[
                        closure_object_actor_indices_t
                        if args.surface_fixed_approach
                        else object_actor_indices_t,
                        :3,
                    ]
                    - approach_start_positions,
                    dim=1,
                )
            )
            if args.surface_fixed_approach:
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
                    raise RuntimeError(
                        "Failed to release fixed surface-approach objects"
                    )
                gym.refresh_actor_root_state_tensor(sim)
            sync_start_positions = root[object_actor_indices_t, :3].clone()
            stopped_groups = torch.zeros_like(ready_mask)
            contact_streak = torch.zeros(
                (count, len(contact_groups)),
                device=device,
                dtype=torch.int32,
            )
            for closure_step in range(args.closure_steps):
                eligible = surface_all_ready_tensor & ~surface_sync_pushed_tensor
                if args.surface_normal_target_closure:
                    target_points_world, target_normals_world = (
                        object_assigned_targets_world(object_actor_indices_t)
                    )
                    pad_points = surface_pad_world_points()
                    control_points, _ = selected_pad_points(
                        pad_points,
                        target_points_world,
                        normal_selected_pad_indices,
                    )
                    desired_delta = (
                        -float(args.surface_normal_close_speed_m_s)
                        / float(args.steps_per_second)
                        * target_normals_world
                    )
                    apply_pad_cartesian_step(
                        command_matrix,
                        eligible[:, None] & ~stopped_groups,
                        desired_delta,
                        control_points,
                    )
                else:
                    for group_index, dof_indices in enumerate(
                        active_dof_indices_by_group
                    ):
                        moving = eligible & ~stopped_groups[:, group_index]
                        if bool(moving.any().item()):
                            rows = torch.nonzero(
                                moving, as_tuple=False
                            ).squeeze(1)
                            current = command_matrix[
                                rows[:, None], dof_indices[None, :]
                            ]
                            upper = close_limit_matrix[
                                rows[:, None], dof_indices[None, :]
                            ]
                            command_matrix[
                                rows[:, None], dof_indices[None, :]
                            ] = torch.minimum(current + joint_step, upper)
                gym.set_dof_position_target_tensor(
                    sim,
                    gymtorch.unwrap_tensor(command_matrix.reshape(-1)),
                )
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                if args.surface_normal_target_closure:
                    target_points_world, target_normals_world = (
                        object_assigned_targets_world(object_actor_indices_t)
                    )
                    pad_points = surface_pad_world_points()
                    control_points, _ = selected_pad_points(
                        pad_points,
                        target_points_world,
                        normal_selected_pad_indices,
                    )
                    clearances = (
                        (control_points - target_points_world)
                        * target_normals_world
                    ).sum(dim=2)
                else:
                    clearances = surface_pad_clearances()
                surface_pad_clearance_tensor.copy_(clearances)
                surface_min_pad_clearance_tensor.copy_(
                    torch.minimum(surface_min_pad_clearance_tensor, clearances)
                )
                gym.refresh_net_contact_force_tensor(sim)
                hand_force_norm = torch.linalg.norm(
                    net_contact_force[hand_body_indices_t], dim=2
                )
                object_force_norm = torch.linalg.norm(
                    net_contact_force[object_body_indices_t], dim=1
                )
                for group_index, group in enumerate(contact_groups):
                    distal_force = hand_force_norm[
                        :, pad_local_body_indices[group_index]
                    ]
                    peak_group_force_tensor[:, group_index] = torch.maximum(
                        peak_group_force_tensor[:, group_index], distal_force
                    )
                    contacting = (
                        distal_force >= args.contact_stop_force_threshold
                    ) & (
                        object_force_norm >= args.contact_stop_force_threshold
                    ) & surface_all_ready_tensor
                    contact_streak[:, group_index] = torch.where(
                        contacting,
                        contact_streak[:, group_index] + 1,
                        torch.zeros_like(contact_streak[:, group_index]),
                    )
                    newly_stopped = (
                        contact_streak[:, group_index]
                        >= args.contact_confirm_steps
                    ) & ~stopped_groups[:, group_index]
                    if bool(newly_stopped.any().item()):
                        rows = torch.nonzero(
                            newly_stopped, as_tuple=False
                        ).squeeze(1)
                        dof_indices = active_dof_indices_by_group[group_index]
                        command_matrix[
                            rows[:, None], dof_indices[None, :]
                        ] = torch.minimum(
                            command_matrix[
                                rows[:, None], dof_indices[None, :]
                            ] + args.surface_sync_preload_rad,
                            close_limit_matrix[
                                rows[:, None], dof_indices[None, :]
                            ],
                        )
                        stopped_groups[newly_stopped, group_index] = True
                        contact_steps_tensor[newly_stopped, group_index] = (
                            closure_step + 1
                        )
                gym.refresh_actor_root_state_tensor(sim)
                sync_displacement = torch.linalg.norm(
                    root[object_actor_indices_t, :3] - sync_start_positions,
                    dim=1,
                )
                surface_sync_pushed_tensor |= (
                    sync_displacement
                    > args.surface_sync_max_closure_displacement_m
                )
                render_viewer()
                capture_videos.physics_step += 1
                capture_videos("surface_sync")
                capture_states.physics_step += 1
                capture_states("surface_sync")
                capture_closure_telemetry("surface_sync", dynamic_object=True)
                done = (
                    ~surface_all_ready_tensor
                    | surface_sync_pushed_tensor
                    | stopped_groups.all(dim=1)
                )
                if bool(done.all().item()):
                    break
            gym.refresh_actor_root_state_tensor(sim)
            surface_sync_displacement_tensor.copy_(
                torch.linalg.norm(
                    root[object_actor_indices_t, :3]
                    - sync_start_positions,
                    dim=1,
                )
            )
            evaluation_targets_tensor = command_matrix.reshape(-1).clone()
        else:
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
                if args.contact_aware_closure:
                    closure_matrix = closure_targets.view(count, dof_count).clone()
                    for group_index, group in enumerate(contact_groups):
                        frozen = stopped_groups[:, group_index]
                        if bool(frozen.any().item()):
                            rows = torch.nonzero(frozen, as_tuple=False).squeeze(1)
                            dof_indices = torch.as_tensor(
                                group["dof_indices"], device=device, dtype=torch.long
                            )
                            closure_matrix[
                                rows[:, None], dof_indices[None, :]
                            ] = evaluation_matrix[
                                rows[:, None], dof_indices[None, :]
                            ]
                    closure_targets = closure_matrix.reshape(-1)
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(closure_targets)
                )
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                if args.contact_aware_closure:
                    gym.refresh_net_contact_force_tensor(sim)
                    hand_force_norm = torch.linalg.norm(
                        net_contact_force[hand_body_indices_t], dim=2
                    )
                    for group_index, group in enumerate(contact_groups):
                        group_force = hand_force_norm[
                            :, group["body_indices"]
                        ].amax(dim=1)
                        peak_group_force_tensor[:, group_index] = torch.maximum(
                            peak_group_force_tensor[:, group_index], group_force
                        )
                        contacting = (
                            group_force >= args.contact_stop_force_threshold
                        )
                        contact_streak[:, group_index] = torch.where(
                            contacting,
                            contact_streak[:, group_index] + 1,
                            torch.zeros_like(contact_streak[:, group_index]),
                        )
                        newly_stopped = (
                            (contact_streak[:, group_index] >= args.contact_confirm_steps)
                            & ~stopped_groups[:, group_index]
                        )
                        if bool(newly_stopped.any().item()):
                            rows = torch.nonzero(
                                newly_stopped, as_tuple=False
                            ).squeeze(1)
                            dof_indices = torch.as_tensor(
                                group["dof_indices"], device=device, dtype=torch.long
                            )
                            current = closure_matrix[
                                rows[:, None], dof_indices[None, :]
                            ]
                            proposed = current + args.contact_preload_fraction * (
                                closing_delta_matrix[
                                    rows[:, None], dof_indices[None, :]
                                ]
                            )
                            lower = torch.minimum(
                                outer_matrix[rows[:, None], dof_indices[None, :]],
                                inner_matrix[rows[:, None], dof_indices[None, :]],
                            )
                            upper = torch.maximum(
                                outer_matrix[rows[:, None], dof_indices[None, :]],
                                inner_matrix[rows[:, None], dof_indices[None, :]],
                            )
                            evaluation_matrix[
                                rows[:, None], dof_indices[None, :]
                            ] = torch.maximum(torch.minimum(proposed, upper), lower)
                            stopped_groups[newly_stopped, group_index] = True
                            contact_steps_tensor[newly_stopped, group_index] = (
                                closure_step + 1
                            )
                render_viewer()
                capture_videos.physics_step += 1
                capture_videos("closure")
                capture_states.physics_step += 1
                capture_states("closure")
                capture_closure_telemetry("closure")
        for _ in range(args.pre_release_hold_steps):
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(evaluation_targets_tensor)
            )
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            render_viewer()
            capture_videos.physics_step += 1
            capture_videos("pre_release_hold")
            capture_states.physics_step += 1
            capture_states("pre_release_hold")
            capture_closure_telemetry("pre_release_hold")
        capture_videos("closure_end", force=True)
        capture_states("closure_end", force=True)
        gym.refresh_actor_root_state_tensor(sim)
        inner_object_positions = (
            root[
                object_actor_indices_t
                if args.surface_fixed_approach
                else closure_object_actor_indices_t,
                :3,
            ]
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
                sim, gymtorch.unwrap_tensor(evaluation_targets_tensor)
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
        contact_steps = (
            contact_steps_tensor.detach().cpu().numpy()
            if contact_steps_tensor is not None
            else None
        )
        peak_group_forces = (
            peak_group_force_tensor.detach().cpu().numpy()
            if peak_group_force_tensor is not None
            else None
        )
        surface_sync_arrays = None
        if args.surface_sync_closure:
            surface_sync_arrays = {
                "ready_steps": surface_ready_steps_tensor.detach().cpu().numpy(),
                "contact_steps": contact_steps,
                "initial_clearance": (
                    surface_initial_pad_clearance_tensor.detach().cpu().numpy()
                ),
                "ready_clearance": (
                    surface_ready_pad_clearance_tensor.detach().cpu().numpy()
                ),
                "final_clearance": (
                    surface_pad_clearance_tensor.detach().cpu().numpy()
                ),
                "minimum_clearance": (
                    surface_min_pad_clearance_tensor.detach().cpu().numpy()
                ),
                "nonpad_collision": (
                    surface_nonpad_collision_tensor.detach().cpu().numpy()
                ),
                "severe_overshoot": (
                    surface_severe_overshoot_tensor.detach().cpu().numpy()
                ),
                "all_ready": surface_all_ready_tensor.detach().cpu().numpy(),
                "approach_pushed": (
                    surface_approach_pushed_tensor.detach().cpu().numpy()
                ),
                "sync_pushed": (
                    surface_sync_pushed_tensor.detach().cpu().numpy()
                ),
                "approach_displacement": (
                    surface_approach_displacement_tensor.detach().cpu().numpy()
                ),
                "sync_displacement": (
                    surface_sync_displacement_tensor.detach().cpu().numpy()
                ),
            }

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
        if args.only_direction is not None:
            directions = tuple(
                item for item in directions if item[0] == args.only_direction
            )
        elif args.max_directions is not None:
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
                    sim, gymtorch.unwrap_tensor(evaluation_targets_tensor)
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

        def surface_sync_summary(index: int) -> dict | None:
            if surface_sync_arrays is None:
                return None
            names = [str(group["name"]) for group in contact_groups]
            ready_steps = surface_sync_arrays["ready_steps"][index]
            contact_steps_row = surface_sync_arrays["contact_steps"][index]
            nonpad = surface_sync_arrays["nonpad_collision"][index]
            overshoot = surface_sync_arrays["severe_overshoot"][index]
            all_ready = bool(surface_sync_arrays["all_ready"][index])
            approach_pushed = bool(
                surface_sync_arrays["approach_pushed"][index]
            )
            sync_pushed = bool(surface_sync_arrays["sync_pushed"][index])
            contacted_count = int((contact_steps_row >= 0).sum())
            if bool(nonpad.any()):
                outcome = "nonpad_collision_during_approach"
            elif approach_pushed:
                outcome = "ball_pushed_during_independent_approach"
            elif bool(overshoot.any()):
                outcome = "pad_overshot_waiting_band"
            elif not all_ready:
                outcome = "not_all_pads_reachable"
            elif sync_pushed:
                outcome = "ball_pushed_during_synchronized_close"
            elif contacted_count < len(contact_groups):
                outcome = "incomplete_pad_contact"
            else:
                outcome = "five_pad_contact_completed"
            return {
                "enabled": True,
                "schema": (
                    "contactdiff-assigned-normal-fixed-stage-v1"
                    if args.surface_normal_target_closure
                    else "contactdiff-mesh-surface-sync-v1"
                ),
                "prepared_outer_inner_targets_used": False,
                "object_surface_model": (
                    "assigned_contact_points_and_object_normals"
                    if args.surface_normal_target_closure
                    else "deterministic_collision_mesh_samples_with_face_normals"
                ),
                "fixed_object_during_waiting_pose_stage": bool(
                    args.surface_fixed_approach
                ),
                "synchronized_closure_direction": (
                    "assigned_contact_inward_normal_damped_least_squares"
                    if args.surface_normal_target_closure
                    else "positive_flexion_joint_direction"
                ),
                "normal_approach_speed_m_s": float(
                    args.surface_normal_approach_speed_m_s
                ),
                "normal_close_speed_m_s": float(
                    args.surface_normal_close_speed_m_s
                ),
                "normal_position_tolerance_m": float(
                    args.surface_normal_position_tolerance_m
                ),
                "mesh_surface_sample_count": int(
                    args.surface_sync_mesh_samples
                ),
                "median_target_radius_m": float(sphere_radii[index]),
                "ready_clearance_m": float(args.surface_ready_clearance_m),
                "braking_trigger_clearance_m": float(
                    args.surface_ready_clearance_m
                    + args.surface_ready_braking_margin_m
                ),
                "joint_speed_rad_s": float(
                    args.surface_sync_joint_speed_rad_s
                ),
                "start_open_from_q_contact_rad": float(
                    args.surface_sync_start_open_rad
                ),
                "all_pads_ready": all_ready,
                "ready_pad_count": int((ready_steps >= 0).sum()),
                "contacted_pad_count": contacted_count,
                "controller_completed": bool(
                    all_ready
                    and contacted_count == len(contact_groups)
                    and not approach_pushed
                    and not sync_pushed
                    and not bool(nonpad.any())
                    and not bool(overshoot.any())
                ),
                "outcome": outcome,
                "pad_ready_step": {
                    name: (int(value) if value >= 0 else None)
                    for name, value in zip(names, ready_steps)
                },
                "pad_contact_step": {
                    name: (int(value) if value >= 0 else None)
                    for name, value in zip(names, contact_steps_row)
                },
                "initial_pad_clearance_m": {
                    name: float(value)
                    for name, value in zip(
                        names,
                        surface_sync_arrays["initial_clearance"][index],
                    )
                },
                "waiting_pad_clearance_m": {
                    name: (float(value) if math.isfinite(float(value)) else None)
                    for name, value in zip(
                        names,
                        surface_sync_arrays["ready_clearance"][index],
                    )
                },
                "minimum_pad_clearance_m": {
                    name: float(value)
                    for name, value in zip(
                        names,
                        surface_sync_arrays["minimum_clearance"][index],
                    )
                },
                "nonpad_collision_digits": [
                    name for name, active in zip(names, nonpad) if active
                ],
                "overshoot_digits": [
                    name for name, active in zip(names, overshoot) if active
                ],
                "approach_object_displacement_m": float(
                    surface_sync_arrays["approach_displacement"][index]
                ),
                "synchronized_close_object_displacement_m": float(
                    surface_sync_arrays["sync_displacement"][index]
                ),
            }
        if viewer is not None and args.viewer_final_hold_seconds > 0.0:
            deadline = time.monotonic() + float(args.viewer_final_hold_seconds)
            while viewer is not None and time.monotonic() < deadline:
                render_viewer()
                time.sleep(1.0 / 60.0)
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
                    "closure_object_displacement_m": float(
                        np.linalg.norm(outer_to_inner_vectors[index])
                    ),
                    "contact_aware_closure": ({
                        "enabled": True,
                        "group_contact_step": {
                            str(group["name"]): (
                                int(contact_steps[index, group_index])
                                if contact_steps[index, group_index] >= 0
                                else None
                            )
                            for group_index, group in enumerate(contact_groups)
                        },
                        "group_peak_force_n": {
                            str(group["name"]): float(
                                peak_group_forces[index, group_index]
                            )
                            for group_index, group in enumerate(contact_groups)
                        },
                        "contacted_group_count": int(
                            (contact_steps[index] >= 0).sum()
                        ),
                    } if args.contact_aware_closure else None),
                    "surface_sync_closure": surface_sync_summary(index),
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
                        if index in video_writers
                        else None
                    ),
                    "video_frames": video_frame_counts[index],
                }
            )
        for writer in video_writers.values():
            writer.release()
        if args.state_dir is not None:
            args.state_dir.mkdir(parents=True, exist_ok=True)
            body_names = list(gym.get_asset_rigid_body_names(hand_asset)) + [
                "object"
            ]
            for index in sorted(state_indices):
                sample = samples[index]
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
                    candidate_rank=np.asarray(int(sample["candidate_rank"])),
                    particle_index=np.asarray(
                        int(sample.get("particle_index", -1))
                    ),
                    diffusion_contacts_object=np.asarray(
                        diffusion_contacts[index], dtype=np.float32
                    ),
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
            "tensor_pipeline": (
                "CPU" if args.cpu_physics or args.cpu_tensor_pipeline else "GPU"
            ),
            "dt": 1.0 / args.steps_per_second,
            "steps_per_second": args.steps_per_second,
            "substeps": args.substeps,
            "solver_type": "TGS",
            "solver_position_iterations": args.solver_position_iterations,
            "solver_velocity_iterations": args.solver_velocity_iterations,
            "contact_offset_m": args.contact_offset,
            "experimental_low_contact_offset": (
                args.experimental_low_contact_offset
            ),
            "rest_offset_m": args.rest_offset,
            "gravity_mps2": 0.0,
            "robot_friction": args.robot_friction,
            "object_friction": args.object_friction,
            "disabled_hand_object_collision_body_prefixes": list(
                args.disable_hand_object_collision_body_prefix
            ),
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
            "contact_aware_closure": {
                "enabled": args.contact_aware_closure,
                "controller": "per-ShadowHand-digit contact stop",
                "detection_source": (
                    "GPU per-rigid-body net contact force with hand "
                    "self-collision filtered"
                ),
                "hand_self_collision_disabled": args.contact_aware_closure,
                "force_threshold_n": args.contact_stop_force_threshold,
                "confirm_steps": args.contact_confirm_steps,
                "preload_fraction": args.contact_preload_fraction,
                "pre_release_hold_steps": args.pre_release_hold_steps,
            },
            "inner_hold_steps": args.inner_hold_steps,
            "closure_ab_experiment": args.closure_ab_experiment,
            "closure_object_mode": args.closure_object_mode,
            "closure_object_transition": (
                "fixed assigned-contact staging; zero-velocity dynamic release "
                "before synchronized inward-normal closure"
                if args.surface_fixed_approach
                else
                "fixed-base collision actor parked at inner via GPU root-state "
                "tensor; zero-velocity dynamic actor moved to the identical pose"
                if args.closure_object_mode == "fixed_until_inner"
                else "dynamic from outer through force evaluation"
            ),
            "surface_sync_controller": {
                "enabled": args.surface_sync_closure,
                "fixed_approach": args.surface_fixed_approach,
                "assigned_normal_targeting": args.surface_normal_target_closure,
                "waiting_clearance_m": args.surface_ready_clearance_m,
                "normal_approach_speed_m_s": args.surface_normal_approach_speed_m_s,
                "normal_close_speed_m_s": args.surface_normal_close_speed_m_s,
                "normal_position_tolerance_m": (
                    args.surface_normal_position_tolerance_m
                ),
                "normal_damping": args.surface_normal_damping,
            },
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
            "only_direction": args.only_direction,
            "partial_disturbance_prescreen": (
                args.max_directions is not None
                or args.only_direction is not None
            ),
            "directions": (
                [args.only_direction]
                if args.only_direction is not None
                else [
                    name
                    for name, _ in (
                        DIRECTIONS_CEDEX
                        if args.direction_order == "cedex"
                        else DIRECTIONS_GENDEX
                    )[: args.max_directions]
                ]
            ),
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
                "free"
                if args.viewer_free_camera
                else f"object_{args.viewer_camera_follow_mode}_follow"
            ),
            "viewer_focus_sample": args.viewer_focus_sample,
            "viewer_camera_follow_mode": args.viewer_camera_follow_mode,
            "viewer_show_reference_grid": bool(
                args.viewer and args.viewer_show_reference_grid
            ),
            "viewer_final_hold_seconds": float(
                args.viewer_final_hold_seconds
            ),
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
