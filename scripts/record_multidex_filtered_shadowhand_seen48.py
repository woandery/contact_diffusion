#!/usr/bin/env python3
"""Replay filtered MultiDex grasps for Barrett, ShadowHand, or EzGripper.

Each video contains at most eight poses in a 2x4 grid.  The environments share
a single Isaac Gym simulation, so the videos show the closure and the original
D(R,O) six-direction disturbance protocol in lockstep.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from collections import defaultdict, deque
from pathlib import Path

# Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCK_ROOT = PROJECT_ROOT.parent
DEFAULT_DRO_ROOT = MCK_ROOT / "DRO-Grasp"
DEFAULT_FILTERED = (
    DEFAULT_DRO_ROOT / "data/MultiDex_filtered/shadowhand/shadowhand.pt"
)
DEFAULT_RAW = DEFAULT_DRO_ROOT / "data/MultiDex/shadowhand/shadowhand.pt"
DEFAULT_SEEN = PROJECT_ROOT / "configs/gendex_seen48_objects.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/multidex_filtered_shadowhand_seen48_videos"
)

DIRECTIONS = (
    ("+X", (1.0, 0.0, 0.0)),
    ("+Y", (0.0, 1.0, 0.0)),
    ("+Z", (0.0, 0.0, 1.0)),
    ("-X", (-1.0, 0.0, 0.0)),
    ("-Y", (0.0, -1.0, 0.0)),
    ("-Z", (0.0, 0.0, -1.0)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dro-root", type=Path, default=DEFAULT_DRO_ROOT)
    parser.add_argument(
        "--robot-name",
        choices=("barrett", "shadowhand", "ezgripper"),
        default="shadowhand",
    )
    parser.add_argument("--filtered", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--seen-file", type=Path, default=DEFAULT_SEEN)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-object", type=int, default=32)
    parser.add_argument("--samples-per-video", type=int, default=8)
    parser.add_argument("--only-object", action="append", default=[])
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--use-gpu-physx",
        action="store_true",
        help="Use GPU PhysX (required by some headless compute-platform graphics stacks).",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-stride", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sim_parameters(*, use_gpu_physx: bool) -> gymapi.SimParams:
    params = gymapi.SimParams()
    params.dt = 0.01
    params.substeps = 2
    params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = use_gpu_physx
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 8
    params.physx.num_velocity_iterations = 0
    params.physx.num_threads = 4
    params.physx.num_subscenes = 0
    params.physx.contact_offset = 0.01
    params.physx.rest_offset = 0.0
    return params


def load_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def evenly_spaced_indices(count: int, take: int) -> list[int]:
    if count < take:
        raise ValueError(f"need {take} samples, found only {count}")
    if take == 1:
        return [count // 2]
    return [round(index * (count - 1) / (take - 1)) for index in range(take)]


def q_key(q: torch.Tensor, object_name: str) -> tuple[str, bytes]:
    return object_name, q.detach().cpu().contiguous().numpy().tobytes()


def select_samples(
    filtered_path: Path,
    raw_path: Path,
    object_names: list[str],
    samples_per_object: int,
    robot_name: str,
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    filtered = load_payload(filtered_path)["metadata"]
    raw = load_payload(raw_path)["metadata"]
    source_indices: dict[tuple[str, bytes], deque[int]] = defaultdict(deque)
    for source_index, (q, object_name, hand_name) in enumerate(raw):
        if hand_name == robot_name:
            source_indices[q_key(q, object_name)].append(source_index)

    grouped: dict[str, list[tuple[torch.Tensor, int, int]]] = defaultdict(list)
    unmatched = 0
    for filtered_index, (q, object_name, hand_name) in enumerate(filtered):
        if hand_name != robot_name:
            continue
        candidates = source_indices[q_key(q, object_name)]
        if not candidates:
            unmatched += 1
            continue
        grouped[object_name].append((q, filtered_index, candidates.popleft()))
    if unmatched:
        raise RuntimeError(f"{unmatched} filtered poses did not match raw MultiDex")

    selected: dict[str, list[dict]] = {}
    counts: dict[str, int] = {}
    for object_name in object_names:
        rows = grouped.get(object_name, [])
        counts[object_name] = len(rows)
        take = min(len(rows), samples_per_object)
        indices = evenly_spaced_indices(len(rows), take) if take else []
        selected[object_name] = [
            {
                "q_rot6d": rows[index][0],
                "filtered_index": rows[index][1],
                "source_index": rows[index][2],
                "object_rank": index,
            }
            for index in indices
        ]
    return selected, counts


def load_selection(
    path: Path,
) -> tuple[str | None, list[str], dict[str, list[dict]], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    supported_schemas = {
        "multidex-filtered-shadowhand-seen48-selection-v1",
        "multidex-filtered-shadowhand-object-selection-v2",
        "multidex-filtered-multihand-object-selection-v3",
        "cmap-filtered-shadowhand-object-selection-v1",
    }
    if payload.get("schema") not in supported_schemas:
        raise ValueError(f"Unsupported selection schema in {path}")
    object_names = []
    selected = {}
    counts = {}
    for row in payload["objects"]:
        object_name = str(row["object_name"])
        object_names.append(object_name)
        counts[object_name] = int(row["filtered_samples_available"])
        normalized_samples = []
        for sample in row["samples"]:
            if "q_rot6d" in sample:
                q_state = sample["q_rot6d"]
                dataset_kind = "multidex_filtered"
            elif "q_euler" in sample:
                q_state = sample["q_euler"]
                dataset_kind = "cmap_filtered"
            else:
                raise ValueError(
                    f"Selection sample for {object_name} has neither q_rot6d nor q_euler"
                )
            normalized_samples.append(
                {
                    **sample,
                    "q_state": torch.as_tensor(q_state, dtype=torch.float32),
                    "dataset_kind": dataset_kind,
                }
            )
        selected[object_name] = normalized_samples
    return payload.get("robot_name"), object_names, selected, counts


def sample_index_label(sample: dict) -> tuple[str, int]:
    if "source_index" in sample:
        return "source", int(sample["source_index"])
    if "cmap_filtered_index" in sample:
        return "cmap", int(sample["cmap_filtered_index"])
    raise ValueError("Sample has no reproducible dataset index")


def video_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    capture = cv2.VideoCapture(str(path))
    valid = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    capture.release()
    return valid


def annotate_tile(
    image: np.ndarray,
    *,
    object_name: str,
    sample_number: int,
    index_label: str,
    index_value: int,
    phase: str,
) -> np.ndarray:
    # Isaac Gym returns RGB(A), while OpenCV's video writer expects BGR.
    tile = np.ascontiguousarray(image[:, :, :3][:, :, ::-1])
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tile.shape[1], 43), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, tile, 0.22, 0.0, tile)
    cv2.putText(
        tile,
        f"#{sample_number:02d} {index_label}={index_value}",
        (7, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        f"{object_name} | {phase}",
        (7, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (90, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def render_object(
    *,
    robot_name: str,
    object_name: str,
    samples: list[dict],
    sample_number_offset: int,
    dro_root: Path,
    output_video: Path,
    output_report: Path,
    width: int,
    height: int,
    fps: float,
    stride: int,
    device_id: int,
    use_gpu_physx: bool,
) -> dict:
    # Insert D(R,O) ahead of this repository's own ``utils`` package.
    sys.path.insert(0, str(dro_root))
    # D(R,O)'s controller imports ``viser`` only for its standalone debug
    # entrypoint.  The minimal local Isaac Gym environment intentionally does
    # not install that UI package, and the controller path below never uses it.
    sys.modules.setdefault("viser", types.ModuleType("viser"))
    from utils.controller import controller
    from utils.hand_model import create_hand_model

    q_state = torch.stack(
        [row.get("q_state", row.get("q_rot6d")) for row in samples]
    )
    outer_q, inner_q = controller(robot_name, q_state)
    hand = create_hand_model(robot_name, device="cpu")
    joint_names = list(hand.get_joint_orders())
    batch_size = len(samples)

    gym = gymapi.acquire_gym()
    sim = gym.create_sim(
        device_id,
        device_id,
        gymapi.SIM_PHYSX,
        sim_parameters(use_gpu_physx=use_gpu_physx),
    )
    if sim is None:
        raise RuntimeError("Failed to create local Isaac Gym CPU PhysX simulation")
    gym.set_light_parameters(
        sim,
        0,
        gymapi.Vec3(1.0, 0.98, 0.95),
        gymapi.Vec3(0.72, 0.72, 0.72),
        gymapi.Vec3(-0.45, -0.35, -1.0),
    )

    data_root = dro_root / "data/data_urdf"
    robot_root = data_root / "robot"
    object_root = data_root / "object"
    robot_meta = json.loads(
        (robot_root / "urdf_assets_meta.json").read_text(encoding="utf-8")
    )
    robot_file = str(robot_meta["urdf_path"][robot_name])
    robot_prefix = "data/data_urdf/robot/"
    if robot_file.startswith(robot_prefix):
        robot_file = robot_file[len(robot_prefix) :]
    dataset_name, object_token = object_name.split("+")
    object_file = (
        f"{dataset_name}/{object_token}/coacd_decomposed_object_one_link.urdf"
    )

    robot_options = gymapi.AssetOptions()
    robot_options.disable_gravity = True
    robot_options.fix_base_link = True
    robot_options.collapse_fixed_joints = True
    object_options = gymapi.AssetOptions()
    object_options.override_com = True
    object_options.override_inertia = True
    object_options.density = 500.0
    robot_asset = gym.load_asset(sim, str(robot_root), robot_file, robot_options)
    object_asset = gym.load_asset(sim, str(object_root), object_file, object_options)
    if robot_asset is None or object_asset is None:
        gym.destroy_sim(sim)
        raise RuntimeError(f"Failed to load assets for {object_name}")

    envs = []
    object_handles = []
    robot_handles = []
    cameras = []
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = width
    camera_properties.height = height
    camera_properties.horizontal_fov = 55.0
    camera_properties.enable_tensors = False
    for env_index in range(batch_size):
        env = gym.create_env(
            sim,
            gymapi.Vec3(-2.0, -2.0, -2.0),
            gymapi.Vec3(2.0, 2.0, 2.0),
            4,
        )
        object_handle = gym.create_actor(
            env, object_asset, gymapi.Transform(), f"object_{env_index}", env_index
        )
        robot_handle = gym.create_actor(
            env, robot_asset, gymapi.Transform(), f"{robot_name}_{env_index}", env_index
        )
        for actor_handle in (object_handle, robot_handle):
            shapes = gym.get_actor_rigid_shape_properties(env, actor_handle)
            for shape in shapes:
                shape.friction = 3.0
            gym.set_actor_rigid_shape_properties(env, actor_handle, shapes)
        hand_color = gymapi.Vec3(0.38, 0.68, 0.92)
        for body_index in range(gym.get_actor_rigid_body_count(env, robot_handle)):
            gym.set_rigid_body_color(
                env, robot_handle, body_index, gymapi.MESH_VISUAL, hand_color
            )
        gym.set_rigid_body_color(
            env,
            object_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(0.95, 0.55, 0.12),
        )
        properties = gym.get_actor_dof_properties(env, robot_handle)
        properties["driveMode"].fill(gymapi.DOF_MODE_POS)
        properties["stiffness"].fill(1000.0)
        properties["damping"].fill(200.0)
        gym.set_actor_dof_properties(env, robot_handle, properties)
        camera = gym.create_camera_sensor(env, camera_properties)
        gym.set_camera_location(
            camera,
            env,
            gymapi.Vec3(0.48, 0.39, 0.31),
            gymapi.Vec3(0.0, 0.0, 0.035),
        )
        envs.append(env)
        object_handles.append(object_handle)
        robot_handles.append(robot_handle)
        cameras.append(camera)

    isaac_order = []
    for name in joint_names:
        index = gym.find_actor_dof_index(
            envs[0], robot_handles[0], name, gymapi.DOMAIN_ACTOR
        )
        if index < 0:
            gym.destroy_sim(sim)
            raise ValueError(f"{robot_name} URDF is missing joint {name}")
        isaac_order.append(index)
    for env_index, (env, robot_handle) in enumerate(zip(envs, robot_handles)):
        outer_isaac = np.empty(len(joint_names), dtype=np.float32)
        inner_isaac = np.empty(len(joint_names), dtype=np.float32)
        for urdf_index, isaac_index in enumerate(isaac_order):
            outer_isaac[isaac_index] = float(outer_q[env_index, urdf_index])
            inner_isaac[isaac_index] = float(inner_q[env_index, urdf_index])
        states = gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
        states["pos"] = outer_isaac
        gym.set_actor_dof_states(env, robot_handle, states, gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env, robot_handle, inner_isaac)

    gym.prepare_sim(sim)
    rigid_tensor = gym.acquire_rigid_body_state_tensor(sim)
    rigid = gymtorch.wrap_tensor(rigid_tensor)
    robot_body_count = gym.get_asset_rigid_body_count(robot_asset)
    object_body_count = gym.get_asset_rigid_body_count(object_asset)
    total_body_count = robot_body_count + object_body_count
    if object_body_count != 1:
        gym.destroy_sim(sim)
        raise ValueError(f"Expected one object body, got {object_body_count}")
    body_properties = gym.get_actor_rigid_body_properties(envs[0], object_handles[0])
    object_mass = float(sum(item.mass for item in body_properties))
    object_force = 0.5 * object_mass

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    grid_width = width * 4
    grid_height = height * 2
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid_width, grid_height)
    )
    if not writer.isOpened():
        gym.destroy_sim(sim)
        raise RuntimeError(f"Could not open video writer: {output_video}")

    simulation_step = 0
    frame_count = 0

    def object_positions() -> np.ndarray:
        gym.fetch_results(sim, True)
        gym.refresh_rigid_body_state_tensor(sim)
        return rigid[::total_body_count, :3].cpu().numpy().copy()

    def capture(phase: str, *, force: bool = False) -> None:
        nonlocal simulation_step, frame_count
        if force or simulation_step % stride == 0:
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            tiles = []
            for sample_number, (env, camera, sample) in enumerate(
                zip(envs, cameras, samples), start=sample_number_offset + 1
            ):
                index_label, index_value = sample_index_label(sample)
                image = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)
                rgba = np.asarray(image, dtype=np.uint8).reshape(height, width, 4)
                tiles.append(
                    annotate_tile(
                        rgba,
                        object_name=object_name,
                        sample_number=sample_number,
                        index_label=index_label,
                        index_value=index_value,
                        phase=phase,
                    )
                )
            while len(tiles) < 8:
                blank = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "No additional",
                    (12, height // 2 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (130, 130, 130),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    blank,
                    "filtered success",
                    (12, height // 2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (130, 130, 130),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(blank)
            grid = np.concatenate(
                (np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:], axis=1)),
                axis=0,
            )
            writer.write(grid)
            frame_count += 1
        simulation_step += 1

    direction_rows: list[dict] = []
    try:
        capture("initial / open", force=True)
        for _ in range(100):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            capture("closing")
        closure_positions = object_positions()
        overall_start = closure_positions.copy()
        segment_start = closure_positions.copy()
        force_tensor = torch.zeros(
            (batch_size, total_body_count, 3), dtype=torch.float32
        )
        for direction_name, direction in DIRECTIONS:
            force_tensor.zero_()
            force_tensor[:, 0, :] = object_force * torch.tensor(direction)
            for _ in range(100):
                gym.apply_rigid_body_force_tensors(
                    sim,
                    gymtorch.unwrap_tensor(force_tensor),
                    None,
                    gymapi.ENV_SPACE,
                )
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                capture(f"disturbance {direction_name}")
            endpoint = object_positions()
            direction_rows.append(
                {
                    "direction": direction_name,
                    "endpoint_m": endpoint.tolist(),
                    "segment_displacement_m": np.linalg.norm(
                        endpoint - segment_start, axis=1
                    ).tolist(),
                    "cumulative_displacement_m": np.linalg.norm(
                        endpoint - overall_start, axis=1
                    ).tolist(),
                }
            )
            segment_start = endpoint
        capture("final", force=True)
    finally:
        writer.release()
        gym.destroy_sim(sim)

    final_displacements = np.asarray(
        direction_rows[-1]["cumulative_displacement_m"], dtype=np.float64
    )
    segment_matrix = np.asarray(
        [row["segment_displacement_m"] for row in direction_rows], dtype=np.float64
    ).T
    final_success = final_displacements <= 0.02
    strict_success = (segment_matrix <= 0.02).all(axis=1)
    sample_reports = []
    for index, sample in enumerate(samples):
        sample_report = {
            "sample_number": sample_number_offset + index + 1,
            "object_rank": int(sample["object_rank"]),
            "closure_position_m": closure_positions[index].tolist(),
            "final_displacement_m": float(final_displacements[index]),
            "maximum_segment_displacement_m": float(segment_matrix[index].max()),
            "local_final_success": bool(final_success[index]),
            "local_strict_six_direction_success": bool(strict_success[index]),
        }
        for field in ("source_index", "filtered_index", "cmap_filtered_index"):
            if field in sample:
                sample_report[field] = int(sample[field])
        sample_reports.append(sample_report)
    dataset_kind = str(samples[0].get("dataset_kind", "multidex_filtered"))
    report = {
        "schema": "filtered-multihand-isaacgym-video-v3",
        "dataset_kind": dataset_kind,
        "robot_name": robot_name,
        "object_name": object_name,
        "filtered_samples_available": None,
        "samples_recorded": batch_size,
        "video": str(output_video.resolve()),
        "video_frames": frame_count,
        "video_fps": fps,
        "grid": {"rows": 2, "columns": 4, "tile_width": width, "tile_height": height},
        "local_final_successes": int(final_success.sum()),
        "local_strict_six_direction_successes": int(strict_success.sum()),
        "samples": sample_reports,
        "directions": direction_rows,
        "protocol": {
            "source": "D(R,O) Isaac Gym filtering protocol",
            "physics": (
                "GPU PhysX compatibility replay"
                if use_gpu_physx
                else "local CPU PhysX compatibility replay"
            ),
            "gpu_physx": use_gpu_physx,
            "gravity": False,
            "robot_friction": 3.0,
            "object_friction": 3.0,
            "object_density": 500.0,
            "steps_per_second": 100,
            "substeps": 2,
            "closure_steps": 100,
            "direction_seconds": 1.0,
            "direction_order": [name for name, _ in DIRECTIONS],
            "acceleration_mps2": 0.5,
            "success_threshold_m": 0.02,
            "robot_urdf": robot_file,
            "object_urdf": object_file,
        },
    }
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    if args.samples_per_object <= 0:
        raise ValueError("--samples-per-object must be positive")
    if not 1 <= args.samples_per_video <= 8:
        raise ValueError("--samples-per-video must be between 1 and 8")
    if args.video_stride <= 0 or args.video_fps <= 0:
        raise ValueError("video stride and FPS must be positive")

    filtered_path = args.filtered or (
        args.dro_root / "data/MultiDex_filtered" / args.robot_name / f"{args.robot_name}.pt"
    )
    raw_path = args.raw or (
        args.dro_root / "data/MultiDex" / args.robot_name / f"{args.robot_name}.pt"
    )

    if args.selection_file is not None:
        selection_robot, object_names, selected, counts = load_selection(
            args.selection_file.resolve()
        )
        if selection_robot is not None and selection_robot != args.robot_name:
            raise ValueError(
                f"Selection is for {selection_robot}, not --robot-name={args.robot_name}"
            )
    else:
        seen_payload = json.loads(args.seen_file.read_text(encoding="utf-8"))
        object_names = [str(name) for name in seen_payload["train"]]
        selected, counts = select_samples(
            filtered_path.resolve(),
            raw_path.resolve(),
            object_names,
            args.samples_per_object,
            args.robot_name,
        )
    if args.only_object:
        requested = set(args.only_object)
        unknown = sorted(requested - set(object_names))
        if unknown:
            raise ValueError(f"Objects are not in seen-48: {unknown}")
        object_names = [name for name in object_names if name in requested]
    if args.max_objects is not None:
        object_names = object_names[: args.max_objects]

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "videos").mkdir(exist_ok=True)
    (args.output_root / "reports").mkdir(exist_ok=True)
    (args.output_root / "logs").mkdir(exist_ok=True)
    (args.output_root / "status").mkdir(exist_ok=True)

    manifest_rows = []
    started = time.monotonic()
    for object_index, object_name in enumerate(object_names, start=1):
        token = object_name.replace("+", "__")
        report_path = args.output_root / "reports" / f"{token}.json"
        status_path = args.output_root / "status" / f"{token}.status"
        samples = selected[object_name]
        if not samples:
            reason = "no_filtered_success"
            status_path.write_text(f"{reason}\n", encoding="utf-8")
            row = {
                "object_name": object_name,
                "status": reason,
                "filtered_samples_available": counts[object_name],
                "samples_requested": args.samples_per_object,
            }
            manifest_rows.append(row)
            print(
                f"[{object_index}/{len(object_names)}] {object_name}: "
                f"skipped ({counts[object_name]} filtered samples)",
                flush=True,
            )
            continue

        status_path.write_text("running\n", encoding="utf-8")
        object_started = time.monotonic()
        print(
            f"[{object_index}/{len(object_names)}] {object_name}: recording "
            f"{len(samples)}/{counts[object_name]} filtered successes",
            flush=True,
        )
        try:
            part_reports = []
            chunks = [
                samples[start : start + args.samples_per_video]
                for start in range(0, len(samples), args.samples_per_video)
            ]
            for part_index, chunk in enumerate(chunks, start=1):
                stem = f"{token}_{args.robot_name}_part{part_index:02d}_{len(chunk)}poses"
                video = args.output_root / "videos" / f"{stem}.mp4"
                part_report_path = args.output_root / "reports" / f"{stem}.json"
                if args.resume and video_is_valid(video) and part_report_path.is_file():
                    part_report = json.loads(
                        part_report_path.read_text(encoding="utf-8")
                    )
                    print(
                        f"  part {part_index}/{len(chunks)}: resume-skip",
                        flush=True,
                    )
                else:
                    part_report = render_object(
                        robot_name=args.robot_name,
                        object_name=object_name,
                        samples=chunk,
                        sample_number_offset=(part_index - 1) * args.samples_per_video,
                        dro_root=args.dro_root.resolve(),
                        output_video=video,
                        output_report=part_report_path,
                        width=args.width,
                        height=args.height,
                        fps=args.video_fps,
                        stride=args.video_stride,
                        device_id=args.device_id,
                        use_gpu_physx=args.use_gpu_physx,
                    )
                    print(
                        f"  part {part_index}/{len(chunks)}: "
                        f"{part_report['local_final_successes']}/{len(chunk)} final success",
                        flush=True,
                    )
                part_reports.append(part_report)

            report = {
                "schema": "filtered-multihand-object-recording-v3",
                "dataset_kind": "multidex_filtered",
                "robot_name": args.robot_name,
                "object_name": object_name,
                "filtered_samples_available": counts[object_name],
                "samples_requested": args.samples_per_object,
                "samples_recorded": sum(row["samples_recorded"] for row in part_reports),
                "videos_recorded": len(part_reports),
                "local_final_successes": sum(
                    row["local_final_successes"] for row in part_reports
                ),
                "local_strict_six_direction_successes": sum(
                    row["local_strict_six_direction_successes"] for row in part_reports
                ),
                "videos": [row["video"] for row in part_reports],
                "parts": part_reports,
                "elapsed_seconds": time.monotonic() - object_started,
            }
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            status_path.write_text("complete\n", encoding="utf-8")
            manifest_rows.append({"object_name": object_name, "status": "complete", **report})
            print(
                f"[{object_index}/{len(object_names)}] {object_name}: complete, "
                f"local final success {report['local_final_successes']}/"
                f"{report['samples_recorded']}, "
                f"{report['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except Exception as error:
            status_path.write_text("failed\n", encoding="utf-8")
            manifest_rows.append(
                {
                    "object_name": object_name,
                    "status": "failed",
                    "error": repr(error),
                    "filtered_samples_available": counts[object_name],
                }
            )
            raise
        finally:
            manifest_dataset_kind = str(
                samples[0].get("dataset_kind", "multidex_filtered")
            ) if samples else "unknown"
            manifest = {
                "schema": "filtered-multihand-recordings-v3",
                "dataset_kind": manifest_dataset_kind,
                "robot_name": args.robot_name,
                "selection_file": (
                    str(args.selection_file.resolve())
                    if args.selection_file is not None
                    else None
                ),
                "filtered_dataset": str(filtered_path.resolve()),
                "raw_dataset": str(raw_path.resolve()),
                "seen_file": str(args.seen_file.resolve()),
                "samples_per_object": args.samples_per_object,
                "samples_per_video": args.samples_per_video,
                "objects_requested": len(object_names),
                "objects_processed": len(manifest_rows),
                "elapsed_seconds": time.monotonic() - started,
                "objects": manifest_rows,
            }
            (args.output_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
