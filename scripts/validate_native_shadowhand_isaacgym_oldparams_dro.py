#!/usr/bin/env python3
"""Validate D(R,O) native-hand grasps with the complete old Gym protocol."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from isaacgym import gymapi, gymtorch
import torch


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gendex-root", type=Path, required=True)
    parser.add_argument("--native-hand-root", type=Path, required=True)
    parser.add_argument(
        "--native-hand-urdf", default="shadow_hand_right_glb.urdf"
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-object", action="append")
    parser.add_argument("--max-samples-per-object", type=int)
    parser.add_argument("--progress-every", type=int, default=16)
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
    parser.add_argument("--direction-seconds", type=float, default=5.0 / 6.0)
    parser.add_argument("--direction-order", choices=("gendex", "cedex"), default="gendex")
    parser.add_argument("--success-mode", choices=("final", "per_direction"), default="per_direction")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--acceleration", type=float, default=0.5)
    parser.add_argument("--robot-friction", type=float, default=10.0)
    parser.add_argument("--object-friction", type=float, default=10.0)
    parser.add_argument("--object-density", type=float, default=10000.0)
    parser.add_argument("--object-linear-damping", type=float, default=10.0)
    parser.add_argument("--object-angular-damping", type=float, default=100.0)
    parser.add_argument("--joint-stiffness", type=float, default=400.0)
    parser.add_argument("--joint-damping", type=float, default=400.0)
    parser.add_argument("--joint-armature", type=float, default=0.01)
    parser.add_argument("--joint-velocity", type=float, default=0.8)
    parser.add_argument("--solver-position-iterations", type=int, default=4)
    parser.add_argument("--solver-velocity-iterations", type=int, default=0)
    parser.add_argument("--contact-offset", type=float, default=0.01)
    parser.add_argument("--rest-offset", type=float, default=0.0)
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument(
        "--object-source",
        choices=("gendex", "prepared"),
        default="gendex",
        help="Load GenDex object URDFs or build a one-link URDF from the prepared raw mesh.",
    )
    return parser.parse_args()


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
    params.physx.num_threads = 0
    params.physx.use_gpu = True
    params.physx.num_subscenes = 0
    params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    params.use_gpu_pipeline = True
    sim = gym.create_sim(args.device_id, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Isaac Gym failed to create the GPU PhysX simulation")
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
    options.density = args.object_density
    options.fix_base_link = True
    options.disable_gravity = True
    options.flip_visual_attachments = False
    options.armature = args.joint_armature
    options.use_mesh_materials = True
    options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    options.override_com = True
    options.override_inertia = True
    options.vhacd_enabled = True
    options.vhacd_params = gymapi.VhacdParams()
    options.vhacd_params.resolution = 1_000_000
    return options


def object_asset_options(args: argparse.Namespace) -> gymapi.AssetOptions:
    options = gymapi.AssetOptions()
    options.density = args.object_density
    options.linear_damping = args.object_linear_damping
    options.angular_damping = args.object_angular_damping
    options.fix_base_link = False
    options.disable_gravity = True
    options.use_mesh_materials = True
    options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    options.override_com = True
    options.override_inertia = True
    options.vhacd_enabled = True
    options.vhacd_params = gymapi.VhacdParams()
    options.vhacd_params.resolution = 1_000_000
    return options


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
    samples = list(object_group["samples"])
    if args.max_samples_per_object is not None:
        samples = samples[: max(0, args.max_samples_per_object)]
    count = len(samples)
    if not count:
        return []

    sim = make_sim(gym, args)
    try:
        movable_urdf = ensure_movable_native_urdf(
            args.native_hand_root.resolve(), args.native_hand_urdf
        )
        hand_asset = gym.load_asset(
            sim,
            str(movable_urdf.parent),
            movable_urdf.name,
            hand_asset_options(args),
        )
        if args.object_source == "prepared":
            object_mesh_path = Path(object_group["object_mesh"]).resolve()
            object_urdf = prepared_object_urdf(
                object_mesh_path,
                args.output.resolve().parent / "prepared_object_urdfs",
                object_name,
            )
            object_root = object_urdf.parent
            object_file = object_urdf.name
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
            object_asset_options(args),
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
        per_row = max(1, math.ceil(math.sqrt(count)))
        envs = []
        object_actor_indices = []
        object_body_indices = []
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
            dof_state = np.zeros(dof_count, dtype=gymapi.DofState.dtype)
            dof_state["pos"] = outer[reorder]
            gym.set_actor_dof_states(env, hand_actor, dof_state, gymapi.STATE_ALL)

            object_pose = gymapi.Transform()
            object_actor = gym.create_actor(
                env, object_asset, object_pose, "object", env_index, 0, 0
            )
            object_shapes = gym.get_actor_rigid_shape_properties(env, object_actor)
            for shape in object_shapes:
                shape.friction = args.object_friction
                shape.restitution = 0.0
            gym.set_actor_rigid_shape_properties(env, object_actor, object_shapes)
            object_actor_indices.append(
                gym.get_actor_index(env, object_actor, gymapi.DOMAIN_SIM)
            )
            object_body_indices.append(
                gym.find_actor_rigid_body_index(
                    env, object_actor, "object", gymapi.DOMAIN_SIM
                )
            )
            outer_targets.append(outer[reorder])
            inner_targets.append(inner[reorder])

        gym.prepare_sim(sim)
        device = torch.device(f"cuda:{args.device_id}")
        dof_targets = torch.as_tensor(
            np.stack(outer_targets), device=device, dtype=torch.float32
        ).reshape(-1)
        inner_targets_tensor = torch.as_tensor(
            np.stack(inner_targets), device=device, dtype=torch.float32
        ).reshape(-1)
        gym.set_dof_position_target_tensor(
            sim, gymtorch.unwrap_tensor(dof_targets)
        )
        for _ in range(3):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        for _ in range(args.closure_steps):
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(inner_targets_tensor)
            )
            gym.simulate(sim)
            gym.fetch_results(sim, True)

        root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        rigid = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
        object_actor_indices_t = torch.as_tensor(
            object_actor_indices, device=device, dtype=torch.long
        )
        object_body_indices_t = torch.as_tensor(
            object_body_indices, device=device, dtype=torch.long
        )
        object_mesh = trimesh.load(object_mesh_path, force="mesh")
        force_magnitude = (
            float(object_mesh.volume) * args.object_density * args.acceleration
        )
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
                    "force_magnitude_n": force_magnitude,
                }
            )
        return results
    finally:
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
    if prepared["hand"] not in {
        "contactdiff_shadowhand",
        "shadowhand",
        "barrett",
        "gendex_barrett",
        "contactdiff_barrett",
    }:
        raise ValueError(
            "Prepared manifest must use a supported ShadowHand or Barrett hand"
        )
    groups = list(prepared["objects"])
    prepared_joint_names = list(prepared["joint_names"])
    for group in groups:
        group["_joint_names"] = prepared_joint_names
    if args.only_object:
        selected = set(args.only_object)
        groups = [group for group in groups if group["object_name"] in selected]
    expected_trials = sum(
        min(
            len(group["samples"]),
            args.max_samples_per_object
            if args.max_samples_per_object is not None
            else len(group["samples"]),
        )
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
                min(
                    len(group["samples"]),
                    args.max_samples_per_object
                    if args.max_samples_per_object is not None
                    else len(group["samples"]),
                )
                for group in groups
                if group["object_name"] == row["object_name"]
            ),
            -1,
        )
    }
    report = {
        "schema": "contactdiff-dro-native-hand-oldparams-isaacgym-v1",
        "hand": prepared["hand"],
        "simulator": "Isaac Gym Preview 4 GPU PhysX",
        "prepared": str(args.prepared.resolve()),
        "native_hand_urdf": str(
            (args.native_hand_root / args.native_hand_urdf).resolve()
        ),
        "protocol": {
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
            "object_linear_damping": args.object_linear_damping,
            "object_angular_damping": args.object_angular_damping,
            "joint_stiffness": args.joint_stiffness,
            "joint_damping": args.joint_damping,
            "joint_armature": args.joint_armature,
            "virtual_root_joint_names": list(VIRTUAL_ROOT_JOINTS),
            "virtual_root_stiffness": args.virtual_root_stiffness,
            "virtual_root_damping": args.virtual_root_damping,
            "joint_velocity": (
                args.joint_velocity
                if args.joint_velocity >= 0.0
                else "inherit URDF/importer"
            ),
            "closure_steps": args.closure_steps,
            "direction_seconds": args.direction_seconds,
            "direction_order": args.direction_order,
            "directions": [
                name
                for name, _ in (
                    DIRECTIONS_CEDEX
                    if args.direction_order == "cedex"
                    else DIRECTIONS_GENDEX
                )
            ],
            "steps_per_direction": max(
                1, round(args.steps_per_second * args.direction_seconds)
            ),
            "acceleration_mps2": args.acceleration,
            "force_magnitude": "object STL volume * density * acceleration",
            "success_mode": args.success_mode,
            "success_threshold_m": args.threshold,
            "ground_enabled": not args.no_ground,
            "object_source": args.object_source,
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
