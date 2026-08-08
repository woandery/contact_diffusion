#!/usr/bin/env python3
"""Validate Contact Diffusion OOD10 grasps with D(R,O) Isaac Gym."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch

import numpy as np
import torch
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.isaac_validator import IsaacValidator
from utils.controller import controller
from utils.hand_model import create_hand_model


OBJECT_NAMES = {
    "contactdb_apple": "contactdb+apple",
    "contactdb_camera": "contactdb+camera",
    "contactdb_cylinder_medium": "contactdb+cylinder_medium",
    "contactdb_door_knob": "contactdb+door_knob",
    "contactdb_rubber_duck": "contactdb+rubber_duck",
    "contactdb_water_bottle": "contactdb+water_bottle",
    "ycb_055_baseball": "ycb+baseball",
    "ycb_016_pear": "ycb+pear",
    "ycb_010_potted_meat_can": "ycb+potted_meat_can",
    "ycb_005_tomato_soup_can": "ycb+tomato_soup_can",
}


class CompatibleIsaacValidator(IsaacValidator):
    """D(R,O) validator with tensor acquisition delayed until prepare_sim.

    Isaac Gym Preview 4 can terminate natively on newer drivers when state
    tensors are acquired immediately after create_sim, before any environment
    exists.  Apart from that lifecycle adjustment, these settings and the
    actor setup are copied from D(R,O)'s IsaacValidator.
    """

    def __init__(
        self,
        robot_name,
        joint_orders,
        batch_size,
        gpu=0,
        is_filter=False,
        use_gui=False,
        robot_friction=3.0,
        object_friction=3.0,
        steps_per_sec=100,
        grasp_step=100,
        debug_interval=0.01,
    ):
        self.gym = gymapi.acquire_gym()
        self.robot_name = robot_name
        self.joint_orders = joint_orders
        self.batch_size = batch_size
        self.gpu = gpu
        self.is_filter = is_filter
        self.robot_friction = robot_friction
        self.object_friction = object_friction
        self.steps_per_sec = steps_per_sec
        self.grasp_step = grasp_step
        self.debug_interval = debug_interval

        self.envs = []
        self.robot_handles = []
        self.object_handles = []
        self.robot_asset = None
        self.object_asset = None
        self.rigid_body_num = None
        self.object_force = None
        self.urdf2isaac_order = None
        self.isaac2urdf_order = None

        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1 / steps_per_sec
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.01
        self.sim_params.physx.rest_offset = 0.0

        graphics_device = self.gpu if use_gui else -1
        self.sim = self.gym.create_sim(
            self.gpu, graphics_device, gymapi.SIM_PHYSX, self.sim_params
        )
        if self.sim is None:
            raise RuntimeError("D(R,O) Isaac Gym create_sim failed")
        print("[stage] create_sim complete", flush=True)
        self._rigid_body_states = None
        self._dof_states = None

        self.viewer = None
        self.has_viewer = bool(use_gui)
        if use_gui:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 1920
            self.camera_props.height = 1080
            self.camera_props.use_collision_geometry = True
            self.viewer = self.gym.create_viewer(self.sim, self.camera_props)
            self.gym.viewer_camera_look_at(
                self.viewer,
                None,
                gymapi.Vec3(1, 0, 0),
                gymapi.Vec3(0, 0, 0),
            )

        self.robot_asset_options = gymapi.AssetOptions()
        self.robot_asset_options.disable_gravity = True
        self.robot_asset_options.fix_base_link = True
        self.robot_asset_options.collapse_fixed_joints = True

        self.object_asset_options = gymapi.AssetOptions()
        self.object_asset_options.override_com = True
        self.object_asset_options.override_inertia = True
        self.object_asset_options.density = 500

    def set_actor_pose_dof(self, q):
        self.gym.prepare_sim(self.sim)
        self._rigid_body_states = self.gym.acquire_rigid_body_state_tensor(
            self.sim
        )
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(actor_root_state)
        root_state[:] = torch.tensor(
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            dtype=torch.float32,
        )
        self.gym.set_actor_root_state_tensor(self.sim, actor_root_state)

        outer_q, inner_q = controller(self.robot_name, q)
        for env_idx, env in enumerate(self.envs):
            robot_handle = self.robot_handles[env_idx]
            dof_states_initial = self.gym.get_actor_dof_states(
                env, robot_handle, gymapi.STATE_ALL
            ).copy()
            dof_states_initial["pos"] = outer_q[
                env_idx, self.urdf2isaac_order
            ]
            self.gym.set_actor_dof_states(
                env, robot_handle, dof_states_initial, gymapi.STATE_ALL
            )

            dof_states_target = self.gym.get_actor_dof_states(
                env, robot_handle, gymapi.STATE_ALL
            ).copy()
            dof_states_target["pos"] = inner_q[
                env_idx, self.urdf2isaac_order
            ]
            self.gym.set_actor_dof_position_targets(
                env, robot_handle, dof_states_target["pos"]
            )


def load_q_batch(
    path: Path,
    object_id: str,
    joint_orders: list[str],
    expected_gripper: str,
):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = sorted(
        (
            row for row in payload["records"]
            if row["object_id"] == object_id
        ),
        key=lambda row: int(row["sample_index"]),
    )
    expected_count = int(payload["selection"]["samples_per_object"])
    if len(records) != expected_count:
        raise ValueError(
            f"{object_id}: expected {expected_count}, got {len(records)}"
        )
    unexpected_grippers = sorted(
        {str(row.get("gripper")) for row in records}
        - {expected_gripper}
    )
    if unexpected_grippers:
        raise ValueError(
            f"{object_id}: expected gripper {expected_gripper}, "
            f"got {unexpected_grippers}"
        )
    expected_indices = list(range(expected_count))
    indices = [int(row["sample_index"]) for row in records]
    if indices != expected_indices:
        raise ValueError(f"{object_id}: unexpected sample indices {indices}")
    values = []
    energies = []
    for row in records:
        fk = row["fk"]
        candidates = fk["candidates"]
        if len(candidates) != 1 or int(candidates[0]["rank"]) != 0:
            raise ValueError(f"{row['record_id']}: expected rank-0 only")
        candidate = candidates[0]
        pose = np.asarray(candidate["root_pose"], dtype=np.float64)
        euler = Rotation.from_matrix(pose[:3, :3]).as_euler("XYZ")
        source_joints = {
            name: float(value)
            for name, value in zip(
                fk["joint_names"], candidate["joint_positions"]
            )
        }
        missing = [name for name in joint_orders[6:] if name not in source_joints]
        if missing:
            raise ValueError(f"{row['record_id']}: missing joints {missing}")
        q = np.concatenate(
            (
                pose[:3, 3],
                euler,
                np.asarray(
                    [source_joints[name] for name in joint_orders[6:]],
                    dtype=np.float64,
                ),
            )
        )
        values.append(q)
        energies.append(float(candidate["optimization_score"]))
    return payload, records, torch.as_tensor(
        np.stack(values), dtype=torch.float32
    ), energies


def object_positions(simulator: IsaacValidator) -> torch.Tensor:
    simulator.gym.fetch_results(simulator.sim, True)
    simulator.gym.refresh_rigid_body_state_tensor(simulator.sim)
    states = gymtorch.wrap_tensor(simulator._rigid_body_states)
    return states[::simulator.rigid_body_num, :3].clone().cpu()


def run_diagnostics(simulator: IsaacValidator):
    for _ in range(simulator.grasp_step):
        simulator.gym.simulate(simulator.sim)
    closure_position = object_positions(simulator)
    overall_start = closure_position.clone()

    force_tensor = torch.zeros(
        [len(simulator.envs), simulator.rigid_body_num, 3],
        dtype=torch.float32,
    )
    directions = (
        ("+x", (1.0, 0.0, 0.0)),
        ("+y", (0.0, 1.0, 0.0)),
        ("+z", (0.0, 0.0, 1.0)),
        ("-x", (-1.0, 0.0, 0.0)),
        ("-y", (0.0, -1.0, 0.0)),
        ("-z", (0.0, 0.0, -1.0)),
    )
    segment_displacements = []
    cumulative_displacements = []
    endpoints = []
    segment_start = overall_start
    for _, direction in directions:
        forces = force_tensor.clone()
        direction_tensor = torch.as_tensor(direction, dtype=torch.float32)
        forces[:, 0, :] = simulator.object_force.reshape(-1, 1) * direction_tensor
        for _ in range(simulator.steps_per_sec):
            simulator.gym.apply_rigid_body_force_tensors(
                simulator.sim,
                gymtorch.unwrap_tensor(forces),
                None,
                gymapi.ENV_SPACE,
            )
            simulator.gym.simulate(simulator.sim)
            simulator.gym.fetch_results(simulator.sim, True)
        endpoint = object_positions(simulator)
        segment_displacements.append((endpoint - segment_start).norm(dim=-1))
        cumulative_displacements.append((endpoint - overall_start).norm(dim=-1))
        endpoints.append(endpoint)
        segment_start = endpoint

    segments = torch.stack(segment_displacements, dim=1)
    cumulative = torch.stack(cumulative_displacements, dim=1)
    final_displacement = cumulative[:, -1]
    maximum_segment = segments.max(dim=1).values
    final_success = final_displacement <= 0.02
    strict_success = (segments <= 0.02).all(dim=1)
    return {
        "directions": directions,
        "closure_position": closure_position,
        "segments": segments,
        "cumulative": cumulative,
        "endpoints": torch.stack(endpoints, dim=1),
        "final_displacement": final_displacement,
        "maximum_segment": maximum_segment,
        "final_success": final_success,
        "strict_success": strict_success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--robot-name",
        choices=("barrett", "shadowhand"),
        default="barrett",
    )
    parser.add_argument("--object-id", choices=tuple(OBJECT_NAMES), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    robot_name = args.robot_name
    expected_gripper = {
        "barrett": "Barrett",
        "shadowhand": "shadow_hand",
    }[robot_name]
    hand = create_hand_model(robot_name, device="cpu")
    joint_orders = list(hand.get_joint_orders())
    payload, records, q_batch, energies = load_q_batch(
        args.candidates.resolve(),
        args.object_id,
        joint_orders,
        expected_gripper,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[:args.limit]
        q_batch = q_batch[:args.limit]
        energies = energies[:args.limit]
    object_name = OBJECT_NAMES[args.object_id]
    outer_q, inner_q = controller(robot_name, q_batch)

    simulator = CompatibleIsaacValidator(
        robot_name=robot_name,
        joint_orders=joint_orders,
        batch_size=len(records),
        gpu=args.gpu,
        is_filter=False,
        use_gui=False,
    )
    print("[stage] validator constructed", flush=True)
    data_root = ROOT / "data/data_urdf"
    robot_meta = json.loads(
        (data_root / "robot/urdf_assets_meta.json").read_text(encoding="utf-8")
    )
    robot_file = robot_meta["urdf_path"][robot_name]
    robot_prefix = "data/data_urdf/robot/"
    if robot_file.startswith(robot_prefix):
        robot_file = robot_file[len(robot_prefix):]
    dataset_name, object_token = object_name.split("+")
    object_file = (
        f"{dataset_name}/{object_token}/"
        "coacd_decomposed_object_one_link.urdf"
    )
    try:
        print(f"[stage] loading assets robot={robot_file} object={object_file}", flush=True)
        simulator.set_asset(
            robot_path=str(data_root / "robot"),
            robot_file=robot_file,
            object_path=str(data_root / "object"),
            object_file=object_file,
        )
        print("[stage] assets loaded", flush=True)
        simulator.create_envs()
        print("[stage] environments created", flush=True)
        simulator.set_actor_pose_dof(q_batch)
        print("[stage] actor state initialized", flush=True)
        diagnostics = run_diagnostics(simulator)
        print("[stage] diagnostics complete", flush=True)
    finally:
        simulator.destroy()

    results = []
    direction_names = [name for name, _ in diagnostics["directions"]]
    for row_index, record in enumerate(records):
        results.append({
            "object_name": args.object_id,
            "source_index": int(record["sample_index"]),
            "record_id": record["record_id"],
            "optimization_score": energies[row_index],
            "closure_position_m": diagnostics["closure_position"][row_index].tolist(),
            "outer_to_inner_joint_l2_rad": float(
                torch.norm(inner_q[row_index, 6:] - outer_q[row_index, 6:]).item()
            ),
            "direction_order": direction_names,
            "segment_displacements_m": diagnostics["segments"][row_index].tolist(),
            "cumulative_displacements_m": diagnostics["cumulative"][row_index].tolist(),
            "final_displacement_m": float(diagnostics["final_displacement"][row_index].item()),
            "maximum_segment_displacement_m": float(diagnostics["maximum_segment"][row_index].item()),
            "dro_success": bool(diagnostics["final_success"][row_index].item()),
            "strict_six_direction_success": bool(diagnostics["strict_success"][row_index].item()),
        })
    dro_successes = sum(row["dro_success"] for row in results)
    strict_successes = sum(row["strict_six_direction_success"] for row in results)
    report = {
        "schema": "contactdiff-ood10-dro-isaacgym-v1",
        "method": "ContactDiffusion-new-FK-matched64x32-top1",
        "robot_name": robot_name,
        "gripper": expected_gripper,
        "optimization_steps": int(payload["optimization_steps"]),
        "particles_per_contact_set": int(payload["particles"]),
        "object_name": args.object_id,
        "trials": len(results),
        "dro_successes": dro_successes,
        "dro_success_rate": dro_successes / len(results),
        "strict_six_direction_successes": strict_successes,
        "strict_six_direction_success_rate": strict_successes / len(results),
        "protocol": {
            "simulator": "Isaac Gym Preview 4 GPU PhysX",
            "source": "D(R,O) validation/isaac_validator.py and utils/controller.py",
            "compatibility_adjustments": [
                "headless graphics device disabled with create_sim(gpu, -1, ...)",
                "state tensors acquired after environment creation and prepare_sim",
            ],
            "steps_per_second": 100,
            "substeps": 2,
            "gravity_mps2": 0.0,
            "solver_type": "TGS",
            "solver_position_iterations": 8,
            "solver_velocity_iterations": 0,
            "contact_offset_m": 0.01,
            "rest_offset_m": 0.0,
            "robot_friction": 3.0,
            "object_friction": 3.0,
            "object_density_kg_m3": 500.0,
            "joint_stiffness": 1000.0,
            "joint_damping": 200.0,
            "fixed_base_link": True,
            "outer_open_fraction": 0.25,
            "inner_close_fraction": 0.15,
            "closure_steps": 100,
            "direction_seconds": 1.0,
            "direction_order": direction_names,
            "acceleration_mps2": 0.5,
            "success_threshold_m": 0.02,
            "dro_success_metric": "final displacement after sequential six directions",
            "diagnostic_strict_metric": "every direction segment displacement <= threshold",
        },
        "candidates": str(args.candidates.resolve()),
        "gpu": args.gpu,
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "object_name": args.object_id,
        "trials": len(results),
        "dro_successes": dro_successes,
        "strict_successes": strict_successes,
        "elapsed_seconds": report["elapsed_seconds"],
        "output": str(args.output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
