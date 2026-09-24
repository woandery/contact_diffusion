import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import trimesh.transformations as tr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from isaacgymenvs.utils.torch_jit_utils import to_torch
from isaacgymenvs.tasks.fetch.fetch_ptd import FetchPointCloudBase
from isaacgymenvs.tasks.fetch.fetch_mesh_pyompl import FetchMeshPyompl, image_to_video


class FetchPtdDROPyompl(FetchPointCloudBase, FetchMeshPyompl):
    """FetchBench baseline using D(R,O) grasps and a mounted dexterous hand."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)
        if self.hand_name not in ("barrett", "shadowhand"):
            raise ValueError(f"D(R,O) task requires Barrett or ShadowHand, got {self.hand_name}")
        if self.gripper_control_type != "position":
            raise ValueError("D(R,O) task requires position-controlled hand joints")

        actor_dofs = list(self.gym.get_actor_dof_names(self.envs[0], self.robots[0]))
        self.sim_hand_joint_names = actor_dofs[self.arm_dof_count:]
        self.current_outer_q = None
        self.current_inner_q = None

    @staticmethod
    def _q_root_pose(q):
        q = np.asarray(q, dtype=np.float64)
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("XYZ", q[3:6]).as_matrix()
        pose[:3, 3] = q[:3]
        return pose

    def _map_hand_q(self, q, dro_joint_names):
        q = np.asarray(q, dtype=np.float32)[6:]
        by_name = dict(zip(dro_joint_names, q))
        missing = [name for name in self.sim_hand_joint_names if name not in by_name]
        if missing:
            raise ValueError(f"D(R,O) output is missing hand joints: {missing}")
        return np.asarray([by_name[name] for name in self.sim_hand_joint_names], dtype=np.float32)

    def _run_dro(self, goal_pc):
        dro_cfg = self.cfg["solution"]["dro"]
        cache_dir = Path(dro_cfg["cache_dir"]).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        scene_name = Path(self.scene_config_path[0]).name
        stem = f"{scene_name}_task{self.get_task_idx():03d}_{self.hand_name}"
        input_path = cache_dir / f"{stem}_goal.npy"
        output_path = cache_dir / f"{stem}_grasps.json"
        np.save(input_path, goal_pc.astype(np.float32))
        inference_seed = int(self.cfg.get("seed", 42)) + self.get_task_idx()

        reuse_cache = bool(dro_cfg.get("reuse_cache", True))
        cache_matches = False
        if output_path.exists() and reuse_cache:
            try:
                cached = json.loads(output_path.read_text())
                cache_matches = (
                    cached.get("hand") == self.hand_name
                    and cached.get("sampled_points") == int(dro_cfg["points"])
                    and cached.get("optimization_steps") == int(dro_cfg["optimization_steps"])
                    and cached.get("seed") == inference_seed
                    and len(cached.get("records", [])) == int(dro_cfg["candidates"])
                )
            except (OSError, ValueError, TypeError):
                cache_matches = False

        if not cache_matches:
            command = [
                str(Path(dro_cfg["python"]).resolve()),
                str(Path(dro_cfg["inference_script"]).resolve()),
                "--dro-root", str(Path(dro_cfg["root"]).resolve()),
                "--checkpoint", str(Path(dro_cfg["checkpoint"]).resolve()),
                "--input", str(input_path),
                "--output", str(output_path),
                "--hand", self.hand_name,
                "--candidates", str(dro_cfg["candidates"]),
                "--points", str(dro_cfg["points"]),
                "--optimization-steps", str(dro_cfg["optimization_steps"]),
                "--seed", str(inference_seed),
                "--device", str(dro_cfg.get("device", "cpu")),
            ]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode != 0:
                raise RuntimeError(
                    "D(R,O) inference failed:\n"
                    f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
                )
        return json.loads(output_path.read_text())

    def generate_dro_candidates(self):
        clouds = self.get_camera_data(
            tensor_ptd=True,
            ptd_in_robot_base=True,
            segmented_ptd=True,
        )["camera_pointcloud_seg"]
        goal_pc = clouds[0]["goal"].cpu().numpy()
        scene_pc = clouds[0]["scene"].cpu().numpy()
        if len(goal_pc) < int(self.cfg["solution"]["dro"]["min_goal_points"]):
            raise RuntimeError(f"Only {len(goal_pc)} target points are visible")

        result = self._run_dro(goal_pc)
        scene_tree = cKDTree(scene_pc) if len(scene_pc) else None
        clearance = float(self.cfg["solution"]["dro"]["scene_clearance"])
        candidates = []
        for record in result["records"]:
            outer_pc = np.asarray(record["outer_pc"], dtype=np.float32)
            scene_distance = np.inf
            if scene_tree is not None:
                scene_distance = float(scene_tree.query(outer_pc, k=1)[0].min())
            predict_q = np.asarray(record["predict_q"], dtype=np.float32)
            candidate = {
                "index": int(record["candidate"]),
                "pose": self._q_root_pose(predict_q),
                "outer_q": self._map_hand_q(record["outer_q"], result["joint_names"]),
                "inner_q": self._map_hand_q(record["inner_q"], result["joint_names"]),
                "scene_distance": scene_distance,
                "pointcloud_clear": scene_distance >= clearance,
            }
            candidates.append(candidate)
        candidates.sort(key=lambda item: item["scene_distance"], reverse=True)
        return candidates, len(goal_pc)

    def _pregrasp_pose(self, grasp_pose):
        direction = np.asarray(self.cfg["solution"]["dro"]["pregrasp_direction"], dtype=float)
        direction /= np.linalg.norm(direction)
        return grasp_pose @ tr.translation_matrix(direction * self.cfg["solution"]["pre_grasp_offset"])

    def _approach_offset(self):
        direction = np.asarray(self.cfg["solution"]["dro"]["pregrasp_direction"], dtype=float)
        direction /= np.linalg.norm(direction)
        return -direction * self.cfg["solution"]["pre_grasp_offset"] \
            * self.cfg["solution"]["grasp_overshoot_ratio"]

    def _pad_arm_traj(self, traj):
        traj = to_torch(np.asarray(traj), device=self.device, dtype=torch.float)
        stabilization = traj[-1:].repeat(10, 1)
        return torch.cat([traj, stabilization], dim=0).unsqueeze(0)

    def plan_to_dro_pregrasp(self, candidates):
        q_start = self.states["q"][0, :self.arm_dof_count].cpu().numpy()
        attempts = []
        for candidate in candidates:
            if not candidate["pointcloud_clear"]:
                attempts.append({"candidate": candidate["index"], "reason": "pointcloud_collision"})
                continue
            self.motion_generator[0].set_hand_configuration(candidate["outer_q"])
            target = self._pregrasp_pose(candidate["pose"])
            success, traj, selected_pose = self.motion_generator[0].plan_goalset(
                q_start, np.expand_dims(target, axis=0), ret_goal=True
            )
            attempts.append({
                "candidate": candidate["index"],
                "scene_distance": candidate["scene_distance"],
                "planned": bool(success),
                "diagnostics": getattr(
                    self.motion_generator[0], "last_plan_diagnostics", {}
                ),
            })
            if success:
                return self._pad_arm_traj(traj), candidate, selected_pose, attempts
        return None, None, None, attempts

    def _follow_arm_traj(self, traj, hand_q):
        hand_target = to_torch(hand_q, device=self.device, dtype=torch.float).unsqueeze(0)
        for step in range(traj.shape[1]):
            command = {
                "joint_state": traj[:, step].clone(),
                "gripper_state": hand_target,
            }
            for _ in range(self.cfg["solution"]["num_step_repeat_per_plan_dt"]):
                self.pre_phy_step(command)
                self.env_physics_step()
                self.post_phy_step()
                rgb, _ = self.get_camera_image(rgb=True, seg=False)
                self.log_video(rgb)

    def _move_hand(self, target_q, steps):
        start_q = self.states["q"][:, self.arm_dof_count:].clone()
        target_q = to_torch(target_q, device=self.device, dtype=torch.float).unsqueeze(0)
        arm_q = self.states["q"][:, :self.arm_dof_count].clone()
        for alpha in torch.linspace(0.0, 1.0, steps, device=self.device):
            hand_q = start_q * (1.0 - alpha) + target_q * alpha
            self.pre_phy_step({"joint_state": arm_q, "gripper_state": hand_q})
            self.env_physics_step()
            self.post_phy_step()
            rgb, _ = self.get_camera_image(rgb=True, seg=False)
            self.log_video(rgb)

    def _plan_fetch(self, hand_q):
        self.motion_generator[0].set_hand_configuration(hand_q)
        q_start = self.states["q"][0, :self.arm_dof_count].cpu().numpy()
        target_pos = [[-0.2, -0.25, 0.66], [-0.2, 0.25, 0.66]]
        target_quat = [[0, 0.707, -0.707, 0], [0, 0.707, 0.707, 0]]
        targets = np.stack([
            tr.translation_matrix(pos) @ tr.quaternion_matrix(quat)
            for pos, quat in zip(target_pos, target_quat)
        ])
        success, traj, pose = self.motion_generator[0].plan_goalset(
            q_start, targets, ret_goal=True
        )
        return (self._pad_arm_traj(traj) if success else None), bool(success), pose

    def _capture_failure_hold(self, steps=30):
        for _ in range(steps):
            self.env_physics_step()
            self.post_phy_step()
            rgb, _ = self.get_camera_image(rgb=True, seg=False)
            self.log_video(rgb)

    def repeat(self):
        arm_q = self.states["q"][:, :self.arm_dof_count].clone()
        hand_q = to_torch(
            self.current_inner_q,
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)
        for _ in range(self.cfg["solution"]["eval_steps"]):
            self.pre_phy_step({"joint_state": arm_q, "gripper_state": hand_q})
            self.env_physics_step()
            self.post_phy_step()
            rgb, _ = self.get_camera_image(rgb=True, seg=False)
            self.log_video(rgb)

    def solve(self):
        log = {"hand": self.hand_name}
        self.set_target_color()
        self._solution_video = []
        self._video_frame = 0
        computing_time = 0.0

        for _ in range(self._init_steps):
            self.env_physics_step()
            self.post_phy_step()
        rgb, _ = self.get_camera_image(rgb=True, seg=False)
        self.log_video(rgb)

        try:
            start = time.time()
            candidates, visible_points = self.generate_dro_candidates()
            computing_time += time.time() - start
            log["visible_goal_points"] = visible_points
            log["dro_candidates"] = len(candidates)
            log["pointcloud_clear_candidates"] = sum(c["pointcloud_clear"] for c in candidates)

            self.update_pyompl_world_collider_pose(attach_goal_obj=False)
            start = time.time()
            traj, chosen, pose, attempts = self.plan_to_dro_pregrasp(candidates)
            computing_time += time.time() - start
            log["candidate_attempts"] = attempts
            log["pre_grasp_plan_success"] = [traj is not None]
            if traj is None:
                log["failure_stage"] = "pregrasp_planning"
                self._capture_failure_hold()
                return image_to_video(self._solution_video), log

            self.current_outer_q = chosen["outer_q"]
            self.current_inner_q = chosen["inner_q"]
            log["selected_candidate"] = chosen["index"]
            log["selected_scene_distance"] = chosen["scene_distance"]

            self._move_hand(self.current_outer_q, self.cfg["solution"]["gripper_steps"] // 2)
            self._follow_arm_traj(traj, self.current_outer_q)
            log["pre_grasp_execute_error"] = self.get_end_effect_error([pose])

            self.follow_cartesian_linear_motion(self._approach_offset(), gripper_state=0)
            self._move_hand(self.current_inner_q, self.cfg["solution"]["gripper_steps"])
            log["grasp_finger_obj_contact"] = self.finger_goal_obj_contact()

            if self.cfg["solution"]["retract_offset"] > 0:
                self.follow_cartesian_linear_motion(
                    np.array([0, 0, self.cfg["solution"]["retract_offset"]]),
                    gripper_state=0,
                    eef_frame=False,
                )
                log["retract_finger_obj_contact"] = self.finger_goal_obj_contact()

            self.update_pyompl_world_collider_pose(
                attach_goal_obj=self.cfg["solution"]["attach_goal_obj"]
            )
            start = time.time()
            fetch_traj, fetch_success, fetch_pose = self._plan_fetch(self.current_inner_q)
            computing_time += time.time() - start
            log["fetch_plan_success"] = [fetch_success]
            if fetch_success:
                self._follow_arm_traj(fetch_traj, self.current_inner_q)
                log["fetch_execute_error"] = self.get_end_effect_error([fetch_pose])
            else:
                log["failure_stage"] = "fetch_planning"

            self.repeat()
            log["end_finger_obj_contact"] = self.finger_goal_obj_contact()
        except Exception as exc:
            log["failure_stage"] = "exception"
            log["exception"] = f"{type(exc).__name__}: {exc}"
            print(log["exception"])
            self._capture_failure_hold()
        finally:
            log["traj_length"] = self._traj_length.cpu().numpy()
            log["computing_time"] = [computing_time]
            self.set_default_color()

        return image_to_video(self._solution_video), log
