"""Render standalone D(R,O) Barrett/ShadowHand grasps in FetchBench scenes."""

from __future__ import annotations

import json
import hashlib
import subprocess
import time
from pathlib import Path

import imageio.v3 as iio
import imageio.v2 as iio_v2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from isaacgym import gymapi

from isaacgymenvs.tasks.fetch.fetch_base import image_to_video
from isaacgymenvs.tasks.fetch.fetch_ptd import FetchPointCloudBase
from isaacgymenvs.utils.torch_jit_utils import quat_apply, to_torch


VIRTUAL_JOINT_NAMES = [
    "virtual_joint_x",
    "virtual_joint_y",
    "virtual_joint_z",
    "virtual_joint_roll",
    "virtual_joint_pitch",
    "virtual_joint_yaw",
]


class FetchPtdDRORender(FetchPointCloudBase):
    """Point cloud -> D(R,O) -> static hand placement, with no arm or fetch."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)
        if self.hand_name not in ("barrett", "shadowhand"):
            raise ValueError(f"Unsupported D(R,O) render hand: {self.hand_name}")
        if self.arm_dof_count != 6:
            raise ValueError("Standalone D(R,O) hands require six virtual root DOFs")

        self.isaac_dof_names = list(
            self.gym.get_actor_dof_names(self.envs[0], self.robots[0])
        )
        if self.isaac_dof_names[:6] != VIRTUAL_JOINT_NAMES:
            raise ValueError(
                "Standalone hand URDF has an unexpected virtual-joint order: "
                f"{self.isaac_dof_names[:6]}"
            )
        self._style_hand()
        self.last_render_log = None
        self._visualization_target_origin = None

    def _style_hand(self) -> None:
        """Give the standalone hand a visible, deterministic render color."""
        color = list(self.cfg["solution"].get("hand_color", [0.85, 0.55, 0.65]))
        if len(color) != 3:
            raise ValueError(f"hand_color must contain three RGB values, got {color}")
        visual_color = gymapi.Vec3(*[float(channel) for channel in color])
        for env, actor in zip(self.envs, self.robots):
            body_count = self.gym.get_actor_rigid_body_count(env, actor)
            for body_index in range(body_count):
                self.gym.set_rigid_body_color(
                    env, actor, body_index, gymapi.MESH_VISUAL, visual_color
                )

    def _aim_visualization_camera(self) -> None:
        """Aim only the non-point-cloud camera at the selected target."""
        if bool(self.cfg["solution"].get("physics_only", False)):
            return
        if not bool(self.cfg["solution"].get("target_closeup_camera", True)):
            return
        offset = list(
            self.cfg["solution"].get("target_closeup_offset", [-0.55, 0.0, 0.18])
        )
        if len(offset) != 3:
            raise ValueError(
                f"target_closeup_offset must contain three values, got {offset}"
            )
        target_index = int(self.task_obj_index[0][self.get_task_idx()].cpu())
        target = self.states["obj_pos"][0][target_index].cpu().numpy()
        max_tracking = self.cfg["solution"].get(
            "target_closeup_max_tracking_displacement", None
        )
        if self._visualization_target_origin is not None and max_tracking is not None:
            maximum = float(max_tracking)
            if maximum < 0.0:
                raise ValueError(
                    "target_closeup_max_tracking_displacement must be non-negative"
                )
            origin = np.asarray(self._visualization_target_origin, dtype=np.float32)
            displacement = target - origin
            distance = float(np.linalg.norm(displacement))
            if distance > maximum and distance > 1.0e-12:
                target = origin + displacement * (maximum / distance)
        camera = target + np.asarray(offset, dtype=np.float32)
        self.gym.set_camera_location(
            self.cameras[0][-1],
            self.envs[0],
            gymapi.Vec3(*camera.tolist()),
            gymapi.Vec3(*target.tolist()),
        )

    def _artifact_dir(self) -> Path:
        root = Path(self.cfg["solution"]["artifact_dir"]).resolve()
        scene_name = Path(self.scene_config_path[0]).name
        return root / scene_name / f"task_{self.get_task_idx():03d}" / self.hand_name

    def _goal_pointcloud(self, visible_goal_pc: np.ndarray) -> np.ndarray:
        """Choose the D(R,O) target cloud, always expressed in robot-base frame."""
        override_value = self.cfg["solution"].get("goal_pointcloud_override", None)
        if not override_value:
            goal_pc = np.asarray(visible_goal_pc, dtype=np.float32).reshape(-1, 3)
            source = "segmented_camera_partial"
            source_path = None
        else:
            source_path = Path(str(override_value)).resolve()
            goal_pc = np.asarray(np.load(source_path), dtype=np.float32).reshape(-1, 3)
            source = str(
                self.cfg["solution"].get(
                    "goal_pointcloud_override_source", "override_pointcloud"
                )
            )

        goal_pc = np.ascontiguousarray(goal_pc[np.isfinite(goal_pc).all(axis=1)])
        min_points = int(self.cfg["solution"]["dro"]["min_goal_points"])
        if len(goal_pc) < min_points:
            raise RuntimeError(
                f"D(R,O) target cloud has only {len(goal_pc)} valid points; "
                f"at least {min_points} are required"
            )
        self._goal_pointcloud_info = {
            "source": source,
            "source_path": None if source_path is None else str(source_path),
            "camera_visible_points": int(len(visible_goal_pc)),
            "input_points": int(len(goal_pc)),
            "sha256": hashlib.sha256(goal_pc.tobytes()).hexdigest(),
            "coordinate_frame": "robot_base",
        }
        return goal_pc

    def _run_dro(self, goal_pc: np.ndarray) -> dict:
        dro_cfg = self.cfg["solution"]["dro"]
        artifact_dir = self._artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        input_path = artifact_dir / "goal_pointcloud_robot_base.npy"
        output_path = artifact_dir / "dro_candidates.json"
        np.save(input_path, goal_pc.astype(np.float32))

        configured_seed = dro_cfg.get("seed")
        inference_seed = (
            int(configured_seed)
            if configured_seed is not None
            else int(self.cfg.get("seed", 42)) + self.get_task_idx()
        )
        expected = {
            "hand": self.hand_name,
            "sampled_points": int(dro_cfg["points"]),
            "optimization_steps": int(dro_cfg["optimization_steps"]),
            "seed": inference_seed,
            "barrett_mimic_enforced": self.hand_name == "barrett",
            "joint_limits_clamped": True,
            "joint_limit_margin": 1.0e-5,
            "pose_pointclouds_serialized": True,
            "checkpoint": str(Path(dro_cfg["checkpoint"]).resolve()),
            "input_sha256": hashlib.sha256(
                np.ascontiguousarray(goal_pc, dtype=np.float32).tobytes()
            ).hexdigest(),
        }
        cache_matches = False
        if output_path.exists() and bool(dro_cfg.get("reuse_cache", True)):
            try:
                cached = json.loads(output_path.read_text())
                cache_matches = all(cached.get(key) == value for key, value in expected.items())
                cache_matches &= len(cached.get("records", [])) == int(dro_cfg["candidates"])
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
            (artifact_dir / "dro_stdout.txt").write_text(process.stdout)
            (artifact_dir / "dro_stderr.txt").write_text(process.stderr)
            if process.returncode != 0:
                raise RuntimeError(
                    "D(R,O) inference failed:\n"
                    f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
                )
        return json.loads(output_path.read_text())

    def _full_q(self, result: dict, record: dict, pose_name: str) -> np.ndarray:
        q = np.asarray(record[pose_name], dtype=np.float32)
        source_names = VIRTUAL_JOINT_NAMES + list(result["joint_names"])
        if len(source_names) != len(q):
            raise ValueError(
                f"D(R,O) returned {len(q)} q values for {len(source_names)} joints"
            )
        by_name = dict(zip(source_names, q))
        missing = [name for name in self.isaac_dof_names if name not in by_name]
        if missing:
            raise ValueError(f"D(R,O) output is missing Isaac joints: {missing}")
        return np.asarray([by_name[name] for name in self.isaac_dof_names], dtype=np.float32)

    def _full_named_q(self, joint_names: list[str], values: list[float]) -> np.ndarray:
        """Map a prepared ContactDiffusion pose to Isaac by joint name."""
        q = np.asarray(values, dtype=np.float32)
        if len(joint_names) != len(q):
            raise ValueError(
                f"Prepared pose has {len(q)} values for {len(joint_names)} joints"
            )
        by_name = dict(zip(joint_names, q))
        missing = [name for name in self.isaac_dof_names if name not in by_name]
        if missing:
            raise ValueError(
                f"Prepared ContactDiffusion pose is missing Isaac joints: {missing}"
            )
        return np.asarray(
            [by_name[name] for name in self.isaac_dof_names], dtype=np.float32
        )

    def _load_external_lift_candidates(self, prepared_path: Path) -> tuple[dict, list]:
        """Load and rank prepared ContactDiffusion candidates for physics trials.

        The preparation manifest intentionally contains only the compact FK
        metrics. Environment metrics are joined back from its immutable source
        candidate file so the point-cloud clearance threshold is a hard gate
        before PhysX is allowed to move the hand.
        """
        payload = json.loads(prepared_path.read_text())
        expected_hands = {
            "barrett": {"barrett", "gendex_barrett"},
            "shadowhand": {"shadowhand", "shadow_hand"},
        }
        source_hand = str(payload.get("hand", "")).lower()
        if source_hand not in expected_hands[self.hand_name]:
            raise ValueError(
                f"Prepared hand {source_hand!r} does not match {self.hand_name!r}"
            )

        object_name = str(self.cfg["solution"].get("external_object_name", ""))
        groups = list(payload.get("objects", []))
        if object_name:
            groups = [group for group in groups if group["object_name"] == object_name]
        if len(groups) != 1:
            raise ValueError(f"Expected exactly one prepared object, found {len(groups)}")
        samples = list(groups[0].get("samples", []))
        if not samples:
            raise RuntimeError("Prepared ContactDiffusion manifest has no samples")

        source_path = Path(str(payload["candidate_file"]))
        if not source_path.is_absolute():
            source_path = (prepared_path.parent / source_path).resolve()
        source = json.loads(source_path.read_text())
        source_metrics = {}
        for source_index, record in enumerate(source.get("records", [])):
            logical_index = int(record.get("sample_index", source_index))
            for candidate in record.get("fk", {}).get("candidates", []):
                key = (
                    logical_index,
                    int(candidate["rank"]),
                    int(candidate["particle"]),
                )
                source_metrics[key] = candidate

        maximum_violation = float(
            self.cfg["solution"].get("external_max_environment_violation", 0.0001)
        )
        require_precomputed_environment = bool(
            self.cfg["solution"].get(
                "external_require_precomputed_environment", True
            )
        )
        scored = []
        for manifest_index, sample in enumerate(samples):
            key = (
                int(sample["source_index"]),
                int(sample["candidate_rank"]),
                int(sample["particle_index"]),
            )
            metrics = dict(sample.get("fk_metrics", {}))
            metrics.update(source_metrics.get(key, {}))
            environment_metric_available = bool(
                "environment_max_violation_m" in metrics
                and np.isfinite(float(metrics["environment_max_violation_m"]))
            )
            environment_violation = (
                float(metrics["environment_max_violation_m"])
                if environment_metric_available
                else float("inf")
            )
            precomputed_environment_pass = bool(
                environment_metric_available
                and environment_violation <= maximum_violation
            )
            # ``external_require_precomputed_environment=false`` means that
            # the point-cloud metric is diagnostic/ranking-only.  Runtime
            # PhysX contacts remain a hard success gate below.  Previously an
            # available-but-failing metric was still rejected here, making the
            # option ineffective exactly when a metric was present.
            environment_pass = bool(
                precomputed_environment_pass
                if require_precomputed_environment
                else True
            )
            contact_error = float(metrics.get("assigned_contact_error_m", float("inf")))
            penetration = float(metrics.get("max_penetration_m", float("inf")))
            self_collision = float(metrics.get("max_self_collision_m", 0.0))
            scored.append({
                "index": manifest_index,
                "scene_clearance": max(0.0, 0.005 - environment_violation),
                "scene_collision_ratio": float(
                    metrics.get("environment_collision_fraction", 0.0)
                ),
                # Keep candidate ordering stable when toggling the hard gate:
                # the precomputed proxy still ranks candidates, while
                # ``environment_constraint_pass`` alone controls whether it
                # may veto success.
                "collision_free": (
                    precomputed_environment_pass
                    if environment_metric_available
                    else environment_pass
                ),
                "environment_constraint_pass": environment_pass,
                "precomputed_environment_constraint_pass": (
                    precomputed_environment_pass
                ),
                "environment_metric_available": environment_metric_available,
                "environment_gate_source": (
                    "precomputed_pointcloud_hard_gate"
                    if environment_metric_available and require_precomputed_environment
                    else (
                        "missing_rejected"
                        if require_precomputed_environment
                        else (
                            "runtime_physx_with_precomputed_diagnostic"
                            if environment_metric_available
                            else "runtime_physx_only"
                        )
                    )
                ),
                "environment_max_violation_m": environment_violation,
                "target_coverage": max(0.0, 1.0 - contact_error / 0.02),
                "target_mean_distance": contact_error,
                "object_max_penetration_m": penetration,
                "self_collision_m": self_collision,
                "record": sample,
                "fk_metrics": metrics,
            })

        scored.sort(key=lambda item: (
            not item["environment_constraint_pass"],
            item["target_mean_distance"] + item["object_max_penetration_m"],
            item["object_max_penetration_m"],
            item["index"],
        ))
        result = {
            "method": "contactdiffusion_external_prepared",
            "joint_names": list(payload["joint_names"]),
            "records": samples,
            "prepared_path": str(prepared_path),
            "candidate_file": str(source_path),
            "protocol_id": payload.get("protocol_id"),
            "environment_max_violation_m": maximum_violation,
            "require_precomputed_environment": require_precomputed_environment,
        }
        return result, scored

    def _environment_contact_pairs(self, target_index: int) -> list[list[str]]:
        """Return hand contacts with furniture, ground, or non-target objects."""
        target_name = f"obj_{target_index}"
        environment = {"ground", "scene", "table"}
        environment.update(
            f"obj_{index}" for index in range(self.num_objs) if index != target_index
        )
        non_robot = set(environment)
        non_robot.add(target_name)
        pairs = []
        for body_a, body_b in self.get_robot_contacts()[0]:
            robot_a = body_a not in non_robot
            robot_b = body_b not in non_robot
            if (robot_a and body_b in environment) or (robot_b and body_a in environment):
                pair = [str(body_a), str(body_b)]
                if pair not in pairs and pair[::-1] not in pairs:
                    pairs.append(pair)
        return pairs

    def _render_external_prepared(
        self,
        prepared_path: Path,
        goal_pc: np.ndarray,
        scene_pc: np.ndarray,
        started: float,
    ):
        """Render one or every prepared ContactDiffusion pose in the live scene."""
        payload = json.loads(prepared_path.read_text())
        expected_hands = {
            "barrett": {"barrett", "gendex_barrett"},
            "shadowhand": {"shadowhand", "shadow_hand"},
        }
        source_hand = str(payload.get("hand", "")).lower()
        if source_hand not in expected_hands[self.hand_name]:
            raise ValueError(
                f"Prepared hand {source_hand!r} does not match {self.hand_name!r}"
            )

        object_name = str(
            self.cfg["solution"].get("external_object_name", "")
        )
        groups = list(payload.get("objects", []))
        if object_name:
            groups = [group for group in groups if group["object_name"] == object_name]
        if len(groups) != 1:
            raise ValueError(
                f"Expected exactly one prepared object, found {len(groups)}"
            )
        samples = list(groups[0].get("samples", []))
        sample_index = int(self.cfg["solution"].get("external_sample_index", 0))
        render_all = sample_index == -1
        if sample_index < -1 or sample_index >= len(samples):
            raise IndexError(
                f"external_sample_index={sample_index} outside [0, {len(samples) - 1}]"
            )
        indexed_samples = list(enumerate(samples)) if render_all else [
            (sample_index, samples[sample_index])
        ]
        pose_name = str(
            self.cfg["solution"].get("external_pose", "q_contact_euler")
        )
        if pose_name not in ("q_contact_euler", "outer_q_euler", "inner_q_euler"):
            raise ValueError(f"Unknown external_pose: {pose_name}")

        self._aim_visualization_camera()
        artifact_dir = self._artifact_dir()
        render_steps = max(1, int(self.cfg["solution"]["render_frames"]))

        pointcloud_audit = {"source_available": False}
        source_pc_value = self.cfg["solution"].get("external_pointcloud", None)
        if source_pc_value:
            source_pc_path = Path(str(source_pc_value)).resolve()
            source_pc = np.asarray(np.load(source_pc_path), dtype=np.float32)
            if source_pc.ndim == 3 and source_pc.shape[0] == 1:
                source_pc = source_pc[0]
            source_to_live = cKDTree(goal_pc).query(source_pc, k=1)[0]
            live_to_source = cKDTree(source_pc).query(goal_pc, k=1)[0]
            pointcloud_audit = {
                "source_available": True,
                "source_path": str(source_pc_path),
                "source_points": int(len(source_pc)),
                "live_points": int(len(goal_pc)),
                "exact_array_match": bool(
                    source_pc.shape == goal_pc.shape and np.array_equal(source_pc, goal_pc)
                ),
                "symmetric_mean_nearest_distance_m": float(
                    0.5 * (source_to_live.mean() + live_to_source.mean())
                ),
                "symmetric_max_nearest_distance_m": float(
                    max(source_to_live.max(), live_to_source.max())
                ),
            }

        common_log = {
            "scene": Path(self.scene_config_path[0]).name,
            "task_index": int(self.get_task_idx()),
            "task_label": self.get_task_label()[0],
            "target_object_index": int(
                self.task_obj_index[0][self.get_task_idx()].cpu()
            ),
            "hand": self.hand_name,
            "method": (
                "contactdiffusion_v4_smoke_all_candidates"
                if render_all
                else "contactdiffusion_v4_smoke_eawq_top1"
            ),
            "render_pose": pose_name,
            "prepared_file": str(prepared_path),
            "protocol_id": payload.get("protocol_id"),
            "base_protocol_id": payload.get("base_protocol_id"),
            "checkpoint_step": payload.get("checkpoint_step"),
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "visible_goal_points": int(len(goal_pc)),
            "scene_points": int(len(scene_pc)),
            "pointcloud_audit": pointcloud_audit,
            "simulation_hand_urdf": payload.get("simulation_hand_urdf"),
            "simulation_hand_urdf_sha256": payload.get("simulation_hand_urdf_sha256"),
            "isaac_dof_names": self.isaac_dof_names,
            "artifact_dir": str(artifact_dir),
        }

        all_frames = []
        closeups = []
        rendered = []
        for output_index, (manifest_index, sample) in enumerate(indexed_samples):
            q = self._full_named_q(list(payload["joint_names"]), sample[pose_name])
            self._place_hand(q)
            joint_limit_audit = self._joint_limit_audit(q)
            frames = []
            for _ in range(render_steps):
                frames.append(
                    self.get_camera_data(numpy_rgb=True)["camera_render_raw"]["rgb"][0]
                )
            all_frames.extend(frames)
            closeups.append(frames[-1][-1])

            pose_dir = artifact_dir / (
                f"pose_{output_index:03d}_set_{int(sample['source_index']):03d}"
                f"_rank_{int(sample['candidate_rank']):02d}"
            )
            self._save_render(frames[-1], pose_dir)
            if output_index == 0:
                self._save_render(frames[-1], artifact_dir)
            pose_log = {
                "output_index": output_index,
                "manifest_index": manifest_index,
                "source_index": int(sample["source_index"]),
                "candidate_rank": int(sample["candidate_rank"]),
                "source_candidate_rank": int(
                    sample.get("source_candidate_rank", sample["candidate_rank"])
                ),
                "particle_index": int(sample["particle_index"]),
                "selection_feasible": bool(sample.get("selection_feasible", False)),
                "selection_fallback": bool(sample.get("selection_fallback", False)),
                "eawq_rank_fusion": sample.get("eawq_rank_fusion"),
                "best_energy": sample.get("best_energy"),
                "fk_metrics": sample.get("fk_metrics"),
                "render_q": q.tolist(),
                "joint_limit_audit": joint_limit_audit,
                "artifact_dir": str(pose_dir),
            }
            (pose_dir / "grasp.json").write_text(json.dumps(pose_log, indent=2))
            rendered.append(pose_log)

        if closeups:
            columns = min(4, len(closeups))
            rows = []
            for start in range(0, len(closeups), columns):
                row = closeups[start:start + columns]
                if len(row) < columns:
                    row.extend([np.zeros_like(closeups[0])] * (columns - len(row)))
                rows.append(np.concatenate(row, axis=1))
            iio.imwrite(artifact_dir / "all_candidate_closeups.png", np.concatenate(rows, axis=0))

        first = rendered[0]
        log = {
            **common_log,
            "rendered_sample_count": len(rendered),
            "feasible_sample_count": sum(
                int(item["selection_feasible"]) for item in rendered
            ),
            "source_candidate_rank": first["source_candidate_rank"],
            "particle_index": first["particle_index"],
            "selection_feasible": first["selection_feasible"],
            "selection_fallback": first["selection_fallback"],
            "fk_metrics": first["fk_metrics"],
            "render_q": first["render_q"],
            "joint_limit_audit": first["joint_limit_audit"],
            "rendered_samples": rendered,
            "elapsed_seconds": time.time() - started,
        }
        (artifact_dir / "selected_grasp.json").write_text(
            json.dumps(log, indent=2)
        )
        self.last_render_log = log
        if bool(self.cfg["solution"].get("highlight_target", True)):
            self.set_default_color()

        video = []
        for rgb_cameras in all_frames:
            video.extend(image_to_video([rgb_cameras]))
        return video, log

    def _select_candidate(
        self,
        result: dict,
        scene_pc: np.ndarray,
        goal_pc: np.ndarray,
        pose_name: str,
    ) -> tuple[dict, list]:
        tree = cKDTree(scene_pc) if len(scene_pc) else None
        contact_distance = float(
            self.cfg["solution"].get("target_contact_distance", 0.02)
        )
        min_scene_clearance = float(
            self.cfg["solution"].get("min_scene_clearance", 0.005)
        )
        pose_pc_name = pose_name.replace("_q", "_pc")
        scored = []
        for record in result["records"]:
            outer_pc = np.asarray(record["outer_pc"], dtype=np.float32)
            pose_pc = np.asarray(record[pose_pc_name], dtype=np.float32)
            scene_distances = None if tree is None else tree.query(outer_pc, k=1)[0]
            clearance = np.inf if scene_distances is None else float(scene_distances.min())
            collision_ratio = 0.0 if scene_distances is None else float(
                np.mean(scene_distances < min_scene_clearance)
            )
            target_distances = cKDTree(pose_pc).query(goal_pc, k=1)[0]
            target_coverage = float(np.mean(target_distances < contact_distance))
            target_mean_distance = float(target_distances.mean())
            scored.append({
                "index": int(record["candidate"]),
                "scene_clearance": clearance,
                "scene_collision_ratio": collision_ratio,
                "collision_free": clearance >= min_scene_clearance,
                "target_coverage": target_coverage,
                "target_mean_distance": target_mean_distance,
                "record": record,
            })

        requested = int(self.cfg["solution"].get("candidate_index", -1))
        if requested >= 0:
            matches = [item for item in scored if item["index"] == requested]
            if not matches:
                raise IndexError(f"D(R,O) candidate {requested} was not generated")
            selected = matches[0]
        else:
            # First reject candidates whose open preshape intersects the
            # non-target scene. Among the rest, prefer the hand that covers the
            # largest fraction of target points at the rendered pose.
            selected = max(
                scored,
                key=lambda item: (
                    item["collision_free"],
                    item["target_coverage"],
                    -item["target_mean_distance"],
                    item["scene_clearance"],
                ),
            )
        return selected, scored

    def _place_hand(self, q: np.ndarray) -> None:
        q_tensor = to_torch(q, device=self.device, dtype=self._q.dtype).unsqueeze(0)
        self.teleport_joint_state(q_tensor)
        # One collision-free visual-only step updates FK and the renderer.
        self.env_physics_step()
        self.post_phy_step()

    @staticmethod
    def _radial_pregrasp_q(
        outer_q: np.ndarray,
        object_center: np.ndarray,
        offset_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retreat the open hand from the object without changing orientation."""
        outer_q = np.asarray(outer_q, dtype=np.float32)
        object_center = np.asarray(object_center, dtype=np.float32).reshape(3)
        outward = outer_q[:3].astype(np.float64) - object_center.astype(np.float64)
        norm = float(np.linalg.norm(outward))
        if not np.isfinite(norm) or norm <= 1.0e-9:
            raise ValueError("Cannot define palm-outward pregrasp at object center")
        outward /= norm
        pregrasp_q = outer_q.copy()
        pregrasp_q[:3] += float(offset_m) * outward.astype(np.float32)
        return pregrasp_q, outward.astype(np.float32)

    def _fully_open_q(
        self,
        outer_q: np.ndarray,
        open_joint_positions: dict,
    ) -> np.ndarray:
        """Open flexion DOFs while preserving root, wrist, and abduction."""
        opened_q = np.asarray(outer_q, dtype=np.float32).copy()
        indices = {name: index for index, name in enumerate(self.isaac_dof_names)}
        unknown = sorted(set(open_joint_positions) - set(indices))
        if unknown:
            raise ValueError(f"Open posture contains unknown Isaac joints: {unknown}")
        for name, value in open_joint_positions.items():
            opened_q[indices[name]] = float(value)
        return opened_q

    def _fk_audit(self, record: dict, pose_name: str) -> dict:
        """Compare Isaac rigid-body origins with D(R,O)'s URDF FK origins."""
        key = pose_name.replace("_q", "_link_positions")
        expected = record.get(key, {})
        if not expected:
            return {"available": False}

        names = list(self.gym.get_actor_rigid_body_names(self.envs[0], self.robots[0]))
        states = self.gym.get_actor_rigid_body_states(
            self.envs[0], self.robots[0], gymapi.STATE_POS
        )
        base = self._robot_base_state[0].detach().cpu().numpy()
        base_pos = base[:3]
        base_rotation_inv = Rotation.from_quat(base[3:7]).inv()

        errors = {}
        for body_index, name in enumerate(names):
            if name not in expected:
                continue
            world_pos = np.asarray([
                states["pose"]["p"][body_index][axis] for axis in ("x", "y", "z")
            ])
            local_pos = base_rotation_inv.apply(world_pos - base_pos)
            errors[name] = float(
                np.linalg.norm(local_pos - np.asarray(expected[name], dtype=np.float64))
            )
        worst = sorted(errors.items(), key=lambda item: item[1], reverse=True)
        return {
            "available": True,
            "common_links": len(errors),
            "max_position_error_m": worst[0][1] if worst else None,
            "worst_links": [
                {"link": name, "position_error_m": error}
                for name, error in worst[:8]
            ],
        }

    def _joint_limit_audit(self, requested_q: np.ndarray) -> dict:
        """Verify the values actually held by Isaac against Isaac's limits."""
        actual_q = self._q[0].detach().cpu().numpy()
        lower = self.robot_dof_lower_limits.detach().cpu().numpy()
        upper = self.robot_dof_upper_limits.detach().cpu().numpy()
        margins = np.minimum(actual_q - lower, upper - actual_q)
        closest = np.argsort(margins)[:8]
        return {
            "all_within_limits": bool(np.all(margins >= 0.0)),
            "minimum_limit_margin": float(margins.min()),
            "max_requested_actual_error": float(
                np.max(np.abs(actual_q - requested_q))
            ),
            "closest_joints": [
                {
                    "joint": self.isaac_dof_names[index],
                    "value": float(actual_q[index]),
                    "lower": float(lower[index]),
                    "upper": float(upper[index]),
                    "margin": float(margins[index]),
                }
                for index in closest
            ],
        }

    def _save_render(
        self, rgb_cameras: list[np.ndarray], output: Path | None = None
    ) -> None:
        output = self._artifact_dir() if output is None else output
        output.mkdir(parents=True, exist_ok=True)
        for camera_idx, image in enumerate(rgb_cameras):
            iio.imwrite(output / f"camera_{camera_idx}.png", image)
        if rgb_cameras:
            iio.imwrite(output / "camera_mosaic.png", np.concatenate(rgb_cameras, axis=0))

    def _command_q(self, q: np.ndarray) -> None:
        """Apply one position-control step to the virtual base and hand."""
        q_tensor = to_torch(q, device=self.device, dtype=self._q.dtype).unsqueeze(0)
        self.pre_phy_step({
            "joint_state": q_tensor[:, :self.arm_dof_count],
            "gripper_state": q_tensor[:, self.arm_dof_count:],
        })
        self.env_physics_step()
        self.post_phy_step()

    def _has_target_hand_contact(self, target_index: int) -> bool:
        """Return whether any hand rigid body touches the selected object."""
        target_name = f"obj_{target_index}"
        non_robot = {"ground", "scene", "table"}
        non_robot.update(f"obj_{index}" for index in range(self.num_objs))
        for body_a, body_b in self.get_robot_contacts()[0]:
            if target_name not in (body_a, body_b):
                continue
            other = body_b if body_a == target_name else body_a
            if other not in non_robot:
                return True
        return False

    def _capture_closeup(self) -> np.ndarray:
        """Capture the target-facing visualization camera."""
        # Keep the lifted object and hand in frame throughout the trajectory.
        self._aim_visualization_camera()
        return self.get_camera_data(
            numpy_rgb=True
        )["camera_render_raw"]["rgb"][0][-1]

    def _object_reference_position(self, object_index: int) -> np.ndarray:
        """Return the world-space FetchBench reference/COM point."""
        position = self.states["obj_pos"][0][object_index]
        orientation = self.states["obj_quat"][0][object_index]
        reference = self.obj_ref_point[0][object_index]
        return (position + quat_apply(orientation, reference)).cpu().numpy().copy()

    def validate_lifts(self):
        """Close D(R,O) preshapes, lift them, record videos, and score success."""
        if self.num_envs != 1:
            raise ValueError("D(R,O) lift validation supports one scene at a time")
        shape_count = self.gym.get_asset_rigid_shape_count(self.robot_asset)
        if shape_count <= 0:
            raise RuntimeError(
                "Lift validation requires a collision-enabled hand URDF; "
                "the loaded asset has no rigid shapes"
            )

        started = time.time()
        for _ in range(int(self.cfg["solution"]["settle_steps"])):
            self.env_physics_step()
            self.post_phy_step()
        physics_only = bool(self.cfg["solution"].get("physics_only", False))
        scene_override_value = self.cfg["solution"].get(
            "scene_pointcloud_override", None
        )
        if physics_only:
            # The experiment already serializes the target cloud in robot-base
            # coordinates.  Runtime environment rejection below uses PhysX
            # rigid contacts, so no rendered depth/segmentation is required.
            goal_pc = self._goal_pointcloud(np.empty((0, 3), dtype=np.float32))
            if scene_override_value:
                scene_path = Path(str(scene_override_value)).resolve()
                scene_pc = np.asarray(
                    np.load(scene_path), dtype=np.float32
                ).reshape(-1, 3)
                scene_pc = np.ascontiguousarray(
                    scene_pc[np.isfinite(scene_pc).all(axis=1)]
                )
            else:
                scene_pc = np.empty((0, 3), dtype=np.float32)
        else:
            camera_data = self.get_camera_data(
                numpy_rgb=True,
                tensor_ptd=True,
                ptd_in_robot_base=True,
                segmented_ptd=True,
                ptd_downscale=int(self.cfg["solution"]["pointcloud_downscale"]),
            )
            clouds = camera_data["camera_pointcloud_seg"][0]
            visible_goal_pc = clouds["goal"].cpu().numpy()
            goal_pc = self._goal_pointcloud(visible_goal_pc)
            scene_pc = clouds["scene"].cpu().numpy()
            if scene_override_value:
                scene_path = Path(str(scene_override_value)).resolve()
                scene_pc = np.asarray(
                    np.load(scene_path), dtype=np.float32
                ).reshape(-1, 3)
                scene_pc = np.ascontiguousarray(
                    scene_pc[np.isfinite(scene_pc).all(axis=1)]
                )
        pointcloud_dir = self._artifact_dir()
        pointcloud_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            pointcloud_dir / "goal_pointcloud_robot_base.npy",
            goal_pc.astype(np.float32),
        )
        np.save(
            pointcloud_dir / "scene_pointcloud_robot_base.npy",
            scene_pc.astype(np.float32),
        )
        external_value = self.cfg["solution"].get("external_prepared", None)
        external_mode = bool(external_value)
        if external_mode:
            result, scored = self._load_external_lift_candidates(
                Path(str(external_value)).resolve()
            )
            pose_name = "q_contact_euler"
        else:
            result = self._run_dro(goal_pc)
            pose_name = str(self.cfg["solution"].get("render_pose", "predict_q"))
            _, scored = self._select_candidate(result, scene_pc, goal_pc, pose_name)
        ranked = sorted(
            (item for item in scored if item["collision_free"]),
            key=lambda item: (
                item["target_mean_distance"]
                + float(item.get("object_max_penetration_m", 0.0)),
                float(item.get("object_max_penetration_m", 0.0)),
                -item["scene_clearance"],
            ),
        )
        count = max(1, int(self.cfg["solution"].get("visualize_top_k", 5)))
        if len(ranked) < count:
            if not bool(
                self.cfg["solution"].get("external_validate_infeasible", False)
            ):
                raise RuntimeError(
                    f"Lift validation requested {count} candidates, but only "
                    f"{len(ranked)} clear the non-target scene"
                )
            selected = {int(item["index"]) for item in ranked}
            rejected = sorted(
                (item for item in scored if int(item["index"]) not in selected),
                key=lambda item: (
                    float(item.get("environment_max_violation_m", float("inf"))),
                    item["target_mean_distance"]
                    + float(item.get("object_max_penetration_m", 0.0)),
                    item["index"],
                ),
            )
            ranked.extend(rejected[: count - len(ranked)])
        ranked = ranked[:count]

        lift_cfg = self.cfg["solution"].get("lift", {})
        direct_closure = bool(lift_cfg.get("direct_closure", False))
        pregrasp_offset = float(lift_cfg.get("pregrasp_offset", 0.08))
        pregrasp_max_offset = float(
            lift_cfg.get("pregrasp_max_offset", pregrasp_offset)
        )
        pregrasp_offset_increment = float(
            lift_cfg.get("pregrasp_offset_increment", 0.04)
        )
        pregrasp_hold_steps = int(lift_cfg.get("pregrasp_hold_steps", 12))
        approach_steps = int(lift_cfg.get("approach_steps", 45))
        open_to_outer_steps = int(lift_cfg.get("open_to_outer_steps", 30))
        max_preclosure_displacement = float(
            lift_cfg.get("max_preclosure_object_displacement", 0.02)
        )
        open_joint_positions_by_hand = lift_cfg.get("open_joint_positions", {})
        open_joint_positions = dict(
            open_joint_positions_by_hand.get(self.hand_name, {})
        )
        invalid_legacy_pregrasp = (
            not direct_closure
            and (
                pregrasp_offset < 0.0
                or pregrasp_max_offset < pregrasp_offset
                or pregrasp_offset_increment <= 0.0
                or pregrasp_hold_steps < 0
                or approach_steps <= 0
                or open_to_outer_steps <= 0
                or not open_joint_positions
            )
        )
        if invalid_legacy_pregrasp or max_preclosure_displacement < 0.0:
            raise ValueError(
                "Invalid lift pregrasp or pre-closure displacement configuration"
            )
        close_steps = int(lift_cfg.get("close_steps", 45))
        close_hold_steps = int(lift_cfg.get("close_hold_steps", 20))
        lift_steps = int(lift_cfg.get("lift_steps", 60))
        final_hold_steps = int(lift_cfg.get("final_hold_steps", 30))
        outer_hold_steps = int(lift_cfg.get("outer_hold_steps", 12))
        lift_height = float(lift_cfg.get("height", 0.25))
        success_height = float(lift_cfg.get("success_height", 0.10))
        capture_stride = max(1, int(lift_cfg.get("capture_stride", 2)))
        fps = int(lift_cfg.get("video_fps", 30))
        record_video = bool(lift_cfg.get("record_video", True)) and not physics_only

        target_index = int(self.task_obj_index[0][self.get_task_idx()].cpu())
        object_center = np.asarray(goal_pc, dtype=np.float64).mean(axis=0)
        reject_environment_contacts = bool(
            self.cfg["solution"].get("reject_runtime_environment_contacts", True)
        )
        output = self._artifact_dir() / "lift_validation"
        output.mkdir(parents=True, exist_ok=True)
        trial_logs = []
        playlist_path = (
            output / f"{self.hand_name}_lift_playlist.mp4"
            if record_video else None
        )
        # Stream the playlist while recording.  Keeping every RGB frame in a
        # Python list is acceptable for five-candidate diagnostics but grows to
        # several GiB for the frozen v6 budget of 32 candidates per group.
        playlist_writer = (
            iio_v2.get_writer(playlist_path, fps=fps) if record_video else None
        )

        for rank, item in enumerate(ranked):
            self.reset_task(self.get_task_idx())
            self.set_target_color()
            self._visualization_target_origin = (
                self.states["obj_pos"][0][target_index].cpu().numpy().copy()
            )
            self._aim_visualization_camera()
            initial_object_pos = self._object_reference_position(target_index)

            if external_mode:
                outer_q = self._full_named_q(
                    result["joint_names"], item["record"]["outer_q_euler"]
                )
                inner_q = self._full_named_q(
                    result["joint_names"], item["record"]["inner_q_euler"]
                )
            else:
                outer_q = self._full_q(result, item["record"], "outer_q")
                inner_q = self._full_q(result, item["record"], "inner_q")
            if direct_closure:
                opened_q = outer_q.copy()
                pregrasp_q = outer_q.copy()
                palm_outward = np.zeros(3, dtype=np.float32)
                attempted_pregrasp_offsets = []
                used_pregrasp_offset = 0.0
                self._place_hand(outer_q)
                initial_target_contact = self._has_target_hand_contact(target_index)
            else:
                opened_q = self._fully_open_q(outer_q, open_joint_positions)
                attempted_pregrasp_offsets = []
                used_pregrasp_offset = pregrasp_offset
                while True:
                    pregrasp_q, palm_outward = self._radial_pregrasp_q(
                        opened_q, object_center, used_pregrasp_offset
                    )
                    attempted_pregrasp_offsets.append(float(used_pregrasp_offset))
                    self._place_hand(pregrasp_q)
                    initial_target_contact = self._has_target_hand_contact(target_index)
                    if not initial_target_contact:
                        break
                    next_offset = min(
                        pregrasp_max_offset,
                        used_pregrasp_offset + pregrasp_offset_increment,
                    )
                    if next_offset <= used_pregrasp_offset + 1.0e-9:
                        break
                    used_pregrasp_offset = next_offset
                    # The rejected placement may already have moved the object, so
                    # restore the task before testing the farther start pose.
                    self.reset_task(self.get_task_idx())
                    self.set_target_color()
                    self._aim_visualization_camera()
                    initial_object_pos = self._object_reference_position(target_index)

            frames = [self._capture_closeup()] if record_video else []
            contact_steps = 0
            contact_pregrasp_steps = 0
            contact_approach_steps = 0
            contact_open_to_outer_steps = 0
            contact_lift_steps = 0
            contact_final_hold_steps = 0
            environment_collision_steps = 0
            approach_environment_collision_steps = 0
            environment_collision_pairs = self._environment_contact_pairs(target_index)
            initial_environment_collision = bool(environment_collision_pairs)
            max_object_z = float(initial_object_pos[2])
            object_z_trace = [float(initial_object_pos[2])]

            def advance(
                target_q: np.ndarray,
                step_index: int,
                during_lift: bool,
                during_final_hold: bool = False,
                stage: str = "other",
            ) -> None:
                nonlocal contact_steps, contact_lift_steps
                nonlocal contact_pregrasp_steps, contact_approach_steps
                nonlocal contact_open_to_outer_steps
                nonlocal contact_final_hold_steps, max_object_z
                nonlocal environment_collision_steps, environment_collision_pairs
                nonlocal approach_environment_collision_steps
                self._command_q(target_q)
                touching = self._has_target_hand_contact(target_index)
                environment_pairs = self._environment_contact_pairs(target_index)
                environment_collision_steps += int(bool(environment_pairs))
                if stage == "approach":
                    approach_environment_collision_steps += int(bool(environment_pairs))
                for pair in environment_pairs:
                    if pair not in environment_collision_pairs:
                        environment_collision_pairs.append(pair)
                contact_steps += int(touching)
                if stage == "pregrasp":
                    contact_pregrasp_steps += int(touching)
                if stage == "approach":
                    contact_approach_steps += int(touching)
                if stage == "open_to_outer":
                    contact_open_to_outer_steps += int(touching)
                if during_lift:
                    contact_lift_steps += int(touching)
                if during_final_hold:
                    contact_final_hold_steps += int(touching)
                object_z = float(self._object_reference_position(target_index)[2])
                object_z_trace.append(object_z)
                max_object_z = max(max_object_z, object_z)
                if record_video and step_index % capture_stride == 0:
                    frames.append(self._capture_closeup())

            active_pregrasp_hold_steps = 0 if direct_closure else pregrasp_hold_steps
            active_approach_steps = 0 if direct_closure else approach_steps
            active_open_to_outer_steps = 0 if direct_closure else open_to_outer_steps
            for step in range(active_pregrasp_hold_steps):
                advance(pregrasp_q, step, False, stage="pregrasp")
            object_pos_before_approach = self._object_reference_position(target_index)
            for step in range(active_approach_steps):
                alpha = float(step + 1) / float(active_approach_steps)
                target_q = pregrasp_q + alpha * (opened_q - pregrasp_q)
                advance(target_q, step, False, stage="approach")
            for step in range(active_open_to_outer_steps):
                alpha = float(step + 1) / float(active_open_to_outer_steps)
                target_q = opened_q + alpha * (outer_q - opened_q)
                advance(target_q, step, False, stage="open_to_outer")
            for step in range(outer_hold_steps):
                advance(outer_q, step, False, stage="outer_hold")
            object_pos_before_closure = self._object_reference_position(target_index)
            preclosure_object_displacement = float(
                np.linalg.norm(object_pos_before_closure - initial_object_pos)
            )
            for step in range(close_steps):
                alpha = float(step + 1) / float(close_steps)
                target_q = outer_q + alpha * (inner_q - outer_q)
                advance(target_q, step, False, stage="closure")
            for step in range(close_hold_steps):
                advance(inner_q, step, False, stage="closure_hold")

            lifted_q = inner_q.copy()
            lifted_q[2] += lift_height
            for step in range(lift_steps):
                alpha = float(step + 1) / float(lift_steps)
                target_q = inner_q.copy()
                target_q[2] += alpha * lift_height
                advance(target_q, step, True)
            for step in range(final_hold_steps):
                advance(lifted_q, step, True, True)

            final_object_pos = self._object_reference_position(target_index)
            final_object_lift = float(final_object_pos[2] - initial_object_pos[2])
            max_object_lift = float(max_object_z - initial_object_pos[2])
            actual_base_lift = float(self._q[0, 2].cpu()) - float(outer_q[2])
            required_final_contacts = max(1, final_hold_steps // 2)
            success = bool(
                bool(item.get("environment_constraint_pass", item["collision_free"]))
                and (direct_closure or not initial_target_contact)
                and preclosure_object_displacement <= max_preclosure_displacement
                and final_object_lift >= success_height
                and contact_final_hold_steps >= required_final_contacts
                and (
                    not reject_environment_contacts
                    or (
                        not initial_environment_collision
                        and environment_collision_steps == 0
                    )
                )
            )
            video_path = None
            if record_video:
                frames.extend([frames[-1]] * max(1, fps // 2))
                video_path = output / f"candidate_{item['index']:03d}.mp4"
                iio.imwrite(video_path, np.stack(frames), fps=fps)
                for frame in frames:
                    playlist_writer.append_data(frame)
                separator = np.zeros_like(frames[-1])
                for _ in range(max(1, fps // 3)):
                    playlist_writer.append_data(separator)

            trial = {
                "rank": rank,
                "candidate": item["index"],
                "success": success,
                "initial_object_position": initial_object_pos.tolist(),
                "final_object_position": final_object_pos.tolist(),
                "final_object_lift_m": final_object_lift,
                "max_object_lift_m": max_object_lift,
                "commanded_hand_lift_m": lift_height,
                "actual_hand_lift_m": actual_base_lift,
                "pregrasp_offset_m": used_pregrasp_offset,
                "trajectory_mode": (
                    "direct_outer_to_inner_closure"
                    if direct_closure
                    else "radial_pregrasp_approach_then_closure"
                ),
                "attempted_pregrasp_offsets_m": attempted_pregrasp_offsets,
                "pregrasp_target_clear": bool(not initial_target_contact),
                "pregrasp_q": pregrasp_q.tolist(),
                "opened_q": opened_q.tolist(),
                "open_joint_positions": {
                    str(name): float(value)
                    for name, value in open_joint_positions.items()
                },
                "palm_outward_direction_world": palm_outward.tolist(),
                "palm_outward_direction_definition": (
                    "disabled_direct_closure"
                    if direct_closure
                    else "normalize(open_hand_root - complete_target_cloud_center)"
                ),
                "target_cloud_center_robot_base": object_center.tolist(),
                "initial_target_contact_at_pregrasp": bool(initial_target_contact),
                "contact_steps": contact_steps,
                "contact_pregrasp_steps": contact_pregrasp_steps,
                "contact_approach_steps": contact_approach_steps,
                "contact_open_to_outer_steps": contact_open_to_outer_steps,
                "contact_lift_steps": contact_lift_steps,
                "contact_final_hold_steps": contact_final_hold_steps,
                "required_final_contact_steps": required_final_contacts,
                "precomputed_environment_constraint_pass": bool(
                    item.get(
                        "precomputed_environment_constraint_pass",
                        item.get("environment_constraint_pass", item["collision_free"]),
                    )
                ),
                "precomputed_environment_metric_available": bool(
                    item.get("environment_metric_available", False)
                ),
                "environment_gate_source": item.get("environment_gate_source"),
                "precomputed_environment_max_violation_m": item.get(
                    "environment_max_violation_m"
                ),
                "initial_environment_collision": initial_environment_collision,
                "environment_collision_steps": environment_collision_steps,
                "approach_environment_collision_steps": (
                    approach_environment_collision_steps
                ),
                "environment_collision_pairs": environment_collision_pairs,
                "runtime_environment_collision_rejected": bool(
                    reject_environment_contacts
                ),
                "scene_clearance_m": item["scene_clearance"],
                "target_coverage": item["target_coverage"],
                "outer_q": outer_q.tolist(),
                "inner_q": inner_q.tolist(),
                "object_position_before_approach": (
                    object_pos_before_approach.tolist()
                ),
                "object_position_before_closure": object_pos_before_closure.tolist(),
                "object_displacement_before_closure_m": preclosure_object_displacement,
                "max_preclosure_object_displacement_m": (
                    max_preclosure_displacement
                ),
                "video": None if video_path is None else str(video_path),
                "object_z_trace": object_z_trace,
            }
            (output / f"candidate_{item['index']:03d}.json").write_text(
                json.dumps(trial, indent=2)
            )
            trial_logs.append(trial)

        if playlist_writer is not None:
            playlist_writer.close()
        successes = sum(int(trial["success"]) for trial in trial_logs)
        summary = {
            "scene": Path(self.scene_config_path[0]).name,
            "task_index": int(self.get_task_idx()),
            "hand": self.hand_name,
            "goal_pointcloud": self._goal_pointcloud_info,
            "method": result.get("method", "dro"),
            "prepared_file": result.get("prepared_path"),
            "candidate_file": result.get("candidate_file"),
            "protocol_id": result.get("protocol_id"),
            "dro_checkpoint": None if external_mode else str(
                Path(self.cfg["solution"]["dro"]["checkpoint"]).resolve()
            ),
            "candidate_pool": len(result["records"]),
            "environment_constrained_candidate_pool": sum(
                int(item["collision_free"]) for item in scored
            ),
            "diagnostic_infeasible_candidates": sum(
                int(not trial["precomputed_environment_constraint_pass"])
                for trial in trial_logs
            ),
            "validated_candidates": len(trial_logs),
            "successes": successes,
            "success_rate": float(successes / len(trial_logs)),
            "success_height_m": success_height,
            "lift_height_m": lift_height,
            "trajectory_mode": (
                "direct_outer_to_inner_closure"
                if direct_closure
                else "radial_pregrasp_approach_then_closure"
            ),
            "pregrasp_offset_m": 0.0 if direct_closure else pregrasp_offset,
            "pregrasp_max_offset_m": 0.0 if direct_closure else pregrasp_max_offset,
            "pregrasp_offset_increment_m": pregrasp_offset_increment,
            "pregrasp_hold_steps": active_pregrasp_hold_steps,
            "approach_steps": active_approach_steps,
            "open_to_outer_steps": active_open_to_outer_steps,
            "approach_hand_posture": (
                "disabled_direct_closure"
                if direct_closure
                else "fully_open_flexion_dofs"
            ),
            "max_preclosure_object_displacement_m": max_preclosure_displacement,
            "pregrasp_direction_definition": (
                "disabled_direct_closure"
                if direct_closure
                else "normalize(open_hand_root - complete_target_cloud_center)"
            ),
            "collision_shape_count": int(shape_count),
            "reject_runtime_environment_contacts": reject_environment_contacts,
            "precomputed_max_environment_violation_m": result.get(
                "environment_max_violation_m"
            ),
            "require_precomputed_environment": result.get(
                "require_precomputed_environment", True
            ),
            "elapsed_seconds": time.time() - started,
            "record_video": record_video,
            "physics_only": physics_only,
            "playlist": None if playlist_path is None else str(playlist_path),
            "trials": trial_logs,
            "artifact_dir": str(output),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        self.last_render_log = summary
        return summary

    def solve(self):
        if self.num_envs != 1:
            raise ValueError("Static D(R,O) rendering currently supports one scene at a time")

        started = time.time()
        if bool(self.cfg["solution"].get("highlight_target", True)):
            self.set_target_color()

        for _ in range(int(self.cfg["solution"]["settle_steps"])):
            self.env_physics_step()
            self.post_phy_step()

        camera_data = self.get_camera_data(
            numpy_rgb=True,
            tensor_ptd=True,
            ptd_in_robot_base=True,
            segmented_ptd=True,
            ptd_downscale=int(self.cfg["solution"]["pointcloud_downscale"]),
        )
        clouds = camera_data["camera_pointcloud_seg"][0]
        visible_goal_pc = clouds["goal"].cpu().numpy()
        goal_pc = self._goal_pointcloud(visible_goal_pc)
        scene_pc = clouds["scene"].cpu().numpy()

        artifact_dir = self._artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(artifact_dir / "goal_pointcloud_robot_base.npy", goal_pc.astype(np.float32))
        np.save(artifact_dir / "scene_pointcloud_robot_base.npy", scene_pc.astype(np.float32))

        external_prepared = self.cfg["solution"].get("external_prepared", None)
        if external_prepared:
            return self._render_external_prepared(
                Path(str(external_prepared)).resolve(),
                goal_pc,
                scene_pc,
                started,
            )

        result = self._run_dro(goal_pc)
        pose_name = str(self.cfg["solution"].get("render_pose", "predict_q"))
        if pose_name not in ("predict_q", "outer_q", "inner_q"):
            raise ValueError(f"Unknown render_pose: {pose_name}")
        selected, scored = self._select_candidate(result, scene_pc, goal_pc, pose_name)
        top_k = max(1, int(self.cfg["solution"].get("visualize_top_k", 1)))
        requested = int(self.cfg["solution"].get("candidate_index", -1))
        if requested >= 0:
            render_items = [selected]
        else:
            ranked = sorted(
                scored,
                key=lambda item: (
                    item["collision_free"],
                    item["target_coverage"],
                    -item["target_mean_distance"],
                    item["scene_clearance"],
                ),
                reverse=True,
            )
            feasible = [item for item in ranked if item["collision_free"]]
            if len(feasible) < top_k:
                raise RuntimeError(
                    f"Requested {top_k} feasible visualizations, but only "
                    f"{len(feasible)} of {len(scored)} candidates clear the scene"
                )
            render_items = feasible[:top_k]

        self._aim_visualization_camera()
        all_frames = []
        closeups = []
        rendered_candidates = []
        render_steps = max(1, int(self.cfg["solution"]["render_frames"]))
        for rank, item in enumerate(render_items):
            q = self._full_q(result, item["record"], pose_name)
            self._place_hand(q)
            joint_limit_audit = self._joint_limit_audit(q)
            fk_audit = self._fk_audit(item["record"], pose_name)
            frames = []
            for _ in range(render_steps):
                render = self.get_camera_data(numpy_rgb=True)["camera_render_raw"]["rgb"][0]
                frames.append(render)
            all_frames.extend(frames)

            candidate_dir = artifact_dir / f"candidate_{item['index']:03d}"
            self._save_render(frames[-1], candidate_dir)
            if rank == 0:
                self._save_render(frames[-1], artifact_dir)
            closeups.append(frames[-1][-1])
            candidate_log = {
                "rank": rank,
                "candidate": item["index"],
                "scene_clearance": item["scene_clearance"],
                "target_coverage": item["target_coverage"],
                "target_mean_distance": item["target_mean_distance"],
                "render_q": q.tolist(),
                "joint_limit_audit": joint_limit_audit,
                "fk_audit": fk_audit,
                "artifact_dir": str(candidate_dir),
            }
            (candidate_dir / "grasp.json").write_text(
                json.dumps(candidate_log, indent=2)
            )
            rendered_candidates.append(candidate_log)

        if closeups:
            iio.imwrite(artifact_dir / "top_k_closeups.png", np.concatenate(closeups, axis=1))

        selected = render_items[0]
        q = np.asarray(rendered_candidates[0]["render_q"], dtype=np.float32)
        joint_limit_audit = rendered_candidates[0]["joint_limit_audit"]
        fk_audit = rendered_candidates[0]["fk_audit"]

        selected_log = {
            "scene": Path(self.scene_config_path[0]).name,
            "task_index": int(self.get_task_idx()),
            "task_label": self.get_task_label()[0],
            "target_object_index": int(self.task_obj_index[0][self.get_task_idx()].cpu()),
            "hand": self.hand_name,
            "render_pose": pose_name,
            "selected_candidate": selected["index"],
            "selected_scene_clearance": selected["scene_clearance"],
            "selected_target_coverage": selected["target_coverage"],
            "visible_goal_points": int(len(goal_pc)),
            "camera_visible_goal_points": int(len(visible_goal_pc)),
            "goal_pointcloud": self._goal_pointcloud_info,
            "scene_points": int(len(scene_pc)),
            "isaac_dof_names": self.isaac_dof_names,
            "render_q": q.tolist(),
            "joint_limit_audit": joint_limit_audit,
            "fk_audit": fk_audit,
            "candidate_scores": [
                {
                    "candidate": item["index"],
                    "scene_clearance": item["scene_clearance"],
                    "scene_collision_ratio": item["scene_collision_ratio"],
                    "collision_free": item["collision_free"],
                    "target_coverage": item["target_coverage"],
                    "target_mean_distance": item["target_mean_distance"],
                }
                for item in scored
            ],
            "rendered_candidates": rendered_candidates,
            "elapsed_seconds": time.time() - started,
            "artifact_dir": str(artifact_dir),
        }
        (artifact_dir / "selected_grasp.json").write_text(
            json.dumps(selected_log, indent=2)
        )
        self.last_render_log = selected_log

        if bool(self.cfg["solution"].get("highlight_target", True)):
            self.set_default_color()

        video = []
        for rgb_cameras in all_frames:
            video.extend(image_to_video([rgb_cameras]))
        return video, selected_log
