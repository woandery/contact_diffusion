#!/usr/bin/env python3
"""Reconstruct a metric FetchBench object cloud with SAM 3D Objects.

Only the captured RGB image, target mask, depth-derived pointmap, and segmented
target partial cloud are consumed.  No FetchBench mesh or full-mesh point cloud
is read by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--sam3d-root", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument(
        "--observed-source",
        choices=("capture_fused", "selected_camera"),
        default="capture_fused",
        help=(
            "Choose the observed partial cloud used for SAM3D registration/fusion. "
            "'capture_fused' preserves the legacy multi-camera capture; "
            "'selected_camera' uses only --camera-index's masked RGB-D points."
        ),
    )
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--observed-fraction", type=float, default=0.25)
    parser.add_argument("--opacity-threshold", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--layout-postprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine the predicted SAM3D pose/scale against the RGB-D pointmap and mask.",
    )
    return parser.parse_args()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sample_points(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("Cannot sample an empty point cloud")
    indices = rng.choice(len(points), size=int(count), replace=len(points) < int(count))
    return np.ascontiguousarray(points[indices], dtype=np.float32)


def scalar_value(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().reshape(-1)[0])
    return float(value)


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return rotation, translation, float(residual.max())


def align_reconstruction_to_partial(
    source: np.ndarray,
    target: np.ndarray,
    *,
    visible_fraction: float = 0.25,
    orientation_switch_min_relative_improvement: float = 0.10,
    orientation_switch_max_partial_mean_ratio: float = 1.05,
) -> tuple[np.ndarray, dict]:
    """Align a complete SAM3D surface to a segmented partial cloud.

    SAM3D pose predictions can retain a 180-degree object-pose ambiguity.  A
    single identity-initialized ICP cannot escape that basin, especially for
    bottles and other nearly axial objects.  Evaluate the four proper axis-flip
    hypotheses and rank the refined clouds with a partial-aware bidirectional
    distance instead of trusting source-to-partial ICP fitness alone.
    """
    import open3d as o3d  # pylint: disable=import-outside-toplevel

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_lo, source_hi = np.percentile(source, (2.0, 98.0), axis=0)
    target_lo, target_hi = np.percentile(target, (2.0, 98.0), axis=0)
    source_extent = np.maximum(source_hi - source_lo, 1.0e-4)
    target_extent = np.maximum(target_hi - target_lo, 1.0e-4)
    axis_scale = np.clip(target_extent / source_extent, 0.60, 1.67)
    source_center = np.median(source, axis=0)
    target_center = np.median(target, axis=0)
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    target_diagonal = float(np.linalg.norm(target_extent))
    voxel_size = float(np.clip(target_diagonal / 80.0, 0.002, 0.008))
    target_down = target_cloud.voxel_down_sample(voxel_size)
    orientation_hypotheses = {
        "identity": np.diag([1.0, 1.0, 1.0]),
        "rot_x_180": np.diag([1.0, -1.0, -1.0]),
        "rot_y_180": np.diag([-1.0, 1.0, -1.0]),
        "rot_z_180": np.diag([-1.0, -1.0, 1.0]),
    }
    hypothesis_results = []
    for hypothesis_name, orientation in orientation_hypotheses.items():
        coarse = (
            ((source - source_center[None]) * axis_scale[None]) @ orientation.T
            + target_center[None]
        )
        source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coarse))
        source_down = source_cloud.voxel_down_sample(voxel_size)
        transform = np.eye(4, dtype=np.float64)
        stages = []
        for threshold in (
            max(0.040, voxel_size * 8.0),
            max(0.020, voxel_size * 5.0),
            max(0.010, voxel_size * 3.0),
        ):
            registration = o3d.pipelines.registration.registration_icp(
                source_down,
                target_down,
                float(threshold),
                transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
            )
            transform = registration.transformation
            stages.append(
                {
                    "threshold_m": float(threshold),
                    "fitness": float(registration.fitness),
                    "inlier_rmse_m": float(registration.inlier_rmse),
                }
            )
        refined_candidate = (
            coarse @ transform[:3, :3].T + transform[:3, 3][None]
        )
        refined_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(refined_candidate)
        ).voxel_down_sample(voxel_size)
        partial_to_reconstruction = np.asarray(
            target_down.compute_point_cloud_distance(refined_cloud), dtype=np.float64
        )
        reconstruction_to_partial = np.asarray(
            refined_cloud.compute_point_cloud_distance(target_down), dtype=np.float64
        )
        visible_count = max(
            1,
            min(
                len(reconstruction_to_partial),
                int(round(float(visible_fraction) * len(reconstruction_to_partial))),
            ),
        )
        visible_reconstruction = np.partition(
            reconstruction_to_partial, visible_count - 1
        )[:visible_count]
        partial_mean = float(partial_to_reconstruction.mean())
        visible_mean = float(visible_reconstruction.mean())
        hypothesis_results.append(
            {
                "name": hypothesis_name,
                "orientation": orientation.tolist(),
                "score_m": partial_mean + visible_mean,
                "partial_to_reconstruction_mean_m": partial_mean,
                "partial_to_reconstruction_p90_m": float(
                    np.quantile(partial_to_reconstruction, 0.90)
                ),
                "visible_reconstruction_to_partial_mean_m": visible_mean,
                "transform": transform,
                "stages": stages,
                "refined": refined_candidate,
            }
        )
    hypothesis_results.sort(key=lambda item: item["score_m"])
    raw_best = hypothesis_results[0]
    identity = next(item for item in hypothesis_results if item["name"] == "identity")
    relative_improvement = (
        float(identity["score_m"] - raw_best["score_m"])
        / max(float(identity["score_m"]), 1.0e-12)
    )
    partial_mean_ratio = float(raw_best["partial_to_reconstruction_mean_m"]) / max(
        float(identity["partial_to_reconstruction_mean_m"]), 1.0e-12
    )
    switch_guard_pass = bool(
        raw_best["name"] == "identity"
        or (
            relative_improvement
            >= float(orientation_switch_min_relative_improvement)
            and partial_mean_ratio
            <= float(orientation_switch_max_partial_mean_ratio)
        )
    )
    selected = raw_best if switch_guard_pass else identity
    refined = selected.pop("refined")
    transform = selected["transform"]
    stages = selected["stages"]
    diagnostics = []
    for item in hypothesis_results:
        item.pop("refined", None)
        item["transform"] = item["transform"].tolist()
        diagnostics.append(item)
    return refined.astype(np.float32), {
        "method": (
            "partial_percentile_axis_scale_then_four_pose_hypotheses_"
            "multistage_point_icp"
        ),
        "percentiles": [2.0, 98.0],
        "axis_scale_clamp": [0.60, 1.67],
        "axis_scale": axis_scale.tolist(),
        "source_median_before_robot_base_m": source_center.tolist(),
        "target_partial_median_robot_base_m": target_center.tolist(),
        "voxel_size_m": voxel_size,
        "visible_fraction_for_selection": float(visible_fraction),
        "raw_best_orientation_hypothesis": str(raw_best["name"]),
        "orientation_switch_relative_improvement": relative_improvement,
        "orientation_switch_partial_mean_ratio": partial_mean_ratio,
        "orientation_switch_min_relative_improvement": float(
            orientation_switch_min_relative_improvement
        ),
        "orientation_switch_max_partial_mean_ratio": float(
            orientation_switch_max_partial_mean_ratio
        ),
        "orientation_switch_guard_pass": switch_guard_pass,
        "selected_orientation_hypothesis": str(selected["name"]),
        "selected_orientation": selected["orientation"],
        "selection_score_m": float(selected["score_m"]),
        "partial_to_reconstruction_mean_m": float(
            selected["partial_to_reconstruction_mean_m"]
        ),
        "partial_to_reconstruction_p90_m": float(
            selected["partial_to_reconstruction_p90_m"]
        ),
        "visible_reconstruction_to_partial_mean_m": float(
            selected["visible_reconstruction_to_partial_mean_m"]
        ),
        "icp_transform": transform.tolist(),
        "stages": stages,
        "orientation_hypotheses": diagnostics,
    }


def write_pointcloud_ply(path: Path, points: np.ndarray) -> None:
    import open3d as o3d  # pylint: disable=import-outside-toplevel

    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    )
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write point cloud: {path}")


def isaac_camera_pointmap(depth: np.ndarray, intrinsics: dict) -> np.ndarray:
    height, width = depth.shape
    fy = float(intrinsics["fy"])
    fx = float(intrinsics["fx"])
    cy = float(intrinsics["cy"]) * height / float(intrinsics["height"])
    cx = float(intrinsics["cx"]) * width / float(intrinsics["width"])
    fy *= height / float(intrinsics["height"])
    fx *= width / float(intrinsics["width"])
    rows, columns = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    pointmap = np.stack(
        (
            ((columns - cx) / -fx) * depth,
            ((rows - cy) / fy) * depth,
            depth,
        ),
        axis=-1,
    )
    pointmap[~np.isfinite(depth)] = np.nan
    convention = os.environ.get('SAM3D_CAMERA_CONVENTION', 'isaac_gl_legacy')
    if convention == 'opencv':
        # Isaac depth gives GL (+X right,+Y up,-Z forward). SAM3D's R3
        # input expects CV (+X right,+Y down,+Z forward) before its P3D rotation.
        pointmap *= np.array([1., -1., -1.], dtype=np.float32)
    elif convention != 'isaac_gl_legacy':
        raise ValueError('Unsupported SAM3D_CAMERA_CONVENTION: '+convention)
    return pointmap.astype(np.float32)


def make_rgbd_runtime_config(source: Path, output_dir: Path) -> Path:
    """Disable the unused monocular-depth model for an explicit RGB-D pointmap."""
    config = OmegaConf.load(source)
    checkpoint_dir = source.parent
    for key in tuple(config.keys()):
        value = config.get(key)
        if value and (key.endswith("_config_path") or key.endswith("_ckpt_path")):
            path = Path(str(value))
            if not path.is_absolute():
                config[key] = str((checkpoint_dir / path).resolve())
    config.depth_model = None
    dino_repository = os.environ.get("SAM3D_DINO_REPOSITORY")
    if dino_repository:
        repo = Path(dino_repository).resolve()
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(repo / "hubconf.py")

        def localize(value):
            if isinstance(value, dict):
                if str(value.get("_target_", "")).endswith(".Dino"):
                    value.update(repo_or_dir=str(repo), source="local")
                for child in value.values():
                    localize(child)
            elif isinstance(value, list):
                for child in value:
                    localize(child)

        for key in ("ss_generator_config_path", "slat_generator_config_path"):
            generator = OmegaConf.to_container(OmegaConf.load(config[key]), resolve=True)
            localize(generator)
            target = output_dir / (key + "_local.yaml")
            OmegaConf.save(OmegaConf.create(generator), target)
            config[key] = str(target.resolve())
    runtime_config = output_dir / "pipeline_rgbd_runtime.yaml"
    OmegaConf.save(config, runtime_config)
    return runtime_config


def load_inference(sam3d_root: Path, checkpoint_config: Path, output_dir: Path):
    """Construct once per worker; RGB-D inputs and outputs remain per-view."""
    sys.path.insert(0, str(sam3d_root.resolve() / "notebook"))
    from inference import Inference

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = make_rgbd_runtime_config(checkpoint_config.resolve(), output_dir)
    download = torch.hub.download_url_to_file
    def deny_download(*args, **kwargs):
        raise RuntimeError("SAM3D offline worker requested a download; populate its cache before launch")
    if os.environ.get("SAM3D_DINO_REPOSITORY"):
        torch.hub.download_url_to_file = deny_download
    try:
        return Inference(str(runtime_config), compile=False)
    finally:
        torch.hub.download_url_to_file = download


def reconstruct(args: argparse.Namespace, inference=None) -> dict:
    capture_dir = args.capture_dir.resolve()
    sam3d_root = args.sam3d_root.resolve()
    checkpoint_config = args.checkpoint_config.resolve()
    for path in (capture_dir / "metadata.json", checkpoint_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 <= float(args.observed_fraction) < 1.0:
        raise ValueError("observed-fraction must be in [0, 1)")

    metadata = json.loads((capture_dir / "metadata.json").read_text())
    views = metadata["rgbd_views"]
    camera_index = (
        int(args.camera_index)
        if args.camera_index is not None
        else int(max(views, key=lambda item: item["target_mask_pixels"])["camera_index"])
    )
    matches = [item for item in views if int(item["camera_index"]) == camera_index]
    if len(matches) != 1:
        raise ValueError(f"Camera {camera_index} is unavailable")
    view = matches[0]

    image = np.asarray(
        Image.open(capture_dir / Path(view["rgb_path"]).name).convert("RGB")
    )
    mask = np.asarray(
        Image.open(capture_dir / Path(view["target_mask_path"]).name).convert("L")
    ) > 0
    depth = np.asarray(
        np.load(capture_dir / Path(view["depth_path"]).name), dtype=np.float32
    ).squeeze()
    robot_pointmap = np.asarray(
        np.load(capture_dir / Path(view["pointmap_robot_base_path"]).name),
        dtype=np.float32,
    )
    camera_pointmap = isaac_camera_pointmap(depth, view["intrinsics"])

    sys.path.insert(0, str(sam3d_root / "notebook"))
    from pytorch3d.transforms import Transform3d  # pylint: disable=import-outside-toplevel
    from inference import (  # pylint: disable=import-error,import-outside-toplevel
        make_scene,
    )
    from sam3d_objects.pipeline.inference_pipeline_pointmap import (  # pylint: disable=import-error,import-outside-toplevel
        camera_to_pytorch3d_camera,
    )

    camera_tensor = torch.as_tensor(camera_pointmap, device="cuda", dtype=torch.float32)
    coordinate_rotation = camera_to_pytorch3d_camera(device="cuda").rotation
    pytorch3d_pointmap = Transform3d(device="cuda").rotate(
        coordinate_rotation
    ).transform_points(camera_tensor)

    valid_alignment = np.isfinite(robot_pointmap).all(axis=2) & torch.isfinite(
        pytorch3d_pointmap
    ).all(dim=2).cpu().numpy()
    source_alignment = pytorch3d_pointmap.detach().cpu().numpy()[valid_alignment]
    target_alignment = robot_pointmap[valid_alignment]
    rotation, translation, alignment_max_error = rigid_alignment(
        source_alignment, target_alignment
    )
    if alignment_max_error > 1.0e-4:
        raise RuntimeError(
            f"Camera-to-robot pointmap alignment error is {alignment_max_error:.6g} m"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = make_rgbd_runtime_config(checkpoint_config, output_dir)
    if os.environ.get("SAM3D_INPUT_AUDIT") == "1":
        np.save(output_dir / "audit_pytorch3d_pointmap.npy", pytorch3d_pointmap.detach().cpu().numpy())
    if inference is None:
        inference = load_inference(sam3d_root, checkpoint_config, output_dir)
    rgba_image = inference.merge_mask_to_rgba(image, mask)
    output = inference._pipeline.run(  # pylint: disable=protected-access
        rgba_image,
        None,
        int(args.seed),
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=bool(args.layout_postprocess),
        use_vertex_color=True,
        stage1_inference_steps=None,
        pointmap=pytorch3d_pointmap,
    )
    if not torch.isfinite(output["scale"]).all():
        raise RuntimeError(f"SAM3D nonfinite predicted scale: {output['scale'].detach().cpu().tolist()}")
    gaussian = make_scene(output, in_place=False)
    gaussian_xyz = gaussian.get_xyz.detach().float().cpu().numpy()
    opacity = gaussian.get_opacity.detach().float().cpu().numpy().reshape(-1)
    active = np.isfinite(gaussian_xyz).all(axis=1) & (
        opacity >= float(args.opacity_threshold)
    )
    reconstructed_pytorch3d = gaussian_xyz[active]
    if len(reconstructed_pytorch3d) < 64:
        raise RuntimeError(
            f"SAM3D returned only {len(reconstructed_pytorch3d)} active Gaussian centers"
        )
    reconstructed_robot_pose = reconstructed_pytorch3d @ rotation.T + translation

    if args.observed_source == "selected_camera":
        observed = np.asarray(robot_pointmap[mask], dtype=np.float32).reshape(-1, 3)
    else:
        observed = np.asarray(
            np.load(capture_dir / "target_partial_robot_base.npy"), dtype=np.float32
        ).reshape(-1, 3)
    observed = observed[np.isfinite(observed).all(axis=1)]
    reconstructed_robot, registration = align_reconstruction_to_partial(
        reconstructed_robot_pose,
        observed,
        visible_fraction=float(args.observed_fraction),
    )
    rng = np.random.default_rng(int(args.seed))
    observed_count = int(round(int(args.points) * float(args.observed_fraction)))
    reconstructed_count = int(args.points) - observed_count
    fused = np.concatenate(
        (
            sample_points(reconstructed_robot, reconstructed_count, rng),
            sample_points(observed, observed_count, rng),
        ),
        axis=0,
    ).astype(np.float32)
    rng.shuffle(fused)
    center = fused.mean(axis=0)
    centered = fused - center[None]

    pose_path = output_dir / "sam3d_pose_robot_base.npy"
    raw_path = output_dir / "sam3d_raw_robot_base.npy"
    robot_path = output_dir / "sam3d_fused_robot_base.npy"
    centered_path = output_dir / "sam3d_fused_centered.npy"
    partial_centered_path = output_dir / "camera_partial_sam3d_centered.npy"
    splat_path = output_dir / "sam3d_reconstructed_robot_base.ply"
    np.save(pose_path, reconstructed_robot_pose.astype(np.float32))
    np.save(raw_path, reconstructed_robot.astype(np.float32))
    np.save(robot_path, fused)
    np.save(centered_path, centered.astype(np.float32))
    np.save(partial_centered_path, (observed - center[None]).astype(np.float32))
    write_pointcloud_ply(splat_path, reconstructed_robot)

    robot_from_centered = np.eye(4, dtype=np.float64)
    robot_from_centered[:3, 3] = center
    result = {
        "schema": "fetchbench-sam3d-partial-derived-v1",
        "camera_pointmap_convention": os.environ.get('SAM3D_CAMERA_CONVENTION', 'isaac_gl_legacy'),
        "ground_truth_mesh_used": False,
        "sam3d_root": str(sam3d_root),
        "checkpoint_config": str(checkpoint_config),
        "runtime_config": str(runtime_config),
        "depth_model_disabled_for_explicit_rgbd_pointmap": True,
        "layout_postprocess": bool(args.layout_postprocess),
        "capture_dir": str(capture_dir),
        "camera_index": camera_index,
        "observed_source": str(args.observed_source),
        "target_mask_pixels": int(mask.sum()),
        "seed": int(args.seed),
        "opacity_threshold": float(args.opacity_threshold),
        "sam3d_pose_camera": {
            "rotation_quaternion": output["rotation"].detach().float().cpu().tolist(),
            "translation_m": output["translation"].detach().float().cpu().tolist(),
            "scale": output["scale"].detach().float().cpu().tolist(),
        },
        "layout_iou": (
            scalar_value(output["iou"])
            if "iou" in output
            else None
        ),
        "partial_registration": registration,
        "active_gaussian_points": int(len(reconstructed_robot)),
        "observed_partial_points": int(len(observed)),
        "fused_points": int(len(fused)),
        "observed_fraction": float(args.observed_fraction),
        "center_robot_base_m": center.tolist(),
        "robot_from_pointcloud_frame": robot_from_centered.tolist(),
        "robot_from_centered_object": robot_from_centered.tolist(),
        "coordinate_frame": "object_sam3d_surface_centered",
        "camera_to_robot_alignment_max_error_m": alignment_max_error,
        "bounds_min_robot_base_m": fused.min(axis=0).tolist(),
        "bounds_max_robot_base_m": fused.max(axis=0).tolist(),
        "raw_robot_base": str(raw_path),
        "pose_robot_base_before_partial_registration": str(pose_path),
        "robot_base": str(robot_path),
        "centered": str(centered_path),
        "partial_centered": str(partial_centered_path),
        "gaussian_splat": str(splat_path),
        "reconstructed_pointcloud_ply": str(splat_path),
        "robot_base_sha256": sha256_array(fused),
        "centered_sha256": sha256_array(centered.astype(np.float32)),
    }
    metadata_path = output_dir / "sam3d_fused_centered.json"
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(result, indent=2) + "\n")
    temporary_metadata.replace(metadata_path)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    reconstruct(parse_args())


if __name__ == "__main__":
    main()
