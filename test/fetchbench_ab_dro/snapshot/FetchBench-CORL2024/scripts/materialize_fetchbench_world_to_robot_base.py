#!/usr/bin/env python3
"""Correct mislabeled FetchBench world-frame point clouds into robot-base frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=39)
    parser.add_argument(
        "--skip-sam3d",
        action="store_true",
        help="Transform only RGB-D-derived target/scene point clouds (for DRO).",
    )
    return parser.parse_args()


def quaternion_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((x, y, z, w)))
    if norm <= 1.0e-12:
        raise ValueError("robot-base quaternion has zero norm")
    x, y, z, w = np.asarray((x, y, z, w)) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def transform_points(array: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(array)
    if points.shape[-1] != 3:
        raise ValueError(f"expected [...,3] points, got {points.shape}")
    result = np.asarray(points, dtype=np.float64) @ rotation.T + translation
    result[~np.isfinite(points).all(axis=-1)] = np.nan
    return np.ascontiguousarray(result.astype(np.float32))


def main() -> None:
    args = parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    task_data = np.load(args.task_config.resolve(), allow_pickle=False)
    state = np.asarray(task_data["task_init_state"][args.task_index, 0, :7], dtype=np.float64)
    base_position_world = state[:3]
    base_quaternion_world = state[3:7]
    world_from_base_rotation = quaternion_matrix_xyzw(base_quaternion_world)
    base_from_world_rotation = world_from_base_rotation.T
    base_from_world_translation = -base_from_world_rotation @ base_position_world
    base_from_world = np.eye(4, dtype=np.float64)
    base_from_world[:3, :3] = base_from_world_rotation
    base_from_world[:3, 3] = base_from_world_translation

    legal_source = source / "visibility/legal_partial_views.json"
    legal = json.loads(legal_source.read_text())
    output.mkdir(parents=True, exist_ok=True)
    corrected_rows = []
    audits = []

    for row in legal["views"]:
        view_id = Path(row["rgbd_capture_dir"]).name
        capture_source = source / "visibility/rgbd_views" / view_id
        sam_source = source / "sam3d" / view_id
        capture_output = output / "visibility/rgbd_views" / view_id
        sam_output = output / "sam3d" / view_id
        capture_output.mkdir(parents=True, exist_ok=True)
        sam_output.mkdir(parents=True, exist_ok=True)

        capture_meta = json.loads((capture_source / "metadata.json").read_text())
        if capture_meta.get("coordinate_frame") != "fetchbench_world":
            raise ValueError(
                f"{view_id}: expected source frame fetchbench_world, got "
                f"{capture_meta.get('coordinate_frame')!r}"
            )

        corrected_arrays: dict[str, np.ndarray] = {}
        for name in (
            "target_partial_robot_base.npy",
            "scene_partial_robot_base.npy",
            "camera_00_pointmap_robot_base.npy",
        ):
            source_path = capture_source / name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            corrected = transform_points(
                np.load(source_path), base_from_world_rotation, base_from_world_translation
            )
            np.save(capture_output / name, corrected)
            corrected_arrays[name] = corrected

        target = corrected_arrays["target_partial_robot_base.npy"]
        scene = corrected_arrays["scene_partial_robot_base.npy"]
        target_valid = target.reshape(-1, 3)
        target_valid = target_valid[np.isfinite(target_valid).all(axis=1)]
        scene_valid = scene.reshape(-1, 3)
        scene_valid = scene_valid[np.isfinite(scene_valid).all(axis=1)]
        capture_meta["coordinate_frame"] = "robot_base"
        capture_meta["coordinate_correction"] = {
            "source_coordinate_frame": "fetchbench_world",
            "base_from_world": base_from_world.tolist(),
            "task_config": str(args.task_config.resolve()),
            "task_index": args.task_index,
        }
        capture_meta["target"].update(
            {
                "path": str(capture_output / "target_partial_robot_base.npy"),
                "sha256": sha256_array(target),
                "bounds_min_m": target_valid.min(axis=0).tolist(),
                "bounds_max_m": target_valid.max(axis=0).tolist(),
            }
        )
        capture_meta["scene_without_target_or_robot"].update(
            {
                "path": str(capture_output / "scene_partial_robot_base.npy"),
                "sha256": sha256_array(scene),
                "bounds_min_m": scene_valid.min(axis=0).tolist(),
                "bounds_max_m": scene_valid.max(axis=0).tolist(),
            }
        )
        for view in capture_meta.get("rgbd_views", []):
            pointmap_name = Path(view["pointmap_robot_base_path"]).name
            view["pointmap_robot_base_path"] = str(capture_output / pointmap_name)
        (capture_output / "metadata.json").write_text(
            json.dumps(capture_meta, indent=2) + "\n"
        )

        if args.skip_sam3d:
            corrected_row = dict(row)
            corrected_row["rgbd_capture_dir"] = str(capture_output)
            corrected_rows.append(corrected_row)
            source_target = np.asarray(
                np.load(capture_source / "target_partial_robot_base.npy")
            )
            roundtrip = (
                target.astype(np.float64) @ world_from_base_rotation.T
                + base_position_world
            )
            audits.append(
                {
                    "view_id": view_id,
                    "audit_source": "target_partial",
                    "source_center_world_m": source_target.mean(axis=0).tolist(),
                    "corrected_center_robot_base_m": target.mean(axis=0).tolist(),
                    "roundtrip_max_error_m": float(
                        np.max(np.abs(roundtrip - source_target))
                    ),
                    "corrected_sha256": sha256_array(target),
                }
            )
            continue

        sam_meta = json.loads((sam_source / "sam3d_fused_centered.json").read_text())
        fused_world = np.asarray(np.load(sam_source / "sam3d_fused_robot_base.npy"))
        fused_base = transform_points(
            fused_world, base_from_world_rotation, base_from_world_translation
        )
        fused_path = sam_output / "sam3d_fused_robot_base.npy"
        np.save(fused_path, fused_base)
        center_world = np.asarray(sam_meta["center_robot_base_m"], dtype=np.float64)
        center_base = base_from_world_rotation @ center_world + base_from_world_translation
        robot_from_centered = np.eye(4, dtype=np.float64)
        robot_from_centered[:3, :3] = base_from_world_rotation
        robot_from_centered[:3, 3] = center_base
        sam_meta["center_robot_base_m"] = center_base.tolist()
        sam_meta["robot_from_pointcloud_frame"] = robot_from_centered.tolist()
        sam_meta["robot_from_centered_object"] = robot_from_centered.tolist()
        sam_meta["bounds_min_robot_base_m"] = fused_base.min(axis=0).tolist()
        sam_meta["bounds_max_robot_base_m"] = fused_base.max(axis=0).tolist()
        sam_meta["robot_base"] = str(fused_path)
        sam_meta["robot_base_sha256"] = sha256_array(fused_base)
        sam_meta["coordinate_correction"] = {
            "source_coordinate_frame": "fetchbench_world",
            "source_mislabeled_path": str(sam_source / "sam3d_fused_robot_base.npy"),
            "base_from_world": base_from_world.tolist(),
            "source_center_world_m": center_world.tolist(),
        }
        (sam_output / "sam3d_fused_centered.json").write_text(
            json.dumps(sam_meta, indent=2) + "\n"
        )

        corrected_row = dict(row)
        corrected_row["rgbd_capture_dir"] = str(capture_output)
        corrected_rows.append(corrected_row)
        roundtrip = fused_base.astype(np.float64) @ world_from_base_rotation.T + base_position_world
        audits.append(
            {
                "view_id": view_id,
                "source_center_world_m": fused_world.mean(axis=0).tolist(),
                "corrected_center_robot_base_m": fused_base.mean(axis=0).tolist(),
                "roundtrip_max_error_m": float(np.max(np.abs(roundtrip - fused_world))),
                "corrected_sha256": sha256_array(fused_base),
            }
        )

    corrected_legal = dict(legal)
    corrected_legal["views"] = corrected_rows
    corrected_legal["coordinate_frame"] = "robot_base"
    corrected_legal["coordinate_correction_manifest"] = str(
        output / "coordinate_fix_manifest.json"
    )
    legal_output = output / "visibility/legal_partial_views.json"
    legal_output.parent.mkdir(parents=True, exist_ok=True)
    legal_output.write_text(json.dumps(corrected_legal, indent=2) + "\n")

    manifest = {
        "schema": "fetchbench-world-to-robot-base-fix-v1",
        "source_root": str(source),
        "task_config": str(args.task_config.resolve()),
        "task_index": args.task_index,
        "sam3d_transformed": not args.skip_sam3d,
        "robot_base_position_world": base_position_world.tolist(),
        "robot_base_quaternion_xyzw_world": base_quaternion_world.tolist(),
        "base_from_world": base_from_world.tolist(),
        "views": len(audits),
        "maximum_roundtrip_error_m": max(x["roundtrip_max_error_m"] for x in audits),
        "audits": audits,
    }
    (output / "coordinate_fix_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({k: manifest[k] for k in ("views", "base_from_world", "maximum_roundtrip_error_m")}, indent=2))


if __name__ == "__main__":
    main()
