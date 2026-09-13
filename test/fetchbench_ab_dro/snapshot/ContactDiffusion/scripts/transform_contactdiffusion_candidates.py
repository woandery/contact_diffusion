#!/usr/bin/env python3
"""Transform ContactDiffusion candidates between rigid coordinate frames."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np


POINT_FIELDS = (
    "palm_position",
    "palm_surface_contact",
    "palm_object_contact",
)
POINT_ARRAY_FIELDS = (
    "tip_points",
    "matched_contact_points",
)
DIRECTION_FIELDS = ("palm_object_normal",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--transform-metadata", type=Path, required=True)
    parser.add_argument("--object-pc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-frame", default="object")
    parser.add_argument("--target-frame", default="robot_base")
    return parser.parse_args()


def transform_points(value, matrix: np.ndarray):
    points = np.asarray(value, dtype=np.float64)
    return (points @ matrix[:3, :3].T + matrix[:3, 3]).tolist()


def transform_direction(value, matrix: np.ndarray):
    direction = np.asarray(value, dtype=np.float64) @ matrix[:3, :3].T
    norm = np.linalg.norm(direction)
    return (direction / max(norm, 1.0e-12)).tolist()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.resolve().read_text())
    metadata = json.loads(args.transform_metadata.resolve().read_text())
    transform_key = (
        "robot_from_pointcloud_frame"
        if "robot_from_pointcloud_frame" in metadata
        else "robot_from_object"
    )
    matrix = np.asarray(metadata[transform_key], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {matrix.shape}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError("Transform is not homogeneous")

    output = deepcopy(payload)
    output["coordinate_transform"] = {
        "source_frame": args.source_frame,
        "target_frame": args.target_frame,
        "matrix": matrix.tolist(),
        "matrix_key": transform_key,
        "metadata": str(args.transform_metadata.resolve()),
        "source_candidates": str(args.input.resolve()),
    }
    for record in output["records"]:
        record["object_pc_asset"] = str(args.object_pc.resolve())
        record["source_object_pc_asset"] = payload["records"][0]["object_pc_asset"]
        fk = record["fk"]
        fk["target_contacts"] = transform_points(fk["target_contacts"], matrix)
        for direction_field in ("preferred_root_direction", "contact_patch_direction"):
            if fk.get(direction_field) is not None:
                fk[direction_field] = transform_direction(fk[direction_field], matrix)
        for candidate in fk["candidates"]:
            root_pose = np.asarray(candidate["root_pose"], dtype=np.float64)
            candidate["root_pose"] = (matrix @ root_pose).tolist()
            for field in POINT_FIELDS:
                if candidate.get(field) is not None:
                    candidate[field] = transform_points(candidate[field], matrix)
            for field in POINT_ARRAY_FIELDS:
                if candidate.get(field) is not None:
                    candidate[field] = transform_points(candidate[field], matrix)
            for field in DIRECTION_FIELDS:
                if candidate.get(field) is not None:
                    candidate[field] = transform_direction(candidate[field], matrix)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
