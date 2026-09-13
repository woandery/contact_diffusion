#!/usr/bin/env python3
"""Delete D(R,O) candidates whose open preshape intersects a partial scene."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scene-pointcloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clearance", type=float, default=0.005)
    return parser.parse_args()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.resolve().read_text())
    scene = np.asarray(np.load(args.scene_pointcloud.resolve()), dtype=np.float32)
    scene = scene.reshape(-1, 3)
    scene = np.ascontiguousarray(scene[np.isfinite(scene).all(axis=1)])
    if not len(scene):
        raise ValueError("Environment point cloud is empty")
    if args.clearance <= 0.0:
        raise ValueError("Clearance must be positive")

    tree = cKDTree(scene)
    metrics = []
    accepted = []
    for record in payload.get("records", []):
        outer = np.asarray(record["outer_pc"], dtype=np.float32).reshape(-1, 3)
        outer = outer[np.isfinite(outer).all(axis=1)]
        if not len(outer):
            raise ValueError(f"Candidate {record.get('candidate')} has no outer points")
        distances = tree.query(outer, k=1)[0]
        clearance = float(distances.min())
        collision_fraction = float(np.mean(distances < args.clearance))
        collision_free = bool(clearance >= args.clearance)
        row = {
            "candidate": int(record["candidate"]),
            "scene_clearance_m": clearance,
            "scene_collision_fraction": collision_fraction,
            "collision_free": collision_free,
        }
        metrics.append(row)
        if collision_free:
            accepted.append(record)

    output = dict(payload)
    output["records"] = accepted
    output["environment_filter"] = {
        "method": "outer_pc_to_partial_scene_nearest_neighbor",
        "clearance_m": float(args.clearance),
        "scene_pointcloud": str(args.scene_pointcloud.resolve()),
        "scene_points": int(len(scene)),
        "scene_sha256": sha256_array(scene),
        "generated_candidates": len(metrics),
        "retained_candidates": len(accepted),
        "deleted_candidates": len(metrics) - len(accepted),
        "accepted_indices": [row["candidate"] for row in metrics if row["collision_free"]],
        "deleted_indices": [row["candidate"] for row in metrics if not row["collision_free"]],
        "candidate_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["environment_filter"], indent=2))


if __name__ == "__main__":
    main()
