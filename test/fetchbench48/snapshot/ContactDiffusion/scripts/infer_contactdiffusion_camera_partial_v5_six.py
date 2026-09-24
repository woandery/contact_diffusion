#!/usr/bin/env python3
"""Run the upstream v5 inference entry with a real camera partial condition.

The upstream entry already separates the AR model condition from its
normalization and surface arguments internally, but its CLI currently exposes
only full input or a synthetic crop.  This adapter replaces only the condition
passed to ``sample_contact_targets``.  The object point cloud supplied to the
upstream entry is a reconstructed full-surface cloud (SAM 3D Objects in this
experiment) and is used for canonical normalization, nearest-surface FK
targets, normal estimation, and penetration energy.  No ground-truth object
mesh is required by this adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def value_after(arguments: list[str], option: str) -> str:
    try:
        return arguments[arguments.index(option) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Required upstream argument is missing: {option}") from error


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--partial-object-pc", type=Path, required=True)
    parser.add_argument("--partial-points", type=int, default=2048)
    parser.add_argument("--partial-sampling-seed", type=int, default=20260902)
    adapter_args, upstream_args = parser.parse_known_args()

    source = Path(
        os.environ.get(
            "CONTACTDIFF_INFER_SOURCE",
            Path(__file__).resolve().parents[2]
            / "ContactDiffusion"
            / "scripts"
            / "infer_local_contactdiffusion_grasp.py",
        )
    ).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    raw_partial = np.asarray(np.load(adapter_args.partial_object_pc), dtype=np.float32)
    raw_partial = raw_partial.reshape(-1, raw_partial.shape[-1])[:, :3]
    raw_partial = raw_partial[np.isfinite(raw_partial).all(axis=1)]
    if len(raw_partial) == 0:
        raise ValueError("Camera partial point cloud is empty")
    rng = np.random.default_rng(adapter_args.partial_sampling_seed)
    indices = rng.choice(
        len(raw_partial),
        size=int(adapter_args.partial_points),
        replace=len(raw_partial) < int(adapter_args.partial_points),
    )
    sampled_partial = np.ascontiguousarray(raw_partial[indices])

    spec = importlib.util.spec_from_file_location("contactdiff_v5_upstream", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_sample = module.sample_contact_targets

    def sample_with_camera_partial(model, condition_pc_world, **kwargs):
        camera_condition = torch.as_tensor(
            sampled_partial,
            device=condition_pc_world.device,
            dtype=condition_pc_world.dtype,
        )
        return original_sample(model, camera_condition, **kwargs)

    module.sample_contact_targets = sample_with_camera_partial
    original_argv = sys.argv
    try:
        sys.argv = [str(source), *upstream_args]
        module.main()
    finally:
        sys.argv = original_argv

    output_path = Path(value_after(upstream_args, "--output")).resolve()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    partial_metadata = {
        "mode": "camera_segmented_partial",
        "asset": str(adapter_args.partial_object_pc.resolve()),
        "source_point_count": int(len(raw_partial)),
        "source_unique_point_count": int(len(np.unique(raw_partial, axis=0))),
        "output_point_count": int(len(sampled_partial)),
        "sampling_seed": int(adapter_args.partial_sampling_seed),
        "sampling_with_replacement": bool(len(raw_partial) < len(sampled_partial)),
        "source_xyz_sha256": sha256_bytes(raw_partial),
        "sampled_indices_sha256": sha256_bytes(indices.astype(np.int64)),
        "sampled_xyz_sha256": sha256_bytes(sampled_partial),
        "bounds_min_m": raw_partial.min(axis=0).tolist(),
        "bounds_max_m": raw_partial.max(axis=0).tolist(),
    }
    payload["camera_partial_adapter"] = {
        "schema": "contactdiff-camera-partial-v5-adapter-v1",
        "upstream_source": str(source),
        "condition": partial_metadata,
        "normalization_reference": "reconstructed_full_surface_pointcloud",
        "fk_surface_reference": "reconstructed_full_surface_pointcloud",
        "penetration_geometry": "reconstructed_full_surface_pointcloud",
    }
    for record in payload.get("records", []):
        record["condition_observation"] = "camera_segmented_partial"
        record["partial_seed"] = int(adapter_args.partial_sampling_seed)
        record["partial_observation"] = partial_metadata
        normalization = record.get("source_model_normalization")
        if isinstance(normalization, dict):
            normalization["reference"] = "reconstructed_full_surface_pointcloud"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
