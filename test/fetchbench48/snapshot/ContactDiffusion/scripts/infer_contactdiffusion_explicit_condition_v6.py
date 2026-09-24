#!/usr/bin/env python3
"""Run ContactDiffusion with an explicit, reproducibly sampled condition cloud.

This adapter changes only the point cloud passed to ``sample_contact_targets``.
The upstream ``--object-pc`` remains the independent reference for
normalization, contact projection, FK geometry, normals, and penetration.
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
    parser.add_argument("--condition-object-pc", type=Path, required=True)
    parser.add_argument("--condition-points", type=int, default=2048)
    parser.add_argument("--condition-sampling-seed", type=int, default=20260905)
    parser.add_argument("--condition-mode", required=True)
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

    raw_condition = np.asarray(
        np.load(adapter_args.condition_object_pc), dtype=np.float32
    )
    raw_condition = raw_condition.reshape(-1, raw_condition.shape[-1])[:, :3]
    raw_condition = raw_condition[np.isfinite(raw_condition).all(axis=1)]
    if len(raw_condition) == 0:
        raise ValueError("Condition point cloud is empty")
    rng = np.random.default_rng(adapter_args.condition_sampling_seed)
    indices = rng.choice(
        len(raw_condition),
        size=int(adapter_args.condition_points),
        replace=len(raw_condition) < int(adapter_args.condition_points),
    )
    sampled_condition = np.ascontiguousarray(raw_condition[indices])

    spec = importlib.util.spec_from_file_location("contactdiff_v6_upstream", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_sample = module.sample_contact_targets

    def sample_with_explicit_condition(model, condition_pc_world, **kwargs):
        condition = torch.as_tensor(
            sampled_condition,
            device=condition_pc_world.device,
            dtype=condition_pc_world.dtype,
        )
        return original_sample(model, condition, **kwargs)

    module.sample_contact_targets = sample_with_explicit_condition
    original_argv = sys.argv
    try:
        sys.argv = [str(source), *upstream_args]
        module.main()
    finally:
        sys.argv = original_argv

    output_path = Path(value_after(upstream_args, "--output")).resolve()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    condition_metadata = {
        "mode": adapter_args.condition_mode,
        "asset": str(adapter_args.condition_object_pc.resolve()),
        "source_point_count": int(len(raw_condition)),
        "source_unique_point_count": int(len(np.unique(raw_condition, axis=0))),
        "output_point_count": int(len(sampled_condition)),
        "sampling_seed": int(adapter_args.condition_sampling_seed),
        "sampling_with_replacement": bool(
            len(raw_condition) < len(sampled_condition)
        ),
        "source_xyz_sha256": sha256_bytes(raw_condition),
        "sampled_indices_sha256": sha256_bytes(indices.astype(np.int64)),
        "sampled_xyz_sha256": sha256_bytes(sampled_condition),
        "bounds_min_m": raw_condition.min(axis=0).tolist(),
        "bounds_max_m": raw_condition.max(axis=0).tolist(),
    }
    payload["explicit_condition_adapter"] = {
        "schema": "contactdiff-explicit-condition-v6-adapter-v1",
        "upstream_source": str(source),
        "condition": condition_metadata,
        "normalization_reference": "reconstructed_full_surface_pointcloud",
        "fk_surface_reference": "reconstructed_full_surface_pointcloud",
        "penetration_geometry": "reconstructed_full_surface_pointcloud",
    }
    for record in payload.get("records", []):
        record["condition_observation"] = adapter_args.condition_mode
        record["condition_seed"] = int(adapter_args.condition_sampling_seed)
        record["condition_input"] = condition_metadata
        normalization = record.get("source_model_normalization")
        if isinstance(normalization, dict):
            normalization["reference"] = "reconstructed_full_surface_pointcloud"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
