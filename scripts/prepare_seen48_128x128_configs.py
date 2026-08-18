#!/usr/bin/env python3
"""Materialize frozen 128-set x 128-particle seen-48 FK configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--barrett-source", type=Path, required=True)
    parser.add_argument("--shadow-source", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    asset_root = args.asset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_root = run_root / "provenance" / "configs"
    output_root.mkdir(parents=True, exist_ok=True)
    if not (asset_root / "manifest.json").is_file():
        raise FileNotFoundError(asset_root / "manifest.json")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    outputs: dict[str, dict] = {}
    for hand, source in (
        ("barrett", args.barrett_source.resolve()),
        ("shadowhand", args.shadow_source.resolve()),
    ):
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config["paths"]["dataset_root"] = str(asset_root)
        config["paths"]["contact_format_root"] = str(asset_root)
        config["paths"]["object_pc_root"] = str(asset_root / "object_pcs")
        config["paths"]["object_mesh_root"] = str(asset_root / "object_meshes")
        config["baseline"]["contact_sets_per_object"] = 128
        config["baseline"]["particles_per_contact_set"] = 128
        config["baseline"]["optimization_steps"] = 400
        config["baseline"]["retained_per_contact_set"] = 128
        config["fk_optimization"]["particles"] = 128
        config["fk_optimization"]["steps"] = 400
        config["diffusion"]["run_dir"] = str(checkpoint.parent.parent)
        config["diffusion"]["checkpoints"] = [
            str(Path("checkpoints") / checkpoint.name)
        ]
        output = output_root / f"{hand}_seen48_128x128_steps400.yaml"
        output.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        outputs[hand] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "path": str(output),
            "sha256": sha256(output),
        }

    protocol = {
        "schema": "contactdiff-seen48-full-particle-stability-v1",
        "hands": ["barrett", "shadowhand"],
        "objects": 48,
        "diffusion_sets_per_object_hand": 128,
        "fk_particles_per_set": 128,
        "fk_optimization_steps": 400,
        "physx_repeats_per_grasp": 1000,
        "unique_grasps_per_hand": 48 * 128 * 128,
        "unique_grasps_total": 2 * 48 * 128 * 128,
        "physx_trials_per_hand": 48 * 128 * 128 * 1000,
        "physx_trials_total": 2 * 48 * 128 * 128 * 1000,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "candidate_configs": outputs,
        "diffusion": {
            "steps": 50,
            "sampler": "ddim",
            "project_to_surface": True,
            "seed": 20260808,
        },
        "fk": {
            "initialization": "enveloping",
            "envelope_side_weight": 5.0,
            "envelope_approach_weight": 2.0,
            "envelope_cosine_margin": 0.5,
            "selection_min_envelope_cosine": 0.5,
            "selection_min_approach_cosine": 0.8,
            "selection_max_penetration_m": 0.007,
            "preferred_root_direction": [0.0, 0.0, 1.0],
        },
        "closure": {"outer_fraction": 0.10, "inner_fraction": 0.20},
        "stability": {
            "pose_perturbation": False,
            "physics_parameter_perturbation": False,
            "repeat_order": "deterministic SHA256-seeded permutation",
            "streaming_aggregation": True,
            "raw_result_retention": False,
        },
    }
    protocol_path = run_root / "provenance" / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"configs": outputs, "protocol": str(protocol_path)}, indent=2))


if __name__ == "__main__":
    main()
