#!/usr/bin/env python3
"""Generate contacts from a checkpoint and solve gripper pose/DOFs with differentiable FK."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "PYTORCH_KERNEL_CACHE_PATH", str(REPO_ROOT / ".cache" / "torch" / "kernels")
)

import torch
import yaml
from omegaconf import OmegaConf


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.contact_dataset import ContactFormatDataset, build_contact_format_dataset
from models.diffusion import ContactDiffusion
from utils.multigripper_fk import (
    load_gripper_from_calibration,
    optimize_gripper_to_contacts,
    sample_object_surface,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multigripper_fk_isaac.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--grippers", nargs="*", default=None)
    parser.add_argument("--samples-per-gripper", type=int, default=1)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val"],
        choices=("train", "val", "test"),
        help="Dataset splits searched in order when selecting distinct objects.",
    )
    parser.add_argument(
        "--objects-per-gripper",
        type=int,
        default=None,
        help="Select this many distinct object IDs per gripper; use all available if fewer exist.",
    )
    parser.add_argument(
        "--samples-per-object",
        type=int,
        default=1,
        help="Independent diffusion/FK samples generated for each selected object.",
    )
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--optimization-steps", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument(
        "--contact-weight",
        type=float,
        default=None,
        help="Override fk_optimization.contact_weight for controlled comparisons.",
    )
    parser.add_argument(
        "--penetration-weight",
        type=float,
        default=None,
        help="Override fk_optimization.penetration_weight; use 0 for an ablation.",
    )
    parser.add_argument(
        "--max-penetration-weight",
        type=float,
        default=None,
        help="Penalize the deepest surface point so sparse deep intersections are not averaged away.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", default="outputs/multigripper_fk/fk_candidates.json")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def dataset_for_gripper(
    cfg, model_cfg, name: str, split: str, max_samples: int | None
) -> ContactFormatDataset:
    dataset = build_contact_format_dataset(
        root_dir=str(model_cfg.dataset.root_dir),
        dataset_dir=list(model_cfg.dataset.dataset_dirs),
        split=split,
        n=int(cfg["grippers"][name]["n"]),
        num_points=int(model_cfg.dataset.num_points),
        contact_field=str(model_cfg.dataset.contact_field),
        load_cmap=False,
        load_qpos=False,
        normalize=bool(model_cfg.dataset.normalize),
        split_fractions=tuple(model_cfg.dataset.split_fractions),
        split_names=tuple(model_cfg.dataset.split_names),
        max_samples=max_samples,
        seed=int(model_cfg.train.seed),
        index_cache_dir=str(resolve(model_cfg.dataset.index_cache_dir)),
        shard_cache_size=int(model_cfg.dataset.shard_cache_size),
        object_pc_asset_keys=tuple(model_cfg.dataset.object_pc_asset_keys),
        success_only=True,
        allowed_grippers=[name],
        native_n_filter=True,
    )
    if not isinstance(dataset, ContactFormatDataset):
        raise TypeError(f"Expected one ContactFormatDataset for {name}, got {type(dataset).__name__}")
    return dataset


def select_distinct_objects(
    cfg,
    model_cfg,
    name: str,
    splits: list[str],
    limit: int,
) -> list[tuple[str, ContactFormatDataset, int, dict]]:
    """Choose one deterministic representative row for each distinct object."""
    selected: list[tuple[str, ContactFormatDataset, int, dict]] = []
    seen: set[str] = set()
    for split in splits:
        dataset = dataset_for_gripper(cfg, model_cfg, name, split, max_samples=None)
        for dataset_index, offset in enumerate(dataset.offsets):
            row = dataset._read_row(int(offset))
            object_id = str(
                row.get("object_id")
                or row.get("object_name")
                or row.get("object_code")
                or ""
            )
            if not object_id or object_id in seen:
                continue
            seen.add(object_id)
            selected.append((split, dataset, dataset_index, row))
            if len(selected) >= limit:
                return selected
    return selected


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if args.checkpoint:
        checkpoint_path = resolve(args.checkpoint)
    else:
        checkpoint_path = (
            resolve(config["diffusion"]["run_dir"])
            / config["diffusion"]["checkpoints"][1]
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(checkpoint["config"])
    device = torch.device(args.device)
    model = ContactDiffusion.from_config(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    selected = args.grippers or list(config["grippers"])
    particles = int(args.particles or config["fk_optimization"]["particles"])
    optimization_steps = int(args.optimization_steps or config["fk_optimization"]["steps"])
    diffusion_steps = int(args.diffusion_steps or config["diffusion"]["num_steps"])
    contact_weight = float(
        args.contact_weight
        if args.contact_weight is not None
        else config["fk_optimization"].get("contact_weight", 1.0)
    )
    penetration_weight = float(
        args.penetration_weight
        if args.penetration_weight is not None
        else config["fk_optimization"].get("penetration_weight", 0.0)
    )
    max_penetration_weight = float(
        args.max_penetration_weight
        if args.max_penetration_weight is not None
        else config["fk_optimization"].get("max_penetration_weight", 0.0)
    )
    calibration_path = resolve(config["paths"]["tip_offset_calibration"])
    output = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "device": str(device),
        "particles": particles,
        "optimization_steps": optimization_steps,
        "diffusion_steps": diffusion_steps,
        "fk_energy": {
            "contact_weight": contact_weight,
            "penetration_weight": penetration_weight,
            "max_penetration_weight": max_penetration_weight,
            "surface_points_per_link": int(
                config["fk_optimization"].get("surface_points_per_link", 0)
            ),
            "object_surface_points": int(
                config["fk_optimization"].get("object_surface_points", 1024)
            ),
        },
        "selection": {
            "splits": list(args.splits),
            "objects_per_gripper_requested": args.objects_per_gripper,
            "samples_per_object": int(args.samples_per_object),
        },
        "records": [],
    }

    for gripper_index, name in enumerate(selected):
        spec = config["grippers"][name]
        fk_gripper = load_gripper_from_calibration(
            name, config, calibration_path, device=device
        )
        initial_joints = torch.as_tensor(spec["opened_dofs"], device=device, dtype=torch.float32)
        if args.objects_per_gripper is None:
            dataset = dataset_for_gripper(
                config,
                model_cfg,
                name,
                args.splits[0],
                max(int(args.samples_per_gripper), 1),
            )
            object_rows = [
                (
                    args.splits[0],
                    dataset,
                    index,
                    dataset._read_row(int(dataset.offsets[index])),
                )
                for index in range(min(int(args.samples_per_gripper), len(dataset)))
            ]
            samples_per_object = 1
        else:
            object_rows = select_distinct_objects(
                config,
                model_cfg,
                name,
                list(args.splits),
                max(int(args.objects_per_gripper), 1),
            )
            samples_per_object = max(int(args.samples_per_object), 1)
            if len(object_rows) < int(args.objects_per_gripper):
                print(
                    f"WARNING: {name} has only {len(object_rows)} distinct objects "
                    f"across splits={list(args.splits)}; requested {args.objects_per_gripper}.",
                    flush=True,
                )

        output.setdefault("selected_object_counts", {})[name] = len(object_rows)
        for object_index, (source_split, dataset, dataset_index, row) in enumerate(object_rows):
            item = dataset[dataset_index]
            object_pc = item["object_pc"].to(device)
            mesh_path = Path(row["asset_object_mesh"])
            if not mesh_path.is_absolute():
                mesh_path = Path(model_cfg.dataset.root_dir) / mesh_path
            collision_points_np, collision_normals_np = sample_object_surface(
                mesh_path,
                int(config["fk_optimization"].get("object_surface_points", 1024)),
                seed=int(args.seed) + 1_000_003 * gripper_index + 1_009 * object_index,
            )
            collision_points = torch.as_tensor(
                collision_points_np, device=device, dtype=torch.float32
            )
            collision_normals = torch.as_tensor(
                collision_normals_np, device=device, dtype=torch.float32
            )
            for sample_index in range(samples_per_object):
                sample_seed = (
                    int(args.seed)
                    + 1_000_003 * gripper_index
                    + 1_009 * object_index
                    + sample_index
                )
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                with torch.inference_mode():
                    contacts = model.sample(
                        object_pc=object_pc.unsqueeze(0),
                        num_contacts=int(spec["n"]),
                        dc=3,
                        num_steps=diffusion_steps,
                        sampler=str(config["diffusion"]["sampler"]),
                        project_to_surface=True,
                    )[0]
                solved = optimize_gripper_to_contacts(
                    fk_gripper,
                    contacts,
                    object_pc,
                    initial_joints,
                    particles=particles,
                    steps=optimization_steps,
                    learning_rate=float(config["fk_optimization"]["learning_rate"]),
                    contact_weight=contact_weight,
                    penetration_weight=penetration_weight,
                    max_penetration_weight=max_penetration_weight,
                    translation_regularization=float(
                        config["fk_optimization"]["translation_regularization"]
                    ),
                    seed=sample_seed,
                    top_k=int(args.top_k),
                    object_surface_points=collision_points,
                    object_surface_normals=collision_normals,
                )
                output["records"].append(
                    {
                        "gripper": name,
                        "n": int(spec["n"]),
                        "source_split": source_split,
                        "object_index": object_index,
                        "sample_index": sample_index,
                        "dataset_index": dataset_index,
                        "sample_seed": sample_seed,
                        "record_id": row.get("record_id"),
                        "object_id": row.get("object_id"),
                        "object_mesh": str(mesh_path),
                        "object_pc_asset": row.get("asset_object_pc"),
                        "fk": solved,
                    }
                )
                print(
                    f"{name} object={object_index} sample={sample_index} "
                    f"{row.get('object_id')}: best contact chamfer="
                    f"{solved['candidates'][0]['contact_chamfer_m']:.6f} m, "
                    f"mean/max penetration="
                    f"{solved['candidates'][0]['mean_penetration_m']:.6f}/"
                    f"{solved['candidates'][0]['max_penetration_m']:.6f} m",
                    flush=True,
                )

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
