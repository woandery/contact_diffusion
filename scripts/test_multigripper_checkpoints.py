#!/usr/bin/env python3
"""Smoke-test ContactDiffusion checkpoints on n=2/3/5 validation samples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.contact_dataset import build_contact_format_dataset
from models.diffusion import ContactDiffusion, project_contacts_to_surface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="outputs/contact_diffusion_multigripper_success_h100_n235",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Checkpoint paths or names relative to RUN_DIR/checkpoints. Defaults to best_val.pt and latest.pt.",
    )
    parser.add_argument("--n-values", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--samples-per-n", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_checkpoints(run_dir: Path, values: list[str] | None) -> list[Path]:
    values = values or ["best_val.pt", "latest.pt"]
    paths = []
    for value in values:
        path = Path(value)
        if not path.is_absolute():
            path = run_dir / "checkpoints" / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def balanced_items(dataset: Dataset, count: int) -> list[dict]:
    count = min(int(count), len(dataset))
    if not isinstance(dataset, ConcatDataset):
        return [dataset[i] for i in range(count)]
    items: list[dict] = []
    max_child_size = max(len(child) for child in dataset.datasets)
    for row in range(max_child_size):
        for child in dataset.datasets:
            if row < len(child):
                items.append(child[row])
                if len(items) == count:
                    return items
    return items


def build_validation_dataset(cfg, n: int, count: int) -> Dataset:
    cache_dir = Path(str(cfg.dataset.index_cache_dir))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    return build_contact_format_dataset(
        root_dir=str(cfg.dataset.root_dir),
        dataset_dir=list(cfg.dataset.dataset_dirs),
        split="val",
        n=int(n),
        num_points=int(cfg.dataset.num_points),
        contact_field=str(cfg.dataset.contact_field),
        load_cmap=False,
        load_qpos=False,
        normalize=bool(cfg.dataset.normalize),
        split_fractions=tuple(cfg.dataset.split_fractions),
        split_names=tuple(cfg.dataset.split_names),
        max_samples=max(int(count), 1),
        seed=int(cfg.train.seed),
        index_cache_dir=str(cache_dir),
        shard_cache_size=int(cfg.dataset.shard_cache_size),
        object_pc_asset_keys=tuple(cfg.dataset.object_pc_asset_keys),
        success_only=bool(cfg.dataset.success_only),
        max_projection_distance=cfg.dataset.max_projection_distance,
        allowed_grippers=list(cfg.dataset.allowed_grippers),
        native_n_filter=bool(cfg.dataset.native_n_filter),
    )


def unordered_chamfer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    distance = torch.cdist(a, b)
    return distance.min(dim=2).values.mean(dim=1) + distance.min(dim=1).values.mean(dim=1)


def evaluate_n(model, cfg, n: int, args, device: torch.device) -> dict:
    dataset = build_validation_dataset(cfg, n=n, count=args.samples_per_n)
    items = balanced_items(dataset, args.samples_per_n)
    object_pc = torch.stack([item["object_pc"] for item in items]).to(device, non_blocking=True)
    gt = torch.stack([item["contacts"] for item in items]).to(device, non_blocking=True)

    # Use identical diffusion noise for the same n across checkpoints so that
    # checkpoint-to-checkpoint metric differences are directly comparable.
    sample_seed = int(args.seed) + 1009 * int(n)
    torch.manual_seed(sample_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(sample_seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        raw = model.sample(
            object_pc=object_pc,
            num_contacts=int(n),
            dc=3,
            num_steps=int(args.num_steps),
            sampler=str(args.sampler),
            project_to_surface=False,
        )
        projected = project_contacts_to_surface(raw, object_pc)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    raw_surface_distance = torch.cdist(raw, object_pc).min(dim=2).values
    projected_indices = torch.cdist(projected, object_pc).argmin(dim=2)
    unique_counts = torch.tensor(
        [int(torch.unique(row).numel()) for row in projected_indices],
        dtype=torch.float32,
    )
    chamfer = unordered_chamfer(projected, gt)
    return {
        "n": int(n),
        "sample_seed": sample_seed,
        "num_samples": len(items),
        "grippers": [str(item.get("robot_name", "unknown")) for item in items],
        "finite": bool(torch.isfinite(raw).all().item()),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(items) / max(elapsed, 1e-12),
        "raw_surface_distance_mean": float(raw_surface_distance.mean().item()),
        "raw_surface_distance_max": float(raw_surface_distance.max().item()),
        "projected_unique_contact_ratio_mean": float((unique_counts / float(n)).mean().item()),
        "projected_to_gt_chamfer_mean": float(chamfer.mean().item()),
        "gpu_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoints = resolve_checkpoints(run_dir, args.checkpoints)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "compute_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sampler": args.sampler,
        "num_steps": int(args.num_steps),
        "checkpoints": [],
    }
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(checkpoint["config"])
        model = ContactDiffusion.from_config(cfg).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        n_results = [evaluate_n(model, cfg, n, args, device) for n in args.n_values]
        report["checkpoints"].append(
            {
                "path": str(checkpoint_path),
                "step": int(checkpoint.get("step", -1)),
                "best_val": float(checkpoint.get("best_val", float("nan"))),
                "n_results": n_results,
            }
        )
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "checkpoint_tests" / "smoke_test_4090.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
