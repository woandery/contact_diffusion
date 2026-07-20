#!/usr/bin/env python3
"""Train standalone contact diffusion."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter

from datasets.contact_dataset import GroupedContactDataLoaders, build_grouped_contact_loaders
from models import ContactDiffusion


def load_config(path: str):
    return OmegaConf.load(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_cfg: str) -> torch.device:
    if device_cfg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_cfg)


def to_device(batch, device):
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def get_conditioned_object_pc(batch, cfg):
    object_pc = batch["object_pc"]
    if bool(getattr(cfg.dataset, "condition_on_cmap", False)):
        if "cmap" not in batch:
            raise KeyError("dataset.condition_on_cmap=true requires batch['cmap']")
        return torch.cat([object_pc, batch["cmap"]], dim=-1)
    return object_pc


def build_loaders(cfg, split: str, shuffle: bool):
    return build_grouped_contact_loaders(
        root_dir=cfg.dataset.root_dir,
        split=split,
        n_values=cfg.dataset.n_values,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        load_cmap=cfg.dataset.load_cmap,
        load_qpos=cfg.dataset.load_qpos,
        normalize=cfg.dataset.normalize,
        dataset_type=getattr(cfg.dataset, "type", "npz_v0"),
        dataset_dir=getattr(cfg.dataset, "dataset_dir", None),
        num_points=getattr(cfg.dataset, "num_points", 2048),
        contact_field=getattr(cfg.dataset, "contact_field", "contact_points"),
        split_fractions=getattr(cfg.dataset, "split_fractions", (0.98, 0.01, 0.01)),
        split_names=getattr(cfg.dataset, "split_names", ("train", "val", "test")),
        max_samples=getattr(cfg.dataset, "max_samples", None),
        split_max_samples=getattr(cfg.dataset, "split_max_samples", None),
        seed=int(cfg.train.seed),
        index_cache_dir=getattr(cfg.dataset, "index_cache_dir", ".cache/contactdiffusion/manifest_offsets"),
        shard_cache_size=getattr(cfg.dataset, "shard_cache_size", 4),
        shuffle=shuffle,
        drop_last=shuffle,
        pin_memory=str(cfg.train.device) == "cuda",
    )


def weighted_loss(losses: Dict[str, tuple]) -> torch.Tensor:
    return sum(weight * value for weight, value in losses.values())


def loss_items(losses: Dict[str, tuple]) -> Dict[str, float]:
    return {key: float(value.detach().cpu()) for key, (_, value) in losses.items()}


def save_checkpoint(path: Path, model, optimizer, step: int, cfg, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0)), float(ckpt.get("best_val", math.inf))


@torch.no_grad()
def validate(model, val_loaders, cfg, device, writer=None, step: int = 0):
    model.eval()
    results = {}
    mean_accum = {}
    mean_count = 0
    for n, loader in val_loaders.items():
        totals = {}
        count = 0
        max_batches = int(getattr(cfg.train, "val_batches_per_n", 10))
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = to_device(batch, device)
            outputs, losses, _ = model.training_step(
                object_pc=get_conditioned_object_pc(batch, cfg),
                contacts=batch["contacts"],
                num_contacts=int(n),
            )
            total = weighted_loss(losses)
            totals["total"] = totals.get("total", 0.0) + float(total.detach().cpu())
            for key, value in loss_items(losses).items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        if count == 0:
            continue
        n_result = {key: value / count for key, value in totals.items()}
        results[int(n)] = n_result
        for key, value in n_result.items():
            mean_accum[key] = mean_accum.get(key, 0.0) + value
            if writer is not None:
                writer.add_scalar(f"val/loss_{key}_n{n}", value, step)
        mean_count += 1
    means = {}
    if mean_count > 0:
        for key, value in mean_accum.items():
            means[key] = value / mean_count
            if writer is not None:
                writer.add_scalar(f"val/loss_{key}_mean", means[key], step)
    model.train()
    return results, means


def train(args):
    cfg = load_config(args.config)
    if args.dataset_dir is not None:
        cfg.dataset.root_dir = args.dataset_dir
    if args.resume is not None:
        cfg.train.resume = args.resume
    if args.max_steps is not None:
        cfg.train.max_steps = int(args.max_steps)
    if args.device is not None:
        cfg.train.device = args.device
    seed_everything(int(cfg.train.seed))

    output_dir = Path(cfg.train.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    device = choose_device(str(cfg.train.device))
    print(
        f"Starting training: config={args.config}, dataset={cfg.dataset.root_dir}, "
        f"n_values={list(cfg.dataset.n_values)}, device={device}, max_steps={cfg.train.max_steps}",
        flush=True,
    )
    train_loaders = build_loaders(cfg, split="train", shuffle=True)
    val_loaders = build_loaders(cfg, split="val", shuffle=False)
    grouped_train = GroupedContactDataLoaders(
        train_loaders, n_sampling=getattr(cfg.train, "n_sampling", "uniform")
    )

    model = ContactDiffusion.from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    start_step = 0
    best_val = math.inf
    if getattr(cfg.train, "resume", None):
        start_step, best_val = load_checkpoint(cfg.train.resume, model, optimizer)
        print(f"Resumed from {cfg.train.resume} at step {start_step}, best_val={best_val:.6g}")

    writer = SummaryWriter(output_dir / "tensorboard")
    max_steps = int(cfg.train.max_steps)
    log_every = int(cfg.train.log_every)
    val_every = int(cfg.train.val_every)
    save_every = int(cfg.train.save_every)
    grad_clip = float(cfg.train.grad_clip_norm)

    model.train()
    for step in range(start_step + 1, max_steps + 1):
        n, batch = grouped_train.next()
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs, losses, _ = model.training_step(
            object_pc=get_conditioned_object_pc(batch, cfg),
            contacts=batch["contacts"],
            num_contacts=int(n),
        )
        loss = weighted_loss(losses)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        if grad_clip > 0:
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
        optimizer.step()

        values = loss_items(losses)
        writer.add_scalar("train/loss_total", float(loss.detach().cpu()), step)
        writer.add_scalar(f"train/loss_total_n{n}", float(loss.detach().cpu()), step)
        for key, value in values.items():
            writer.add_scalar(f"train/loss_{key}", value, step)
            writer.add_scalar(f"train/loss_{key}_n{n}", value, step)

        if step % log_every == 0 or step == 1:
            print(
                f"[step {step:06d}] n={n} loss={float(loss.detach().cpu()):.6f} "
                f"noise={values.get('noise', 0.0):.6f} "
                f"chamfer={values.get('chamfer', 0.0):.6f}",
                flush=True,
            )

        if step % val_every == 0:
            val_by_n, val_mean = validate(model, val_loaders, cfg, device, writer, step)
            print(f"[step {step:06d}] validation by n: {val_by_n}")
            val_total = val_mean.get("total", math.inf)
            if val_total < best_val:
                best_val = val_total
                save_checkpoint(ckpt_dir / "best_val.pt", model, optimizer, step, cfg, best_val)
            model.train()

        if step % save_every == 0 or step == max_steps:
            save_checkpoint(ckpt_dir / f"step_{step:08d}.pt", model, optimizer, step, cfg, best_val)
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, step, cfg, best_val)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train standalone contact diffusion.")
    parser.add_argument("--config", default="configs/contact_diffusion_barrett_n3.yaml")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
