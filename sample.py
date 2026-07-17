#!/usr/bin/env python3
"""Sample contact sets from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.contact_dataset import ContactDatasetV0, contact_collate_fn
from models import ContactDiffusion
from models.losses import (
    chamfer_loss_contacts,
    diversity_loss_contacts,
    pairwise_contact_distances,
    surface_loss_contacts,
)
from train import choose_device, get_conditioned_object_pc, load_checkpoint, to_device


@torch.no_grad()
def sample(args):
    cfg = OmegaConf.load(args.config)
    if args.dataset_dir is not None:
        cfg.dataset.root_dir = args.dataset_dir
    device = choose_device(str(cfg.train.device))
    dataset = ContactDatasetV0(
        root_dir=cfg.dataset.root_dir,
        split=args.split,
        n=args.n,
        load_cmap=bool(getattr(cfg.dataset, "condition_on_cmap", False)),
        load_qpos=False,
        normalize=cfg.dataset.normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=contact_collate_fn,
    )
    model = ContactDiffusion.from_config(cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    saved = 0
    for batch in loader:
        batch = to_device(batch, device)
        object_pc_xyz = batch["object_pc"]
        object_pc_model = get_conditioned_object_pc(batch, cfg)
        gt_contacts = batch["contacts"]
        sampled = model.sample(
            object_pc=object_pc_model,
            num_contacts=args.n,
            dc=cfg.model.dc,
            num_steps=args.num_steps,
            sampler=getattr(cfg.sampling, "sampler", "ddpm"),
            project_to_surface=args.project_to_surface,
        )
        metrics = {
            "surface_distance": float(surface_loss_contacts(sampled, object_pc_xyz, squared=False).cpu()),
            "pairwise_contact_distance": float(pairwise_contact_distances(sampled).mean().cpu()),
            "chamfer_to_gt": float(chamfer_loss_contacts(sampled, gt_contacts).cpu()),
            "diversity": float(diversity_loss_contacts(sampled, sigma=cfg.loss.diversity_sigma).cpu()),
        }
        all_metrics.append(metrics)

        object_pc_np = object_pc_xyz.detach().cpu().numpy()
        gt_np = gt_contacts.detach().cpu().numpy()
        sampled_np = sampled.detach().cpu().numpy()
        for i in range(object_pc_np.shape[0]):
            if args.max_objects > 0 and saved >= args.max_objects:
                break
            np.savez_compressed(
                output_dir / f"sample_{saved:08d}.npz",
                object_pc=object_pc_np[i].astype(np.float32),
                gt_contacts=gt_np[i].astype(np.float32),
                sampled_contacts=sampled_np[i].astype(np.float32),
                object_name=np.array(batch["object_name"][i]),
                robot_name=np.array(batch["robot_name"][i]),
                n=np.array(args.n, dtype=np.int64),
            )
            saved += 1
        if args.max_objects > 0 and saved >= args.max_objects:
            break

    summary = {
        "num_saved": saved,
        "metrics_mean": {
            key: float(np.mean([m[key] for m in all_metrics])) for key in all_metrics[0]
        }
        if all_metrics
        else {},
    }
    with open(output_dir / "sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Sample contact diffusion contacts.")
    parser.add_argument("--config", default="configs/contact_diffusion_barrett_n3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_to_surface", action="store_true")
    parser.add_argument("--max_objects", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    sample(parse_args())
