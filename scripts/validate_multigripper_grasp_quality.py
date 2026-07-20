#!/usr/bin/env python3
"""Validate grasp-quality metrics on MultiGripperGrasp contact-format data.

This script does not train a model. It reuses the validation-time quality module
with fixed contact generators so the metric can be sanity-checked against known
contact variants:

- gt: ground-truth object-side contact_points from the dataset.
- jitter: gt contacts with Gaussian perturbation.
- random_surface: random object point-cloud points.
- far: gt contacts translated far away from the object.

Expected behavior for a useful metric is that gt has high projection validity,
far has zero projection validity, and degenerate duplicate-index contacts are
rejected by the low-level indexed evaluator. Force-closure/epsilon are geometric
heuristics and may not perfectly separate all successful dataset labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.contact_dataset import build_contact_format_dataset
from utils.multi_contact_grasp_quality import (
    PointCloudMultiContactGraspEvaluator,
    estimate_point_cloud_normals,
    match_predicted_contacts_to_point_cloud,
)
from utils.validation_grasp_quality import evaluate_generated_contact_batch


class FixedContactModel:
    def __init__(self, contacts: torch.Tensor):
        self.contacts = contacts.detach().clone().float()

    def sample(self, object_pc, num_contacts, **_kwargs):
        contacts = self.contacts.to(object_pc.device)
        if contacts.shape[0] != object_pc.shape[0]:
            contacts = contacts[: object_pc.shape[0]]
        return contacts


def make_quality_config(args) -> SimpleNamespace:
    return SimpleNamespace(
        samples_per_batch=int(args.num_samples),
        dc=3,
        num_steps=1,
        sampler="ddim",
        max_projection_distance=args.max_projection_distance,
        max_projection_distance_factor=float(args.max_projection_distance_factor),
        require_precomputed_normals=False,
        normal_k_neighbors=int(args.normal_k_neighbors),
        normal_viewpoint=None,
        friction_coef=float(args.friction_coef),
        num_cone_faces=int(args.num_cone_faces),
        soft_fingers=bool(args.soft_fingers),
        finger_radius=float(args.finger_radius),
        torque_scaling=None,
        fixed_total_force_budget=True,
    )


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def collect_direct_metrics(object_pc: torch.Tensor, contacts: torch.Tensor, args) -> Dict[str, object]:
    clouds = object_pc.detach().cpu().numpy()
    gts = contacts.detach().cpu().numpy()
    gt_eps, gt_fc, random_eps, random_fc = [], [], [], []
    duplicate_valid = []
    duplicate_reasons = []
    projection_distances = []
    rng = np.random.default_rng(int(args.seed) + 991)

    for cloud, gt in zip(clouds, gts):
        normals = estimate_point_cloud_normals(cloud, k_neighbors=int(args.normal_k_neighbors))
        matched = match_predicted_contacts_to_point_cloud(
            gt,
            cloud,
            max_projection_distance=args.max_projection_distance,
            max_distance_factor=float(args.max_projection_distance_factor),
        )
        if matched["projection_distances"].size:
            projection_distances.extend(matched["projection_distances"].tolist())
        if matched["valid"]:
            result = PointCloudMultiContactGraspEvaluator.evaluate(
                cloud,
                matched["predicted_indices"],
                normals,
                friction_coef=float(args.friction_coef),
                num_cone_faces=int(args.num_cone_faces),
                soft_fingers=bool(args.soft_fingers),
                finger_radius=float(args.finger_radius),
                fixed_total_force_budget=True,
            )
            gt_eps.append(float(result["epsilon"]) if result["valid"] else 0.0)
            gt_fc.append(float(result["force_closure"]) if result["valid"] else 0.0)

            dup_indices = np.asarray([matched["predicted_indices"][0], matched["predicted_indices"][0]], dtype=np.int64)
            dup = PointCloudMultiContactGraspEvaluator.evaluate(
                cloud,
                dup_indices,
                normals,
                friction_coef=float(args.friction_coef),
                num_cone_faces=int(args.num_cone_faces),
                soft_fingers=bool(args.soft_fingers),
                finger_radius=float(args.finger_radius),
                fixed_total_force_budget=True,
            )
            duplicate_valid.append(float(bool(dup["valid"])))
            duplicate_reasons.append(str(dup.get("failure_reason", "")))

        rand_indices = rng.choice(len(cloud), size=gt.shape[0], replace=False)
        rand = PointCloudMultiContactGraspEvaluator.evaluate(
            cloud,
            rand_indices,
            normals,
            friction_coef=float(args.friction_coef),
            num_cone_faces=int(args.num_cone_faces),
            soft_fingers=bool(args.soft_fingers),
            finger_radius=float(args.finger_radius),
            fixed_total_force_budget=True,
        )
        random_eps.append(float(rand["epsilon"]) if rand["valid"] else 0.0)
        random_fc.append(float(rand["force_closure"]) if rand["valid"] else 0.0)

    return {
        "gt_epsilon": summarize(gt_eps),
        "gt_force_closure_rate": float(np.mean(gt_fc)) if gt_fc else 0.0,
        "random_surface_epsilon": summarize(random_eps),
        "random_surface_force_closure_rate": float(np.mean(random_fc)) if random_fc else 0.0,
        "gt_projection_distance": summarize(projection_distances),
        "duplicate_index_valid_rate": float(np.mean(duplicate_valid)) if duplicate_valid else 0.0,
        "duplicate_index_failure_examples": sorted(set(duplicate_reasons))[:5],
    }


def write_report(path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# MultiGripperGrasp Grasp Quality Metric Validation",
        "",
        "## Setup",
        "",
        "- dataset_root: `{}`".format(payload["dataset_root"]),
        "- dataset_dirs: `{}`".format(payload["dataset_dirs"]),
        "- grippers: `{}`".format(payload["allowed_grippers"]),
        "- n: `{}`".format(payload["n"]),
        "- success_only: `{}`".format(payload["success_only"]),
        "- samples: `{}`".format(payload["num_samples"]),
        "",
        "## Validation-Time Metric Results",
        "",
        "| variant | projection valid | quality valid | force closure | epsilon mean | projection mean | projection max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in payload["validation_metrics"].items():
        lines.append(
            "| {name} | {proj:.3f} | {qvalid:.3f} | {fc:.3f} | {eps:.6f} | {pmean:.6f} | {pmax:.6f} |".format(
                name=name,
                proj=metrics.get("grasp_projection_valid_rate", 0.0),
                qvalid=metrics.get("grasp_quality_valid_rate", 0.0),
                fc=metrics.get("grasp_force_closure_rate", 0.0),
                eps=metrics.get("grasp_epsilon_mean", 0.0),
                pmean=metrics.get("grasp_projection_distance_mean", 0.0),
                pmax=metrics.get("grasp_projection_distance_max", 0.0),
            )
        )
    direct = payload["direct_metrics"]
    lines.extend(
        [
            "",
            "## Direct Indexed Evaluator Checks",
            "",
            "- GT force-closure rate: `{:.3f}`".format(direct["gt_force_closure_rate"]),
            "- Random-surface force-closure rate: `{:.3f}`".format(direct["random_surface_force_closure_rate"]),
            "- Duplicate-index valid rate: `{:.3f}`".format(direct["duplicate_index_valid_rate"]),
            "- Duplicate-index failure examples: `{}`".format(direct["duplicate_index_failure_examples"]),
            "",
            "## Interpretation",
            "",
            "- The projection gate is effective if `far` has zero projection validity.",
            "- The metric rejects exact duplicate indexed contacts if duplicate-index valid rate is zero.",
            "- Force-closure and epsilon are heuristic geometric scores computed from estimated point-cloud normals; they should be used as ranking/filtering signals, not as proof of simulator success.",
            "- If GT and random-surface scores are close, the metric is not sufficiently discriminative for this dataset/config and should be calibrated with friction, normal orientation, force budget, or simulator labels.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/inspire/qb-ilm2/project/zhanghanbo/public/wjx/grasp/graspdata_end")
    parser.add_argument("--dataset-dirs", nargs="+", default=["contact_format/v0/multigrippergrasp/by_gripper/*"])
    parser.add_argument("--allowed-grippers", nargs="+", default=["franka_panda"])
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--success-only", action="store_true", default=True)
    parser.add_argument("--include-failures", action="store_false", dest="success_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter-sigma", type=float, default=0.02)
    parser.add_argument("--far-offset", type=float, default=1.0)
    parser.add_argument("--max-projection-distance", type=float, default=None)
    parser.add_argument("--max-projection-distance-factor", type=float, default=20.0)
    parser.add_argument("--normal-k-neighbors", type=int, default=30)
    parser.add_argument("--friction-coef", type=float, default=0.5)
    parser.add_argument("--num-cone-faces", type=int, default=8)
    parser.add_argument("--soft-fingers", action="store_true", default=True)
    parser.add_argument("--finger-radius", type=float, default=0.005)
    parser.add_argument("--output-dir", default="outputs/grasp_quality_validation")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_contact_format_dataset(
        root_dir=args.dataset_root,
        dataset_dir=args.dataset_dirs,
        split=args.split,
        n=args.n,
        num_points=2048,
        contact_field="contact_points",
        load_cmap=False,
        load_qpos=False,
        normalize=False,
        split_fractions=(0.98, 0.01, 0.01),
        split_names=("train", "val", "test"),
        max_samples=args.num_samples,
        seed=args.seed,
        index_cache_dir=".cache/contactdiffusion/manifest_offsets",
        shard_cache_size=8,
        object_pc_asset_keys=("asset_object_pc", "object_pc_asset", "object_point_cloud_asset", "pc_asset"),
        success_only=args.success_only,
        allowed_grippers=args.allowed_grippers,
        native_n_filter=True,
    )
    items = [dataset[i] for i in range(min(args.num_samples, len(dataset)))]
    object_pc = torch.stack([item["object_pc"] for item in items], dim=0)
    contacts = torch.stack([item["contacts"] for item in items], dim=0)

    rng = torch.Generator().manual_seed(int(args.seed))
    jitter = contacts + torch.randn(contacts.shape, generator=rng) * float(args.jitter_sigma)
    far = contacts + float(args.far_offset)
    random_contacts = []
    for cloud in object_pc:
        indices = torch.randperm(cloud.shape[0], generator=rng)[: args.n]
        random_contacts.append(cloud[indices])
    random_surface = torch.stack(random_contacts, dim=0)

    quality_cfg = make_quality_config(args)
    variants = {
        "gt": contacts,
        "jitter": jitter,
        "random_surface": random_surface,
        "far": far,
    }
    validation_metrics = {}
    for name, variant in variants.items():
        validation_metrics[name] = evaluate_generated_contact_batch(
            FixedContactModel(variant),
            object_pc,
            object_pc,
            args.n,
            quality_cfg,
            object_normals=None,
            seed=args.seed,
        )

    direct_metrics = collect_direct_metrics(object_pc, contacts, args)
    payload = {
        "dataset_root": args.dataset_root,
        "dataset_dirs": args.dataset_dirs,
        "allowed_grippers": args.allowed_grippers,
        "n": args.n,
        "split": args.split,
        "success_only": args.success_only,
        "num_samples": len(items),
        "quality_config": vars(quality_cfg),
        "validation_metrics": validation_metrics,
        "direct_metrics": direct_metrics,
    }
    json_path = out_dir / "multigripper_grasp_quality_validation.json"
    md_path = out_dir / "multigripper_grasp_quality_validation.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
