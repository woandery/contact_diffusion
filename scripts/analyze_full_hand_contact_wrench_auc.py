#!/usr/bin/env python3
"""Evaluate palm and non-distal phalanx contacts on existing particle labels.

Each physical hand link contributes at most one closest surface representative.
Representatives are spatially suppressed, capped, and evaluated under the same
global wrench budget, so dense mesh sampling cannot manufacture grasp quality.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from analyze_execution_aware_wrench_auc import (
    contact_activation,
    metric_statistics,
    top1_statistics,
    weighted_graspqp_residual,
)
from utils.multi_contact_grasp_quality import PointCloudMultiContactGraspEvaluator
from utils.multigripper_fk import load_gripper_from_calibration
from utils.point_cloud_geometry import estimate_point_cloud_geometry


CONTACT_THRESHOLDS_M = (0.002, 0.005, 0.010)
SPATIAL_SUPPRESSION_M = 0.003
MAX_CONTACT_REGIONS = 8
ROOT = Path(__file__).resolve().parents[1]


def resolve_relocated_asset(path: str | Path) -> Path:
    """Resolve an absolute provenance path after moving the repository."""
    recorded = Path(path)
    if recorded.is_file():
        return recorded.resolve()
    for anchor in ("outputs", "configs"):
        try:
            anchor_index = recorded.parts.index(anchor)
        except ValueError:
            continue
        relocated = ROOT.joinpath(*recorded.parts[anchor_index:])
        if relocated.is_file():
            return relocated.resolve()
    raise FileNotFoundError(recorded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-particle-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qp-device", default="cuda:3")
    parser.add_argument("--qp-batch-size", type=int, default=1024)
    parser.add_argument("--qp-iterations", type=int, default=80)
    return parser.parse_args()


def load_baseline(path: Path) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["hand"],
                row["object_name"],
                int(row["source_index"]),
                int(row["candidate_rank"]),
            )
            values = {
                "selection_feasible": row["selection_feasible"].lower() == "true",
                "valid_simulation": row["valid_simulation"].lower() == "true",
                "final_success": row["final_success"].lower() == "true",
                "candidate_rank": int(row["candidate_rank"]),
                "old_eawq_weighted_mean_residual": float(row["eawq_weighted_mean_residual"]),
            }
            if "all_epsilon_mean_friction" in row:
                values["old_all_epsilon_mean_friction"] = float(
                    row["all_epsilon_mean_friction"]
                )
            if "eawq_composite" in row:
                values["old_eawq_composite"] = float(row["eawq_composite"])
            result[key] = values
    return result


def retained_link_names(gripper: Any) -> list[str]:
    excluded = {"world", "wrist", "forearm"}
    names = [name for name in gripper.surface_link_names if name.lower() not in excluded]
    if len(names) < 2:
        raise ValueError(f"Too few physical contact links: {names}")
    return names


def transform_link_surfaces(
    gripper: Any,
    joints: torch.Tensor,
    poses: torch.Tensor,
    link_names: list[str],
) -> dict[str, np.ndarray]:
    """Transform local link surfaces with candidate pose @ FK link transform."""

    transforms = gripper.chain.forward_kinematics(joints)
    batch = joints.shape[0]
    output = {}
    with torch.no_grad():
        for link in link_names:
            link_matrix = gripper._frame_matrix(transforms, link, batch)
            local = gripper.surface_points_local[link]
            chain_points = torch.einsum("bij,nj->bni", link_matrix[:, :3, :3], local)
            chain_points = chain_points + link_matrix[:, None, :3, 3]
            object_points = torch.einsum(
                "bij,bnj->bni", poses[:, :3, :3], chain_points
            ) + poses[:, None, :3, 3]
            output[link] = object_points.cpu().numpy()
    return output


def suppress_regions(
    candidates: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    eligible = [item for item in candidates if item["gap"] <= threshold]
    eligible.sort(key=lambda item: (-item["activation"], item["gap"], item["link"]))
    selected: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for item in eligible:
        if item["point_index"] in used_indices:
            continue
        if any(
            np.linalg.norm(item["point"] - kept["point"]) < SPATIAL_SUPPRESSION_M
            for kept in selected
        ):
            continue
        selected.append(item)
        used_indices.add(item["point_index"])
        if len(selected) >= MAX_CONTACT_REGIONS:
            break
    return selected


def zero_quality() -> dict[str, float]:
    return {
        "epsilon_mu06": 0.0,
        "force_closure_mu06": 0.0,
    }


def deterministic_quality(
    object_points: np.ndarray,
    normals: np.ndarray,
    indices: np.ndarray,
    center: np.ndarray,
) -> dict[str, float]:
    result = PointCloudMultiContactGraspEvaluator.evaluate(
        object_points,
        indices,
        normals,
        friction_coef=0.6,
        num_cone_faces=8,
        soft_fingers=True,
        finger_radius=0.005,
        center_of_mass=center,
        fixed_total_force_budget=True,
    )
    return {
        "epsilon_mu06": float(result.get("epsilon", 0.0)) if result.get("valid") else 0.0,
        "force_closure_mu06": float(bool(result.get("force_closure", False))),
    }


def process_candidate_file(
    candidate_path_string: str,
    baseline_path_string: str,
    ranking_only: bool = False,
) -> list[dict[str, Any]]:
    candidate_path = Path(candidate_path_string)
    hand = candidate_path.parent.name
    object_name = candidate_path.stem
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    records = payload["records"]
    gripper_name = records[0]["gripper"]
    config_path = resolve_relocated_asset(payload["config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gripper = load_gripper_from_calibration(
        gripper_name,
        config,
        config["paths"]["tip_offset_calibration"],
        device="cpu",
    )
    links = retained_link_names(gripper)
    tip_links = set(gripper.tip_links)
    palm_link = gripper.palm_surface_link
    baseline = load_baseline(Path(baseline_path_string))

    pc_path = resolve_relocated_asset(records[0]["object_pc_asset"])
    object_points = np.asarray(np.load(pc_path, allow_pickle=False), dtype=np.float64)[:, :3]
    geometry = estimate_point_cloud_geometry(object_points, k_neighbors=30)
    normals = np.asarray(geometry["normals"], dtype=np.float64)
    confidence = np.asarray(geometry["confidence"], dtype=np.float64)
    tree = cKDTree(object_points)
    center = object_points.mean(axis=0)
    characteristic_length = float(
        np.linalg.norm(object_points.max(axis=0) - object_points.min(axis=0))
    )

    flat_candidates = []
    for source_fallback, record in enumerate(records):
        source_index = int(record.get("sample_index", source_fallback))
        if list(record["fk"]["joint_names"]) != list(gripper.joint_names):
            raise ValueError(f"Joint order mismatch in {candidate_path}")
        for candidate in record["fk"]["candidates"]:
            flat_candidates.append((source_index, candidate))
    joints = torch.as_tensor(
        np.asarray([item[1]["joint_positions"] for item in flat_candidates]),
        dtype=torch.float32,
    )
    poses = torch.as_tensor(
        np.asarray([item[1]["root_pose"] for item in flat_candidates]),
        dtype=torch.float32,
    )
    link_surfaces = transform_link_surfaces(gripper, joints, poses, links)

    # Verify the reconstruction convention against the stored distal-surface
    # point.  A stored match must be one of the reconstructed distal samples.
    for probe in (0, len(flat_candidates) // 2, len(flat_candidates) - 1):
        stored = np.asarray(flat_candidates[probe][1]["matched_contact_points"])
        reconstructed = np.stack(
            [link_surfaces[link][probe] for link in gripper.tip_links], axis=0
        )
        error = np.linalg.norm(reconstructed - stored[:, None, :], axis=2).min(axis=1)
        if float(error.max()) > 2.0e-5:
            raise ValueError(
                f"FK surface reconstruction mismatch {hand}/{object_name}: {error.max()}"
            )

    nearest_by_link = {}
    for link in links:
        surface = link_surfaces[link]
        distances, point_indices = tree.query(surface.reshape(-1, 3), k=1)
        distances = distances.reshape(surface.shape[:2])
        point_indices = point_indices.reshape(surface.shape[:2])
        best_surface = distances.argmin(axis=1)
        batch_indices = np.arange(len(flat_candidates))
        nearest_by_link[link] = {
            "gap": distances[batch_indices, best_surface],
            "point_index": point_indices[batch_indices, best_surface].astype(np.int64),
        }

    output = []
    for index, (source_index, candidate) in enumerate(flat_candidates):
        region_candidates = []
        for link in links:
            gap = float(nearest_by_link[link]["gap"][index])
            point_index = int(nearest_by_link[link]["point_index"][index])
            normal_confidence = float(confidence[point_index])
            activation = float(
                contact_activation(
                    np.asarray([gap]), np.asarray([normal_confidence])
                )[0]
            )
            region_candidates.append(
                {
                    "link": link,
                    "gap": gap,
                    "point_index": point_index,
                    "point": object_points[point_index],
                    "normal": normals[point_index],
                    "confidence": normal_confidence,
                    "activation": activation,
                    "is_tip": link in tip_links,
                    "is_palm": link == palm_link,
                }
            )

        key = (hand, object_name, source_index, int(candidate["rank"]))
        if key not in baseline:
            raise KeyError(f"Missing baseline row {key}")
        row = {"hand": hand, "object_name": object_name, "source_index": source_index, **baseline[key]}
        row["candidate_rank"] = int(candidate["rank"])
        row["available_surface_links"] = len(links)
        row["closest_any_link_gap_m"] = float(min(item["gap"] for item in region_candidates))

        selected_by_threshold = {}
        thresholds = (0.010,) if ranking_only else CONTACT_THRESHOLDS_M
        for threshold in thresholds:
            selected = suppress_regions(region_candidates, threshold=threshold)
            selected_by_threshold[threshold] = selected
            label = f"full_{int(round(threshold * 1000))}mm"
            row[f"{label}_contact_regions"] = len(selected)
            row[f"{label}_extra_nondistal_regions"] = sum(not item["is_tip"] for item in selected)
            row[f"{label}_palm_regions"] = sum(item["is_palm"] for item in selected)
            row[f"{label}_mean_gap_m"] = (
                float(np.mean([item["gap"] for item in selected])) if selected else math.inf
            )
            row[f"{label}_reliability_geomean"] = (
                float(np.exp(np.mean(np.log([item["activation"] for item in selected]))))
                if selected
                else 0.0
            )
            if not ranking_only:
                if len(selected) >= 2:
                    indices = np.asarray(
                        [item["point_index"] for item in selected], dtype=np.int64
                    )
                    quality = deterministic_quality(
                        object_points,
                        normals,
                        indices,
                        center,
                    )
                else:
                    quality = zero_quality()
                row.update({f"{label}_{name}": value for name, value in quality.items()})

        qp_selected = selected_by_threshold[0.010]
        count = len(qp_selected)
        contact_array = np.zeros((MAX_CONTACT_REGIONS, 3), dtype=np.float32)
        normal_array = np.zeros((MAX_CONTACT_REGIONS, 3), dtype=np.float32)
        activation_array = np.zeros(MAX_CONTACT_REGIONS, dtype=np.float32)
        if count:
            contact_array[:count] = np.asarray([item["point"] for item in qp_selected])
            normal_array[:count] = np.asarray([item["normal"] for item in qp_selected])
            activation_array[:count] = np.asarray([item["activation"] for item in qp_selected])
        # Keep padded lever arms at the object center. Their zero activations
        # make their primitive wrenches exactly zero.
        contact_array[count:] = center
        normal_array[count:, 2] = 1.0
        row["qp_contacts"] = contact_array
        row["qp_normals"] = normal_array
        row["qp_activations"] = activation_array
        row["qp_center"] = center.astype(np.float32)
        row["object_characteristic_length_m"] = characteristic_length
        output.append(row)
    return output


def add_full_hand_qp(
    rows: list[dict[str, Any]], device_name: str, batch_size: int, iterations: int
) -> None:
    device = torch.device(
        device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu"
    )
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        contacts = torch.as_tensor(np.stack([row["qp_contacts"] for row in batch]), device=device)
        normals = torch.as_tensor(np.stack([row["qp_normals"] for row in batch]), device=device)
        activations = torch.as_tensor(np.stack([row["qp_activations"] for row in batch]), device=device)
        centers = torch.as_tensor(np.stack([row["qp_center"] for row in batch]), device=device)
        scales = torch.as_tensor(
            [row["object_characteristic_length_m"] for row in batch],
            dtype=contacts.dtype,
            device=device,
        )
        with torch.no_grad():
            residual, mean_residual, minimum_singular = weighted_graspqp_residual(
                contacts,
                normals,
                activations,
                centers,
                scales,
                iterations=iterations,
            )
        values = [item.cpu().numpy() for item in (residual, mean_residual, minimum_singular)]
        for index, row in enumerate(batch):
            row["full_hand_weighted_worst_residual"] = float(values[0][index])
            row["full_hand_weighted_mean_residual"] = float(values[1][index])
            row["full_hand_weighted_min_singular"] = float(values[2][index])
            for key in ("qp_contacts", "qp_normals", "qp_activations", "qp_center"):
                del row[key]


FEATURE_DIRECTIONS = {
    "candidate_rank": "lower",
    "old_eawq_weighted_mean_residual": "lower",
    "old_all_epsilon_mean_friction": "higher",
    "old_eawq_composite": "higher",
    "full_hand_weighted_mean_residual": "lower",
    "full_hand_weighted_worst_residual": "lower",
    "full_hand_weighted_min_singular": "higher",
    "full_2mm_epsilon_mu06": "higher",
    "full_2mm_force_closure_mu06": "higher",
    "full_5mm_epsilon_mu06": "higher",
    "full_5mm_force_closure_mu06": "higher",
    "full_10mm_epsilon_mu06": "higher",
    "full_10mm_force_closure_mu06": "higher",
    "full_2mm_contact_regions": "higher",
    "full_5mm_contact_regions": "higher",
    "full_10mm_contact_regions": "higher",
    "full_5mm_extra_nondistal_regions": "higher",
    "full_10mm_extra_nondistal_regions": "higher",
    "full_5mm_reliability_geomean": "higher",
    "full_10mm_reliability_geomean": "higher",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    baseline_path = args.baseline_particle_metrics.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted((run_root / "candidates").glob("*/*.json"))
    rows = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(process_candidate_file, str(path), str(baseline_path)): path
            for path in paths
        }
        for future in as_completed(futures):
            path = futures[future]
            chunk = future.result()
            rows.extend(chunk)
            print(f"full-hand {path.parent.name}/{path.stem}: {len(chunk)}", flush=True)
    rows.sort(key=lambda row: (row["hand"], row["object_name"], row["source_index"], row["candidate_rank"]))
    if len(rows) != 40960:
        raise ValueError(f"Expected 40960 rows, got {len(rows)}")
    for hand in sorted({row["hand"] for row in rows}):
        hand_rows = [row for row in rows if row["hand"] == hand]
        add_full_hand_qp(
            hand_rows, args.qp_device, int(args.qp_batch_size), int(args.qp_iterations)
        )
        print(f"full-hand QP {hand}: {len(hand_rows)}", flush=True)

    scopes = {"overall": rows}
    for hand in sorted({row["hand"] for row in rows}):
        scopes[f"hand/{hand}"] = [row for row in rows if row["hand"] == hand]
    auc_rows = []
    top1_rows = []
    for scope, scope_rows in scopes.items():
        for feature, direction in FEATURE_DIRECTIONS.items():
            auc_rows.append({"scope": scope, **metric_statistics(scope_rows, feature, direction)})
            top1_rows.append({"scope": scope, **top1_statistics(scope_rows, feature, direction)})
    write_csv(output_dir / "metric_auc.csv", auc_rows)
    write_csv(output_dir / "top1_reranking.csv", top1_rows)
    with gzip.open(output_dir / "particle_metrics.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "full-hand-contact-wrench-auc-v1",
        "run_root": str(run_root),
        "trial_count": len(rows),
        "thresholds_m": list(CONTACT_THRESHOLDS_M),
        "spatial_suppression_m": SPATIAL_SUPPRESSION_M,
        "max_contact_regions": MAX_CONTACT_REGIONS,
        "friction_coefficient": 0.6,
        "auc": auc_rows,
        "top1": top1_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    top_lookup = {(row["scope"], row["feature"]): row for row in top1_rows}
    lines = [
        "# 全手表面接触 EAWQ 统计",
        "",
        "本实验将掌心、近节、中节、末节等所有物理手部表面 link 纳入接触候选。",
        f"每个 link 只保留最近表面点，3 mm 内候选做空间抑制，最多保留{MAX_CONTACT_REGIONS}个接触区域。",
        "",
        "| Scope | 指标 | 全局 AUC | 有效仿真 AUC | 组内宏 AUC | Pair-weighted AUC | top1 | 相对 rank0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    selected_features = [
        "candidate_rank",
        "old_eawq_weighted_mean_residual",
        "old_all_epsilon_mean_friction",
        "full_hand_weighted_mean_residual",
        "full_hand_weighted_worst_residual",
        "full_2mm_epsilon_mu06",
        "full_5mm_epsilon_mu06",
        "full_10mm_epsilon_mu06",
        "full_5mm_contact_regions",
        "full_10mm_extra_nondistal_regions",
    ]
    for scope in scopes:
        for feature in selected_features:
            auc = next(row for row in auc_rows if row["scope"] == scope and row["feature"] == feature)
            top = top_lookup[(scope, feature)]
            lines.append(
                f"| {scope} | `{feature}` | {auc['global_auc_protocol']:.4f} | "
                f"{auc['global_auc_valid_only']:.4f} | {auc['macro_within_set_auc']:.4f} | "
                f"{auc['pair_weighted_within_set_auc']:.4f} | {100*top['top1_success_rate']:.2f}% | "
                f"{top['delta_percentage_points']:+.2f} pp |"
            )
    lines.extend([
        "",
        "## 接触区域数量",
        "",
        "| 执行器 | 2 mm mean | 5 mm mean | 10 mm mean | 10 mm额外非末节 mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for hand in sorted({row["hand"] for row in rows}):
        hand_rows = scopes[f"hand/{hand}"]
        lines.append(
            f"| {hand} | {np.mean([row['full_2mm_contact_regions'] for row in hand_rows]):.2f} | "
            f"{np.mean([row['full_5mm_contact_regions'] for row in hand_rows]):.2f} | "
            f"{np.mean([row['full_10mm_contact_regions'] for row in hand_rows]):.2f} | "
            f"{np.mean([row['full_10mm_extra_nondistal_regions'] for row in hand_rows]):.2f} |"
        )
    (output_dir / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
