#!/usr/bin/env python3
"""Evaluate execution-aware wrench metrics on an all-particle simulation run.

The existing candidate GraspQP score is built mostly from the diffusion target
contacts.  This analysis instead projects every FK-realized hand contact onto
the oriented object cloud, then evaluates friction robustness and a six-force
QP after attenuating contacts by their realized gap and normal confidence.

No success label is used to construct or tune the metrics.  Labels are read
only after all analytic scores have been computed and are used for AUC and
top-1 re-ranking evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from utils.multi_contact_grasp_quality import PointCloudMultiContactGraspEvaluator
from utils.point_cloud_geometry import estimate_point_cloud_geometry


FRICTION_GRID = (0.4, 0.6, 0.8)
GAP_SIGMA_M = 0.01
EPSILON_SCALE = 0.05
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--qp-device", default="cuda:0")
    parser.add_argument("--qp-batch-size", type=int, default=4096)
    parser.add_argument("--qp-iterations", type=int, default=80)
    return parser.parse_args()


def load_results(run_root: Path, hand: str, object_name: str) -> dict[tuple[int, int], dict]:
    result_map: dict[tuple[int, int], dict] = {}
    paths = sorted((run_root / "results" / hand / object_name).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No result shards for {hand}/{object_name}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            key = (int(row["source_index"]), int(row["candidate_rank"]))
            if key in result_map:
                raise ValueError(f"Duplicate result key {hand}/{object_name}/{key}")
            result_map[key] = row
    return result_map


def quality_for_indices(
    points: np.ndarray,
    normals: np.ndarray,
    indices: np.ndarray,
    center: np.ndarray,
    *,
    soft_fingers: bool,
) -> dict[str, float]:
    epsilons: list[float] = []
    closures: list[float] = []
    for friction in FRICTION_GRID:
        result = PointCloudMultiContactGraspEvaluator.evaluate(
            points,
            indices,
            normals,
            friction_coef=friction,
            num_cone_faces=8,
            soft_fingers=soft_fingers,
            finger_radius=0.005,
            center_of_mass=center,
            fixed_total_force_budget=True,
        )
        epsilons.append(float(result.get("epsilon", 0.0)) if result.get("valid") else 0.0)
        closures.append(float(bool(result.get("force_closure", False))))
    values = np.asarray(epsilons, dtype=np.float64)
    return {
        "epsilon_mu04": float(values[0]),
        "epsilon_mu06": float(values[1]),
        "epsilon_mu08": float(values[2]),
        "epsilon_mean_friction": float(values.mean()),
        "epsilon_q10_friction": float(np.percentile(values, 10)),
        "friction_force_closure_rate": float(np.mean(closures)),
    }


def contact_activation(gaps: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    gap_term = np.exp(-0.5 * np.square(gaps / GAP_SIGMA_M))
    confidence_term = 0.25 + 0.75 * np.clip(confidence, 0.0, 1.0)
    return np.clip(gap_term * confidence_term, 1.0e-8, 1.0)


def process_candidate_file(
    candidate_path_string: str,
    run_root_string: str,
    load_simulation_labels: bool = True,
    expected_particles_per_set: int = 32,
    ranking_only: bool = False,
) -> list[dict[str, Any]]:
    candidate_path = Path(candidate_path_string)
    run_root = Path(run_root_string)
    hand = candidate_path.parent.name
    object_name = candidate_path.stem
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    result_map = (
        load_results(run_root, hand, object_name)
        if load_simulation_labels
        else {}
    )
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No candidate records in {candidate_path}")

    point_cloud_path = resolve_relocated_asset(records[0]["object_pc_asset"])
    points = np.asarray(np.load(point_cloud_path, allow_pickle=False), dtype=np.float64)[:, :3]
    geometry = estimate_point_cloud_geometry(points, k_neighbors=30)
    normals = np.asarray(geometry["normals"], dtype=np.float64)
    normal_confidence = np.asarray(geometry["confidence"], dtype=np.float64)
    tree = cKDTree(points)
    center = points.mean(axis=0)
    characteristic_length = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    soft_fingers = True
    output: list[dict[str, Any]] = []

    for source_fallback, record in enumerate(records):
        source_index = int(record.get("sample_index", source_fallback))
        candidates = record["fk"]["candidates"]
        for candidate in candidates:
            rank = int(candidate["rank"])
            result = result_map.get((source_index, rank), {})
            if load_simulation_labels and not result:
                raise KeyError(f"Missing result {hand}/{object_name}/{source_index}/{rank}")

            realized = np.asarray(candidate["matched_contact_points"], dtype=np.float64)
            finger_gaps, finger_indices = tree.query(realized, k=1)
            finger_indices = np.asarray(finger_indices, dtype=np.int64)
            finger_confidence = normal_confidence[finger_indices]

            palm_query = np.asarray(candidate["palm_surface_contact"], dtype=np.float64)
            palm_gap_query, palm_index_query = tree.query(palm_query, k=1)
            palm_index = int(palm_index_query)
            palm_gap = float(candidate.get("palm_unsigned_distance_m", palm_gap_query))
            if not np.isfinite(palm_gap):
                palm_gap = float(palm_gap_query)
            palm_confidence = float(
                candidate.get("palm_object_normal_confidence", normal_confidence[palm_index])
            )

            all_indices = np.concatenate((finger_indices, np.asarray([palm_index], dtype=np.int64)))
            finger_unique = len(np.unique(finger_indices)) == len(finger_indices)
            all_unique = len(np.unique(all_indices)) == len(all_indices)
            if not ranking_only:
                zero_quality = {
                    "epsilon_mu04": 0.0,
                    "epsilon_mu06": 0.0,
                    "epsilon_mu08": 0.0,
                    "epsilon_mean_friction": 0.0,
                    "epsilon_q10_friction": 0.0,
                    "friction_force_closure_rate": 0.0,
                }
                finger_quality = (
                    quality_for_indices(
                        points, normals, finger_indices, center, soft_fingers=soft_fingers
                    )
                    if finger_unique
                    else zero_quality
                )
                all_quality = (
                    quality_for_indices(
                        points, normals, all_indices, center, soft_fingers=soft_fingers
                    )
                    if all_unique
                    else zero_quality
                )

            gaps = np.concatenate((finger_gaps, np.asarray([palm_gap])))
            confidences = np.concatenate((finger_confidence, np.asarray([palm_confidence])))
            activations = contact_activation(gaps, confidences)
            reliability_geomean = float(np.exp(np.mean(np.log(activations))))

            strict_success = bool(
                result.get("strict_six_direction_success", result.get("strict_success", False))
            )
            row: dict[str, Any] = {
                "hand": hand,
                "object_name": object_name,
                "source_index": source_index,
                "candidate_rank": rank,
                "particle_index": int(candidate.get("particle", rank)),
                "selection_feasible": bool(candidate.get("selection_feasible", False)),
                "valid_simulation": bool(result.get("valid_simulation", True)),
                "final_success": bool(result.get("final_success", result.get("success", False))),
                "strict_success": strict_success,
                "simulation_labels_available": load_simulation_labels,
                "optimization_score": float(candidate.get("optimization_score", math.nan)),
                "assigned_contact_error_m": float(candidate.get("assigned_contact_error_m", math.nan)),
                "contact_chamfer_m": float(candidate.get("contact_chamfer_m", math.nan)),
                "cvar_penetration_m": float(candidate.get("cvar_penetration_m", math.nan)),
                "max_penetration_m": float(candidate.get("max_penetration_m", math.nan)),
                "palm_unsigned_distance_m": palm_gap,
                "existing_graspqp_score": float(candidate.get("graspqp_score", math.nan)),
                "existing_graspqp_residual": float(candidate.get("graspqp_wrench_residual", math.nan)),
                "existing_graspqp_min_singular": float(candidate.get("graspqp_min_singular_value", math.nan)),
                "finger_gap_mean_m": float(np.mean(finger_gaps)),
                "finger_gap_max_m": float(np.max(finger_gaps)),
                "finger_gap_p90_m": float(np.percentile(finger_gaps, 90)),
                "finger_active_fraction_5mm": float(np.mean(finger_gaps <= 0.005)),
                "finger_active_fraction_10mm": float(np.mean(finger_gaps <= 0.010)),
                "realized_normal_confidence_mean": float(np.mean(confidences)),
                "realized_normal_confidence_min": float(np.min(confidences)),
                "contact_reliability_geomean": reliability_geomean,
                "contact_activation_min": float(np.min(activations)),
                "finger_projection_unique": finger_unique,
                "all_projection_unique": all_unique,
                "object_characteristic_length_m": characteristic_length,
                "wrench_contacts": points[all_indices].astype(np.float32),
                "wrench_normals": normals[all_indices].astype(np.float32),
                "wrench_activations": activations.astype(np.float32),
                "wrench_center": center.astype(np.float32),
            }
            if not ranking_only:
                row.update({f"finger_{key}": value for key, value in finger_quality.items()})
                row.update({f"all_{key}": value for key, value in all_quality.items()})
            output.append(row)

    expected = len(records) * int(expected_particles_per_set)
    if len(output) != expected:
        raise ValueError(f"Expected {expected} rows for {hand}/{object_name}, got {len(output)}")
    return output


def weighted_graspqp_residual(
    contacts: torch.Tensor,
    outward_normals: torch.Tensor,
    activations: torch.Tensor,
    centers: torch.Tensor,
    torque_scales: torch.Tensor,
    *,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normal = torch.nn.functional.normalize(outward_normals, dim=2)
    reference_x = torch.zeros_like(normal)
    reference_x[..., 0] = 1.0
    reference_y = torch.zeros_like(normal)
    reference_y[..., 1] = 1.0
    reference = torch.where((normal[..., 0].abs() > 0.9).unsqueeze(2), reference_y, reference_x)
    tangent_one = torch.nn.functional.normalize(torch.cross(normal, reference, dim=2), dim=2)
    tangent_two = torch.cross(normal, tangent_one, dim=2)
    angles = torch.arange(8, device=contacts.device, dtype=contacts.dtype) * (2.0 * math.pi / 8.0)
    tangential = (
        torch.cos(angles)[None, None, :, None] * tangent_one[:, :, None, :]
        + torch.sin(angles)[None, None, :, None] * tangent_two[:, :, None, :]
    )
    rays = torch.nn.functional.normalize(-normal[:, :, None, :] + 0.8 * tangential, dim=3)
    rays = rays * activations[:, :, None, None]
    lever = (contacts - centers[:, None, :]) / torque_scales[:, None, None].clamp_min(1.0e-6)
    torque = torch.cross(lever[:, :, None, :].expand_as(rays), rays, dim=3)
    wrench = torch.cat((rays, torque), dim=3).reshape(contacts.shape[0], -1, 6).transpose(1, 2)
    gram = torch.bmm(wrench.transpose(1, 2), wrench)
    disturbances = torch.zeros((6, 6), device=contacts.device, dtype=contacts.dtype)
    disturbances[0, 0], disturbances[1, 0] = 1.0, -1.0
    disturbances[2, 1], disturbances[3, 1] = 1.0, -1.0
    disturbances[4, 2], disturbances[5, 2] = 1.0, -1.0
    coefficients = torch.zeros(
        (wrench.shape[0], 6, wrench.shape[2]), device=contacts.device, dtype=contacts.dtype
    )
    step_size = 0.45 / gram.square().sum(dim=(1, 2)).sqrt().clamp_min(1.0e-6)
    for _ in range(int(iterations)):
        response = torch.einsum("bwr,bdr->bdw", wrench, coefficients)
        error = response + disturbances.unsqueeze(0)
        gradient = 2.0 * torch.einsum("bwr,bdw->bdr", wrench, error) + 1.0e-3 * coefficients
        coefficients = torch.clamp(coefficients - step_size[:, None, None] * gradient, min=0.0)
    response = torch.einsum("bwr,bdr->bdw", wrench, coefficients)
    directional = (response + disturbances.unsqueeze(0)).square().sum(dim=2)
    residual = directional.max(dim=1).values
    minimum_singular = torch.linalg.svdvals(wrench)[:, -1]
    mean_residual = directional.mean(dim=1)
    return residual, mean_residual, minimum_singular


def add_qp_metrics(
    rows: list[dict[str, Any]],
    device_name: str,
    batch_size: int,
    iterations: int,
    ranking_only: bool = False,
) -> None:
    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        contacts = torch.as_tensor(np.stack([row["wrench_contacts"] for row in batch]), device=device)
        normals = torch.as_tensor(np.stack([row["wrench_normals"] for row in batch]), device=device)
        activations = torch.as_tensor(np.stack([row["wrench_activations"] for row in batch]), device=device)
        centers = torch.as_tensor(np.stack([row["wrench_center"] for row in batch]), device=device)
        scales = torch.as_tensor(
            [row["object_characteristic_length_m"] for row in batch], device=device, dtype=contacts.dtype
        )
        with torch.no_grad():
            weighted = weighted_graspqp_residual(
                contacts, normals, activations, centers, scales, iterations=iterations
            )
            unweighted = None
            if not ranking_only:
                unweighted = weighted_graspqp_residual(
                    contacts,
                    normals,
                    torch.ones_like(activations),
                    centers,
                    scales,
                    iterations=iterations,
                )
        tensors = weighted if unweighted is None else (*weighted, *unweighted)
        arrays = [value.detach().cpu().numpy() for value in tensors]
        for index, row in enumerate(batch):
            row["eawq_weighted_worst_residual"] = float(arrays[0][index])
            row["eawq_weighted_mean_residual"] = float(arrays[1][index])
            row["eawq_weighted_min_singular"] = float(arrays[2][index])
            if not ranking_only:
                row["realized_qp_worst_residual"] = float(arrays[3][index])
                row["realized_qp_mean_residual"] = float(arrays[4][index])
                row["realized_qp_min_singular"] = float(arrays[5][index])
                epsilon_term = row["all_epsilon_q10_friction"] / (
                    row["all_epsilon_q10_friction"] + EPSILON_SCALE
                )
                row["eawq_composite"] = (
                    row["contact_reliability_geomean"]
                    * (row["all_friction_force_closure_rate"] + epsilon_term)
                    / (1.0 + row["eawq_weighted_worst_residual"])
                )
            for key in ("wrench_contacts", "wrench_normals", "wrench_activations", "wrench_center"):
                del row[key]


FEATURE_DIRECTIONS = {
    "candidate_rank": "lower",
    "optimization_score": "lower",
    "assigned_contact_error_m": "lower",
    "contact_chamfer_m": "lower",
    "cvar_penetration_m": "lower",
    "max_penetration_m": "lower",
    "palm_unsigned_distance_m": "lower",
    "existing_graspqp_score": "lower",
    "existing_graspqp_residual": "lower",
    "existing_graspqp_min_singular": "higher",
    "finger_gap_mean_m": "lower",
    "finger_gap_max_m": "lower",
    "finger_active_fraction_5mm": "higher",
    "finger_active_fraction_10mm": "higher",
    "realized_normal_confidence_mean": "higher",
    "contact_reliability_geomean": "higher",
    "contact_activation_min": "higher",
    "finger_epsilon_q10_friction": "higher",
    "finger_friction_force_closure_rate": "higher",
    "all_epsilon_q10_friction": "higher",
    "all_epsilon_mean_friction": "higher",
    "all_friction_force_closure_rate": "higher",
    "realized_qp_worst_residual": "lower",
    "realized_qp_min_singular": "higher",
    "eawq_weighted_worst_residual": "lower",
    "eawq_weighted_mean_residual": "lower",
    "eawq_weighted_min_singular": "higher",
    "eawq_composite": "higher",
}


def binary_auc(labels: np.ndarray, values: np.ndarray, direction: str) -> float:
    mask = np.isfinite(values)
    labels = labels[mask].astype(bool)
    values = values[mask]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return math.nan
    oriented = values if direction == "higher" else -values
    ranks = rankdata(oriented, method="average")
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def pair_auc(success: np.ndarray, failure: np.ndarray, direction: str) -> float:
    delta = success[:, None] - failure[None, :]
    better = delta > 0 if direction == "higher" else delta < 0
    return float((better.sum() + 0.5 * (delta == 0).sum()) / delta.size)


def metric_statistics(rows: list[dict[str, Any]], feature: str, direction: str) -> dict[str, Any]:
    labels = np.asarray([row["final_success"] for row in rows], dtype=bool)
    values = np.asarray([row[feature] for row in rows], dtype=np.float64)
    valid = np.asarray([row["valid_simulation"] for row in rows], dtype=bool)
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["hand"], row["object_name"], row["source_index"])].append(index)
    set_aucs: list[float] = []
    pair_counts: list[int] = []
    for indices in groups.values():
        group_labels = labels[indices]
        group_values = values[indices]
        finite = np.isfinite(group_values)
        success = group_values[finite & group_labels]
        failure = group_values[finite & ~group_labels]
        if success.size and failure.size:
            set_aucs.append(pair_auc(success, failure, direction))
            pair_counts.append(int(success.size * failure.size))
    auc_array = np.asarray(set_aucs, dtype=np.float64)
    weights = np.asarray(pair_counts, dtype=np.float64)
    return {
        "feature": feature,
        "direction": direction,
        "global_auc_protocol": binary_auc(labels, values, direction),
        "global_auc_valid_only": binary_auc(labels[valid], values[valid], direction),
        "mixed_sets": int(len(auc_array)),
        "macro_within_set_auc": float(auc_array.mean()) if len(auc_array) else math.nan,
        "pair_weighted_within_set_auc": (
            float(np.average(auc_array, weights=weights)) if len(auc_array) else math.nan
        ),
        "median_within_set_auc": float(np.median(auc_array)) if len(auc_array) else math.nan,
        "sets_auc_at_least_0_6_fraction": float(np.mean(auc_array >= 0.6)) if len(auc_array) else math.nan,
    }


def choose_top1(group_rows: list[dict[str, Any]], feature: str, direction: str) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float, int]:
        value = float(row[feature])
        if not np.isfinite(value):
            value = -math.inf if direction == "higher" else math.inf
        oriented = -value if direction == "higher" else value
        return (-float(bool(row["selection_feasible"])), oriented, int(row["candidate_rank"]))

    return min(group_rows, key=key)


def top1_statistics(rows: list[dict[str, Any]], feature: str, direction: str) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["hand"], row["object_name"], row["source_index"])].append(row)
    selected = [choose_top1(group, feature, direction) for group in groups.values()]
    baseline = [min(group, key=lambda row: int(row["candidate_rank"])) for group in groups.values()]
    success = np.asarray([row["final_success"] for row in selected], dtype=bool)
    baseline_success = np.asarray([row["final_success"] for row in baseline], dtype=bool)
    return {
        "feature": feature,
        "direction": direction,
        "sets": len(selected),
        "top1_success_rate": float(success.mean()),
        "baseline_rank0_success_rate": float(baseline_success.mean()),
        "delta_percentage_points": float(100.0 * (success.mean() - baseline_success.mean())),
        "selected_invalid_rate": float(np.mean([not row["valid_simulation"] for row in selected])),
        "selection_changed_fraction": float(
            np.mean([a["candidate_rank"] != b["candidate_rank"] for a, b in zip(selected, baseline)])
        ),
        "rescued_sets": int(np.sum(success & ~baseline_success)),
        "lost_sets": int(np.sum(~success & baseline_success)),
    }


def subset_rows(rows: list[dict[str, Any]], hand: str | None = None, object_name: str | None = None) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if (hand is None or row["hand"] == hand)
        and (object_name is None or row["object_name"] == object_name)
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_paths = sorted((run_root / "candidates").glob("*/*.json"))
    if not candidate_paths:
        raise FileNotFoundError(f"No candidate files under {run_root}")

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(process_candidate_file, str(path), str(run_root)): path
            for path in candidate_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            chunk = future.result()
            rows.extend(chunk)
            print(f"geometry {path.parent.name}/{path.stem}: {len(chunk)}", flush=True)
    rows.sort(key=lambda row: (row["hand"], row["object_name"], row["source_index"], row["candidate_rank"]))
    if len(rows) != 40960:
        raise ValueError(f"Expected 40960 particle records, got {len(rows)}")

    for hand in sorted({row["hand"] for row in rows}):
        hand_rows = [row for row in rows if row["hand"] == hand]
        add_qp_metrics(hand_rows, args.qp_device, int(args.qp_batch_size), int(args.qp_iterations))
        print(f"QP {hand}: {len(hand_rows)}", flush=True)

    scopes = {"overall": rows}
    for hand in sorted({row["hand"] for row in rows}):
        scopes[f"hand/{hand}"] = subset_rows(rows, hand=hand)

    auc_rows: list[dict[str, Any]] = []
    top1_rows: list[dict[str, Any]] = []
    for scope, scope_rows_value in scopes.items():
        for feature, direction in FEATURE_DIRECTIONS.items():
            auc_row = metric_statistics(scope_rows_value, feature, direction)
            auc_row = {"scope": scope, **auc_row}
            auc_rows.append(auc_row)
            top1_row = top1_statistics(scope_rows_value, feature, direction)
            top1_rows.append({"scope": scope, **top1_row})

    per_object_rows: list[dict[str, Any]] = []
    for hand in sorted({row["hand"] for row in rows}):
        for object_name in sorted({row["object_name"] for row in rows if row["hand"] == hand}):
            object_rows = subset_rows(rows, hand=hand, object_name=object_name)
            auc = metric_statistics(object_rows, "eawq_composite", "higher")
            top1 = top1_statistics(object_rows, "eawq_composite", "higher")
            per_object_rows.append(
                {
                    "hand": hand,
                    "object_name": object_name,
                    "trials": len(object_rows),
                    "success_rate": float(np.mean([row["final_success"] for row in object_rows])),
                    "invalid_rate": float(np.mean([not row["valid_simulation"] for row in object_rows])),
                    "eawq_global_auc": auc["global_auc_protocol"],
                    "eawq_macro_within_set_auc": auc["macro_within_set_auc"],
                    "eawq_top1_success_rate": top1["top1_success_rate"],
                    "baseline_rank0_success_rate": top1["baseline_rank0_success_rate"],
                    "delta_percentage_points": top1["delta_percentage_points"],
                }
            )

    write_csv(output_dir / "metric_auc.csv", auc_rows)
    write_csv(output_dir / "top1_reranking.csv", top1_rows)
    write_csv(output_dir / "per_object_eawq.csv", per_object_rows)
    particle_fields = [key for key in rows[0].keys()]
    with gzip.open(output_dir / "particle_metrics.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=particle_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": "execution-aware-wrench-auc-v1",
        "run_root": str(run_root),
        "trial_count": len(rows),
        "contact_sets": len({(row["hand"], row["object_name"], row["source_index"]) for row in rows}),
        "friction_grid": list(FRICTION_GRID),
        "gap_sigma_m": GAP_SIGMA_M,
        "epsilon_scale": EPSILON_SCALE,
        "label_usage": "labels used only for post-hoc AUC and top1 evaluation",
        "auc": auc_rows,
        "top1": top1_rows,
        "per_object": per_object_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    overall_auc = sorted(
        [row for row in auc_rows if row["scope"] == "overall"],
        key=lambda row: row["macro_within_set_auc"],
        reverse=True,
    )
    top1_lookup = {(row["scope"], row["feature"]): row for row in top1_rows}
    report = [
        "# 40,960 粒子执行感知抓取质量与成功率统计",
        "",
        f"- 粒子记录：{len(rows):,}",
        f"- Contact sets：{summary['contact_sets']:,}",
        "- 成功标签：Isaac Gym `final_success`；无效仿真按协议计为失败",
        "- 指标构造过程不读取成功标签；标签只用于事后 AUC 与重排序成功率",
        "",
        "## 新指标定义",
        "",
        "EAWQ 使用 FK 实际手部接触位置的最近物体表面点与法向，而不是 diffusion 目标接触点。",
        f"接触激活为 `exp(-0.5*(gap/{GAP_SIGMA_M:.3f})^2) * (0.25 + 0.75*normal_confidence)`。",
        f"在摩擦系数 `{FRICTION_GRID}` 下计算 epsilon 与力闭合率，并用接触激活缩放摩擦锥后计算六方向最坏 QP 残差。",
        "组合分数越大越好：`reliability * (P_FC + epsilon_q10/(epsilon_q10+0.05)) / (1+weighted_residual)`。",
        "",
        "## 整体 AUC 与 top1 重排序",
        "",
        "组内 AUC 在同一 contact set 内比较成功/失败粒子，0.5 为随机；这是候选排序最直接的指标。",
        "",
        "| 指标 | 全局 AUC | 有效仿真 AUC | 组内宏 AUC | Pair-weighted AUC | 可行性门控后 top1 | 相对 rank0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_auc:
        top1 = top1_lookup[("overall", row["feature"])]
        report.append(
            f"| `{row['feature']}` | {row['global_auc_protocol']:.4f} | "
            f"{row['global_auc_valid_only']:.4f} | {row['macro_within_set_auc']:.4f} | "
            f"{row['pair_weighted_within_set_auc']:.4f} | {100*top1['top1_success_rate']:.2f}% | "
            f"{top1['delta_percentage_points']:+.2f} pp |"
        )
    report.extend([
        "",
        "## 分执行器 EAWQ",
        "",
        "| 执行器 | 粒子成功率 | 全局 AUC | 组内宏 AUC | rank0 top1 | EAWQ top1 | 变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for hand in sorted({row["hand"] for row in rows}):
        hand_rows = scopes[f"hand/{hand}"]
        auc = next(row for row in auc_rows if row["scope"] == f"hand/{hand}" and row["feature"] == "eawq_composite")
        top1 = top1_lookup[(f"hand/{hand}", "eawq_composite")]
        report.append(
            f"| {hand} | {100*np.mean([row['final_success'] for row in hand_rows]):.2f}% | "
            f"{auc['global_auc_protocol']:.4f} | {auc['macro_within_set_auc']:.4f} | "
            f"{100*top1['baseline_rank0_success_rate']:.2f}% | {100*top1['top1_success_rate']:.2f}% | "
            f"{top1['delta_percentage_points']:+.2f} pp |"
        )
    report.extend([
        "",
        "## 分物体 EAWQ",
        "",
        "| 执行器 | 物体 | 粒子成功率 | 全局 AUC | 组内宏 AUC | rank0 top1 | EAWQ top1 | 变化 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in per_object_rows:
        report.append(
            f"| {row['hand']} | {row['object_name']} | {100*row['success_rate']:.2f}% | "
            f"{row['eawq_global_auc']:.4f} | {row['eawq_macro_within_set_auc']:.4f} | "
            f"{100*row['baseline_rank0_success_rate']:.2f}% | {100*row['eawq_top1_success_rate']:.2f}% | "
            f"{row['delta_percentage_points']:+.2f} pp |"
        )
    (output_dir / "REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
