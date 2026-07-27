#!/usr/bin/env python3
"""Create training curves and an interactive contact-generation report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.contact_dataset import build_contact_format_dataset
from models.diffusion import ContactDiffusion
from utils.multi_contact_grasp_quality import (
    PointCloudMultiContactGraspEvaluator,
    estimate_point_cloud_normals,
    match_predicted_contacts_to_point_cloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-visual-samples", type=int, default=12)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--sampler", default="ddim", choices=("ddim", "ddpm"))
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoothing-window", type=int, default=200)
    return parser.parse_args()


def load_scalar_series(event_dir: Path) -> tuple[dict[str, list[tuple[int, float]]], list[str]]:
    event_files = sorted(event_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime_ns)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files in {event_dir}")
    merged: dict[str, dict[int, float]] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            by_step = merged.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                by_step[int(event.step)] = float(event.value)
    return (
        {tag: sorted(values.items()) for tag, values in merged.items()},
        [path.name for path in event_files],
    )


def rolling_mean(points: list[tuple[int, float]], window: int) -> tuple[np.ndarray, np.ndarray]:
    steps = np.asarray([step for step, _ in points], dtype=np.int64)
    values = np.asarray([value for _, value in points], dtype=np.float64)
    window = max(1, min(int(window), len(values)))
    if window == 1:
        return steps, values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return steps[window - 1 :], np.convolve(values, kernel, mode="valid")


def svg_text(x, y, value, size=12, anchor="start", color="#17212b", weight=400):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-weight="{weight}">{html.escape(str(value))}</text>'
    )


def svg_polyline(points, x_map, y_map, color, width=1.5, opacity=1.0, dash=None):
    coords = " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in points if np.isfinite(y))
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
    )


def svg_panel(
    elements,
    x,
    y,
    width,
    height,
    title,
    y_label,
    y_min,
    y_max,
    *,
    log_y=False,
    x_max=52000,
):
    plot_x, plot_y = x + 62, y + 34
    plot_w, plot_h = width - 82, height - 74
    if log_y:
        low, high = math.log10(y_min), math.log10(y_max)
        y_map = lambda value: plot_y + plot_h * (1.0 - (math.log10(max(value, y_min)) - low) / (high - low))
        exponents = range(math.floor(low), math.ceil(high) + 1)
        y_ticks = [(10.0**power, f"1e{power}") for power in exponents if y_min <= 10.0**power <= y_max]
    else:
        span = max(y_max - y_min, 1e-12)
        y_map = lambda value: plot_y + plot_h * (1.0 - (value - y_min) / span)
        y_ticks = [(y_min + span * i / 4.0, f"{y_min + span * i / 4.0:.3g}") for i in range(5)]
    x_map = lambda value: plot_x + plot_w * float(value) / float(x_max)
    elements.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="#fff" stroke="#d8dee5"/>')
    elements.append(svg_text(x + 14, y + 23, title, size=15, weight=600))
    x_ticks = (0, 10000, 20000, 30000, 40000, 50000, 52000)
    for value in x_ticks:
        px = x_map(value)
        elements.append(f'<line x1="{px:.2f}" y1="{plot_y}" x2="{px:.2f}" y2="{plot_y + plot_h}" stroke="#e8ecf0"/>')
        elements.append(svg_text(px, plot_y + plot_h + 19, f"{value // 1000}k", size=10, anchor="middle", color="#66717d"))
    for value, label in y_ticks:
        py = y_map(value)
        elements.append(f'<line x1="{plot_x}" y1="{py:.2f}" x2="{plot_x + plot_w}" y2="{py:.2f}" stroke="#e8ecf0"/>')
        elements.append(svg_text(plot_x - 8, py + 4, label, size=10, anchor="end", color="#66717d"))
    for step in (20000, 40000):
        px = x_map(step)
        elements.append(f'<line x1="{px:.2f}" y1="{plot_y}" x2="{px:.2f}" y2="{plot_y + plot_h}" stroke="#6b7280" stroke-dasharray="5 5"/>')
    elements.append(svg_text(x + 15, y + height / 2, y_label, size=10, anchor="middle", color="#66717d"))
    return x_map, y_map, (plot_x, plot_y, plot_w, plot_h)


def write_svg(output_path, width, height, title, elements):
    document = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f5f7f9"/>',
        svg_text(width / 2, 28, title, size=18, anchor="middle", weight=650),
        *elements,
        '</svg>',
    ]
    output_path.write_text("\n".join(document), encoding="utf-8")


def plot_losses(series: dict[str, list[tuple[int, float]]], output_path: Path, window: int) -> None:
    elements = []
    panels = [
        ("train/loss_noise", "val/loss_noise_n2", "Noise loss", "#176b87", 52),
        ("train/loss_chamfer", "val/loss_chamfer_n2", "Chamfer loss", "#c2410c", 462),
    ]
    for train_tag, val_tag, title, color, y in panels:
        points = series.get(train_tag, [])
        if not points:
            raise KeyError(f"Missing TensorBoard tag {train_tag}")
        positive = [value for _, value in points if value > 0]
        val_points = series.get(val_tag, [])
        positive.extend(value for _, value in val_points if value > 0)
        y_min = 10 ** math.floor(math.log10(max(min(positive), 1e-8)))
        y_max = 10 ** math.ceil(math.log10(max(positive)))
        x_map, y_map, bounds = svg_panel(
            elements, 35, y, 1330, 380, title, "loss (log)", y_min, y_max, log_y=True
        )
        elements.append(svg_polyline(points, x_map, y_map, color, width=0.55, opacity=0.22))
        smooth_steps, smooth_values = rolling_mean(points, window)
        smooth = list(zip(smooth_steps.tolist(), smooth_values.tolist()))
        elements.append(svg_polyline(smooth, x_map, y_map, color, width=2.2))
        if val_points:
            elements.append(svg_polyline(val_points, x_map, y_map, "#111827", width=1.0))
            for step, value in val_points:
                elements.append(f'<circle cx="{x_map(step):.2f}" cy="{y_map(value):.2f}" r="2.6" fill="#111827"/>')
        legend_y = bounds[1] + 17
        elements.append(svg_text(bounds[0] + bounds[2] - 355, legend_y, "raw / step", size=10, color=color))
        elements.append(svg_text(bounds[0] + bounds[2] - 255, legend_y, f"rolling mean ({window})", size=10, color=color, weight=650))
        elements.append(svg_text(bounds[0] + bounds[2] - 100, legend_y, "validation", size=10, color="#111827"))
    write_svg(output_path, 1400, 870, "ContactDiffusion n=2 Franka losses (0-52k)", elements)


def plot_grasp_metrics(series: dict[str, list[tuple[int, float]]], output_path: Path) -> None:
    elements = []
    definitions = [
        (70, 55, "Force-closure rate", "rate", 0.0, 1.0, [
            ("val/grasp_force_closure_rate_n2", "all samples", "#15803d", 1.0, None),
            ("val/grasp_force_closure_rate_valid_n2", "projection-valid", "#7c3aed", 1.0, "5 4"),
        ]),
        (735, 55, "Ferrari-Canny epsilon", "epsilon", 0.0, None, [
            ("val/grasp_epsilon_mean_n2", "epsilon mean", "#b45309", 1.0, None),
            ("val/grasp_epsilon_p10_valid_n2", "p10 valid", "#be123c", 1.0, "5 4"),
        ]),
        (70, 455, "Validation gates", "rate", 0.0, 1.0, [
            ("val/grasp_projection_valid_rate_n2", "projection valid", "#0369a1", 1.0, None),
            ("val/grasp_quality_valid_rate_n2", "quality valid", "#4338ca", 1.0, "5 4"),
        ]),
        (735, 455, "Projection distance", "distance (cm)", 0.0, None, [
            ("val/grasp_projection_distance_mean_n2", "mean", "#0f766e", 100.0, None),
            ("val/grasp_projection_distance_max_n2", "max", "#dc2626", 100.0, None),
        ]),
    ]
    for x, y, title, label, y_min, y_max, lines in definitions:
        all_values = [value * scale for tag, _, _, scale, _ in lines for _, value in series.get(tag, [])]
        if y_max is None:
            y_max = max(all_values, default=1.0) * 1.12
            if title == "Projection distance":
                y_max = max(y_max, 3.2)
        x_map, y_map, bounds = svg_panel(elements, x, y, 600, 355, title, label, y_min, y_max)
        for line_index, (tag, line_label, color, scale, dash) in enumerate(lines):
            points = [(step, value * scale) for step, value in series.get(tag, [])]
            elements.append(svg_polyline(points, x_map, y_map, color, width=1.7, dash=dash))
            for step, value in points:
                elements.append(f'<circle cx="{x_map(step):.2f}" cy="{y_map(value):.2f}" r="2.7" fill="{color}"/>')
            elements.append(svg_text(bounds[0] + 12 + line_index * 145, bounds[1] + 17, line_label, size=10, color=color, weight=600))
        if title == "Projection distance":
            gate_y = y_map(3.0)
            elements.append(f'<line x1="{bounds[0]}" y1="{gate_y:.2f}" x2="{bounds[0] + bounds[2]}" y2="{gate_y:.2f}" stroke="#111827" stroke-dasharray="2 4"/>')
            elements.append(svg_text(bounds[0] + bounds[2] - 5, gate_y - 5, "3 cm gate", size=9, anchor="end"))
    write_svg(output_path, 1400, 850, "ContactDiffusion n=2 Franka grasp-quality validation (0-52k)", elements)


def write_scalars_csv(series: dict[str, list[tuple[int, float]]], output_path: Path) -> None:
    selected = {
        tag: points
        for tag, points in series.items()
        if tag.startswith("train/loss_") or tag.startswith("val/loss_") or tag.startswith("val/grasp_")
    }
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tag", "step", "value"))
        for tag in sorted(selected):
            for step, value in selected[tag]:
                writer.writerow((tag, step, f"{value:.12g}"))


def build_dataset(cfg, max_samples: int):
    return build_contact_format_dataset(
        root_dir=cfg.dataset.root_dir,
        dataset_dir=cfg.dataset.dataset_dirs,
        split="val",
        n=2,
        num_points=int(cfg.dataset.num_points),
        contact_field=str(cfg.dataset.contact_field),
        load_cmap=False,
        load_qpos=False,
        normalize=bool(cfg.dataset.normalize),
        split_fractions=tuple(cfg.dataset.split_fractions),
        split_names=tuple(cfg.dataset.split_names),
        max_samples=max_samples,
        seed=int(cfg.train.seed),
        index_cache_dir=str(cfg.dataset.index_cache_dir),
        shard_cache_size=int(cfg.dataset.shard_cache_size),
        object_pc_asset_keys=tuple(cfg.dataset.object_pc_asset_keys),
        success_only=bool(cfg.dataset.success_only),
        max_projection_distance=cfg.dataset.max_projection_distance,
        allowed_grippers=list(cfg.dataset.allowed_grippers),
        native_n_filter=bool(cfg.dataset.native_n_filter),
    )


def choose_diverse_items(dataset, count: int) -> list[dict]:
    candidate_count = min(len(dataset), max(count * 24, 256))
    candidate_indices = np.linspace(0, len(dataset) - 1, candidate_count, dtype=np.int64)
    groups: dict[str, list[dict]] = {}
    for index in candidate_indices:
        item = dataset[int(index)]
        object_name = item.get("object_name") or item["path"]
        groups.setdefault(object_name, []).append(item)
    selected = []
    depth = 0
    while len(selected) < count:
        added = False
        for items in groups.values():
            if depth < len(items):
                selected.append(items[depth])
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
        depth += 1
    return selected


def evaluate_contact_set(cloud, raw_contacts, normals, quality_cfg) -> tuple[dict, np.ndarray, np.ndarray]:
    match = match_predicted_contacts_to_point_cloud(
        raw_contacts,
        cloud,
        max_projection_distance=quality_cfg.get("max_projection_distance"),
        max_distance_factor=float(quality_cfg.get("max_projection_distance_factor", 4.0)),
    )
    indices = np.asarray(match.get("predicted_indices", []), dtype=np.int64)
    projected = cloud[indices] if len(indices) else np.empty((0, 3), dtype=np.float64)
    contact_normals = normals[indices] if len(indices) else np.empty((0, 3), dtype=np.float64)
    if not match["valid"]:
        return (
            {
                "valid": False,
                "force_closure": False,
                "epsilon": 0.0,
                "projection_distances": np.asarray(match["projection_distances"]),
                "failure_reason": match.get("failure_reason", "projection failed"),
            },
            projected,
            contact_normals,
        )
    result = PointCloudMultiContactGraspEvaluator.evaluate(
        cloud,
        indices,
        normals,
        friction_coef=float(quality_cfg.get("friction_coef", 0.5)),
        num_cone_faces=int(quality_cfg.get("num_cone_faces", 8)),
        soft_fingers=bool(quality_cfg.get("soft_fingers", True)),
        finger_radius=float(quality_cfg.get("finger_radius", 0.005)),
        torque_scaling=quality_cfg.get("torque_scaling"),
        center_of_mass=None,
        fixed_total_force_budget=bool(quality_cfg.get("fixed_total_force_budget", True)),
    )
    result = dict(result)
    result["projection_distances"] = np.asarray(match["projection_distances"])
    return result, projected, contact_normals


def rounded(array, digits=5):
    return np.round(np.asarray(array, dtype=np.float64), digits).tolist()


def generate_contact_samples(ckpt_path: Path, args: argparse.Namespace) -> tuple[list[dict], dict]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    checkpoint_step = int(checkpoint.get("step", 0))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    dataset = build_dataset(cfg, max_samples=max(512, args.num_visual_samples * 32))
    items = choose_diverse_items(dataset, int(args.num_visual_samples))
    clouds_tensor = torch.stack([item["object_pc"] for item in items]).to(device)
    model = ContactDiffusion.from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        predicted = model.sample(
            object_pc=clouds_tensor,
            num_contacts=2,
            dc=3,
            num_steps=int(args.num_steps),
            sampler=str(args.sampler),
            project_to_surface=False,
        )
    predictions = predicted.detach().cpu().numpy()
    quality_cfg = OmegaConf.to_container(cfg.validation.grasp_quality, resolve=True)
    visual_samples = []
    for index, (item, raw_contacts) in enumerate(zip(items, predictions)):
        cloud = item["object_pc"].numpy().astype(np.float64)
        gt_contacts = item["contacts"].numpy().astype(np.float64)
        if "object_normals" in item:
            normals = item["object_normals"].numpy().astype(np.float64)
            normal_source = "asset"
        else:
            normals = estimate_point_cloud_normals(
                cloud,
                k_neighbors=int(quality_cfg.get("normal_k_neighbors", 30)),
                viewpoint=quality_cfg.get("normal_viewpoint"),
            )
            normal_source = "PCA k=30"
        result, projected, contact_normals = evaluate_contact_set(
            cloud, raw_contacts, normals, quality_cfg
        )
        gt_result, gt_projected, _ = evaluate_contact_set(cloud, gt_contacts, normals, quality_cfg)
        projection_distances = np.asarray(result.get("projection_distances", []), dtype=np.float64)
        bbox_diagonal = float(np.linalg.norm(cloud.max(axis=0) - cloud.min(axis=0)))
        visual_samples.append(
            {
                "index": index,
                "object_name": item.get("object_name") or f"sample_{index:02d}",
                "robot_name": item.get("robot_name", "franka_panda"),
                "source": item.get("path", ""),
                "point_cloud": rounded(cloud),
                "raw_contacts": rounded(raw_contacts, 6),
                "projected_contacts": rounded(projected, 6),
                "contact_normals": rounded(contact_normals, 6),
                "gt_contacts": rounded(gt_projected, 6),
                "center": rounded(cloud.mean(axis=0), 6),
                "bbox_diagonal": bbox_diagonal,
                "projection_valid": bool(result.get("valid", False)),
                "force_closure": bool(result.get("force_closure", False)),
                "epsilon": float(result.get("epsilon", 0.0)),
                "projection_mean_m": float(projection_distances.mean()) if projection_distances.size else 0.0,
                "projection_max_m": float(projection_distances.max()) if projection_distances.size else 0.0,
                "failure_reason": result.get("failure_reason"),
                "gt_force_closure": bool(gt_result.get("force_closure", False)),
                "gt_epsilon": float(gt_result.get("epsilon", 0.0)),
                "normal_source": normal_source,
            }
        )
    metadata = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": checkpoint_step,
        "num_samples": len(visual_samples),
        "num_contacts": 2,
        "sampler": args.sampler,
        "sampling_steps": int(args.num_steps),
        "seed": int(args.seed),
        "friction_coef": float(quality_cfg.get("friction_coef", 0.5)),
        "num_cone_faces": int(quality_cfg.get("num_cone_faces", 8)),
        "projection_gate_m": quality_cfg.get("max_projection_distance"),
        "force_closure_count": sum(sample["force_closure"] for sample in visual_samples),
        "projection_valid_count": sum(sample["projection_valid"] for sample in visual_samples),
        "epsilon_mean": float(np.mean([sample["epsilon"] for sample in visual_samples])),
    }
    return visual_samples, metadata


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#f5f7f9;--panel:#fff;--ink:#17212b;--muted:#66717d;--line:#d8dee5;--ok:#137a4e;--bad:#b42318;--accent:#176b87}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;letter-spacing:0}}
header{{padding:22px 28px 18px;background:#fff;border-bottom:1px solid var(--line)}} h1{{font-size:22px;margin:0 0 7px}} .subtitle{{font-size:13px;color:var(--muted)}}
main{{max-width:1480px;margin:0 auto;padding:18px 22px 36px}} .summary{{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:10px;margin-bottom:14px}}
.metric{{background:var(--panel);border:1px solid var(--line);padding:12px 14px;border-radius:6px}} .metric b{{display:block;font-size:20px;margin-top:3px}} .metric span{{font-size:12px;color:var(--muted)}}
.workspace{{display:grid;grid-template-columns:minmax(540px,1.6fr) minmax(300px,.7fr);gap:14px}} .viewer,.side{{background:var(--panel);border:1px solid var(--line);border-radius:6px}}
.toolbar{{height:54px;display:flex;align-items:center;gap:8px;padding:8px 10px;border-bottom:1px solid var(--line);flex-wrap:wrap}} button,select{{height:34px;border:1px solid #bcc5cf;background:#fff;color:var(--ink);border-radius:5px;padding:0 10px;font:inherit;font-size:13px}} button{{cursor:pointer}} button:hover{{border-color:#6b7b8a}} select{{min-width:220px;flex:1}}
.check{{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:4px;white-space:nowrap}} canvas{{display:block;width:100%;height:min(68vh,720px);background:#fff;cursor:grab}} canvas:active{{cursor:grabbing}}
.side{{padding:16px}} .status{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:12px}} .status h2{{font-size:17px;margin:0;overflow-wrap:anywhere}} .badge{{border-radius:999px;padding:5px 9px;color:#fff;font-weight:700;font-size:11px;white-space:nowrap}} .ok{{background:var(--ok)}} .bad{{background:var(--bad)}}
.details{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line)}} .details div{{background:#fff;padding:10px}} .details span{{display:block;color:var(--muted);font-size:11px}} .details b{{display:block;font-size:14px;margin-top:3px;overflow-wrap:anywhere}}
.legend{{display:flex;gap:13px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:14px 0}} .swatch{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}}
.sample-table{{margin-top:14px;background:#fff;border:1px solid var(--line);border-radius:6px;overflow:auto}} table{{width:100%;border-collapse:collapse;font-size:12px}} th,td{{text-align:left;padding:9px 11px;border-bottom:1px solid #e8ecf0;white-space:nowrap}} th{{color:var(--muted);font-weight:600;background:#fafbfc}} tbody tr{{cursor:pointer}} tbody tr:hover,tbody tr.active{{background:#eef6f8}}
.source{{margin-top:12px;font-size:11px;color:var(--muted);overflow-wrap:anywhere}} @media(max-width:900px){{.summary{{grid-template-columns:1fr 1fr}}.workspace{{grid-template-columns:1fr}}canvas{{height:58vh}}}}
</style>
</head>
<body>
<header><h1>{title}</h1><div class="subtitle">Checkpoint step {step} · n=2 · {sampler_upper} {sampling_steps} steps · self-contained canvas report</div></header>
<main>
<section class="summary">
  <div class="metric"><span>Force closure</span><b id="overallFc">{fc_count}/{sample_count}</b></div>
  <div class="metric"><span>Projection valid</span><b>{valid_count}/{sample_count}</b></div>
  <div class="metric"><span>Mean epsilon</span><b>{epsilon_mean}</b></div>
  <div class="metric"><span>Friction coefficient</span><b>{friction}</b></div>
</section>
<section class="workspace">
  <div class="viewer">
    <div class="toolbar">
      <button id="prev" title="Previous sample">&#8592;</button><button id="next" title="Next sample">&#8594;</button>
      <select id="sampleSelect" aria-label="Sample"></select><button id="reset" title="Reset view">Reset</button>
      <label class="check"><input id="showRaw" type="checkbox" checked>Raw</label>
      <label class="check"><input id="showProjected" type="checkbox" checked>Projected</label>
      <label class="check"><input id="showGt" type="checkbox">GT</label>
      <label class="check"><input id="showNormals" type="checkbox" checked>Normals</label>
    </div>
    <canvas id="canvas"></canvas>
  </div>
  <aside class="side">
    <div class="status"><h2 id="objectName"></h2><span id="fcBadge" class="badge"></span></div>
    <div class="details">
      <div><span>Epsilon</span><b id="epsilon"></b></div><div><span>GT epsilon</span><b id="gtEpsilon"></b></div>
      <div><span>Projection mean</span><b id="projMean"></b></div><div><span>Projection max</span><b id="projMax"></b></div>
      <div><span>Projection gate</span><b>{projection_gate_cm} cm</b></div><div><span>GT force closure</span><b id="gtFc"></b></div>
      <div><span>Normals</span><b id="normalSource"></b></div><div><span>Gripper</span><b id="gripper"></b></div>
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:#80909e"></i>Object PC</span><span><i class="swatch" style="background:#f59e0b"></i>Raw generated</span>
      <span><i class="swatch" style="background:#dc2626"></i>Projected</span><span><i class="swatch" style="background:#7c3aed"></i>GT</span><span><i class="swatch" style="background:#15803d"></i>Normal</span>
    </div>
    <div class="source" id="source"></div>
  </aside>
</section>
<section class="sample-table"><table><thead><tr><th>#</th><th>Object</th><th>Projection valid</th><th>Force closure</th><th>Epsilon</th><th>Projection max</th><th>GT closure</th></tr></thead><tbody id="sampleRows"></tbody></table></section>
</main>
<script id="reportData" type="application/json">{report_json}</script>
<script>
const report=JSON.parse(document.getElementById('reportData').textContent), samples=report.samples;
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'); let current=0,yaw=-0.65,pitch=0.35,zoom=1.0,drag=false,lastX=0,lastY=0;
const select=document.getElementById('sampleSelect'); samples.forEach((s,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`${{String(i+1).padStart(2,'0')}} · ${{s.object_name||'unnamed'}}`;select.appendChild(o)}});
const fmt=(v,d=5)=>Number(v).toExponential(d), cm=v=>`${{(100*Number(v)).toFixed(2)}} cm`;
function resize(){{const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);canvas.width=Math.max(1,Math.floor(r.width*d));canvas.height=Math.max(1,Math.floor(r.height*d));ctx.setTransform(d,0,0,d,0,0);draw()}}
function rotate(p,c){{let x=p[0]-c[0],y=p[1]-c[1],z=p[2]-c[2];let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);let x1=cy*x+sy*z,z1=-sy*x+cy*z;return [x1,cp*y-sp*z1,sp*y+cp*z1]}}
function projector(s){{const w=canvas.clientWidth,h=canvas.clientHeight,c=s.center,scale=Math.min(w,h)*0.39*zoom/Math.max(s.bbox_diagonal,.001);return p=>{{const q=rotate(p,c),persp=4/(4-q[2]/Math.max(s.bbox_diagonal,.001));return [w/2+q[0]*scale*persp,h/2-q[1]*scale*persp,q[2]]}}}}
function dot(p,r,color,stroke=null){{ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);if(color){{ctx.fillStyle=color;ctx.fill()}}if(stroke){{ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke()}}}}
function draw(){{if(!samples.length)return;const s=samples[current],project=projector(s),w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);const cloud=s.point_cloud.map(p=>[p,project(p)]).sort((a,b)=>a[1][2]-b[1][2]);for(const [,q] of cloud)dot(q,1.15,'rgba(100,119,134,.48)');
if(document.getElementById('showGt').checked)for(const p of s.gt_contacts)dot(project(p),7,'rgba(124,58,237,.18)','#7c3aed');
if(document.getElementById('showProjected').checked){{const q=s.projected_contacts.map(project);if(q.length===2){{ctx.beginPath();ctx.moveTo(q[0][0],q[0][1]);ctx.lineTo(q[1][0],q[1][1]);ctx.strokeStyle=s.force_closure?'#137a4e':'#b42318';ctx.lineWidth=2.5;ctx.stroke()}}for(const p of q)dot(p,7,'#dc2626','#fff')}}
if(document.getElementById('showRaw').checked)for(const p of s.raw_contacts)dot(project(p),5,'#f59e0b','#fff');
if(document.getElementById('showNormals').checked){{const L=s.bbox_diagonal*.16;s.projected_contacts.forEach((p,i)=>{{const n=s.contact_normals[i];if(!n)return;const a=project(p),b=project([p[0]+n[0]*L,p[1]+n[1]*L,p[2]+n[2]*L]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle='#15803d';ctx.lineWidth=2;ctx.stroke();dot(b,2.5,'#15803d')}})}}}}
function update(){{const s=samples[current];select.value=current;document.getElementById('objectName').textContent=s.object_name||`Sample ${{current+1}}`;const b=document.getElementById('fcBadge');b.textContent=s.force_closure?'FORCE CLOSURE':'NOT CLOSED';b.className='badge '+(s.force_closure?'ok':'bad');document.getElementById('epsilon').textContent=fmt(s.epsilon);document.getElementById('gtEpsilon').textContent=fmt(s.gt_epsilon);document.getElementById('projMean').textContent=cm(s.projection_mean_m);document.getElementById('projMax').textContent=cm(s.projection_max_m);document.getElementById('gtFc').textContent=s.gt_force_closure?'YES':'NO';document.getElementById('normalSource').textContent=s.normal_source;document.getElementById('gripper').textContent=s.robot_name;document.getElementById('source').textContent=s.source;document.querySelectorAll('#sampleRows tr').forEach((r,i)=>r.classList.toggle('active',i===current));draw()}}
const tbody=document.getElementById('sampleRows');samples.forEach((s,i)=>{{const tr=document.createElement('tr');tr.innerHTML=`<td>${{i+1}}</td><td></td><td>${{s.projection_valid?'YES':'NO'}}</td><td>${{s.force_closure?'YES':'NO'}}</td><td>${{fmt(s.epsilon,3)}}</td><td>${{cm(s.projection_max_m)}}</td><td>${{s.gt_force_closure?'YES':'NO'}}</td>`;tr.children[1].textContent=s.object_name||'unnamed';tr.onclick=()=>{{current=i;update()}};tbody.appendChild(tr)}});
select.onchange=()=>{{current=Number(select.value);update()}};document.getElementById('prev').onclick=()=>{{current=(current-1+samples.length)%samples.length;update()}};document.getElementById('next').onclick=()=>{{current=(current+1)%samples.length;update()}};document.getElementById('reset').onclick=()=>{{yaw=-.65;pitch=.35;zoom=1;draw()}};
for(const id of ['showRaw','showProjected','showGt','showNormals'])document.getElementById(id).onchange=draw;
canvas.onpointerdown=e=>{{drag=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)}};canvas.onpointermove=e=>{{if(!drag)return;yaw+=(e.clientX-lastX)*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-lastY)*.008));lastX=e.clientX;lastY=e.clientY;draw()}};canvas.onpointerup=()=>drag=false;canvas.onpointercancel=()=>drag=false;canvas.onwheel=e=>{{e.preventDefault();zoom=Math.max(.35,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()}},{{passive:false}};
window.addEventListener('resize',resize);resize();update();
</script>
</body></html>"""


def write_html(samples: list[dict], metadata: dict, output_path: Path) -> None:
    payload = json.dumps({"metadata": metadata, "samples": samples}, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    gate = metadata.get("projection_gate_m")
    document = HTML_TEMPLATE.format(
        title="ContactDiffusion 52k contact generation",
        step=metadata["checkpoint_step"],
        sampler_upper=html.escape(str(metadata["sampler"]).upper()),
        sampling_steps=metadata["sampling_steps"],
        fc_count=metadata["force_closure_count"],
        valid_count=metadata["projection_valid_count"],
        sample_count=metadata["num_samples"],
        epsilon_mean=f"{metadata['epsilon_mean']:.4e}",
        friction=f"{metadata['friction_coef']:.2f}",
        projection_gate_cm=f"{float(gate) * 100.0:.1f}" if gate is not None else "adaptive",
        report_json=payload,
    )
    output_path.write_text(document, encoding="utf-8")


def metric_value(series, tag, step=None):
    points = series.get(tag, [])
    if not points:
        return None
    if step is None:
        return points[-1][1]
    by_step = dict(points)
    return by_step.get(step)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "checkpoints" / "latest.pt"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "visualizations_52k"
    output_dir.mkdir(parents=True, exist_ok=True)

    series, event_files = load_scalar_series(run_dir / "tensorboard")
    plot_losses(series, output_dir / "loss_noise_chamfer_0_52k.svg", args.smoothing_window)
    plot_grasp_metrics(series, output_dir / "grasp_quality_metrics_0_52k.svg")
    write_scalars_csv(series, output_dir / "training_scalars_0_52k.csv")
    samples, sample_metadata = generate_contact_samples(checkpoint, args)
    write_html(samples, sample_metadata, output_dir / "contact_generation_52k.html")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "event_files": event_files,
        "deduplication": "later event file wins for duplicate tag+step",
        "train_noise_points": len(series.get("train/loss_noise", [])),
        "train_chamfer_points": len(series.get("train/loss_chamfer", [])),
        "validation_points": len(series.get("val/grasp_force_closure_rate_n2", [])),
        "step_52000": {
            "train_noise": metric_value(series, "train/loss_noise", 52000),
            "train_chamfer": metric_value(series, "train/loss_chamfer", 52000),
            "val_noise": metric_value(series, "val/loss_noise_n2", 52000),
            "val_chamfer": metric_value(series, "val/loss_chamfer_n2", 52000),
            "force_closure_rate": metric_value(series, "val/grasp_force_closure_rate_n2", 52000),
            "epsilon_mean": metric_value(series, "val/grasp_epsilon_mean_n2", 52000),
            "projection_valid_rate": metric_value(series, "val/grasp_projection_valid_rate_n2", 52000),
        },
        "visual_samples": sample_metadata,
    }
    (output_dir / "visualization_summary_52k.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote visualizations to {output_dir}")


if __name__ == "__main__":
    main()
