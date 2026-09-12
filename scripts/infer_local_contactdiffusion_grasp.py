#!/usr/bin/env python3
"""Run ContactDiffusion and differentiable gripper FK from local object assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
cache_root = REPO_ROOT / ".cache"
matplotlib_cache = cache_root / "matplotlib"
torch_kernel_cache = cache_root / "torch" / "kernels"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
torch_kernel_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault(
    "PYTORCH_KERNEL_CACHE_PATH", str(torch_kernel_cache)
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from omegaconf import OmegaConf

if os.environ.get("CONTACTDIFF_MISH_COMPOSITE", "0") == "1":
    # Some otherwise healthy CUDA devices fail in the fused Mish kernel from
    # the legacy PyTorch build used on the remote platform. This is the exact
    # Mish formula expressed with ordinary CUDA ops and remains differentiable.
    import torch.nn.functional as torch_functional

    def _composite_mish(input_tensor, inplace=False):
        del inplace
        return input_tensor * torch.tanh(torch_functional.softplus(input_tensor))

    torch_functional.mish = _composite_mish

from models.diffusion import ContactDiffusion
from models.autoregressive_contact_diffusion import AutoregressiveContactDiffusion
from utils.multigripper_fk import (
    load_gripper_from_calibration,
    optimize_gripper_to_contacts,
    ordered_joint_values,
    resolve_local_grasp_approach_axis,
    resolve_local_palm_axis,
)
from utils.point_cloud_geometry import (
    estimate_point_cloud_geometry,
    prepare_point_cloud_inputs,
)
from utils.contact_set_guidance import (
    MATCHER_VERSION,
    matched_random_surface_contacts,
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/multigripper_fk_isaac_local_physics.yaml"
    )
    parser.add_argument(
        "--assignment-temperature",
        type=float,
        default=0.005,
        help="Soft one-to-one contact assignment temperature in metres.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/isaacsim_local_assets/training_run/checkpoints/step_00050000.pt",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/isaacsim_local_assets/fixed_three_grippers_candidates_step50000_local.json",
        help="Existing local manifest used to resolve object IDs and XYZ point clouds.",
    )
    parser.add_argument("--object-id", action="append", default=[])
    parser.add_argument("--all-objects", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Do not append objects discovered in the local asset cache.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed gripper/object/sample keys from --output.",
    )
    parser.add_argument("--object-pc", default=None)
    parser.add_argument(
        "--inference-object-observation",
        choices=("checkpoint", "full", "synthetic_partial"),
        default="checkpoint",
        help=(
            "Object condition used at inference. 'checkpoint' follows the "
            "checkpoint dataset mode; 'full' explicitly disables synthetic "
            "partial cropping while preserving the checkpoint normalization."
        ),
    )
    parser.add_argument(
        "--object-mesh",
        default=None,
        help="Optional simulator asset; never used by FK optimization energies.",
    )
    parser.add_argument("--grippers", nargs="+", default=["franka_panda"])
    parser.add_argument("--samples-per-object", type=int, default=1)
    parser.add_argument(
        "--autoregressive-fk-target",
        choices=("nearest_2048", "raw_xyz"),
        default="nearest_2048",
        help=(
            "For AR checkpoints, optimize FK against either the nearest point "
            "in the complete 2048-point object cloud or the unprojected raw XYZ."
        ),
    )
    parser.add_argument(
        "--contact-target-mode",
        choices=("diffusion", "matched_random"),
        default="diffusion",
        help=(
            "Use projected ContactDiffusion contacts or a deterministic random "
            "surface set matched to their rotation-free spread statistics."
        ),
    )
    parser.add_argument(
        "--contact-weight",
        type=float,
        default=None,
        help=(
            "Optional override for fk_optimization.contact_weight. This is "
            "intended for paired contact-energy ablations that keep the "
            "checkpoint, contacts, initialization, and all other FK terms fixed."
        ),
    )
    parser.add_argument(
        "--contact-normal-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for opposing outward surface-normal alignment between "
            "each matched distal fingertip sample and generated contact."
        ),
    )
    parser.add_argument(
        "--object-surface-distance-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the sum of selected hand-contact distances to the "
            "object point-cloud surface (DexGraspNet E_dis port)."
        ),
    )
    parser.add_argument("--penetration-weight", type=float)
    parser.add_argument("--self-collision-weight", type=float)
    parser.add_argument("--self-collision-clearance", type=float)
    parser.add_argument("--joint-regularization", type=float)
    parser.add_argument(
        "--disable-palm-selection-gate",
        action="store_true",
        help="Do not use the configured palm-distance gate for final ranking.",
    )
    parser.add_argument(
        "--dexgraspnet-sum-reductions",
        action="store_true",
        help=(
            "Use DexGraspNet's sum scale for sampled penetration and "
            "self-penetration energies instead of the project's mean/CVaR scale."
        ),
    )
    parser.add_argument(
        "--contact-geometry-mode",
        choices=("tip_point", "distal_surface", "distal_pad"),
        default=None,
        help="Override the FK contact region; distal_pad is kinematically calibrated.",
    )
    parser.add_argument("--pad-points-per-finger", type=int, default=12)
    parser.add_argument(
        "--pad-softmin-temperature-m", type=float, default=0.002
    )
    parser.add_argument("--closing-direction-weight", type=float, default=0.0)
    parser.add_argument("--closing-direction-margin", type=float, default=0.5)
    parser.add_argument("--sweep-collision-weight", type=float, default=0.0)
    parser.add_argument("--sweep-samples", type=int, default=4)
    parser.add_argument("--sweep-points-per-link", type=int, default=12)
    parser.add_argument("--pad-exclusion-radius-m", type=float, default=0.0)
    parser.add_argument("--closure-outer-fraction", type=float, default=0.10)
    parser.add_argument("--closure-inner-fraction", type=float, default=0.20)
    parser.add_argument(
        "--contact-priority",
        action="store_true",
        help="Use three-stage bottleneck pad-contact optimization.",
    )
    parser.add_argument(
        "--contact-feasibility-threshold-m", type=float, default=0.010
    )
    parser.add_argument("--contact-priority-target-m", type=float, default=0.005)
    parser.add_argument(
        "--contact-priority-temperature-m", type=float, default=0.002
    )
    parser.add_argument("--contact-barrier-weight", type=float, default=1000.0)
    parser.add_argument("--contact-stage1-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--contact-stage2-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--require-contact-feasibility",
        action="store_true",
        help="Retain only candidates whose five pad contacts all pass the threshold.",
    )
    parser.add_argument(
        "--initialization-contact-source",
        choices=("target", "diffusion"),
        default="target",
        help=(
            "Contacts used only to initialize FK particles. Use 'diffusion' "
            "for both A and B to obtain identical paired initial states while "
            "changing the optimized target contacts."
        ),
    )
    parser.add_argument(
        "--source-diffusion-candidates",
        type=Path,
        help=(
            "Optional diffusion-variant candidate JSON supplying the exact "
            "paired source contacts. This can also freeze diffusion contacts "
            "across FK contact-weight ablations."
        ),
    )
    parser.add_argument(
        "--warm-start-candidates",
        type=Path,
        help=(
            "Candidate JSON supplying root/joint states for staged FK refinement. "
            "The selected record must contain exactly --particles candidates."
        ),
    )
    parser.add_argument(
        "--replay-source-diffusion-rng",
        action="store_true",
        help=(
            "When frozen contacts come from --source-diffusion-candidates, run "
            "and discard the corresponding diffusion sample first. This "
            "reproduces the source run's CUDA RNG position before FK "
            "initialization, allowing exact paired initial states."
        ),
    )
    parser.add_argument(
        "--matched-random-candidates",
        type=int,
        default=4096,
        help="Number of random surface sets searched for the matched control.",
    )
    parser.add_argument(
        "--matched-random-seed-offset",
        type=int,
        default=7919,
        help="Independent deterministic seed offset for matched-random targets.",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help=(
            "First sample index to generate. This allows disjoint deterministic "
            "sample ranges to be distributed across multiple GPUs."
        ),
    )
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--optimization-steps", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument(
        "--cedex-palm-axis",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Explicit local palmar surface normal for a paired experiment. "
            "By default the embodiment value declared in the config is used."
        ),
    )
    parser.add_argument(
        "--grasp-approach-axis",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Local finger-extension/enveloping axis. It is independent of "
            "the physical palmar surface normal."
        ),
    )
    parser.add_argument(
        "--grasp-approach-target-mode",
        choices=("object_center", "contact_plane_normal"),
        default="object_center",
        help=(
            "Orient the grasp axis either toward the object center or along "
            "the inward normal of the generated-contact regression plane."
        ),
    )
    parser.add_argument(
        "--grasp-approach-plane-sides",
        choices=("inward_only", "bilateral"),
        default="inward_only",
        help=(
            "For a fitted contact plane, use only the inward normal or retain "
            "a deterministic 50/50 inward/opposite particle split."
        ),
    )
    parser.add_argument(
        "--fk-initialization",
        choices=("kabsch", "cedex", "enveloping"),
        default="kabsch",
        help=(
            "FK particle initialization. The default preserves the existing "
            "contact-set Kabsch path; cedex enables center-facing palm poses; "
            "enveloping mixes Kabsch and top-down palm hypotheses."
        ),
    )
    parser.add_argument("--envelope-side-weight", type=float, default=0.0)
    parser.add_argument("--envelope-approach-weight", type=float, default=2.0)
    parser.add_argument("--envelope-cosine-margin", type=float, default=0.25)
    parser.add_argument("--selection-min-envelope-cosine", type=float, default=0.5)
    parser.add_argument("--selection-min-approach-cosine", type=float, default=0.8)
    parser.add_argument(
        "--enable-legacy-cosine-feasibility-gates",
        action="store_true",
        help=(
            "Re-enable the historical envelope/approach cosine Top-K hard "
            "gates for an explicit legacy baseline replay."
        ),
    )
    parser.add_argument(
        "--preferred-root-direction",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Optional world-space direction from object center to hand root. "
            "By default the root is placed opposite the generated patch."
        ),
    )
    parser.add_argument(
        "--cedex-cleanup-steps",
        type=int,
        default=0,
        help=(
            "Final CEDex-style penetration-only refinement iterations. "
            "Use 50 to match CEDex's final cleanup phase."
        ),
    )
    parser.add_argument(
        "--cedex-joint-init-fraction",
        type=float,
        default=0.5,
        help="Fraction of the open-to-closed joint range sampled at initialization.",
    )
    parser.add_argument(
        "--cedex-cleanup-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate used during the final CEDex physical cleanup.",
    )
    parser.add_argument(
        "--cedex-contact-guard",
        type=float,
        default=0.015,
        help=(
            "Sparse-contact Chamfer hinge in metres retained during CEDex "
            "cleanup; adapts CEDex's dense-map SPF to ContactDiffusion."
        ),
    )
    parser.add_argument(
        "--object-surface-points",
        type=int,
        default=None,
        help="Deprecated; the full XYZ point cloud is retained for FK geometry.",
    )
    parser.add_argument(
        "--selection-max-penetration",
        type=float,
        default=None,
        help=(
            "Prefer final FK particles whose sampled maximum hand-object "
            "penetration is at most this many metres. If none pass, select "
            "the least-penetrating particles."
        ),
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--force-closure-weight",
        type=float,
        default=None,
        help=(
            "Enable the execution-aware FC seventh energy term without "
            "changing the frozen six-term YAML; omitted means YAML/default 0."
        ),
    )
    parser.add_argument("--force-closure-start-fraction", type=float, default=0.75)
    parser.add_argument("--force-closure-gap-sigma-m", type=float, default=0.01)
    parser.add_argument("--force-closure-qp-iterations", type=int, default=20)
    parser.add_argument("--force-closure-qp-weight", type=float, default=1.0)
    parser.add_argument("--force-closure-dfc-weight", type=float, default=0.1)
    parser.add_argument("--force-closure-coverage-weight", type=float, default=0.25)
    parser.add_argument("--force-closure-target-fraction", type=float)
    parser.add_argument(
        "--force-closure-ramp-mode",
        choices=("linear", "constant"),
        default="linear",
    )
    parser.add_argument(
        "--force-closure-formulation",
        choices=("legacy", "dexgraspnet", "graspqp"),
        default="legacy",
    )
    force_closure_palm = parser.add_mutually_exclusive_group()
    force_closure_palm.add_argument(
        "--force-closure-include-palm",
        dest="force_closure_include_palm",
        action="store_true",
    )
    force_closure_palm.add_argument(
        "--no-force-closure-include-palm",
        dest="force_closure_include_palm",
        action="store_false",
    )
    parser.set_defaults(force_closure_include_palm=True)
    parser.add_argument("--force-closure-friction-coefficient", type=float)
    parser.add_argument("--force-closure-cone-edges", type=int)
    parser.add_argument(
        "--force-closure-max-force-coefficient", type=float, default=50.0
    )
    parser.add_argument("--force-closure-torque-weight", type=float, default=5.0)
    parser.add_argument("--force-closure-svd-gain", type=float, default=0.1)
    parser.add_argument(
        "--selection-rank-mode",
        choices=("optimization", "graspqp"),
        default=None,
        help=(
            "Final particle ranking. By default enveloping load-bearing runs "
            "retain legacy GraspQP ranking; pass optimization to exclude it."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--hand-index-offset",
        type=int,
        default=0,
        help="Global hand index added to the local gripper enumeration for seeding.",
    )
    parser.add_argument(
        "--object-index-offset",
        type=int,
        default=0,
        help="Global object index added to the local object enumeration for seeding.",
    )
    parser.add_argument(
        "--output",
        default="outputs/isaacsim_local_inference/local_step50000_candidates.json",
    )
    return parser.parse_args()


def load_manifest_objects(path: Path, dataset_root: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects: list[dict] = []
    seen: set[str] = set()
    for record in payload.get("records", []):
        object_id = str(record.get("object_id") or "")
        if not object_id or object_id in seen:
            continue
        mesh_value = record.get("object_mesh")
        mesh = None if not mesh_value else resolve(mesh_value)
        pc_value = Path(record["object_pc_asset"])
        pc = pc_value if pc_value.is_absolute() else dataset_root / pc_value
        objects.append(
            {
                "object_id": object_id,
                "object_mesh": None if mesh is None else mesh.resolve(),
                "object_pc": pc.resolve(),
            }
        )
        seen.add(object_id)
    if not objects:
        raise ValueError(f"No object records found in {path}")
    return objects


def discover_paired_local_objects(config: dict) -> list[dict]:
    paths = config.get("paths", {})
    pc_root_value = paths.get("object_pc_root")
    mesh_root_value = paths.get("object_mesh_root")
    if not pc_root_value:
        return []
    pc_root = resolve(pc_root_value)
    mesh_root = None if not mesh_root_value else resolve(mesh_root_value)
    objects = []
    point_clouds = sorted(pc_root.glob("*.npy")) + sorted(pc_root.glob("*.npz"))
    for pc in point_clouds:
        mesh = None if mesh_root is None else mesh_root / f"{pc.stem}.obj"
        objects.append(
            {
                "object_id": pc.stem,
                "object_mesh": (
                    None if mesh is None or not mesh.is_file() else mesh.resolve()
                ),
                "object_pc": pc.resolve(),
            }
        )
    return objects


def select_objects(args: argparse.Namespace, config: dict) -> list[dict]:
    explicit = args.object_pc is not None or args.object_mesh is not None
    if explicit:
        if not args.object_pc:
            raise ValueError("--object-pc is required; --object-mesh is optional")
        pc = resolve(args.object_pc)
        mesh = None if not args.object_mesh else resolve(args.object_mesh)
        object_id = args.object_id[0] if args.object_id else pc.stem
        return [{"object_id": object_id, "object_pc": pc, "object_mesh": mesh}]

    objects = load_manifest_objects(
        resolve(args.manifest), resolve(config["paths"]["dataset_root"])
    )
    if args.manifest_only:
        if args.all_objects:
            return objects
        requested = args.object_id
        if not requested:
            return objects[:1]
        by_id = {item["object_id"]: item for item in objects}
        missing = [value for value in requested if value not in by_id]
        if missing:
            raise KeyError(f"Object IDs not present in manifest: {missing}")
        return [by_id[value] for value in requested]
    seen = {item["object_id"] for item in objects}
    objects.extend(
        item
        for item in discover_paired_local_objects(config)
        if item["object_id"] not in seen
    )
    if args.all_objects:
        return objects
    requested = args.object_id
    if not requested:
        return objects[:1]
    by_id = {item["object_id"]: item for item in objects}
    missing = [value for value in requested if value not in by_id]
    if missing:
        raise KeyError(f"Object IDs not present in manifest: {missing}")
    return [by_id[value] for value in requested]


def load_object_pc(
    path: Path, num_points: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the model-sized cloud and the unabridged XYZ geometry cloud."""

    return prepare_point_cloud_inputs(
        path, model_point_count=int(num_points), seed=int(seed)
    )


def synthetic_partial_observation(
    full_object_pc: torch.Tensor,
    *,
    output_points: int,
    keep_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    """Mirror the training dataset's deterministic synthetic partial crop.

    A random view direction keeps the front-most ``keep_ratio`` of the full
    cloud and resamples that visible subset to the model's fixed point count.
    Normalization is deliberately not performed here: partial-AR inference
    must normalize this observation with the complete cloud's center/radius.
    """

    if full_object_pc.ndim != 2 or full_object_pc.shape[1] != 3:
        raise ValueError(
            "full_object_pc must have shape (M, 3), got "
            f"{tuple(full_object_pc.shape)}"
        )
    if len(full_object_pc) == 0:
        raise ValueError("full_object_pc must not be empty")
    if int(output_points) <= 0:
        raise ValueError("output_points must be positive")
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")

    full_xyz = (
        full_object_pc.detach().cpu().to(dtype=torch.float32).numpy()
    )
    rng = np.random.default_rng(int(seed))
    direction = rng.normal(size=3).astype(np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    centered = full_xyz - full_xyz.mean(axis=0, keepdims=True)
    depth = centered @ direction
    keep = max(
        1,
        min(len(full_xyz), int(np.ceil(len(full_xyz) * float(keep_ratio)))),
    )
    if keep == len(full_xyz):
        source_indices = np.arange(len(full_xyz), dtype=np.int64)
    else:
        source_indices = np.argpartition(depth, len(full_xyz) - keep)[-keep:]
        source_indices = np.asarray(source_indices, dtype=np.int64)
    relative_indices = rng.choice(
        keep,
        size=int(output_points),
        replace=keep < int(output_points),
    ).astype(np.int64)
    sampled_indices = source_indices[relative_indices]
    partial = torch.as_tensor(
        full_xyz[sampled_indices],
        device=full_object_pc.device,
        dtype=full_object_pc.dtype,
    )
    metadata = {
        "mode": "synthetic_partial",
        "seed": int(seed),
        "keep_ratio": float(keep_ratio),
        "source_point_count": int(len(full_xyz)),
        "visible_source_point_count": int(keep),
        "output_point_count": int(output_points),
        "view_direction": direction.tolist(),
        "visible_source_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(source_indices).tobytes()
        ).hexdigest(),
        "sampled_full_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(sampled_indices).tobytes()
        ).hexdigest(),
    }
    return partial, metadata


@torch.inference_mode()
def sample_contact_targets(
    model: torch.nn.Module,
    condition_pc_world: torch.Tensor,
    *,
    normalization_reference_world: torch.Tensor | None = None,
    surface_pc_world: torch.Tensor | None = None,
    num_contacts: int,
    num_steps: int,
    sampler: str,
    normalize_model_inputs: bool,
    autoregressive: bool,
    autoregressive_fk_target: str = "nearest_2048",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Sample contacts while separating condition, normalization and surface.

    The AR checkpoint was trained in a centered unit-radius frame and predicts
    unconstrained XYZ.  The frozen v4 grasp pipeline, however, consumes object
    surface targets. We therefore normalize the condition using the complete
    reference cloud, denormalize AR outputs, and project onto the complete
    surface cloud. Defaults preserve the legacy full-cloud call behavior.
    """

    normalization_reference_world = (
        condition_pc_world
        if normalization_reference_world is None
        else normalization_reference_world
    )
    surface_pc_world = (
        condition_pc_world if surface_pc_world is None else surface_pc_world
    )

    if normalize_model_inputs:
        center = normalization_reference_world.mean(dim=0)
        centered_reference = normalization_reference_world - center.unsqueeze(0)
        scale = torch.linalg.vector_norm(centered_reference, dim=-1).max()
        if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
            scale = normalization_reference_world.new_tensor(1.0)
        model_object_pc = (condition_pc_world - center.unsqueeze(0)) / scale
    else:
        center = normalization_reference_world.new_zeros(3)
        scale = normalization_reference_world.new_tensor(1.0)
        model_object_pc = condition_pc_world

    raw_model_contacts = model.sample(
        object_pc=model_object_pc.unsqueeze(0),
        num_contacts=int(num_contacts),
        dc=3,
        num_steps=int(num_steps),
        sampler=str(sampler),
        project_to_surface=not autoregressive,
    )[0, :, :3]
    raw_world_contacts = raw_model_contacts * scale + center.unsqueeze(0)

    if autoregressive:
        distances = torch.cdist(
            raw_world_contacts.unsqueeze(0), surface_pc_world.unsqueeze(0)
        )[0]
        nearest_distance, nearest_index = distances.min(dim=1)
        if autoregressive_fk_target == "nearest_2048":
            target_contacts = surface_pc_world[nearest_index]
        elif autoregressive_fk_target == "raw_xyz":
            target_contacts = raw_world_contacts
        else:
            raise ValueError(
                f"unsupported AR FK target: {autoregressive_fk_target}"
            )
    else:
        target_contacts = raw_world_contacts
        nearest_distance = torch.linalg.vector_norm(
            target_contacts
            - surface_pc_world[
                torch.cdist(
                    target_contacts.unsqueeze(0), surface_pc_world.unsqueeze(0)
                )[0].argmin(dim=1)
            ],
            dim=-1,
        )

    normalization = {
        "enabled": bool(normalize_model_inputs),
        "center_m": center.detach().cpu().tolist(),
        "scale_m": float(scale.detach().cpu()),
        "reference": (
            "full_object_pc" if normalize_model_inputs else "not_applied"
        ),
    }
    return target_contacts, raw_world_contacts, nearest_distance, normalization


def training_observation_supports(dataset, observation: str) -> bool:
    """Check actual mixture support without changing the inference condition."""
    mode = str(dataset.get("object_observation_mode", "full"))
    if mode == "mixed_full_synthetic_real_partial":
        real = float(dataset.get("real_partial_probability", 0.0))
        synthetic = float(dataset.get(
            "synthetic_partial_probability", dataset.get("partial_probability", 0.0)
        ))
        if not (0.0 <= real <= 1.0 and 0.0 <= synthetic <= 1.0
                and real + synthetic <= 1.0):
            raise ValueError("Invalid full/synthetic/real-partial mixture probabilities")
        return {
            "full": real + synthetic < 1.0,
            "synthetic_partial": synthetic > 0.0,
            "real_partial": real > 0.0,
        }.get(observation, False)
    if mode == "mixed_full_synthetic_partial":
        partial = float(dataset.get("partial_probability", 0.0))
        return (
            observation == "full" and partial < 1.0
            or observation == "synthetic_partial" and partial > 0.0
        )
    return mode == observation


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    checkpoint_path = resolve(args.checkpoint)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_cfg = OmegaConf.create(checkpoint["config"])
    is_autoregressive = (
        checkpoint.get("model_type") == "autoregressive_contact_diffusion"
    )
    normalize_model_inputs = bool(model_cfg.dataset.normalize)
    checkpoint_object_observation_mode = str(
        model_cfg.dataset.get("object_observation_mode", "full")
    )
    object_observation_mode = (
        checkpoint_object_observation_mode
        if args.inference_object_observation == "checkpoint"
        else str(args.inference_object_observation)
    )
    partial_keep_ratio = float(
        model_cfg.dataset.get("synthetic_partial_keep_ratio", 1.0)
    )
    partial_normalization_reference = str(
        model_cfg.dataset.get("partial_normalization_reference", "visible")
    )
    uses_synthetic_partial = bool(
        is_autoregressive and object_observation_mode == "synthetic_partial"
    )
    checkpoint_uses_synthetic_partial = bool(
        is_autoregressive
        and checkpoint_object_observation_mode
        in {
            "synthetic_partial", "mixed_full_synthetic_partial",
            "mixed_full_synthetic_real_partial",
        }
    )
    if checkpoint_uses_synthetic_partial:
        if int(model_cfg.model.get("object_input_dim", -1)) != 3:
            raise ValueError("partial-AR v5 requires model.object_input_dim=3")
        if list(model_cfg.dataset.get("n_values", [])) != [2, 3, 5]:
            raise ValueError("AR v5 inference requires dataset.n_values=[2, 3, 5]")
        if str(model_cfg.diffusion.get("prediction_type", "")) != "v_prediction":
            raise ValueError("AR v5 inference requires diffusion v_prediction")
        if not normalize_model_inputs:
            raise ValueError("partial-AR v5 requires dataset.normalize=true")
        if partial_normalization_reference != "full":
            raise ValueError(
                "partial-AR v5 requires partial_normalization_reference=full"
            )
        if not 0.0 < partial_keep_ratio <= 1.0:
            raise ValueError("synthetic_partial_keep_ratio must be in (0, 1]")
        if (
            checkpoint_object_observation_mode
            == "mixed_full_synthetic_partial"
            and not 0.0
            < float(model_cfg.dataset.get("partial_probability", -1.0))
            < 1.0
        ):
            raise ValueError(
                "mixed full/partial AR checkpoint requires "
                "dataset.partial_probability in (0, 1)"
            )
    if object_observation_mode not in {"full", "synthetic_partial"}:
        raise ValueError(
            "Direct inference supports only full or synthetic_partial object "
            f"observations, got {object_observation_mode!r}"
        )
    condition_observation_label = (
        "synthetic_partial_keep50"
        if uses_synthetic_partial
        and abs(partial_keep_ratio - 0.5) <= 1.0e-12
        else (
            "full_object_pc"
            if object_observation_mode == "full"
            else object_observation_mode
        )
    )
    training_supports_inference_observation = training_observation_supports(
        model_cfg.dataset, object_observation_mode
    )
    if normalize_model_inputs and not is_autoregressive:
        raise ValueError(
            "This direct local entry point currently requires dataset.normalize=false "
            "so the XYZ point cloud and FK coordinates remain in metric units."
        )

    device = torch.device(args.device)
    model_class = (
        AutoregressiveContactDiffusion if is_autoregressive else ContactDiffusion
    )
    model = model_class.from_config(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    selected_grippers = list(args.grippers)
    unknown = [name for name in selected_grippers if name not in config["grippers"]]
    if unknown:
        raise KeyError(f"Unknown grippers: {unknown}")
    objects = select_objects(args, config)
    for item in objects:
        if not item["object_pc"].is_file():
            raise FileNotFoundError(item["object_pc"])
        # ``object_mesh`` is optional simulator provenance. FK generation and
        # every optimization energy consume only the full XYZ point cloud, so
        # a missing mesh must not block point-cloud-only inference.

    particles = int(args.particles or config["fk_optimization"]["particles"])
    optimization_steps = int(
        args.optimization_steps or config["fk_optimization"]["steps"]
    )
    if args.cedex_cleanup_steps != 0:
        raise ValueError(
            "--cedex-cleanup-steps must be 0 for the six-term XYZ-only energy"
        )
    diffusion_steps = int(args.diffusion_steps or config["diffusion"]["num_steps"])
    contact_weight = float(
        config["fk_optimization"].get("contact_weight", 1.0)
        if args.contact_weight is None
        else args.contact_weight
    )
    if contact_weight < 0.0:
        raise ValueError("--contact-weight must be non-negative")
    contact_normal_weight = float(args.contact_normal_weight)
    if contact_normal_weight < 0.0:
        raise ValueError("--contact-normal-weight must be non-negative")
    if args.object_surface_distance_weight < 0.0:
        raise ValueError("--object-surface-distance-weight must be non-negative")
    penetration_weight = float(
        config["fk_optimization"].get("penetration_weight", 0.0)
        if args.penetration_weight is None
        else args.penetration_weight
    )
    self_collision_weight = float(
        config["fk_optimization"].get("self_collision_weight", 0.0)
        if args.self_collision_weight is None
        else args.self_collision_weight
    )
    self_collision_clearance = float(
        config["fk_optimization"].get("self_collision_clearance", 0.002)
        if args.self_collision_clearance is None
        else args.self_collision_clearance
    )
    joint_regularization = float(
        config["fk_optimization"].get("joint_regularization", 1.0e-3)
        if args.joint_regularization is None
        else args.joint_regularization
    )
    if min(
        penetration_weight,
        self_collision_weight,
        self_collision_clearance,
        joint_regularization,
    ) < 0.0:
        raise ValueError("energy weights and self-collision clearance must be non-negative")
    force_closure_weight = float(
        config["fk_optimization"].get("force_closure_weight", 0.0)
        if args.force_closure_weight is None
        else args.force_closure_weight
    )
    calibration_path = resolve(config["paths"]["tip_offset_calibration"])
    num_points = int(model_cfg.dataset.num_points)

    output = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_model_type": str(
            checkpoint.get("model_type", "joint_set_contact_diffusion")
        ),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "input_mode": "xyz_point_cloud_only",
        "device": str(device),
        "particles": particles,
        "optimization_steps": optimization_steps,
        "diffusion_steps": diffusion_steps,
        "contact_generator": {
            "autoregressive": bool(is_autoregressive),
            "normalize_model_inputs": bool(normalize_model_inputs),
            "condition_observation": condition_observation_label,
            "checkpoint_object_observation_mode": (
                checkpoint_object_observation_mode
            ),
            "training_inference_observation_match": bool(
                object_observation_mode == checkpoint_object_observation_mode
            ),
            "training_supports_inference_observation": (
                training_supports_inference_observation
            ),
            "condition_points": int(model_cfg.dataset.num_points),
            "synthetic_partial_keep_ratio": (
                partial_keep_ratio if uses_synthetic_partial else None
            ),
            "normalization_reference": (
                "full_object_pc"
                if normalize_model_inputs
                else "not_applied"
            ),
            "raw_representation": "free_xyz_m",
            "model_project_to_surface": not bool(is_autoregressive),
            "surface_projection_reference": (
                "full_2048_object_pc" if is_autoregressive else "model_native"
            ),
            "fk_target_adapter": (
                (
                    "nearest_full_2048_object_point"
                    if args.autoregressive_fk_target == "nearest_2048"
                    else "raw_free_xyz_no_projection"
                )
                if is_autoregressive
                else "model_native_surface_projection"
            ),
        },
        "fk_energy": {
            "formula": (
                "wc*contact + wn*opposing_contact_normal + "
                "wd*closing_direction + wsw*nonpad_sweep_collision + "
                "wp*pc_penetration_mean_cvar + "
                "ws*self_collision_mean_cvar + wa*approach + "
                "wq*joint_prior + wpd*palm_unsigned_distance + "
                "wfc*execution_aware_force_closure"
            ),
            "contact_weight": contact_weight,
            "object_surface_distance_weight": float(
                args.object_surface_distance_weight
            ),
            "dexgraspnet_sum_reductions": bool(
                args.dexgraspnet_sum_reductions
            ),
            "contact_normal_weight": contact_normal_weight,
            "contact_normal_target": "opposing_outward_surface_normals",
            "contact_geometry_mode": (
                args.contact_geometry_mode
                or (
                    "distal_surface"
                    if args.fk_initialization == "enveloping"
                    else "tip_point"
                )
            ),
            "pad_points_per_finger": int(args.pad_points_per_finger),
            "pad_softmin_temperature_m": float(
                args.pad_softmin_temperature_m
            ),
            "closing_direction_weight": float(args.closing_direction_weight),
            "closing_direction_margin": float(args.closing_direction_margin),
            "sweep_collision_weight": float(args.sweep_collision_weight),
            "sweep_samples": int(args.sweep_samples),
            "sweep_points_per_link": int(args.sweep_points_per_link),
            "pad_exclusion_radius_m": float(args.pad_exclusion_radius_m),
            "closure_outer_fraction": float(args.closure_outer_fraction),
            "closure_inner_fraction": float(args.closure_inner_fraction),
            "contact_priority_enabled": bool(args.contact_priority),
            "contact_feasibility_threshold_m": float(
                args.contact_feasibility_threshold_m
            ),
            "contact_priority_target_m": float(
                args.contact_priority_target_m
            ),
            "contact_priority_temperature_m": float(
                args.contact_priority_temperature_m
            ),
            "contact_barrier_weight": float(args.contact_barrier_weight),
            "contact_stage1_fraction": float(args.contact_stage1_fraction),
            "contact_stage2_fraction": float(args.contact_stage2_fraction),
            "contact_feasibility_required": bool(
                args.require_contact_feasibility
            ),
            "penetration_weight": penetration_weight,
            "penetration_cvar_fraction": float(
                config["fk_optimization"].get("penetration_cvar_fraction", 0.1)
            ),
            "penetration_cvar_weight": float(
                config["fk_optimization"].get("penetration_cvar_weight", 1.0)
            ),
            "penetration_depth_mode": str(
                config["fk_optimization"].get(
                    "penetration_depth_mode", "point_to_plane"
                )
            ),
            "penetration_aggregation": str(
                config["fk_optimization"].get(
                    "penetration_aggregation", "mean_cvar"
                )
            ),
            "penetration_confidence_mode": str(
                config["fk_optimization"].get(
                    "penetration_confidence_mode", "weighted"
                )
            ),
            "penetration_gate_metric": str(
                config["fk_optimization"].get(
                    "penetration_gate_metric", "confidence_weighted"
                )
            ),
            "penetration_hinge_threshold_m": float(
                config["fk_optimization"].get(
                    "penetration_hinge_threshold_m", 0.0
                )
            ),
            "penetration_hinge_weight": float(
                config["fk_optimization"].get("penetration_hinge_weight", 0.0)
            ),
            "self_collision_weight": self_collision_weight,
            "self_collision_cvar_fraction": float(
                config["fk_optimization"].get("self_collision_cvar_fraction", 0.25)
            ),
            "self_collision_cvar_weight": float(
                config["fk_optimization"].get("self_collision_cvar_weight", 1.0)
            ),
            "joint_regularization": joint_regularization,
            "self_collision_clearance": self_collision_clearance,
            "self_collision_points_per_link": int(
                config["fk_optimization"].get("self_collision_points_per_link", 12)
            ),
            "surface_points_per_link": int(
                config["fk_optimization"].get("surface_points_per_link", 0)
            ),
            "object_normal_k_neighbors": int(
                config["fk_optimization"].get("object_normal_k_neighbors", 30)
            ),
            "initialization_mode": args.fk_initialization,
            "cedex_cleanup_steps": int(args.cedex_cleanup_steps),
            "cedex_joint_init_fraction": float(
                args.cedex_joint_init_fraction
            ),
            "cedex_cleanup_learning_rate": float(
                args.cedex_cleanup_learning_rate
            ),
            "cedex_contact_guard_m": float(args.cedex_contact_guard),
            "selection_max_penetration_m": args.selection_max_penetration,
            "envelope_approach_weight": float(
                args.envelope_approach_weight
            ),
            "palm_distance_weight": float(
                config["fk_optimization"].get("palm_distance_weight", 0.0)
            ),
            "contact_assignment_mode": (
                "soft_permutation"
                if args.fk_initialization == "enveloping"
                else "chamfer"
            ),
            "assignment_temperature_m": float(
                args.assignment_temperature
            ),
            "preferred_root_direction": (
                None
                if args.preferred_root_direction is None
                else [
                    float(value) for value in args.preferred_root_direction
                ]
            ),
            "load_bearing_pipeline": "unsigned_palm_distance -> optional_GraspQP_rank",
            "selection_min_envelope_cosine": float(
                args.selection_min_envelope_cosine
            ),
            "selection_min_approach_cosine": float(
                args.selection_min_approach_cosine
            ),
            "force_closure_weight": force_closure_weight,
            "force_closure_start_fraction": float(
                args.force_closure_start_fraction
            ),
            "force_closure_gap_sigma_m": float(
                args.force_closure_gap_sigma_m
            ),
            "force_closure_qp_iterations": int(
                args.force_closure_qp_iterations
            ),
            "force_closure_qp_weight": float(args.force_closure_qp_weight),
            "force_closure_dfc_weight": float(
                args.force_closure_dfc_weight
            ),
            "force_closure_coverage_weight": float(
                args.force_closure_coverage_weight
            ),
            "force_closure_target_fraction": args.force_closure_target_fraction,
            "force_closure_ramp_mode": args.force_closure_ramp_mode,
            "selection_rank_mode": (
                args.selection_rank_mode
                or (
                    "graspqp"
                    if args.fk_initialization == "enveloping"
                    else "optimization"
                )
            ),
        },
        "selection": {
            "object_ids": [item["object_id"] for item in objects],
            "sample_start": int(args.sample_start),
            "samples_per_object": int(args.samples_per_object),
            "grippers": selected_grippers,
            "base_seed": int(args.seed),
            "hand_index_offset": int(args.hand_index_offset),
            "object_index_offset": int(args.object_index_offset),
            "seed_formula": (
                "base_seed + 1000003*global_hand_index + "
                "1009*global_object_index + sample_index"
            ),
            "partial_observation_seed_formula": (
                "sample_seed" if uses_synthetic_partial else None
            ),
            "top_k": int(args.top_k),
            "tie_break": "stable_particle_index",
        },
        "contact_target_ab": {
            "mode": str(args.contact_target_mode),
            "autoregressive_fk_target": str(args.autoregressive_fk_target),
            "initialization_contact_source": str(
                args.initialization_contact_source
            ),
            "matched_random_matcher_version": MATCHER_VERSION,
            "matched_random_candidates": int(args.matched_random_candidates),
            "matched_random_seed_offset": int(
                args.matched_random_seed_offset
            ),
            "replay_source_diffusion_rng": bool(
                args.replay_source_diffusion_rng
            ),
            "comparison_scope": (
                "fixed-diffusion-initialization_target-only"
                if args.initialization_contact_source == "diffusion"
                else "end-to-end-per-target-initialization"
            ),
            "success_labels_used": False,
        },
        "records": [],
    }
    source_diffusion_records = None
    if args.source_diffusion_candidates is not None:
        source_path = resolve(args.source_diffusion_candidates)
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if source_payload.get("contact_target_ab", {}).get("mode") != "diffusion":
            raise ValueError("paired source candidate file is not the diffusion arm")
        for field in ("checkpoint_sha256", "config_sha256"):
            if source_payload.get(field) != output.get(field):
                raise ValueError(
                    f"paired source candidate {field} does not match this run"
                )
        source_selection = source_payload.get("selection", {})
        for field in (
            "object_ids",
            "grippers",
            "base_seed",
            "hand_index_offset",
            "object_index_offset",
            "seed_formula",
        ):
            if source_selection.get(field) != output["selection"].get(field):
                raise ValueError(
                    f"paired source candidate selection.{field} does not match "
                    "this run"
                )
        source_diffusion_records = {
            (
                str(record["gripper"]),
                str(record["object_id"]),
                int(record["sample_index"]),
            ): record
            for record in source_payload.get("records", [])
        }
        output["contact_target_ab"]["source_diffusion_candidates"] = str(
            source_path
        )
        output["contact_target_ab"]["source_diffusion_candidates_sha256"] = sha256(
            source_path
        )
    else:
        output["contact_target_ab"]["source_diffusion_candidates"] = None
        output["contact_target_ab"]["source_diffusion_candidates_sha256"] = None
    if args.replay_source_diffusion_rng and source_diffusion_records is None:
        raise ValueError(
            "--replay-source-diffusion-rng requires "
            "--source-diffusion-candidates"
        )
    warm_start_records = None
    if args.warm_start_candidates is not None:
        warm_start_path = resolve(args.warm_start_candidates)
        warm_start_payload = json.loads(
            warm_start_path.read_text(encoding="utf-8")
        )
        warm_start_records = {
            (
                str(record["gripper"]),
                str(record["object_id"]),
                int(record["sample_index"]),
            ): record
            for record in warm_start_payload.get("records", [])
        }
        output["contact_target_ab"]["warm_start_candidates"] = str(
            warm_start_path
        )
        output["contact_target_ab"]["warm_start_candidates_sha256"] = sha256(
            warm_start_path
        )
    else:
        output["contact_target_ab"]["warm_start_candidates"] = None
        output["contact_target_ab"]["warm_start_candidates_sha256"] = None
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        previous_contact_target = dict(previous.get("contact_target_ab", {}))
        # Candidate files written before the RNG-replay diagnostic existed are
        # semantically equivalent to an explicit false value.
        previous_contact_target.setdefault("replay_source_diffusion_rng", False)
        if (
            previous.get("checkpoint") != output["checkpoint"]
            or previous.get("checkpoint_sha256")
            != output["checkpoint_sha256"]
            or previous.get("config_sha256") != output["config_sha256"]
            or previous.get("selection") != output["selection"]
            or previous.get("fk_energy") != output["fk_energy"]
            or previous_contact_target != output["contact_target_ab"]
        ):
            raise ValueError(
                f"Cannot resume {output_path}: run configuration differs"
            )
        output = previous
    completed_keys = {
        (
            str(record["gripper"]),
            str(record["object_id"]),
            int(record["sample_index"]),
        )
        for record in output["records"]
    }

    for gripper_index, name in enumerate(selected_grippers):
        global_hand_index = int(args.hand_index_offset) + gripper_index
        spec = config["grippers"][name]
        use_load_bearing_pipeline = bool(
            args.fk_initialization == "enveloping"
            and spec.get("palm_surface_link")
        )
        fk_gripper = load_gripper_from_calibration(
            name, config, calibration_path, device=device
        )
        initial_joints = torch.as_tensor(
            ordered_joint_values(
                spec["opened_dofs"],
                fk_gripper.joint_names,
                label=f"{name}.opened_dofs",
            ),
            device=device,
            dtype=torch.float32,
        )
        close_direction = torch.as_tensor(
            ordered_joint_values(
                spec["close_dir"],
                fk_gripper.joint_names,
                label=f"{name}.close_dir",
            ),
            device=device,
            dtype=torch.float32,
        )
        axis_spec = dict(spec)
        if args.cedex_palm_axis is not None:
            axis_spec["cedex_palm_axis"] = args.cedex_palm_axis
        if args.grasp_approach_axis is not None:
            axis_spec["grasp_approach_axis"] = args.grasp_approach_axis
        cedex_palm_axis = resolve_local_palm_axis(
            fk_gripper, axis_spec, initial_joints
        )
        grasp_approach_axis = resolve_local_grasp_approach_axis(
            fk_gripper, axis_spec, initial_joints
        )
        for object_index, item in enumerate(objects):
            global_object_index = int(args.object_index_offset) + object_index
            object_seed = (
                int(args.seed)
                + 1_000_003 * global_hand_index
                + 1_009 * global_object_index
            )
            model_object_pc, geometry_object_pc = load_object_pc(
                item["object_pc"], num_points, object_seed
            )
            geometry = estimate_point_cloud_geometry(
                geometry_object_pc,
                k_neighbors=int(
                    config["fk_optimization"].get(
                        "object_normal_k_neighbors", 30
                    )
                ),
            )
            model_object_pc = model_object_pc.to(device)
            geometry_object_pc = geometry_object_pc.to(device)
            geometry_normals = torch.as_tensor(
                geometry["normals"], device=device, dtype=torch.float32
            )
            geometry_confidence = torch.as_tensor(
                geometry["confidence"], device=device, dtype=torch.float32
            )
            sample_start = max(0, int(args.sample_start))
            sample_stop = sample_start + max(1, int(args.samples_per_object))
            for sample_index in range(sample_start, sample_stop):
                record_key = (name, item["object_id"], sample_index)
                if record_key in completed_keys:
                    print(
                        f"{name} object={item['object_id']} "
                        f"sample={sample_index}: resume-skip",
                        flush=True,
                    )
                    continue
                sample_seed = object_seed + sample_index
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                condition_object_pc = model_object_pc
                partial_observation = {
                    "mode": "full",
                    "seed": None,
                    "keep_ratio": 1.0,
                    "source_point_count": int(len(model_object_pc)),
                    "visible_source_point_count": int(len(model_object_pc)),
                    "output_point_count": int(len(model_object_pc)),
                }
                if uses_synthetic_partial:
                    condition_object_pc, partial_observation = (
                        synthetic_partial_observation(
                            model_object_pc,
                            output_points=num_points,
                            keep_ratio=partial_keep_ratio,
                            seed=sample_seed,
                        )
                    )
                if source_diffusion_records is None:
                    (
                        diffusion_contacts,
                        raw_model_contacts_world,
                        model_projection_distance,
                        model_normalization,
                    ) = sample_contact_targets(
                        model,
                        condition_object_pc,
                        normalization_reference_world=model_object_pc,
                        surface_pc_world=model_object_pc,
                        num_contacts=int(spec["n"]),
                        num_steps=diffusion_steps,
                        sampler=str(config["diffusion"]["sampler"]),
                        normalize_model_inputs=normalize_model_inputs,
                        autoregressive=is_autoregressive,
                        autoregressive_fk_target=args.autoregressive_fk_target,
                    )
                else:
                    if args.replay_source_diffusion_rng:
                        # The paired source run sampled contacts immediately
                        # after setting sample_seed. Replaying that call keeps
                        # the subsequent stochastic FK initialization at the
                        # exact same CUDA RNG position while the stored source
                        # contacts remain the actual optimization target.
                        sample_contact_targets(
                            model,
                            condition_object_pc,
                            normalization_reference_world=model_object_pc,
                            surface_pc_world=model_object_pc,
                            num_contacts=int(spec["n"]),
                            num_steps=diffusion_steps,
                            sampler=str(config["diffusion"]["sampler"]),
                            normalize_model_inputs=normalize_model_inputs,
                            autoregressive=is_autoregressive,
                            autoregressive_fk_target=args.autoregressive_fk_target,
                        )
                    source_record = source_diffusion_records.get(record_key)
                    if source_record is None:
                        raise ValueError(
                            "paired diffusion source is missing "
                            f"{name}/{item['object_id']}/sample={sample_index}"
                        )
                    if int(source_record["sample_seed"]) != sample_seed:
                        raise ValueError(
                            f"paired source seed mismatch for {record_key}"
                        )
                    source_partial = source_record.get("partial_observation")
                    if (
                        uses_synthetic_partial
                        and source_partial != partial_observation
                    ):
                        raise ValueError(
                            "paired source partial observation mismatch for "
                            f"{record_key}"
                        )
                    projected_source_contacts = torch.as_tensor(
                        source_record["source_diffusion_contacts"],
                        device=device,
                        dtype=torch.float32,
                    )
                    if projected_source_contacts.shape != (int(spec["n"]), 3):
                        raise ValueError(
                            f"paired source contacts have invalid shape for {record_key}"
                        )
                    raw_model_contacts_world = torch.as_tensor(
                        source_record.get(
                            "source_model_raw_contacts",
                            source_record["source_diffusion_contacts"],
                        ),
                        device=device,
                        dtype=torch.float32,
                    )
                    diffusion_contacts = (
                        raw_model_contacts_world
                        if is_autoregressive
                        and args.autoregressive_fk_target == "raw_xyz"
                        else projected_source_contacts
                    )
                    model_projection_distance = torch.as_tensor(
                        source_record.get(
                            "source_model_projection_distance_m",
                            [0.0] * int(spec["n"]),
                        ),
                        device=device,
                        dtype=torch.float32,
                    )
                    model_normalization = source_record.get(
                        "source_model_normalization",
                        {
                            "enabled": False,
                            "center_m": [0.0, 0.0, 0.0],
                            "scale_m": 1.0,
                        },
                    )
                matched_random = None
                if args.contact_target_mode == "matched_random":
                    matched_random = matched_random_surface_contacts(
                        geometry_object_pc.detach().cpu().numpy(),
                        diffusion_contacts.detach().cpu().numpy(),
                        seed=(
                            int(sample_seed)
                            + int(args.matched_random_seed_offset)
                        ),
                        candidate_count=int(args.matched_random_candidates),
                    )
                    contacts = torch.as_tensor(
                        matched_random["contacts"],
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    contacts = diffusion_contacts
                initialization_contacts = (
                    diffusion_contacts
                    if args.initialization_contact_source == "diffusion"
                    else contacts
                )
                warm_root_poses = None
                warm_joint_states = None
                if warm_start_records is not None:
                    warm_record = warm_start_records.get(record_key)
                    if warm_record is None:
                        raise ValueError(
                            f"warm-start source is missing {record_key}"
                        )
                    warm_candidates = warm_record.get("fk", {}).get(
                        "candidates", []
                    )
                    if len(warm_candidates) != particles:
                        raise ValueError(
                            "warm-start candidate count does not match particles: "
                            f"{len(warm_candidates)} != {particles}"
                        )
                    warm_root_poses = torch.as_tensor(
                        [row["root_pose"] for row in warm_candidates],
                        device=device,
                        dtype=torch.float32,
                    )
                    warm_joint_states = torch.as_tensor(
                        [row["joint_positions"] for row in warm_candidates],
                        device=device,
                        dtype=torch.float32,
                    )
                solved = optimize_gripper_to_contacts(
                    fk_gripper,
                    contacts,
                    geometry_object_pc,
                    initial_joints,
                    particles=particles,
                    steps=optimization_steps,
                    learning_rate=float(config["fk_optimization"]["learning_rate"]),
                    contact_weight=contact_weight,
                    object_surface_distance_weight=float(
                        args.object_surface_distance_weight
                    ),
                    dexgraspnet_sum_reductions=bool(
                        args.dexgraspnet_sum_reductions
                    ),
                    contact_normal_weight=contact_normal_weight,
                    penetration_weight=penetration_weight,
                    penetration_cvar_fraction=float(
                        config["fk_optimization"].get(
                            "penetration_cvar_fraction", 0.1
                        )
                    ),
                    penetration_cvar_weight=float(
                        config["fk_optimization"].get(
                            "penetration_cvar_weight", 1.0
                        )
                    ),
                    penetration_depth_mode=str(
                        config["fk_optimization"].get(
                            "penetration_depth_mode", "point_to_plane"
                        )
                    ),
                    penetration_aggregation=str(
                        config["fk_optimization"].get(
                            "penetration_aggregation", "mean_cvar"
                        )
                    ),
                    penetration_confidence_mode=str(
                        config["fk_optimization"].get(
                            "penetration_confidence_mode", "weighted"
                        )
                    ),
                    penetration_gate_metric=str(
                        config["fk_optimization"].get(
                            "penetration_gate_metric", "confidence_weighted"
                        )
                    ),
                    penetration_hinge_threshold_m=float(
                        config["fk_optimization"].get(
                            "penetration_hinge_threshold_m", 0.0
                        )
                    ),
                    penetration_hinge_weight=float(
                        config["fk_optimization"].get(
                            "penetration_hinge_weight", 0.0
                        )
                    ),
                    object_normal_k_neighbors=int(
                        config["fk_optimization"].get(
                            "object_normal_k_neighbors", 30
                        )
                    ),
                    self_collision_weight=self_collision_weight,
                    self_collision_cvar_fraction=float(
                        config["fk_optimization"].get(
                            "self_collision_cvar_fraction", 0.25
                        )
                    ),
                    self_collision_cvar_weight=float(
                        config["fk_optimization"].get(
                            "self_collision_cvar_weight", 1.0
                        )
                    ),
                    self_collision_clearance=self_collision_clearance,
                    self_collision_points_per_link=int(
                        config["fk_optimization"].get(
                            "self_collision_points_per_link", 12
                        )
                    ),
                    joint_regularization=joint_regularization,
                    seed=sample_seed,
                    top_k=int(args.top_k),
                    object_normals=geometry_normals,
                    object_normal_confidence=geometry_confidence,
                    initialization_mode=args.fk_initialization,
                    cedex_local_palm_axis=cedex_palm_axis,
                    grasp_local_approach_axis=grasp_approach_axis,
                    grasp_approach_target_mode=args.grasp_approach_target_mode,
                    grasp_approach_plane_sides=args.grasp_approach_plane_sides,
                    close_direction=close_direction,
                    cedex_joint_init_fraction=float(
                        args.cedex_joint_init_fraction
                    ),
                    cedex_cleanup_steps=int(args.cedex_cleanup_steps),
                    cedex_cleanup_learning_rate=float(
                        args.cedex_cleanup_learning_rate
                    ),
                    cedex_contact_guard_m=float(
                        args.cedex_contact_guard
                    ),
                    selection_max_penetration_m=(
                        None
                        if args.selection_max_penetration is None
                        else float(args.selection_max_penetration)
                    ),
                    envelope_approach_weight=(
                        float(args.envelope_approach_weight)
                        if args.fk_initialization == "enveloping"
                        else 0.0
                    ),
                    contact_assignment_mode=(
                        "soft_permutation"
                        if args.fk_initialization == "enveloping"
                        else "chamfer"
                    ),
                    preferred_root_direction=(
                        torch.as_tensor(
                            args.preferred_root_direction,
                            device=device,
                            dtype=torch.float32,
                        )
                        if (
                            args.fk_initialization == "enveloping"
                            and args.preferred_root_direction is not None
                        )
                        else None
                    ),
                    assignment_temperature_m=float(
                        args.assignment_temperature
                    ),
                    contact_geometry_mode=(
                        args.contact_geometry_mode
                        or (
                            "distal_surface"
                            if use_load_bearing_pipeline
                            else "tip_point"
                        )
                    ),
                    pad_points_per_finger=int(args.pad_points_per_finger),
                    pad_softmin_temperature_m=float(
                        args.pad_softmin_temperature_m
                    ),
                    closing_direction_weight=float(
                        args.closing_direction_weight
                    ),
                    closing_direction_margin=float(
                        args.closing_direction_margin
                    ),
                    sweep_collision_weight=float(args.sweep_collision_weight),
                    sweep_samples=int(args.sweep_samples),
                    sweep_points_per_link=int(args.sweep_points_per_link),
                    pad_exclusion_radius_m=float(args.pad_exclusion_radius_m),
                    closure_outer_fraction=float(args.closure_outer_fraction),
                    closure_inner_fraction=float(args.closure_inner_fraction),
                    contact_priority_enabled=bool(args.contact_priority),
                    contact_feasibility_threshold_m=float(
                        args.contact_feasibility_threshold_m
                    ),
                    contact_priority_target_m=float(
                        args.contact_priority_target_m
                    ),
                    contact_priority_temperature_m=float(
                        args.contact_priority_temperature_m
                    ),
                    contact_barrier_weight=float(args.contact_barrier_weight),
                    contact_stage1_fraction=float(args.contact_stage1_fraction),
                    contact_stage2_fraction=float(args.contact_stage2_fraction),
                    contact_feasibility_required=bool(
                        args.require_contact_feasibility
                    ),
                    selection_min_envelope_cosine=(
                        float(args.selection_min_envelope_cosine)
                        if args.fk_initialization == "enveloping"
                        else None
                    ),
                    selection_min_approach_cosine=(
                        float(args.selection_min_approach_cosine)
                        if args.fk_initialization == "enveloping"
                        else None
                    ),
                    cosine_feasibility_gates_enabled=bool(
                        args.enable_legacy_cosine_feasibility_gates
                    ),
                    palm_distance_weight=(
                        float(
                            config["fk_optimization"].get(
                                "palm_distance_weight", 0.0
                            )
                        )
                        if use_load_bearing_pipeline
                        else 0.0
                    ),
                    palm_target_distance_m=float(
                        config["fk_optimization"]["palm_target_distance_m"]
                    ),
                    selection_max_palm_distance_m=(
                        float(
                            config["fk_optimization"][
                                "selection_max_palm_distance_m"
                            ]
                        )
                        if (
                            use_load_bearing_pipeline
                            and not args.disable_palm_selection_gate
                        )
                        else None
                    ),
                    graspqp_friction_coefficient=float(
                        config["fk_optimization"][
                            "graspqp_friction_coefficient"
                        ]
                    ),
                    graspqp_cone_edges=int(
                        config["fk_optimization"]["graspqp_cone_edges"]
                    ),
                    graspqp_iterations=int(
                        config["fk_optimization"]["graspqp_iterations"]
                    ),
                    force_closure_weight=force_closure_weight,
                    force_closure_start_fraction=float(
                        args.force_closure_start_fraction
                    ),
                    force_closure_gap_sigma_m=float(
                        args.force_closure_gap_sigma_m
                    ),
                    force_closure_qp_iterations=int(
                        args.force_closure_qp_iterations
                    ),
                    force_closure_qp_weight=float(args.force_closure_qp_weight),
                    force_closure_dfc_weight=float(
                        args.force_closure_dfc_weight
                    ),
                    force_closure_coverage_weight=float(
                        args.force_closure_coverage_weight
                    ),
                    force_closure_target_fraction=args.force_closure_target_fraction,
                    force_closure_ramp_mode=args.force_closure_ramp_mode,
                    force_closure_formulation=args.force_closure_formulation,
                    force_closure_include_palm=args.force_closure_include_palm,
                    force_closure_friction_coefficient=(
                        args.force_closure_friction_coefficient
                    ),
                    force_closure_cone_edges=args.force_closure_cone_edges,
                    force_closure_max_force_coefficient=float(
                        args.force_closure_max_force_coefficient
                    ),
                    force_closure_torque_weight=float(
                        args.force_closure_torque_weight
                    ),
                    force_closure_svd_gain=float(args.force_closure_svd_gain),
                    selection_rank_mode=(
                        args.selection_rank_mode
                        or (
                            "graspqp"
                            if use_load_bearing_pipeline
                            else "optimization"
                        )
                    ),
                    initialization_contacts=initialization_contacts,
                    initial_root_poses=warm_root_poses,
                    initial_joint_states=warm_joint_states,
                )
                best = (
                    solved["candidates"][0]
                    if solved["candidates"]
                    else None
                )
                output["records"].append(
                    {
                        "gripper": name,
                        "n": int(spec["n"]),
                        "source_split": "local",
                        "object_index": object_index,
                        "global_object_index": global_object_index,
                        "gripper_index": gripper_index,
                        "global_hand_index": global_hand_index,
                        "sample_index": sample_index,
                        "dataset_index": None,
                        "sample_seed": sample_seed,
                        "condition_observation": condition_observation_label,
                        "partial_seed": (
                            sample_seed if uses_synthetic_partial else None
                        ),
                        "partial_observation": partial_observation,
                        "record_id": (
                            f"local:{item['object_id']}:{name}:{sample_index}:"
                            f"{args.contact_target_mode}"
                        ),
                        "contact_target_mode": str(args.contact_target_mode),
                        "initialization_contact_source": str(
                            args.initialization_contact_source
                        ),
                        "source_diffusion_contacts": (
                            diffusion_contacts.detach().cpu().tolist()
                        ),
                        "source_model_raw_contacts": (
                            raw_model_contacts_world.detach().cpu().tolist()
                        ),
                        "source_model_projection_distance_m": (
                            model_projection_distance.detach().cpu().tolist()
                        ),
                        "source_model_projection_distance_mean_m": float(
                            model_projection_distance.mean().detach().cpu()
                        ),
                        "source_model_projection_distance_max_m": float(
                            model_projection_distance.max().detach().cpu()
                        ),
                        "source_model_normalization": model_normalization,
                        "target_contacts": contacts.detach().cpu().tolist(),
                        "source_diffusion_contacts_sha256": hashlib.sha256(
                            diffusion_contacts.detach()
                            .cpu()
                            .contiguous()
                            .numpy()
                            .tobytes()
                        ).hexdigest(),
                        "target_contacts_sha256": hashlib.sha256(
                            contacts.detach()
                            .cpu()
                            .contiguous()
                            .numpy()
                            .tobytes()
                        ).hexdigest(),
                        "matched_random": (
                            None
                            if matched_random is None
                            else {
                                "indices": matched_random["indices"],
                                "diagnostics": matched_random["diagnostics"],
                            }
                        ),
                        "object_id": item["object_id"],
                        "object_mesh": (
                            None
                            if item["object_mesh"] is None
                            else str(item["object_mesh"])
                        ),
                        "object_pc_asset": str(item["object_pc"]),
                        "object_geometry_point_count": int(
                            geometry_object_pc.shape[0]
                        ),
                        "object_normal_confidence_mean": float(
                            geometry["mean_confidence"]
                        ),
                        "object_normal_confidence_median": float(
                            geometry["median_confidence"]
                        ),
                        "fk": solved,
                    }
                )
                completed_keys.add(record_key)
                output_path.write_text(
                    json.dumps(output, indent=2), encoding="utf-8"
                )
                if best is None:
                    print(
                        f"{name} object={item['object_id']} "
                        f"sample={sample_index}: no candidate passed the "
                        "required contact-feasibility gate",
                        flush=True,
                    )
                else:
                    print(
                        f"{name} object={item['object_id']} sample={sample_index}: "
                        f"contact={best['contact_chamfer_m']:.6f} m, "
                        f"max_finger_contact="
                        f"{best['max_finger_contact_error_m']:.6f} m, "
                        f"penetration(mean/CVaR/max)={best['mean_penetration_m']:.6f}/"
                        f"{best['cvar_penetration_m']:.6f}/"
                        f"{best['max_penetration_m']:.6f} m, "
                        f"self_collision(mean/CVaR/max)="
                        f"{best['mean_self_collision_m']:.6f}/"
                        f"{best['cvar_self_collision_m']:.6f}/"
                        f"{best['max_self_collision_m']:.6f} m",
                        flush=True,
                    )

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
