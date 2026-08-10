#!/usr/bin/env python3
"""Run ContactDiffusion and differentiable gripper FK from local object assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault(
    "PYTORCH_KERNEL_CACHE_PATH", str(REPO_ROOT / ".cache" / "torch" / "kernels")
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
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
from utils.multigripper_fk import (
    load_gripper_from_calibration,
    optimize_gripper_to_contacts,
    ordered_joint_values,
    sample_object_surface,
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


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
        help="Existing local manifest used only to pair object IDs, meshes and point clouds.",
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
    parser.add_argument("--object-mesh", default=None)
    parser.add_argument("--grippers", nargs="+", default=["franka_panda"])
    parser.add_argument("--samples-per-object", type=int, default=1)
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
        "--fk-initialization",
        choices=("kabsch", "cedex", "enveloping"),
        default="kabsch",
        help=(
            "FK particle initialization. The default preserves the existing "
            "contact-set Kabsch path; cedex enables center-facing palm poses; "
            "enveloping mixes Kabsch and top-down palm hypotheses."
        ),
    )
    parser.add_argument("--envelope-side-weight", type=float, default=5.0)
    parser.add_argument("--envelope-approach-weight", type=float, default=2.0)
    parser.add_argument("--envelope-cosine-margin", type=float, default=0.25)
    parser.add_argument("--selection-min-envelope-cosine", type=float, default=0.5)
    parser.add_argument("--selection-min-approach-cosine", type=float, default=0.8)
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
        help="Override the number of mesh surface points used by FK collision losses.",
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260724)
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
        mesh = resolve(record["object_mesh"])
        pc_value = Path(record["object_pc_asset"])
        pc = pc_value if pc_value.is_absolute() else dataset_root / pc_value
        objects.append(
            {
                "object_id": object_id,
                "object_mesh": mesh.resolve(),
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
    if not pc_root_value or not mesh_root_value:
        return []
    pc_root = resolve(pc_root_value)
    mesh_root = resolve(mesh_root_value)
    objects = []
    for pc in sorted(pc_root.glob("*.npy")):
        mesh = mesh_root / f"{pc.stem}.obj"
        if mesh.is_file():
            objects.append(
                {
                    "object_id": pc.stem,
                    "object_mesh": mesh.resolve(),
                    "object_pc": pc.resolve(),
                }
            )
    return objects


def select_objects(args: argparse.Namespace, config: dict) -> list[dict]:
    explicit = args.object_pc is not None or args.object_mesh is not None
    if explicit:
        if not args.object_pc or not args.object_mesh:
            raise ValueError("--object-pc and --object-mesh must be provided together")
        pc = resolve(args.object_pc)
        mesh = resolve(args.object_mesh)
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


def load_object_pc(path: Path, num_points: int, seed: int) -> torch.Tensor:
    points = np.load(path, allow_pickle=False)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise ValueError(f"Expected non-empty [M, >=3] point cloud at {path}, got {points.shape}")
    points = points[:, :3]
    if not np.isfinite(points).all():
        raise ValueError(f"Point cloud contains NaN or Inf: {path}")
    if len(points) != num_points:
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(len(points), size=num_points, replace=len(points) < num_points)
        points = points[indices]
    return torch.from_numpy(np.ascontiguousarray(points))


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    checkpoint_path = resolve(args.checkpoint)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(checkpoint["config"])
    if bool(model_cfg.dataset.normalize):
        raise ValueError(
            "This direct local entry point currently requires dataset.normalize=false "
            "so FK and mesh coordinates remain in metric units."
        )

    device = torch.device(args.device)
    model = ContactDiffusion.from_config(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    selected_grippers = list(args.grippers)
    unknown = [name for name in selected_grippers if name not in config["grippers"]]
    if unknown:
        raise KeyError(f"Unknown grippers: {unknown}")
    objects = select_objects(args, config)
    for item in objects:
        for key in ("object_pc", "object_mesh"):
            if not item[key].is_file():
                raise FileNotFoundError(item[key])

    particles = int(args.particles or config["fk_optimization"]["particles"])
    optimization_steps = int(
        args.optimization_steps or config["fk_optimization"]["steps"]
    )
    if args.cedex_cleanup_steps > optimization_steps:
        raise ValueError("--cedex-cleanup-steps cannot exceed optimization steps")
    diffusion_steps = int(args.diffusion_steps or config["diffusion"]["num_steps"])
    contact_weight = float(config["fk_optimization"].get("contact_weight", 1.0))
    penetration_weight = float(
        config["fk_optimization"].get("penetration_weight", 0.0)
    )
    max_penetration_weight = float(
        config["fk_optimization"].get("max_penetration_weight", 0.0)
    )
    self_collision_weight = float(
        config["fk_optimization"].get("self_collision_weight", 0.0)
    )
    max_self_collision_weight = float(
        config["fk_optimization"].get("max_self_collision_weight", 0.0)
    )
    calibration_path = resolve(config["paths"]["tip_offset_calibration"])
    num_points = int(model_cfg.dataset.num_points)

    output = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "config": str(config_path),
        "input_mode": "local_object_assets",
        "device": str(device),
        "particles": particles,
        "optimization_steps": optimization_steps,
        "diffusion_steps": diffusion_steps,
        "fk_energy": {
            "contact_weight": contact_weight,
            "penetration_weight": penetration_weight,
            "max_penetration_weight": max_penetration_weight,
            "self_collision_weight": self_collision_weight,
            "max_self_collision_weight": max_self_collision_weight,
            "self_collision_clearance": float(
                config["fk_optimization"].get("self_collision_clearance", 0.002)
            ),
            "self_collision_points_per_link": int(
                config["fk_optimization"].get("self_collision_points_per_link", 12)
            ),
            "surface_points_per_link": int(
                config["fk_optimization"].get("surface_points_per_link", 0)
            ),
            "object_surface_points": int(
                args.object_surface_points
                or config["fk_optimization"].get("object_surface_points", 1024)
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
            "envelope_side_weight": float(args.envelope_side_weight),
            "envelope_approach_weight": float(
                args.envelope_approach_weight
            ),
            "envelope_cosine_margin": float(
                args.envelope_cosine_margin
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
            "load_bearing_pipeline": (
                "palm_sdf+normal_opposition+center_clamp"
                " -> DexGraspNet_DFC -> GraspQP_6dir"
            ),
            "selection_min_envelope_cosine": float(
                args.selection_min_envelope_cosine
            ),
            "selection_min_approach_cosine": float(
                args.selection_min_approach_cosine
            ),
        },
        "selection": {
            "object_ids": [item["object_id"] for item in objects],
            "sample_start": int(args.sample_start),
            "samples_per_object": int(args.samples_per_object),
            "grippers": selected_grippers,
        },
        "records": [],
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            previous.get("checkpoint") != output["checkpoint"]
            or previous.get("selection") != output["selection"]
            or previous.get("fk_energy") != output["fk_energy"]
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
        # Derive the palm-to-fingers direction from the actual calibrated URDF
        # instead of relying on hand-specific hard-coded axes.  The old
        # ShadowHand fallback (-Y) was almost orthogonal to its measured
        # open-hand tip-centroid direction (+Z).
        initial_joint_batch = initial_joints.unsqueeze(0)
        measured_palm_axis = (
            fk_gripper.tip_points_in_aligned_base(
                initial_joint_batch
            ).mean(dim=1)[0]
            - fk_gripper.palm_point_in_aligned_base(
                initial_joint_batch
            )[0]
        )
        measured_palm_axis = measured_palm_axis / torch.linalg.norm(
            measured_palm_axis
        ).clamp_min(1.0e-8)
        cedex_palm_axis = torch.as_tensor(
            spec.get("cedex_palm_axis", measured_palm_axis.tolist()),
            device=device,
            dtype=torch.float32,
        )
        for object_index, item in enumerate(objects):
            object_seed = int(args.seed) + 1_000_003 * gripper_index + 1_009 * object_index
            object_pc = load_object_pc(item["object_pc"], num_points, object_seed).to(device)
            collision_points_np, collision_normals_np = sample_object_surface(
                item["object_mesh"],
                int(
                    args.object_surface_points
                    or config["fk_optimization"].get(
                        "object_surface_points", 1024
                    )
                ),
                seed=object_seed,
            )
            collision_points = torch.as_tensor(
                collision_points_np, device=device, dtype=torch.float32
            )
            collision_normals = torch.as_tensor(
                collision_normals_np, device=device, dtype=torch.float32
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
                    self_collision_weight=self_collision_weight,
                    max_self_collision_weight=max_self_collision_weight,
                    self_collision_clearance=float(
                        config["fk_optimization"].get(
                            "self_collision_clearance", 0.002
                        )
                    ),
                    self_collision_points_per_link=int(
                        config["fk_optimization"].get(
                            "self_collision_points_per_link", 12
                        )
                    ),
                    translation_regularization=float(
                        config["fk_optimization"]["translation_regularization"]
                    ),
                    seed=sample_seed,
                    top_k=int(args.top_k),
                    object_surface_points=collision_points,
                    object_surface_normals=collision_normals,
                    initialization_mode=args.fk_initialization,
                    cedex_local_palm_axis=cedex_palm_axis,
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
                    envelope_side_weight=(
                        float(args.envelope_side_weight)
                        if args.fk_initialization == "enveloping"
                        else 0.0
                    ),
                    envelope_approach_weight=(
                        float(args.envelope_approach_weight)
                        if args.fk_initialization == "enveloping"
                        else 0.0
                    ),
                    envelope_cosine_margin=float(
                        args.envelope_cosine_margin
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
                        "distal_surface"
                        if use_load_bearing_pipeline
                        else "tip_point"
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
                    palm_sdf_weight=(
                        float(config["fk_optimization"]["palm_sdf_weight"])
                        if use_load_bearing_pipeline
                        else 0.0
                    ),
                    palm_target_distance_m=float(
                        config["fk_optimization"]["palm_target_distance_m"]
                    ),
                    palm_normal_opposition_weight=(
                        float(
                            config["fk_optimization"][
                                "palm_normal_opposition_weight"
                            ]
                        )
                        if use_load_bearing_pipeline
                        else 0.0
                    ),
                    palm_normal_max_cosine=float(
                        config["fk_optimization"]["palm_normal_max_cosine"]
                    ),
                    center_clamp_weight=(
                        float(
                            config["fk_optimization"]["center_clamp_weight"]
                        )
                        if use_load_bearing_pipeline
                        else 0.0
                    ),
                    center_clamp_margin_m=float(
                        config["fk_optimization"]["center_clamp_margin_m"]
                    ),
                    dfc_weight=(
                        float(config["fk_optimization"]["dfc_weight"])
                        if use_load_bearing_pipeline
                        else 0.0
                    ),
                    selection_max_palm_distance_m=(
                        float(
                            config["fk_optimization"][
                                "selection_max_palm_distance_m"
                            ]
                        )
                        if use_load_bearing_pipeline
                        else None
                    ),
                    selection_max_dfc_energy=(
                        float(
                            config["fk_optimization"][
                                "selection_max_dfc_energy"
                            ]
                        )
                        if use_load_bearing_pipeline
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
                    selection_rank_mode=(
                        "graspqp"
                        if use_load_bearing_pipeline
                        else "optimization"
                    ),
                )
                best = solved["candidates"][0]
                output["records"].append(
                    {
                        "gripper": name,
                        "n": int(spec["n"]),
                        "source_split": "local",
                        "object_index": object_index,
                        "sample_index": sample_index,
                        "dataset_index": None,
                        "sample_seed": sample_seed,
                        "record_id": f"local:{item['object_id']}:{name}:{sample_index}",
                        "object_id": item["object_id"],
                        "object_mesh": str(item["object_mesh"]),
                        "object_pc_asset": str(item["object_pc"]),
                        "fk": solved,
                    }
                )
                completed_keys.add(record_key)
                output_path.write_text(
                    json.dumps(output, indent=2), encoding="utf-8"
                )
                print(
                    f"{name} object={item['object_id']} sample={sample_index}: "
                    f"contact={best['contact_chamfer_m']:.6f} m, "
                    f"penetration={best['mean_penetration_m']:.6f}/"
                    f"{best['max_penetration_m']:.6f} m, "
                    f"self_collision={best['mean_self_collision_m']:.6f}/"
                    f"{best['max_self_collision_m']:.6f} m",
                    flush=True,
                )

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
