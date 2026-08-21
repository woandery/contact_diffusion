#!/usr/bin/env python3
"""Run ContactDiffusion and differentiable gripper FK from local object assets."""

from __future__ import annotations

import argparse
import hashlib
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
        "--object-mesh",
        default=None,
        help="Optional simulator asset; never used by FK optimization energies.",
    )
    parser.add_argument("--grippers", nargs="+", default=["franka_panda"])
    parser.add_argument("--samples-per-object", type=int, default=1)
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
            "paired source contacts. Intended for matched_random B runs."
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


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    checkpoint_path = resolve(args.checkpoint)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
        if not item["object_pc"].is_file():
            raise FileNotFoundError(item["object_pc"])
        if item["object_mesh"] is not None and not item["object_mesh"].is_file():
            raise FileNotFoundError(item["object_mesh"])

    particles = int(args.particles or config["fk_optimization"]["particles"])
    optimization_steps = int(
        args.optimization_steps or config["fk_optimization"]["steps"]
    )
    if args.cedex_cleanup_steps != 0:
        raise ValueError(
            "--cedex-cleanup-steps must be 0 for the six-term XYZ-only energy"
        )
    diffusion_steps = int(args.diffusion_steps or config["diffusion"]["num_steps"])
    contact_weight = float(config["fk_optimization"].get("contact_weight", 1.0))
    penetration_weight = float(
        config["fk_optimization"].get("penetration_weight", 0.0)
    )
    self_collision_weight = float(
        config["fk_optimization"].get("self_collision_weight", 0.0)
    )
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
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "input_mode": "xyz_point_cloud_only",
        "device": str(device),
        "particles": particles,
        "optimization_steps": optimization_steps,
        "diffusion_steps": diffusion_steps,
        "fk_energy": {
            "formula": (
                "wc*contact + wp*pc_penetration_mean_cvar + "
                "ws*self_collision_mean_cvar + wa*approach + "
                "wq*joint_prior + wpd*palm_unsigned_distance + "
                "wfc*execution_aware_force_closure"
            ),
            "contact_weight": contact_weight,
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
            "joint_regularization": float(
                config["fk_optimization"].get("joint_regularization", 1.0e-3)
            ),
            "self_collision_clearance": float(
                config["fk_optimization"].get("self_collision_clearance", 0.002)
            ),
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
            "top_k": int(args.top_k),
            "tie_break": "stable_particle_index",
        },
        "contact_target_ab": {
            "mode": str(args.contact_target_mode),
            "initialization_contact_source": str(
                args.initialization_contact_source
            ),
            "matched_random_matcher_version": MATCHER_VERSION,
            "matched_random_candidates": int(args.matched_random_candidates),
            "matched_random_seed_offset": int(
                args.matched_random_seed_offset
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
        if args.contact_target_mode != "matched_random":
            raise ValueError(
                "--source-diffusion-candidates is only valid with "
                "--contact-target-mode matched_random"
            )
        source_path = resolve(args.source_diffusion_candidates)
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if source_payload.get("contact_target_ab", {}).get("mode") != "diffusion":
            raise ValueError("paired source candidate file is not the diffusion arm")
        for field in ("checkpoint_sha256", "config_sha256", "selection"):
            if source_payload.get(field) != output.get(field):
                raise ValueError(
                    f"paired source candidate {field} does not match this run"
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
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            previous.get("checkpoint") != output["checkpoint"]
            or previous.get("checkpoint_sha256")
            != output["checkpoint_sha256"]
            or previous.get("config_sha256") != output["config_sha256"]
            or previous.get("selection") != output["selection"]
            or previous.get("fk_energy") != output["fk_energy"]
            or previous.get("contact_target_ab") != output["contact_target_ab"]
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
                if source_diffusion_records is None:
                    with torch.inference_mode():
                        diffusion_contacts = model.sample(
                            object_pc=model_object_pc.unsqueeze(0),
                            num_contacts=int(spec["n"]),
                            dc=3,
                            num_steps=diffusion_steps,
                            sampler=str(config["diffusion"]["sampler"]),
                            project_to_surface=True,
                        )[0]
                else:
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
                    diffusion_contacts = torch.as_tensor(
                        source_record["source_diffusion_contacts"],
                        device=device,
                        dtype=torch.float32,
                    )
                    if diffusion_contacts.shape != (int(spec["n"]), 3):
                        raise ValueError(
                            f"paired source contacts have invalid shape for {record_key}"
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
                solved = optimize_gripper_to_contacts(
                    fk_gripper,
                    contacts,
                    geometry_object_pc,
                    initial_joints,
                    particles=particles,
                    steps=optimization_steps,
                    learning_rate=float(config["fk_optimization"]["learning_rate"]),
                    contact_weight=contact_weight,
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
                    joint_regularization=float(
                        config["fk_optimization"].get(
                            "joint_regularization", 1.0e-3
                        )
                    ),
                    seed=sample_seed,
                    top_k=int(args.top_k),
                    object_normals=geometry_normals,
                    object_normal_confidence=geometry_confidence,
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
                        "graspqp"
                        if use_load_bearing_pipeline
                        else "optimization"
                    ),
                    initialization_contacts=initialization_contacts,
                )
                best = solved["candidates"][0]
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
                print(
                    f"{name} object={item['object_id']} sample={sample_index}: "
                    f"contact={best['contact_chamfer_m']:.6f} m, "
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
