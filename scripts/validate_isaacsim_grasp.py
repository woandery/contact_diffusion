#!/usr/bin/env python3
"""Validate one optimized ContactDiffusion grasp in Isaac Sim/PhysX."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multigripper_fk_isaac.yaml")
    parser.add_argument(
        "--experience",
        default=None,
        help="Optional Isaac Sim .kit experience. Use the minimal visual experience locally.",
    )
    parser.add_argument("--candidates", default="outputs/multigripper_fk/fk_candidates.json")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=0.10)
    parser.add_argument("--open-fraction", type=float, default=0.08)
    parser.add_argument(
        "--overclose-fraction",
        type=float,
        default=None,
        help="Extra closing travel after FK contact; defaults to isaac.overclose_fraction.",
    )
    parser.add_argument("--drive-stiffness", type=float, default=1.0e3)
    parser.add_argument("--drive-damping", type=float, default=1.0e2)
    parser.add_argument("--solver-position-iterations", type=int, default=None)
    parser.add_argument("--solver-velocity-iterations", type=int, default=None)
    parser.add_argument(
        "--max-joint-effort",
        type=float,
        default=None,
        help="Optional effort override; otherwise retain each URDF joint's safe limit.",
    )
    parser.add_argument(
        "--kinematic-close",
        action="store_true",
        help="Hold the object fixed during finger closure instead of allowing self-centering.",
    )
    parser.add_argument("--cpu-physics", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Render headlessly through an Isaac Sim streaming experience.",
    )
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--skip-gravity-test", action="store_true")
    parser.add_argument("--pre-release-hold-seconds", type=float, default=0.0)
    parser.add_argument("--keep-open-seconds", type=float, default=0.0)
    parser.add_argument(
        "--camera-eye",
        nargs=3,
        type=float,
        default=[0.24, 0.24, 0.16],
    )
    parser.add_argument(
        "--camera-target",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.03],
    )
    parser.add_argument(
        "--screenshot",
        default=None,
        help="Optional PNG path used to verify the rendered application frame.",
    )
    parser.add_argument("--output", default="outputs/isaacsim_validation/result.json")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read OBJ vertices/faces and triangulate polygon faces."""
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                fields = line.split()[1:]
                indices = []
                for field in fields:
                    index = int(field.split("/", 1)[0])
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                for offset in range(1, len(indices) - 1):
                    triangles.append([indices[0], indices[offset], indices[offset + 1]])
    if not vertices or not triangles:
        raise ValueError(f"OBJ has no triangle geometry: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an Isaac scalar-first quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return (quat / np.linalg.norm(quat)).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.visualize and args.streaming:
        raise ValueError("--visualize and --streaming are mutually exclusive")
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    candidate_payload = json.loads(resolve(args.candidates).read_text(encoding="utf-8"))
    record = candidate_payload["records"][args.record_index]
    candidates = record["fk"]["candidates"]
    candidate = next(item for item in candidates if int(item["rank"]) == args.candidate_rank)
    gripper_name = record["gripper"]
    gripper_spec = config["grippers"][gripper_name]
    overclose_fraction = float(
        args.overclose_fraction
        if args.overclose_fraction is not None
        else config["isaac"].get("overclose_fraction", 0.02)
    )
    configured_effort = (
        args.max_joint_effort
        if args.max_joint_effort is not None
        else config["isaac"].get("max_joint_effort")
    )
    max_joint_effort = (
        None if configured_effort is None else float(configured_effort)
    )
    solver_position_iterations = int(
        args.solver_position_iterations
        if args.solver_position_iterations is not None
        else config["isaac"].get("solver_position_iterations", 64)
    )
    solver_velocity_iterations = int(
        args.solver_velocity_iterations
        if args.solver_velocity_iterations is not None
        else config["isaac"].get("solver_velocity_iterations", 8)
    )

    portable_root = os.environ.get("CONTACTDIFF_OMNI_PORTABLE_ROOT")
    sys.argv = [sys.argv[0]]
    if portable_root:
        sys.argv.extend(["--portable-root", portable_root])
    omni_logs = Path(os.environ.get("OMNI_LOGS", str(REPO_ROOT / ".cache" / "ov-logs")))
    omni_logs.mkdir(parents=True, exist_ok=True)
    process_tag = f"grasp_gpu{args.gpu_id}_{os.getpid()}"
    runtime_root = Path(
        os.environ.get(
            "CONTACTDIFF_ISAAC_RUNTIME_ROOT",
            str(REPO_ROOT / ".cache" / "isaacsim" / "runtime"),
        )
    )
    runtime_dir = runtime_root / process_tag
    runtime_dir.mkdir(parents=True, exist_ok=False)
    # The URDF importer creates fixed-name intermediate files such as
    # ``palm.tmp.usd`` in the current directory.  One private directory per
    # process prevents collisions when multiple GPUs validate in parallel.
    os.chdir(runtime_dir)
    sys.argv.extend(
        [
            f"--/app/userConfigPath={omni_logs / (process_tag + '_user.config.json')}",
            f"--/log/file={omni_logs / (process_tag + '.log')}",
            f"--/structuredLog/logDirectory={omni_logs / 'structured'}",
            "--/structuredLog/enable=false",
        ]
    )

    import isaacsim as isaacsim_package

    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError:
        from isaacsim import SimulationApp

    if os.environ.get("CONTACTDIFF_SKIP_VIEWPORT_WAIT") == "1":
        # The local RTX 50-series host only needs PhysX. Isaac Sim 4.5 may
        # crash while polling a viewport even in headless mode.
        if hasattr(SimulationApp, "_wait_for_viewport"):
            SimulationApp._wait_for_viewport = lambda self: None

    use_cpu_physics = bool(args.cpu_physics)
    render_enabled = bool(args.visualize or args.streaming)
    renderer_name = os.environ.get(
        "CONTACTDIFF_ISAAC_RENDERER", "MinimalRendering"
    )
    app_config = {
        "headless": True if args.streaming else (
            False if args.visualize else bool(config["isaac"]["headless"])
        ),
        "active_gpu": int(args.gpu_id),
        "physics_gpu": None if use_cpu_physics else int(args.gpu_id),
        "multi_gpu": False,
        "fast_shutdown": True,
        "renderer": renderer_name,
        "minimal_shading_mode": int(
            os.environ.get("CONTACTDIFF_MINIMAL_SHADING_MODE", "2")
        ),
        "disable_viewport_updates": not render_enabled,
        "limit_cpu_threads": 8,
        "sync_loads": False,
        "extra_args": [
            "--ext-folder",
            str(Path(isaacsim_package.__file__).resolve().parent / "extscache"),
            "--ext-folder",
            str(Path(isaacsim_package.__file__).resolve().parent / "extsDeprecated"),
        ],
    }
    if os.environ.get("CONTACTDIFF_SKIP_VIEWPORT_WAIT") == "1":
        app_config["extra_args"].append("--/renderer/enabled=pxr")
    # The grasp validator does not use ROS. Isaac Sim Full enables the ROS2
    # bridge automatically on Linux, and an unavailable ROS runtime can block
    # startup while the extension tears itself down. Keep it disabled by
    # default while allowing an explicit opt-in for ROS-integrated tests.
    if os.environ.get("CONTACTDIFF_ENABLE_ROS_BRIDGE", "0") != "1":
        app_config["extra_args"].append(
            "--/isaac/startup/ros_bridge_extension="
        )
    experience = str(resolve(args.experience)) if args.experience else ""
    simulation_app = SimulationApp(app_config, experience=experience)

    report: dict = {
        "status": "error",
        "checkpoint": candidate_payload.get("checkpoint"),
        "checkpoint_step": candidate_payload.get("checkpoint_step"),
        "gripper": gripper_name,
        "n": int(record["n"]),
        "object_id": record.get("object_id"),
        "source_split": record.get("source_split"),
        "object_index": record.get("object_index"),
        "sample_index": record.get("sample_index"),
        "sample_seed": record.get("sample_seed"),
        "record_index": int(args.record_index),
        "candidate_rank": int(args.candidate_rank),
        "contact_chamfer_m": float(candidate["contact_chamfer_m"]),
        "friction": float(args.friction),
        "mass_kg": float(args.mass),
        "drive_stiffness": float(args.drive_stiffness),
        "drive_damping": float(args.drive_damping),
        "overclose_fraction": overclose_fraction,
        "max_joint_effort": max_joint_effort,
        "solver_position_iterations": solver_position_iterations,
        "solver_velocity_iterations": solver_velocity_iterations,
    }
    output_path = resolve(args.output)
    try:
        import omni.kit.commands
        import omni.usd
        try:
            from isaacsim.core.api import World
            from isaacsim.core.prims import (
                SingleArticulation as Articulation,
                SingleRigidPrim as RigidPrim,
                SingleXFormPrim as XFormPrim,
            )
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core import World
            from omni.isaac.core.articulations import Articulation
            from omni.isaac.core.prims import RigidPrim, XFormPrim
            from omni.isaac.core.utils.types import ArticulationAction
        from omni.physx.scripts import physicsUtils
        from pxr import (
            Gf,
            PhysxSchema,
            Sdf,
            Usd,
            UsdGeom,
            UsdLux,
            UsdPhysics,
            UsdShade,
            Vt,
        )

        physics_dt = float(config["isaac"]["physics_dt"])
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=physics_dt,
            rendering_dt=physics_dt,
            backend="numpy",
            device="cpu" if use_cpu_physics else None,
        )
        world.get_physics_context().set_gravity(0.0)
        stage = omni.usd.get_context().get_stage()

        if render_enabled:
            dome_light = UsdLux.DomeLight.Define(stage, "/World/VisualDomeLight")
            dome_light.CreateIntensityAttr(700.0)
            dome_light.CreateColorAttr(Gf.Vec3f(0.85, 0.90, 1.0))
            key_light = UsdLux.DistantLight.Define(stage, "/World/VisualKeyLight")
            key_light.CreateIntensityAttr(2500.0)
            key_light.CreateAngleAttr(3.0)
            key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.82))
            UsdGeom.Xformable(key_light).AddRotateXYZOp().Set(
                Gf.Vec3f(-35.0, 25.0, -35.0)
            )

        material_path = "/World/PhysicsMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr(float(args.friction))
        material_api.CreateDynamicFrictionAttr(float(args.friction))
        material_api.CreateRestitutionAttr(0.0)

        object_path = "/World/Object"
        vertices, faces = load_obj(Path(record["object_mesh"]))
        object_mesh = UsdGeom.Mesh.Define(stage, object_path)
        object_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        object_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
        object_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
        object_mesh.CreateSubdivisionSchemeAttr("none")
        object_mesh.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.42, 0.82)])
        UsdPhysics.CollisionAPI.Apply(object_mesh.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(object_mesh.GetPrim())
        mesh_collision.CreateApproximationAttr("convexDecomposition")
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(object_mesh.GetPrim())
        # Optionally hold the object fixed while the fingers close.  Dynamic
        # closure is the default so the object can self-center in the grasp.
        kinematic_attr = rigid_body_api.CreateKinematicEnabledAttr(bool(args.kinematic_close))
        physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(object_mesh.GetPrim())
        linear_damping_attr = physx_rigid_body.CreateLinearDampingAttr(10.0)
        angular_damping_attr = physx_rigid_body.CreateAngularDampingAttr(10.0)
        mass_api = UsdPhysics.MassAPI.Apply(object_mesh.GetPrim())
        mass_api.CreateMassAttr(float(args.mass))
        physicsUtils.add_physics_material_to_prim(stage, object_mesh.GetPrim(), material_path)

        gripper_root = resolve(config["paths"]["gripper_root"])
        usd_relative = gripper_spec.get("usd")
        if usd_relative:
            usd_path = gripper_root / usd_relative
            if not usd_path.is_file():
                raise FileNotFoundError(f"Preconverted gripper USD is missing: {usd_path}")
            source_stage = Usd.Stage.Open(str(usd_path))
            source_default_prim = source_stage.GetDefaultPrim()
            if not source_default_prim.IsValid():
                raise RuntimeError(f"Gripper USD has no default prim: {usd_path}")
            robot_root_path = "/World/GripperRoot"
            articulation_path = f"{robot_root_path}/{source_default_prim.GetName()}"
            UsdGeom.Xform.Define(stage, robot_root_path)
            articulation_prim = stage.DefinePrim(articulation_path)
            articulation_prim.GetReferences().AddReference(
                str(usd_path), source_default_prim.GetPath()
            )
            asset_source = "usd"
        else:
            urdf_path = gripper_root / gripper_spec["urdf"]
            asset_cache = (
                resolve(config["paths"]["isaac_asset_cache"])
                / "physics_imports"
                / gripper_name
            )
            asset_cache.mkdir(parents=True, exist_ok=True)
            try:
                # Isaac Sim 6 converter API.
                from isaacsim.asset.importer.urdf import (
                    URDFImporter,
                    URDFImporterConfig,
                )

                import_config = URDFImporterConfig(
                    urdf_path=str(urdf_path),
                    usd_path=str(asset_cache),
                    merge_fixed_joints=False,
                    collision_from_visuals=False,
                    collision_type="Convex Decomposition",
                    allow_self_collision=False,
                    fix_base=True,
                    joint_drive_type="force",
                    joint_target_type="position",
                    override_joint_stiffness=float(args.drive_stiffness),
                    override_joint_damping=float(args.drive_damping),
                    # Keep physics schemas in one generated layer.
                    run_asset_transformer=False,
                )
                converted_path = Path(URDFImporter(import_config).import_urdf())
            except ImportError:
                # Isaac Sim 4.5 command API.
                status, import_config = omni.kit.commands.execute(
                    "URDFCreateImportConfig"
                )
                if not status:
                    raise RuntimeError("URDFCreateImportConfig failed")
                import_config.merge_fixed_joints = False
                import_config.convex_decomp = True
                import_config.set_collision_from_visuals(False)
                import_config.set_self_collision(False)
                import_config.fix_base = True
                converted_path = asset_cache / f"{gripper_name}.usd"
                status, _ = omni.kit.commands.execute(
                    "URDFParseAndImportFile",
                    urdf_path=str(urdf_path),
                    import_config=import_config,
                    dest_path=str(converted_path),
                )
                if not status:
                    raise RuntimeError(
                        f"URDFParseAndImportFile failed for {urdf_path}"
                    )
            source_stage = Usd.Stage.Open(str(converted_path))
            source_default_prim = source_stage.GetDefaultPrim()
            if not source_default_prim.IsValid():
                root_prims = list(source_stage.GetPseudoRoot().GetChildren())
                if not root_prims:
                    raise RuntimeError(
                        f"Converted gripper USD contains no root prim: {converted_path}"
                    )
                source_default_prim = root_prims[0]
            report.update(
                {
                    "converted_gripper_asset": str(converted_path),
                    "converted_source_collision_prim_count": sum(
                        prim.HasAPI(UsdPhysics.CollisionAPI)
                        for prim in source_stage.Traverse()
                    ),
                    "converted_source_rigid_body_prim_count": sum(
                        prim.HasAPI(UsdPhysics.RigidBodyAPI)
                        for prim in source_stage.Traverse()
                    ),
                }
            )
            robot_root_path = "/World/GripperRoot"
            articulation_path = f"{robot_root_path}/{source_default_prim.GetName()}"
            UsdGeom.Xform.Define(stage, robot_root_path)
            articulation_prim = stage.DefinePrim(articulation_path)
            articulation_prim.GetReferences().AddReference(
                str(converted_path), source_default_prim.GetPath()
            )
            asset_source = "urdf_converted"
        if not articulation_prim.IsValid():
            raise RuntimeError(f"Invalid articulation path: {articulation_path}")
        articulation_prims = list(Usd.PrimRange(articulation_prim))
        collision_prim_paths = [
            str(prim.GetPath())
            for prim in articulation_prims
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        rigid_body_prim_paths = [
            str(prim.GetPath())
            for prim in articulation_prims
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        mass_prim_paths = [
            str(prim.GetPath())
            for prim in articulation_prims
            if prim.HasAPI(UsdPhysics.MassAPI)
        ]
        report.update(
            {
                "gripper_collision_prim_count": len(collision_prim_paths),
                "gripper_rigid_body_prim_count": len(rigid_body_prim_paths),
                "gripper_mass_prim_count": len(mass_prim_paths),
                "gripper_collision_prim_paths": collision_prim_paths,
            }
        )
        physics_articulation_roots = [
            prim
            for prim in articulation_prims
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if not physics_articulation_roots:
            raise RuntimeError(
                f"No PhysicsArticulationRootAPI found below {articulation_path}"
            )
        for root_prim in physics_articulation_roots:
            physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
            physx_articulation.CreateSolverPositionIterationCountAttr(
                solver_position_iterations
            )
            physx_articulation.CreateSolverVelocityIterationCountAttr(
                solver_velocity_iterations
            )
        report["physics_articulation_root_paths"] = [
            str(prim.GetPath()) for prim in physics_articulation_roots
        ]
        for prim in Usd.PrimRange(articulation_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                physicsUtils.add_physics_material_to_prim(stage, prim, material_path)

        gripper = world.scene.add(
            Articulation(prim_path=str(articulation_path), name="validated_gripper")
        )
        rigid_object = world.scene.add(RigidPrim(prim_path=object_path, name="grasped_object"))
        root_pose = np.asarray(candidate["root_pose"], dtype=np.float64)
        robot_root_path = str(Sdf.Path(str(articulation_path)).GetParentPath())
        robot_root = XFormPrim(prim_path=robot_root_path, name="validated_gripper_root")
        # A fixed-base URDF creates its world joint during reset.  Position its
        # model root before reset so that the fixed joint is anchored at the
        # optimized pose (setting the articulation after reset is ignored).
        robot_root.set_world_pose(
            position=root_pose[:3, 3].astype(np.float32),
            orientation=matrix_to_wxyz(root_pose[:3, :3]),
        )
        world.reset()
        gripper.set_solver_position_iteration_count(solver_position_iterations)
        gripper.set_solver_velocity_iteration_count(solver_velocity_iterations)
        if render_enabled:
            from omni.kit.viewport.utility import get_active_viewport

            camera_path = "/World/ContactDiffusionCamera"
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.CreateFocalLengthAttr(35.0)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 1000.0))
            eye = Gf.Vec3d(*[float(value) for value in args.camera_eye])
            target = Gf.Vec3d(*[float(value) for value in args.camera_target])
            view_matrix = Gf.Matrix4d().SetLookAt(
                eye,
                target,
                Gf.Vec3d(0.0, 0.0, 1.0),
            )
            UsdGeom.Xformable(camera).AddTransformOp().Set(view_matrix.GetInverse())
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("No active viewport is available for visualization")
            viewport.camera_path = Sdf.Path(camera_path)
            viewport.resolution = (1280, 720)
            for _ in range(30):
                simulation_app.update()
            if os.environ.get("CONTACTDIFF_SHOW_CASE_INFO") == "1":
                import omni.ui as ui

                case_info_window = ui.Window(
                    "ContactDiffusion Sample",
                    width=720,
                    height=150,
                )
                case_info_window.position_x = 20
                case_info_window.position_y = 80
                with case_info_window.frame:
                    with ui.VStack(spacing=4, height=0):
                        ui.Label(
                            f"checkpoint {candidate_payload.get('checkpoint_step')} | "
                            f"{gripper_name} | record {args.record_index} | "
                            f"diffusion sample {record.get('sample_index')} | "
                            f"FK rank {args.candidate_rank}"
                        )
                        ui.Label(f"object: {record.get('object_id')}")
                        ui.Label(
                            f"contact chamfer: "
                            f"{1000.0 * float(candidate['contact_chamfer_m']):.3f} mm | "
                            f"gravity test: {'off (pose preview)' if args.skip_gravity_test else 'on'}"
                        )
        fk_names = list(record["fk"]["joint_names"])
        fk_index = {name: index for index, name in enumerate(fk_names)}
        dof_names = list(gripper.dof_names)
        missing = [name for name in dof_names if name not in fk_index]
        if missing:
            raise RuntimeError(f"Isaac DOFs missing from FK solution: {missing}")
        q_contact_fk = np.asarray(candidate["joint_positions"], dtype=np.float32)
        lower_fk = np.asarray(record["fk"]["joint_lower"], dtype=np.float32)
        upper_fk = np.asarray(record["fk"]["joint_upper"], dtype=np.float32)
        close_dir_fk = np.asarray(gripper_spec["close_dir"], dtype=np.float32)
        if len(close_dir_fk) != len(fk_names):
            raise RuntimeError("close_dir length does not match FK joint order")
        q_contact = np.asarray([q_contact_fk[fk_index[name]] for name in dof_names], dtype=np.float32)
        lower = np.asarray([lower_fk[fk_index[name]] for name in dof_names], dtype=np.float32)
        upper = np.asarray([upper_fk[fk_index[name]] for name in dof_names], dtype=np.float32)
        close_dir = np.asarray([close_dir_fk[fk_index[name]] for name in dof_names], dtype=np.float32)
        span = upper - lower
        q_open = np.clip(q_contact - float(args.open_fraction) * span * close_dir, lower, upper)
        q_closed = np.clip(q_contact + overclose_fraction * span * close_dir, lower, upper)

        gripper.set_joint_positions(q_open)
        controller = gripper.get_articulation_controller()
        controller.switch_control_mode("position")
        controller.set_gains(
            kps=np.full(len(dof_names), float(args.drive_stiffness), dtype=np.float32),
            kds=np.full(len(dof_names), float(args.drive_damping), dtype=np.float32),
        )
        if max_joint_effort is not None:
            controller.set_max_efforts(
                np.full(len(dof_names), max_joint_effort, dtype=np.float32)
            )
        zero_velocity = np.zeros(len(dof_names), dtype=np.float32)

        def hold_action(target: np.ndarray) -> None:
            controller.apply_action(
                ArticulationAction(
                    joint_positions=target,
                    joint_velocities=zero_velocity,
                )
            )

        preclose_position, _ = rigid_object.get_world_pose()
        close_steps = max(1, round(float(config["isaac"]["close_seconds"]) / physics_dt))
        settle_steps = max(1, round(float(config["isaac"]["settle_seconds"]) / physics_dt))
        for step in range(close_steps):
            alpha = float(step + 1) / close_steps
            target = q_open + alpha * (q_closed - q_open)
            hold_action(target)
            world.step(render=bool(render_enabled and step % 4 == 0))
            if args.realtime:
                time.sleep(physics_dt)
        for step in range(settle_steps):
            hold_action(q_closed)
            world.step(render=bool(render_enabled and step % 4 == 0))
            if args.realtime:
                time.sleep(physics_dt)

        # Capture the closed grasp before an optional long interactive hold.
        # Waiting until after the hold would make preview screenshots
        # unavailable for the entire hold duration.
        if render_enabled and args.screenshot:
            screenshot_path = resolve(args.screenshot)
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            import omni.kit.renderer_capture

            renderer_capture = omni.kit.renderer_capture.acquire_renderer_capture_interface()
            renderer_capture.capture_next_frame_swapchain(str(screenshot_path))
            for _ in range(4):
                simulation_app.update()
            renderer_capture.wait_async_capture()
            simulation_app.update()
            report["screenshot"] = str(screenshot_path)
            report["screenshot_written"] = screenshot_path.is_file()

        if render_enabled and args.pre_release_hold_seconds > 0:
            hold_deadline = time.monotonic() + float(args.pre_release_hold_seconds)
            while time.monotonic() < hold_deadline:
                hold_action(q_closed)
                world.step(render=True)
                if args.realtime:
                    time.sleep(physics_dt)

        calibration = json.loads(
            resolve(config["paths"]["tip_offset_calibration"]).read_text(encoding="utf-8")
        )["grippers"][gripper_name]
        xform_cache = UsdGeom.XformCache()
        actual_tip_points = []
        for link_name in gripper_spec["tip_links"]:
            link_prim = stage.GetPrimAtPath(f"{robot_root_path}/{link_name}")
            if not link_prim.IsValid():
                matches = [
                    prim
                    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root_path))
                    if prim.GetName() == link_name
                ]
                if not matches:
                    raise RuntimeError(f"Could not find imported tip link: {link_name}")
                link_prim = matches[0]
            offset = calibration["tips"][link_name]["offset_xyz"]
            point = xform_cache.GetLocalToWorldTransform(link_prim).Transform(
                Gf.Vec3d(*[float(value) for value in offset])
            )
            actual_tip_points.append([float(point[0]), float(point[1]), float(point[2])])
        actual_tips = np.asarray(actual_tip_points, dtype=np.float64)
        actual_joint_positions = np.asarray(gripper.get_joint_positions(), dtype=np.float64)
        target_contacts = np.asarray(record["fk"]["target_contacts"], dtype=np.float64)
        pairwise = np.linalg.norm(actual_tips[:, None, :] - target_contacts[None, :, :], axis=-1)
        actual_tip_chamfer = float(pairwise.min(axis=1).mean() + pairwise.min(axis=0).mean())

        initial_position, initial_orientation = rigid_object.get_world_pose()
        pre_release_drift = float(np.linalg.norm(initial_position - preclose_position))
        if args.skip_gravity_test:
            gravity_steps = 0
        else:
            kinematic_attr.Set(False)
            linear_damping_attr.Set(0.1)
            angular_damping_attr.Set(0.1)
            world.get_physics_context().set_gravity(-9.81)
            gravity_steps = max(1, round(float(config["isaac"]["gravity_test_seconds"]) / physics_dt))
            for step in range(gravity_steps):
                hold_action(q_closed)
                world.step(render=bool(render_enabled and step % 4 == 0))
                if args.realtime:
                    time.sleep(physics_dt)
        final_position, final_orientation = rigid_object.get_world_pose()
        final_joint_positions = np.asarray(gripper.get_joint_positions(), dtype=np.float64)
        joint_tracking_error = np.abs(final_joint_positions - q_closed)
        applied_action = controller.get_applied_action()
        applied_joint_positions = (
            None
            if applied_action is None or applied_action.joint_positions is None
            else np.asarray(applied_action.joint_positions, dtype=np.float64).tolist()
        )
        controller_kps, controller_kds = controller.get_gains()
        controller_max_efforts = controller.get_max_efforts()
        displacement = float(np.linalg.norm(final_position - initial_position))
        vertical_drop = float(initial_position[2] - final_position[2])
        threshold = float(config["isaac"]["failure_displacement_m"])
        report.update(
            {
                "status": "ok",
                "success": (
                    None
                    if args.skip_gravity_test
                    else bool(pre_release_drift <= threshold and displacement <= threshold)
                ),
                "evaluation_skipped": bool(args.skip_gravity_test),
                "valid_initialization": bool(pre_release_drift <= threshold),
                "kinematic_close": bool(args.kinematic_close),
                "failure_displacement_m": threshold,
                "displacement_m": displacement,
                "vertical_drop_m": vertical_drop,
                "pre_release_drift_m": pre_release_drift,
                "actual_tip_chamfer_m": actual_tip_chamfer,
                "actual_tip_points": actual_tip_points,
                "actual_joint_positions": actual_joint_positions.tolist(),
                "final_joint_positions": final_joint_positions.tolist(),
                "max_joint_tracking_error": float(joint_tracking_error.max(initial=0.0)),
                "mean_joint_tracking_error": float(joint_tracking_error.mean()),
                "commanded_closed_joint_positions": q_closed.tolist(),
                "applied_joint_position_targets": applied_joint_positions,
                "controller_kps": np.asarray(controller_kps, dtype=np.float64).tolist(),
                "controller_kds": np.asarray(controller_kds, dtype=np.float64).tolist(),
                "controller_max_efforts": (
                    None
                    if controller_max_efforts is None
                    else np.asarray(controller_max_efforts, dtype=np.float64).tolist()
                ),
                "target_contacts": target_contacts.tolist(),
                "initial_object_position": np.asarray(initial_position).tolist(),
                "final_object_position": np.asarray(final_position).tolist(),
                "initial_object_orientation_wxyz": np.asarray(initial_orientation).tolist(),
                "final_object_orientation_wxyz": np.asarray(final_orientation).tolist(),
                "physics_dt": physics_dt,
                "close_steps": close_steps,
                "settle_steps": settle_steps,
                "gravity_steps": gravity_steps,
                "articulation_path": str(articulation_path),
                "robot_root_path": robot_root_path,
                "gripper_asset_source": asset_source,
                "isaac_dof_names": dof_names,
                "camera_path": camera_path if render_enabled else None,
            }
        )
        if render_enabled and args.keep_open_seconds > 0:
            keep_open_deadline = time.monotonic() + float(args.keep_open_seconds)
            while time.monotonic() < keep_open_deadline and simulation_app.is_running():
                hold_action(q_closed)
                world.step(render=True)
                if args.realtime:
                    time.sleep(physics_dt)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        os.chdir(REPO_ROOT)
        shutil.rmtree(runtime_dir, ignore_errors=True)
        if os.environ.get("CONTACTDIFF_HARD_EXIT_ISAAC") == "1":
            # Each validation trial has its own process. Avoid Isaac Sim 4.5
            # shutdown callbacks that can segfault on the local Blackwell GPU.
            os._exit(0 if report.get("status") == "ok" else 1)
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
