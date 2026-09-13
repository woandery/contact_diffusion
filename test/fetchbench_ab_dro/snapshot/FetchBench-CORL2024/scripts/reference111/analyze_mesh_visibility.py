import csv
import json
import math
import os
import sys
from pathlib import Path

import hydra
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from omegaconf import DictConfig

import isaacgym
from isaacgym import gymapi
import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from render_camera_sweep import (
    _load_support_geometry,
    _quaternion_xyzw_to_matrix,
    _render_camera,
    _unproject_target_points,
)


ELEVATIONS = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
AZIMUTHS = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
RADII = [1.0, 1.5, 2.0]


def _sample_partial_cloud(points, sample_count, seed):
    if len(points) == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=sample_count, replace=len(points) < sample_count)
    return points[indices].astype(np.float32)


def _camera_directions():
    directions = []
    for elevation in ELEVATIONS:
        selected = [0.0] if math.isclose(abs(elevation), 90.0) else AZIMUTHS
        directions.extend((elevation, azimuth) for azimuth in selected)
    return directions


def _camera_position(center, radius, elevation_deg, azimuth_deg):
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    horizontal_radius = radius * math.cos(elevation)
    return np.asarray(center, dtype=np.float64) + np.asarray(
        [
            -horizontal_radius * math.cos(azimuth),
            horizontal_radius * math.sin(azimuth),
            radius * math.sin(elevation),
        ],
        dtype=np.float64,
    )


def _load_scene_objects(asset_path, scene_ref, task_index):
    task_dir = asset_path / "Task" / scene_ref
    with (task_dir / "asset_config.json").open() as stream:
        asset_config = json.load(stream)
    rearrange = np.load(task_dir / "rearrange_config.npz", allow_pickle=True)
    labels = rearrange["object_labels"][task_index]
    records = []
    for index, (config, label) in enumerate(zip(asset_config["object_config"], labels)):
        source_root = str(config["asset_root"])
        marker = "/benchmark_objects/"
        if marker not in source_root:
            raise ValueError(f"Unsupported object asset path: {source_root}")
        relative_root = source_root.split(marker, 1)[1]
        mesh_root = asset_path / "benchmark_objects" / relative_root
        records.append(
            {
                "index": index,
                "name": config["name"],
                "category": Path(relative_root).parts[0],
                "label": str(label),
                "mesh_path": mesh_root / "mesh.obj",
            }
        )
    return records


def _sample_mesh_surface(mesh_path, sample_count, seed):
    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected one mesh in {mesh_path}, got {type(mesh)}")
    np.random.seed(seed)
    points, face_indices = trimesh.sample.sample_surface(mesh, sample_count)
    normals = mesh.face_normals[face_indices]
    return {
        "points_local": np.asarray(points, dtype=np.float64),
        "normals_local": np.asarray(normals, dtype=np.float64),
        "surface_area_m2": float(mesh.area),
        "face_count": int(len(mesh.faces)),
        "vertex_count": int(len(mesh.vertices)),
    }


def _camera_projection(task, env, camera_handle, width, height):
    projection = np.asarray(
        task.gym.get_camera_proj_matrix(task.sim, env, camera_handle),
        dtype=np.float64,
    )
    view = np.asarray(
        task.gym.get_camera_view_matrix(task.sim, env, camera_handle),
        dtype=np.float64,
    )
    fu = 2.0 / projection[0, 0]
    fv = 2.0 / projection[1, 1]
    intrinsics = np.asarray(
        [
            [-width / fu, 0.0, width / 2.0],
            [0.0, height / fv, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    origin = task.gym.get_env_origin(env)
    env_origin = np.asarray([origin.x, origin.y, origin.z], dtype=np.float64)
    return view, intrinsics, env_origin


def _project_surface(points_world, view, intrinsics, env_origin):
    global_points = points_world + env_origin[None, :]
    homogeneous = np.concatenate(
        (global_points, np.ones((len(global_points), 1), dtype=np.float64)), axis=1
    )
    camera_points = homogeneous @ view
    in_front = camera_points[:, 2] < -1e-5
    normalized = camera_points[:, :3] / camera_points[:, 2:3]
    pixels = normalized @ intrinsics.T
    return pixels[:, :2], -camera_points[:, 2], in_front


def _visible_surface_mask(
    points_world,
    normals_world,
    camera_position,
    view,
    intrinsics,
    env_origin,
    depth_m,
    segmentation,
    segmentation_id,
    depth_tolerance_m,
):
    height, width = depth_m.shape
    pixels, sample_depth, in_front = _project_surface(
        points_world, view, intrinsics, env_origin
    )
    columns = np.rint(pixels[:, 0]).astype(np.int64)
    rows = np.rint(pixels[:, 1]).astype(np.int64)
    in_frame = (
        in_front
        & (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )

    toward_camera = np.asarray(camera_position)[None, :] - points_world
    front_facing = np.einsum("ij,ij->i", normals_world, toward_camera) > 0.0
    visible = np.zeros(len(points_world), dtype=bool)
    valid_indices = np.flatnonzero(in_frame & front_facing)
    if len(valid_indices) == 0:
        return visible

    sampled_depth = depth_m[rows[valid_indices], columns[valid_indices]]
    sampled_segmentation = segmentation[rows[valid_indices], columns[valid_indices]]
    # A pixel subtends a larger physical footprint farther from the camera.
    # Scale the tolerance by depth so this geometric estimate is not biased
    # toward near cameras merely because the rendered depth grid is denser.
    adaptive_tolerance = depth_tolerance_m + 0.0038 * sample_depth[valid_indices]
    depth_match = (
        np.isfinite(sampled_depth)
        & (np.abs(sampled_depth - sample_depth[valid_indices]) <= adaptive_tolerance)
        & (sampled_segmentation == segmentation_id)
    )
    visible[valid_indices[depth_match]] = True
    return visible


def _plot_summary(object_results, output_path):
    labels = [item["category"] for item in object_results]
    observable = [item["observable_fraction_full"] * 100.0 for item in object_results]
    best_full = [item["max_single_fraction_full"] * 100.0 for item in object_results]
    best_observable = [
        item["max_single_fraction_observable"] * 100.0 for item in object_results
    ]
    x = np.arange(len(labels))
    width = 0.25
    figure, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    ax.bar(x - width, observable, width, label="observable union / full surface", color="#286f6c")
    ax.bar(x, best_full, width, label="best single view / full surface", color="#e66b45")
    ax.bar(x + width, best_observable, width, label="best single view / observable union", color="#e5ad32")
    ax.set_xticks(x, labels)
    ax.set_ylabel("mesh surface coverage (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("FetchBench camera visibility: equal-area mesh estimate")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def _plot_heatmaps(object_results, records, output_path):
    rows = len(object_results)
    figure, axes = plt.subplots(
        rows,
        len(RADII),
        figsize=(14.5, 2.65 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row_index, object_result in enumerate(object_results):
        object_records = [
            record
            for record in records
            if record["object_index"] == object_result["object_index"]
        ]
        for column_index, radius in enumerate(RADII):
            matrix = np.full((len(ELEVATIONS), len(AZIMUTHS)), np.nan)
            for record in object_records:
                if not math.isclose(record["radius_m"], radius):
                    continue
                elevation_index = ELEVATIONS.index(record["elevation_deg"])
                azimuth_index = AZIMUTHS.index(record["azimuth_deg"])
                matrix[elevation_index, azimuth_index] = (
                    record["fraction_observable"] * 100.0
                )
            ax = axes[row_index, column_index]
            image = ax.imshow(
                matrix,
                origin="lower",
                vmin=0,
                vmax=100,
                cmap="YlGnBu",
                aspect="auto",
            )
            ax.set_xticks(range(len(AZIMUTHS)), [f"{v:+.0f}" for v in AZIMUTHS])
            ax.set_yticks(range(len(ELEVATIONS)), [f"{v:+.0f}" for v in ELEVATIONS])
            ax.set_xlabel("azimuth (deg)")
            if column_index == 0:
                ax.set_ylabel(f"{object_result['category']}\nelevation (deg)")
            if row_index == 0:
                ax.set_title(f"r = {radius:.1f} m")
            for y_index in range(len(ELEVATIONS)):
                for x_index in range(len(AZIMUTHS)):
                    value = matrix[y_index, x_index]
                    if np.isfinite(value) and value > 0.0:
                        ax.text(
                            x_index,
                            y_index,
                            f"{value:.0f}",
                            ha="center",
                            va="center",
                            fontsize=6.5,
                            color="white" if value > 48 else "#17302e",
                        )
    colorbar = figure.colorbar(image, ax=axes, shrink=0.7, pad=0.015)
    colorbar.set_label("single-view coverage / observable surface (%)")
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def _write_csv(records, output_path):
    if not records:
        return
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _outlined_target_image(rgb, segmentation, segmentation_id):
    mask = segmentation == segmentation_id
    interior = (
        mask
        & np.roll(mask, 1, axis=0)
        & np.roll(mask, -1, axis=0)
        & np.roll(mask, 1, axis=1)
        & np.roll(mask, -1, axis=1)
    )
    boundary = mask & ~interior
    thick_boundary = (
        boundary
        | np.roll(boundary, 1, axis=0)
        | np.roll(boundary, -1, axis=0)
        | np.roll(boundary, 1, axis=1)
        | np.roll(boundary, -1, axis=1)
    )
    output = rgb.copy()
    output[thick_boundary] = np.asarray([0, 255, 255], dtype=np.uint8)
    return output


def _plot_all_camera_views(object_result, object_records, output_path):
    record_by_pose = {
        (
            record["radius_m"],
            record["elevation_deg"],
            record["azimuth_deg"],
        ): record
        for record in object_records
    }
    display_elevations = list(reversed(ELEVATIONS))
    figure, axes = plt.subplots(
        len(display_elevations),
        len(AZIMUTHS) * len(RADII),
        figsize=(31.5, 11.4),
        constrained_layout=True,
        squeeze=False,
    )
    for radius_index, radius in enumerate(RADII):
        for elevation_index, elevation in enumerate(display_elevations):
            for azimuth_index, azimuth in enumerate(AZIMUTHS):
                column = radius_index * len(AZIMUTHS) + azimuth_index
                ax = axes[elevation_index, column]
                ax.axis("off")
                record = record_by_pose.get((radius, elevation, azimuth))
                if record is None:
                    continue
                ax.imshow(iio.imread(record["target_outline_image"]))
                percentage = record["fraction_observable"] * 100.0
                ax.set_title(
                    f"{percentage:.1f}%",
                    fontsize=7.2,
                    color="#b42318" if percentage == 0.0 else "#172b2a",
                    pad=1.5,
                )
                if azimuth_index == 0:
                    ax.text(
                        -0.08,
                        0.5,
                        f"el {elevation:+.0f} deg",
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        rotation=90,
                        fontsize=7.5,
                    )
                if elevation_index == len(display_elevations) - 1:
                    ax.text(
                        0.5,
                        -0.08,
                        f"az {azimuth:+.0f}",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=7.0,
                    )
        first_column = radius_index * len(AZIMUTHS)
        middle_column = first_column + len(AZIMUTHS) // 2
        axes[0, middle_column].text(
            0.5,
            1.35,
            f"r = {radius:.1f} m",
            transform=axes[0, middle_column].transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
            weight="semibold",
        )
    figure.suptitle(
        f"{object_result['category']}: all 111 camera views | cyan = exact target mask outline",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=140, facecolor="white")
    plt.close(figure)


def _representative_records(object_records):
    positive = [record for record in object_records if record["fraction_observable"] > 0.0]
    if not positive:
        return []
    selections = []

    def add(label, record):
        pose = (record["radius_m"], record["elevation_deg"], record["azimuth_deg"])
        if pose not in {item[1] for item in selections}:
            selections.append((label, pose, record))

    add("best", max(positive, key=lambda record: record["fraction_observable"]))
    for label, target in (("near 50%", 0.50), ("near 40%", 0.40), ("near 25%", 0.25)):
        add(
            label,
            min(positive, key=lambda record: abs(record["fraction_observable"] - target)),
        )
    add("lowest non-zero", min(positive, key=lambda record: record["fraction_observable"]))
    lower = [record for record in positive if record["elevation_deg"] < 0.0]
    if lower:
        add("best below center", max(lower, key=lambda record: record["fraction_observable"]))
    return [(label, record) for label, _, record in selections]


def _plot_representative_views(object_result, object_records, output_path):
    selections = _representative_records(object_records)
    if not selections:
        return
    columns = min(3, len(selections))
    rows = int(math.ceil(len(selections) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 4.3 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, (label, record) in zip(axes.flat, selections):
        ax.imshow(iio.imread(record["target_outline_image"]))
        ax.set_title(
            f"{label}: {record['fraction_observable'] * 100.0:.1f}%\n"
            f"r={record['radius_m']:.1f} m, el={record['elevation_deg']:+.0f} deg, "
            f"az={record['azimuth_deg']:+.0f} deg",
            fontsize=10,
        )
        ax.axis("off")
    figure.suptitle(
        f"{object_result['category']} representative visibility examples",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=170, facecolor="white")
    plt.close(figure)


def _plot_low_coverage_views(object_result, object_records, output_path):
    bins = [
        (0.0, 1.0, "0-1%"),
        (1.0, 5.0, "1-5%"),
        (5.0, 10.0, "5-10%"),
        (10.0, 15.0, "10-15%"),
    ]
    selections = []
    for lower, upper, label in bins:
        candidates = [
            record
            for record in object_records
            if lower < record["fraction_observable"] * 100.0 <= upper
        ]
        if not candidates:
            continue
        midpoint = (lower + upper) / 2.0
        record = min(
            candidates,
            key=lambda item: abs(item["fraction_observable"] * 100.0 - midpoint),
        )
        selections.append((label, record))
    if not selections:
        return
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (label, record) in zip(axes.flat, selections):
        ax.imshow(iio.imread(record["target_outline_image"]))
        ax.set_title(
            f"{label}: coverage={record['fraction_observable'] * 100.0:.2f}% | "
            f"depth pixels={record['target_depth_pixel_count']:,}\n"
            f"r={record['radius_m']:.1f} m, el={record['elevation_deg']:+.0f} deg, "
            f"az={record['azimuth_deg']:+.0f} deg",
            fontsize=10,
        )
        ax.axis("off")
    figure.suptitle(
        f"{object_result['category']} low-coverage examples | cyan = exact target mask outline",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=170, facecolor="white")
    plt.close(figure)


@hydra.main(
    version_base="1.1",
    config_name="camera_sweep",
    config_path="./isaacgymenvs/config",
)
def launch(cfg: DictConfig):
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    cfg.task.task.scene_config_path = cfg.scene.scene_list
    cfg.task.experiment_name = "mesh_visibility"
    cfg.scene.num_tasks = 1
    cfg.headless = True
    cfg.force_render = True

    asset_path = Path(os.environ["ASSET_PATH"])
    scene_ref = str(cfg.scene.scene_list[0])
    center, _, _, _, support_label, _ = _load_support_geometry(asset_path, scene_ref)
    task_index = int(cfg.camera_sweep.task_index)
    scene_objects = _load_scene_objects(asset_path, scene_ref, task_index)
    selected_indices = [
        int(value)
        for value in os.environ.get("FETCHBENCH_VISIBILITY_OBJECTS", "1,2,3,4,6").split(",")
    ]
    sample_count = int(os.environ.get("FETCHBENCH_VISIBILITY_SAMPLES", "60000"))
    depth_tolerance_m = float(
        os.environ.get("FETCHBENCH_VISIBILITY_DEPTH_TOLERANCE_M", "0.002")
    )
    output_dir = Path(str(cfg.camera_sweep.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview_rgb"
    preview_dir.mkdir(exist_ok=True)
    save_view_images = os.environ.get("FETCHBENCH_SAVE_VIEW_IMAGES", "0") == "1"
    export_partial = os.environ.get("FETCHBENCH_EXPORT_PARTIAL", "0") == "1"
    partial_sample_count = int(os.environ.get("FETCHBENCH_PARTIAL_SAMPLES", "512"))
    partial_depth_min_m = float(
        os.environ.get("FETCHBENCH_PARTIAL_DEPTH_MIN_M", "0.15")
    )
    partial_depth_max_m = float(
        os.environ.get("FETCHBENCH_PARTIAL_DEPTH_MAX_M", "2.5")
    )
    partial_dir = output_dir / "partial_pointclouds"
    if export_partial:
        partial_dir.mkdir(exist_ok=True)
    outlined_view_dirs = {}
    if save_view_images:
        for object_index in selected_indices:
            category = scene_objects[object_index]["category"]
            outlined_view_dirs[object_index] = output_dir / "camera_views" / category
            outlined_view_dirs[object_index].mkdir(parents=True, exist_ok=True)

    task = isaacgymenvs.make(
        cfg.seed,
        cfg.task_name,
        cfg.task.env.numEnvs,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
    )
    task.reset_task(task_index)
    task.set_target_color()
    task.gym.refresh_actor_root_state_tensor(task.sim)
    env = task.envs[0]
    camera_handle = task.cameras[0][0]
    object_segment_base = int(getattr(task, "_object_actor_offset", 3)) + 1

    samples = {}
    for object_index in selected_indices:
        item = scene_objects[object_index]
        mesh_sample = _sample_mesh_surface(
            item["mesh_path"], sample_count, seed=cfg.seed + object_index
        )
        root_pose = task._obj_state[0, object_index, :7].detach().cpu().numpy()
        rotation = _quaternion_xyzw_to_matrix(root_pose[3:7])
        mesh_sample["points_world"] = (
            mesh_sample["points_local"] @ rotation.T + root_pose[:3]
        )
        mesh_sample["normals_world"] = mesh_sample["normals_local"] @ rotation.T
        mesh_sample["root_pose"] = root_pose
        mesh_sample["visible_union"] = np.zeros(sample_count, dtype=bool)
        mesh_sample["item"] = item
        samples[object_index] = mesh_sample

    records = []
    raw_visibility = {}
    raw_target_depth_pixels = {}
    outlined_view_paths = {}
    partial_records = {}
    directions = _camera_directions()
    for radius in RADII:
        for elevation, azimuth in directions:
            position = _camera_position(center, radius, elevation, azimuth)
            task.gym.set_camera_location(
                camera_handle,
                env,
                gymapi.Vec3(*position.tolist()),
                gymapi.Vec3(*center.tolist()),
            )
            rgb, depth_m, segmentation = _render_camera(task, camera_handle)
            height, width = depth_m.shape
            view, intrinsics, env_origin = _camera_projection(
                task, env, camera_handle, width, height
            )
            view_key = (radius, elevation, azimuth)
            raw_visibility[view_key] = {}
            raw_target_depth_pixels[view_key] = {}
            for object_index, sample in samples.items():
                visible = _visible_surface_mask(
                    sample["points_world"],
                    sample["normals_world"],
                    position,
                    view,
                    intrinsics,
                    env_origin,
                    depth_m,
                    segmentation,
                    object_segment_base + object_index,
                    depth_tolerance_m,
                )
                raw_visibility[view_key][object_index] = visible
                raw_target_depth_pixels[view_key][object_index] = int(
                    np.count_nonzero(
                        np.isfinite(depth_m)
                        & (segmentation == object_segment_base + object_index)
                    )
                )
                if export_partial:
                    target_segmentation_id = object_segment_base + object_index
                    points_world, _ = _unproject_target_points(
                        task,
                        env,
                        camera_handle,
                        depth_m,
                        segmentation,
                        target_segmentation_id,
                        partial_depth_min_m,
                        partial_depth_max_m,
                    )
                    root_pose = sample["root_pose"]
                    rotation = _quaternion_xyzw_to_matrix(root_pose[3:7])
                    points_local = (
                        (points_world.astype(np.float64) - root_pose[:3]) @ rotation
                    ).astype(np.float32)
                    view_seed = (
                        int(cfg.seed)
                        + task_index * 1000003
                        + object_index * 10007
                        + int(round(radius * 10.0)) * 1009
                        + int(round(elevation + 90.0)) * 101
                        + int(round(azimuth + 180.0))
                    )
                    sampled_local = _sample_partial_cloud(
                        points_local, partial_sample_count, view_seed
                    )
                    partial_path = partial_dir / (
                        f"task{task_index:02d}_obj{object_index:02d}_"
                        f"r{radius:.1f}_el{elevation:+.0f}_az{azimuth:+.0f}.npz"
                    )
                    if sampled_local is not None:
                        np.savez_compressed(
                            partial_path,
                            object_partial_pc=sampled_local,
                            object_partial_pc_raw=points_local,
                            object_state=root_pose.astype(np.float32),
                            camera_position=position.astype(np.float32),
                            radius_m=np.float32(radius),
                            elevation_deg=np.float32(elevation),
                            azimuth_deg=np.float32(azimuth),
                            segmentation_id=np.int32(target_segmentation_id),
                            depth_range_m=np.asarray(
                                [partial_depth_min_m, partial_depth_max_m],
                                dtype=np.float32,
                            ),
                        )
                    partial_records[(view_key, object_index)] = {
                        "partial_pointcloud": str(partial_path)
                        if sampled_local is not None
                        else "",
                        "partial_raw_point_count": int(len(points_local)),
                    }
                sample["visible_union"] |= visible
                if save_view_images:
                    view_name = (
                        f"r{radius:.1f}_el{elevation:+.0f}_az{azimuth:+.0f}.png"
                    )
                    outlined_path = outlined_view_dirs[object_index] / view_name
                    iio.imwrite(
                        outlined_path,
                        _outlined_target_image(
                            rgb,
                            segmentation,
                            object_segment_base + object_index,
                        ),
                    )
                    outlined_view_paths[(view_key, object_index)] = outlined_path
            if math.isclose(radius, 1.5) and elevation in (-30.0, 0.0, 30.0) and math.isclose(azimuth, 0.0):
                iio.imwrite(
                    preview_dir / f"r{radius:.1f}_el{elevation:+.0f}_az{azimuth:+.0f}.png",
                    rgb,
                )
            print(f"rendered r={radius:.1f} elevation={elevation:+.0f} azimuth={azimuth:+.0f}")

    object_results = []
    for object_index, sample in samples.items():
        union_count = int(np.count_nonzero(sample["visible_union"]))
        observable_fraction_full = union_count / sample_count
        object_records = []
        for (radius, elevation, azimuth), per_object in raw_visibility.items():
            visible_count = int(np.count_nonzero(per_object[object_index]))
            fraction_full = visible_count / sample_count
            fraction_observable = visible_count / union_count if union_count else 0.0
            record = {
                "object_index": object_index,
                "object_name": sample["item"]["name"],
                "category": sample["item"]["category"],
                "radius_m": radius,
                "elevation_deg": elevation,
                "azimuth_deg": azimuth,
                "visible_sample_count": visible_count,
                "target_depth_pixel_count": raw_target_depth_pixels[
                    (radius, elevation, azimuth)
                ][object_index],
                "fraction_full": fraction_full,
                "fraction_observable": fraction_observable,
                "target_outline_image": str(
                    outlined_view_paths.get(
                        ((radius, elevation, azimuth), object_index), ""
                    )
                ),
            }
            record.update(
                partial_records.get(((radius, elevation, azimuth), object_index), {})
            )
            records.append(record)
            object_records.append(record)
        full_values = np.asarray([record["fraction_full"] for record in object_records])
        observable_values = np.asarray(
            [record["fraction_observable"] for record in object_records]
        )
        positive_observable = observable_values[observable_values > 0.0]
        radius_summary = {}
        for radius in RADII:
            radius_values = np.asarray(
                [
                    record["fraction_observable"]
                    for record in object_records
                    if math.isclose(record["radius_m"], radius)
                ]
            )
            radius_summary[str(radius)] = {
                "visible_view_count": int(np.count_nonzero(radius_values > 0.0)),
                "max_fraction_observable": float(radius_values.max()),
                "median_positive_fraction_observable": float(
                    np.median(radius_values[radius_values > 0.0])
                    if np.any(radius_values > 0.0)
                    else 0.0
                ),
            }
        best = object_records[int(np.argmax(observable_values))]
        object_results.append(
            {
                "object_index": object_index,
                "object_name": sample["item"]["name"],
                "category": sample["item"]["category"],
                "label": sample["item"]["label"],
                "mesh_path": str(sample["item"]["mesh_path"]),
                "mesh_surface_area_m2": sample["surface_area_m2"],
                "mesh_face_count": sample["face_count"],
                "sample_count": sample_count,
                "observable_sample_count": union_count,
                "observable_fraction_full": observable_fraction_full,
                "visible_view_count": int(np.count_nonzero(observable_values > 0.0)),
                "max_single_fraction_full": float(full_values.max()),
                "max_single_fraction_observable": float(observable_values.max()),
                "median_positive_fraction_observable": float(
                    np.median(positive_observable) if len(positive_observable) else 0.0
                ),
                "best_view": {
                    "radius_m": best["radius_m"],
                    "elevation_deg": best["elevation_deg"],
                    "azimuth_deg": best["azimuth_deg"],
                },
                "by_radius": radius_summary,
            }
        )

    summary = {
        "scene": str(cfg.scene.name),
        "scene_ref": scene_ref,
        "task_index": task_index,
        "support_label": support_label,
        "support_center_m": center.tolist(),
        "radii_m": RADII,
        "elevation_deg": ELEVATIONS,
        "azimuth_deg": AZIMUTHS,
        "directions_per_radius": len(directions),
        "camera_count": len(RADII) * len(directions),
        "surface_sample_count_per_object": sample_count,
        "depth_tolerance_m": depth_tolerance_m,
        "depth_tolerance_rule": "base tolerance + 0.0038 * sample depth",
        "coverage_definition": {
            "fraction_full": "visible equal-area mesh samples / all mesh samples",
            "fraction_observable": "visible equal-area mesh samples / union visible from all allowed cameras",
        },
        "partial_pointcloud_export": {
            "enabled": export_partial,
            "coordinate_frame": "target asset-local frame (object actor root; not recentered)",
            "sample_count": partial_sample_count,
            "depth_range_m": [partial_depth_min_m, partial_depth_max_m],
            "source": "same rendered depth and exact target segmentation mask as coverage",
        },
        "objects": object_results,
    }
    (output_dir / "mesh_visibility_summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(records, output_dir / "per_view_mesh_visibility.csv")
    if export_partial:
        legal_records = [
            record
            for record in records
            if record["fraction_observable"] >= 0.05
            and record.get("partial_pointcloud")
        ]
        legal_manifest = {
            "scene": str(cfg.scene.name),
            "scene_ref": scene_ref,
            "task_index": task_index,
            "minimum_fraction_observable": 0.05,
            "coordinate_frame": "target asset-local frame (object actor root; not recentered)",
            "depth_range_m": [partial_depth_min_m, partial_depth_max_m],
            "views": legal_records,
        }
        (output_dir / "legal_partial_views.json").write_text(
            json.dumps(legal_manifest, indent=2)
        )
    _plot_summary(object_results, output_dir / "mesh_visibility_summary.png")
    _plot_heatmaps(object_results, records, output_dir / "mesh_visibility_heatmaps.png")
    if save_view_images:
        for object_result in object_results:
            object_records = [
                record
                for record in records
                if record["object_index"] == object_result["object_index"]
            ]
            category = object_result["category"]
            _plot_all_camera_views(
                object_result,
                object_records,
                output_dir / f"{category}_all_111_camera_views.png",
            )
            _plot_representative_views(
                object_result,
                object_records,
                output_dir / f"{category}_representative_views.png",
            )
            _plot_low_coverage_views(
                object_result,
                object_records,
                output_dir / f"{category}_low_coverage_examples.png",
            )
    print(json.dumps(object_results, indent=2))
    print(f"saved mesh visibility analysis to {output_dir}")
    task.exit()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    launch()
