import json
import math
import os
import sys
from pathlib import Path

import hydra
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from omegaconf import DictConfig, OmegaConf

import isaacgym
from isaacgym import gymapi
import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


def _safe_name(value):
    return f"{value:.1f}".replace(".", "p")


def _load_support_geometry(asset_path, scene_ref):
    task_dir = asset_path / "Task" / scene_ref
    with (task_dir / "asset_config.json").open() as f:
        asset_config = json.load(f)

    remote_asset_root = asset_config["scene_config"]["asset_root"]
    marker = "/benchmark_scenes/"
    if marker not in remote_asset_root:
        raise ValueError(f"Expected benchmark scene path, got {remote_asset_root}")
    scene_asset = asset_path / "benchmark_scenes" / remote_asset_root.split(marker, 1)[1]

    with (scene_asset / "support.json").open() as f:
        supports = json.load(f)
    support_label = None
    selected_supports = []
    for candidate in ("on_table", "on_shelf"):
        selected_supports = [item for item in supports if item["label"] == candidate]
        if selected_supports:
            support_label = candidate
            break
    if not selected_supports:
        selected_supports = [
            item for item in supports if not str(item["label"]).endswith("on_floor")
        ]
        support_label = "supported region"
    if not selected_supports:
        raise ValueError(f"No usable support surface in {scene_asset}")

    translated_polygons = []
    for support in selected_supports:
        polygon = json.loads(support["polygon"])
        xy = np.asarray(polygon["coordinates"][0], dtype=np.float64)
        translation = np.asarray(support["translation"], dtype=np.float64)
        translated_polygons.append(xy + translation[:2])
    all_xy = np.concatenate(translated_polygons, axis=0)
    local_min = all_xy.min(axis=0)
    local_max = all_xy.max(axis=0)
    local_center = (local_min + local_max) / 2.0
    support_z_levels = sorted(
        {round(float(item["translation"][2]), 9) for item in selected_supports}
    )
    support_center_z = float(np.mean(support_z_levels))

    # FetchBench maps generated scene coordinates into Isaac Gym with diag(-1, -1, 1, 1).
    world_min = np.asarray([-local_max[0], -local_max[1]], dtype=np.float64)
    world_max = np.asarray([-local_min[0], -local_min[1]], dtype=np.float64)
    center = np.asarray(
        [-local_center[0], -local_center[1], support_center_z], dtype=np.float64
    )
    return (
        center,
        world_min,
        world_max,
        scene_asset,
        support_label,
        support_z_levels,
    )


def _load_object_metadata(asset_path, scene_ref, task_index):
    task_dir = asset_path / "Task" / scene_ref
    with (task_dir / "asset_config.json").open() as stream:
        asset_config = json.load(stream)
    rearrange = np.load(task_dir / "rearrange_config.npz", allow_pickle=True)
    labels = rearrange["object_labels"][task_index]
    objects = []
    for index, (config, label) in enumerate(zip(asset_config["object_config"], labels)):
        asset_parts = Path(config["asset_root"]).parts
        category = asset_parts[-2] if len(asset_parts) >= 2 else config["name"]
        objects.append(
            {
                "index": index,
                "name": config["name"],
                "category": category,
                "label": str(label),
            }
        )
    return objects


def _camera_position(center, radius, height, azimuth_deg):
    dz = float(height - center[2])
    if radius < abs(dz):
        raise ValueError(
            f"radius={radius:.3f} m cannot reach height={height:.3f} m "
            f"from support center z={center[2]:.3f} m"
        )
    horizontal_radius = math.sqrt(max(radius * radius - dz * dz, 0.0))
    theta = math.radians(azimuth_deg)
    position = np.asarray(
        [
            center[0] - horizontal_radius * math.cos(theta),
            center[1] + horizontal_radius * math.sin(theta),
            height,
        ],
        dtype=np.float64,
    )
    return position, horizontal_radius


def _render_camera(task, camera_handle):
    task.gym.fetch_results(task.sim, True)
    task.gym.step_graphics(task.sim)
    task.gym.render_all_camera_sensors(task.sim)
    env = task.envs[0]
    image = task.gym.get_camera_image(task.sim, env, camera_handle, gymapi.IMAGE_COLOR)
    rgb = image.reshape(image.shape[0], -1, 4)[..., :3].copy()
    raw_depth = task.gym.get_camera_image(task.sim, env, camera_handle, gymapi.IMAGE_DEPTH)
    depth_m = -np.asarray(raw_depth, dtype=np.float32).reshape(rgb.shape[:2])
    depth_m[~np.isfinite(depth_m) | (depth_m <= 0.0)] = np.nan
    segmentation = task.gym.get_camera_image(
        task.sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION
    )
    segmentation = np.asarray(segmentation, dtype=np.int32).reshape(rgb.shape[:2])
    return rgb, depth_m, segmentation


def _unproject_target_points(
    task,
    env,
    camera_handle,
    depth_m,
    segmentation,
    goal_segmentation_id,
    depth_min_m,
    depth_max_m,
):
    """Match FetchBench's row-vector depth unprojection exactly."""
    height, width = depth_m.shape
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

    rows, columns = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    uv_one = np.stack((columns, rows, np.ones_like(rows)), axis=-1)
    camera_rays = uv_one @ np.linalg.inv(intrinsics.T)
    valid = (
        np.isfinite(depth_m)
        & (depth_m > depth_min_m)
        & (depth_m < depth_max_m)
        & (segmentation == goal_segmentation_id)
    )
    target_depth = -depth_m[valid].astype(np.float64)
    points_camera = camera_rays[valid] * target_depth[:, None]
    points_camera_h = np.concatenate(
        (points_camera, np.ones((len(points_camera), 1), dtype=np.float64)),
        axis=1,
    )
    points_world = points_camera_h @ np.linalg.inv(view)
    origin = task.gym.get_env_origin(env)
    points_world[:, :3] -= np.asarray([origin.x, origin.y, origin.z])
    return points_world[:, :3].astype(np.float32), valid


def _quaternion_xyzw_to_matrix(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _world_to_object_centered(points_world, object_root_pose, object_center_local):
    translation = np.asarray(object_root_pose[:3], dtype=np.float64)
    rotation = _quaternion_xyzw_to_matrix(object_root_pose[3:7])
    points_local = (np.asarray(points_world, dtype=np.float64) - translation) @ rotation
    return (points_local - np.asarray(object_center_local, dtype=np.float64)).astype(
        np.float32
    )


def _draw_layout(
    ax,
    center,
    support_min,
    support_max,
    positions,
    radius,
    height,
    horizontal_radius,
    angles,
    support_label,
):
    width, depth = support_max - support_min
    rect = Rectangle(
        support_min,
        width,
        depth,
        facecolor="#d8c7a2",
        edgecolor="#6c5938",
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.scatter(
        [center[0]],
        [center[1]],
        marker="x",
        s=80,
        c="#202020",
        linewidths=2,
        label="support center",
    )

    theta = np.radians(np.linspace(-90.0, 90.0, 181))
    ax.plot(
        center[0] - horizontal_radius * np.cos(theta),
        center[1] + horizontal_radius * np.sin(theta),
        linestyle="--",
        color="#8a8a8a",
        linewidth=1.2,
    )
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(positions)))
    for color, angle, position in zip(colors, angles, positions):
        ax.plot([position[0], center[0]], [position[1], center[1]], color=color, linewidth=1.3, alpha=0.75)
        ax.scatter([position[0]], [position[1]], s=55, color=color, edgecolor="black", linewidth=.5)
        ax.annotate(f"{angle:+.0f} deg", position[:2], xytext=(5, 5), textcoords="offset points", fontsize=8)

    dimension_offset = 0.075
    ax.annotate(
        "",
        xy=(support_max[0], support_min[1] - dimension_offset),
        xytext=(support_min[0], support_min[1] - dimension_offset),
        arrowprops={"arrowstyle": "<->", "color": "#4f4028", "linewidth": 1.2},
    )
    ax.text(
        center[0],
        support_min[1] - dimension_offset - 0.025,
        f"support x = {width:.3f} m",
        ha="center",
        va="top",
        fontsize=8,
        color="#4f4028",
    )
    ax.annotate(
        "",
        xy=(support_max[0] + dimension_offset, support_max[1]),
        xytext=(support_max[0] + dimension_offset, support_min[1]),
        arrowprops={"arrowstyle": "<->", "color": "#4f4028", "linewidth": 1.2},
    )
    ax.text(
        support_max[0] + dimension_offset + 0.025,
        center[1],
        f"support y = {depth:.3f} m",
        ha="left",
        va="center",
        rotation=90,
        fontsize=8,
        color="#4f4028",
    )

    pad = max(horizontal_radius * 0.2, 0.18)
    ax.set_xlim(center[0] - horizontal_radius - pad, support_max[0] + pad)
    ax.set_ylim(center[1] - horizontal_radius - pad, center[1] + horizontal_radius + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(
        f"Camera locations around {support_label}\n"
        f"3D r={radius:.1f} m, z={height:.1f} m, horizontal rho={horizontal_radius:.3f} m",
        fontsize=11,
    )
    ax.grid(alpha=.2)


def _draw_elevation(
    ax,
    center,
    radius,
    height,
    horizontal_radius,
    support_label,
    support_z_levels,
):
    support_height = float(center[2])
    height_above_support = height - support_height
    x_max = horizontal_radius * 1.18
    ax.plot([-0.12 * x_max, x_max], [0.0, 0.0], color="#454545", linewidth=2)
    for index, level in enumerate(support_z_levels):
        ax.plot(
            [-0.12 * x_max, x_max],
            [level, level],
            color="#8b6b3f",
            linewidth=1.7,
            label=f"{support_label} z={level:.3f} m" if index == 0 else None,
        )
        ax.text(-0.105 * x_max, level + 0.015, f"z={level:.3f}", fontsize=7)
    if len(support_z_levels) > 1:
        ax.axhline(
            support_height,
            color="#8b6b3f",
            linewidth=1.2,
            linestyle="--",
            label="support-region center",
        )
    ax.plot([0.0, horizontal_radius], [support_height, height], color="#2474a6", linewidth=2)
    ax.scatter([0.0], [support_height], marker="x", s=65, c="#202020", linewidths=2)
    ax.scatter([horizontal_radius], [height], s=65, c="#e24a33", edgecolor="black", linewidth=.6)
    ax.annotate(
        "",
        xy=(horizontal_radius, height),
        xytext=(horizontal_radius, support_height),
        arrowprops={"arrowstyle": "<->", "color": "#e24a33", "linewidth": 1.2},
    )
    ax.text(
        horizontal_radius * 1.015,
        (height + support_height) / 2.0,
        f"vertical offset\n{height_above_support:.3f} m",
        va="center",
        fontsize=8,
        color="#b23927",
    )
    ax.annotate(
        "",
        xy=(0.0, support_height),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "<->", "color": "#6c5938", "linewidth": 1.2},
    )
    ax.text(
        0.018 * x_max,
        support_height / 2.0,
        f"center z\n{support_height:.3f} m",
        va="center",
        fontsize=8,
    )
    ax.text(
        horizontal_radius * 0.48,
        (height + support_height) / 2.0 + 0.06,
        f"3D r = {radius:.3f} m",
        ha="center",
        rotation=math.degrees(math.atan2(height_above_support, horizontal_radius)),
        fontsize=8,
        color="#1d5d84",
    )
    ax.text(horizontal_radius, height + 0.055, f"5 cameras: ground z = {height:.3f} m", ha="right", fontsize=8)
    ax.text(horizontal_radius * 0.5, -0.055, f"horizontal rho = {horizontal_radius:.3f} m", ha="center", va="top", fontsize=8)
    ax.set_xlim(-0.12 * x_max, x_max)
    ax.set_ylim(-0.12, max(height * 1.13, 1.18))
    ax.set_xlabel("radial distance from support center (m)")
    ax.set_ylabel("world z (m)")
    ax.set_title("Side elevation (all 5 overlap)", fontsize=10)
    ax.grid(alpha=.2)


def _save_group_montage(
    path,
    center,
    table_min,
    table_max,
    positions,
    images,
    radius,
    height,
    horizontal_radius,
    angles,
    actor_label,
    support_label,
    support_z_levels,
    target_description,
):
    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.35, 1.0, 1.0, 1.0])
    geometry_grid = grid[:, 0].subgridspec(2, 1, height_ratios=[1.25, 0.85])
    layout_ax = fig.add_subplot(geometry_grid[0, 0])
    _draw_layout(
        layout_ax,
        center,
        table_min,
        table_max,
        positions,
        radius,
        height,
        horizontal_radius,
        angles,
        support_label,
    )
    elevation_ax = fig.add_subplot(geometry_grid[1, 0])
    _draw_elevation(
        elevation_ax,
        center,
        radius,
        height,
        horizontal_radius,
        support_label,
        support_z_levels,
    )

    slots = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]
    for slot, angle, position, image in zip(slots, angles, positions, images):
        ax = fig.add_subplot(grid[slot])
        ax.imshow(image)
        ax.set_title(
            f"azimuth {angle:+.0f} deg\npos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})",
            fontsize=10,
        )
        if float(image.std()) < 1.0:
            ax.text(
                0.5,
                0.5,
                "ALL-WHITE FRAME\nCamera is inside or fully occluded\nby scene/robot geometry",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="#b00020",
                bbox={"facecolor": "white", "edgecolor": "#b00020", "alpha": 0.92, "pad": 8},
            )
        ax.axis("off")

    info_ax = fig.add_subplot(grid[1, 3])
    info_ax.axis("off")
    info_ax.text(
        0.02,
        0.95,
        f"Look-at: {support_label} region center\n"
        f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m\n"
        f"grasp target={target_description} (red)\n"
        f"sphere radius={radius:.3f} m\n"
        f"camera height={height:.3f} m\n"
        f"vertical offset={height-center[2]:.3f} m\n"
        f"horizontal radius={horizontal_radius:.3f} m\n"
        f"actor={actor_label}\n"
        "HFOV=70 deg",
        va="top",
        fontsize=12,
        linespacing=1.5,
    )
    fig.suptitle(f"FetchBench deterministic five-view camera sweep | {actor_label}", fontsize=16)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_depth_files(float_path, millimeter_path, depth_m):
    np.save(float_path, depth_m.astype(np.float32))
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    valid = np.isfinite(depth_m)
    depth_mm[valid] = np.clip(np.rint(depth_m[valid] * 1000.0), 1, 65535).astype(
        np.uint16
    )
    iio.imwrite(millimeter_path, depth_mm)


def _save_depth_montage(
    path,
    center,
    table_min,
    table_max,
    positions,
    depths,
    radius,
    height,
    horizontal_radius,
    angles,
    actor_label,
    display_max_m,
    visible_objects,
    total_supported_objects,
    support_label,
    support_z_levels,
    target_description,
    target_depth_pixels,
):
    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.35, 1.0, 1.0, 1.0])
    geometry_grid = grid[:, 0].subgridspec(2, 1, height_ratios=[1.25, 0.85])
    layout_ax = fig.add_subplot(geometry_grid[0, 0])
    _draw_layout(
        layout_ax,
        center,
        table_min,
        table_max,
        positions,
        radius,
        height,
        horizontal_radius,
        angles,
        support_label,
    )
    elevation_ax = fig.add_subplot(geometry_grid[1, 0])
    _draw_elevation(
        elevation_ax,
        center,
        radius,
        height,
        horizontal_radius,
        support_label,
        support_z_levels,
    )

    slots = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]
    depth_axes = []
    depth_image = None
    cmap = plt.get_cmap("turbo_r").copy()
    cmap.set_bad("black")
    for slot, angle, position, depth in zip(slots, angles, positions, depths):
        ax = fig.add_subplot(grid[slot])
        depth_image = ax.imshow(depth, cmap=cmap, vmin=0.0, vmax=display_max_m)
        valid = depth[np.isfinite(depth)]
        if valid.size:
            stats = f"valid min/median={valid.min():.2f}/{np.median(valid):.2f} m"
        else:
            stats = "no valid depth"
        ax.set_title(
            f"azimuth {angle:+.0f} deg | {stats}\n"
            f"pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})",
            fontsize=9,
        )
        ax.axis("off")
        depth_axes.append(ax)

    info_ax = fig.add_subplot(grid[1, 3])
    info_ax.axis("off")
    info_ax.text(
        0.02,
        0.95,
        "Metric depth (meters)\n"
        "black = invalid / no return\n"
        f"fixed display range=0-{display_max_m:.1f} m\n"
        f"supported objects visible={visible_objects}/{total_supported_objects}\n"
        f"grasp target={target_description}\n"
        f"target valid pixels={target_depth_pixels}\n"
        f"sphere radius={radius:.3f} m\n"
        f"camera height={height:.3f} m\n"
        f"actor={actor_label}\n"
        "HFOV=70 deg",
        va="top",
        fontsize=12,
        linespacing=1.5,
    )
    if depth_image is not None:
        fig.colorbar(
            depth_image,
            ax=depth_axes,
            orientation="horizontal",
            fraction=0.04,
            pad=0.02,
            label="depth from camera (m)",
        )
    fig.suptitle(
        f"FetchBench five-view metric depth | {actor_label}", fontsize=16
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _target_cloud_limit(pointcloud_groups):
    clouds = [
        item["points_object_centered_m"]
        for group in pointcloud_groups.values()
        for item in group
        if len(item["points_object_centered_m"])
    ]
    if not clouds:
        return 0.1
    points = np.concatenate(clouds, axis=0)
    robust_extent = float(np.percentile(np.abs(points), 99.8))
    return max(robust_extent * 1.12, 0.03)


def _format_object_axes(ax, axis_limit):
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("object x (m)", fontsize=8, labelpad=2)
    ax.set_ylabel("object y (m)", fontsize=8, labelpad=2)
    ax.set_zlabel("object z (m)", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=20, azim=-58)
    ax.scatter([0.0], [0.0], [0.0], marker="x", s=40, c="black", linewidths=1.5)


def _save_target_pointcloud_montage(
    path,
    group,
    radius,
    height,
    target_description,
    goal_segmentation_id,
    depth_min_m,
    depth_max_m,
    axis_limit,
):
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, max(len(group), 1)))
    for index, item in enumerate(group):
        ax = fig.add_subplot(2, 3, index + 1, projection="3d")
        points = item["points_object_centered_m"]
        if len(points):
            plot_points = points
            if len(plot_points) > 6000:
                sample = np.linspace(0, len(plot_points) - 1, 6000).astype(int)
                plot_points = plot_points[sample]
            ax.scatter(
                plot_points[:, 0],
                plot_points[:, 1],
                plot_points[:, 2],
                s=1.2,
                c=plot_points[:, 2],
                cmap="viridis",
                alpha=0.75,
                linewidths=0,
            )
        ax.set_title(
            f"azimuth {item['azimuth_deg']:+.0f} deg | N={len(points):,}\n"
            "object-centered partial cloud",
            fontsize=10,
        )
        _format_object_axes(ax, axis_limit)

    fused_ax = fig.add_subplot(2, 3, 6, projection="3d")
    fused_count = 0
    for color, item in zip(colors, group):
        points = item["points_object_centered_m"]
        fused_count += len(points)
        if not len(points):
            continue
        plot_points = points
        if len(plot_points) > 3000:
            sample = np.linspace(0, len(plot_points) - 1, 3000).astype(int)
            plot_points = plot_points[sample]
        fused_ax.scatter(
            plot_points[:, 0],
            plot_points[:, 1],
            plot_points[:, 2],
            s=1.1,
            color=color,
            alpha=0.5,
            linewidths=0,
        )
    fused_ax.set_title(
        f"five-view overlay | total N={fused_count:,}\n"
        "color distinguishes camera azimuth",
        fontsize=10,
    )
    _format_object_axes(fused_ax, axis_limit)
    fig.suptitle(
        f"{target_description} target-only point cloud | "
        f"r={radius:.1f} m, camera z={height:.1f} m\n"
        f"exact mask: segmentation == {goal_segmentation_id}; "
        f"depth gate: {depth_min_m:.2f}-{depth_max_m:.2f} m; origin: object center",
        fontsize=15,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_pointcloud_density_plot(
    path, pointcloud_groups, radii, heights, target_description
):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = plt.cm.turbo(np.linspace(0.12, 0.88, len(heights)))
    annotation_offsets = [(-10, -16), (0, 11), (12, 20)]
    for height_index, (color, height) in enumerate(zip(colors, heights)):
        values = []
        for radius in radii:
            group = pointcloud_groups.get((radius, height), [])
            values.append(
                float(np.mean([len(item["points_object_centered_m"]) for item in group]))
                if group
                else np.nan
            )
        ax.plot(radii, values, marker="o", linewidth=2, color=color, label=f"camera z={height:.1f} m")
        for radius, value in zip(radii, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:,.0f}",
                    (radius, value),
                    xytext=annotation_offsets[height_index],
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
                )
    ax.set_yscale("log")
    ax.set_xticks(radii)
    ax.set_xlabel("camera-to-support-center sphere radius r (m)")
    ax.set_ylabel("mean target points per camera (log scale)")
    ax.set_title(
        f"Observed target point density vs distance | {target_description}\n"
        "One valid target depth pixel produces one 3D point"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_infeasible_montage(
    path, center, radius, height, actor_label, modality, support_label
):
    support_height = float(center[2])
    dz = abs(height - support_height)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    axes[0].plot(
        radius * np.cos(theta),
        support_height + radius * np.sin(theta),
        color="#2474a6",
    )
    axes[0].axhline(height, color="#c23b22", linewidth=2, label=f"requested camera z={height:.3f} m")
    axes[0].axhline(
        support_height,
        color="#8b6b3f",
        linewidth=2,
        label=f"{support_label} center z={support_height:.3f} m",
    )
    axes[0].scatter([0.0], [support_height], marker="x", s=80, color="black")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("horizontal radial distance (m)")
    axes[0].set_ylabel("world z (m)")
    axes[0].set_title("No sphere/height intersection")
    axes[0].grid(alpha=.25)
    axes[0].legend(loc="best")
    axes[1].axis("off")
    axes[1].text(
        0.5,
        0.58,
        "GEOMETRICALLY INFEASIBLE",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#b00020",
    )
    axes[1].text(
        0.5,
        0.38,
        f"|camera z - support center z| = {dz:.3f} m > r = {radius:.3f} m\n"
        "No camera position satisfies both requested values.\n"
        "Radius and height were not changed.",
        ha="center",
        va="center",
        fontsize=14,
        linespacing=1.6,
    )
    fig.suptitle(
        f"{modality} | r={radius:.1f} m, z={height:.1f} m | {actor_label}",
        fontsize=16,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_contact_sheet(path, montage_paths, scene_name, modality, radii, heights):
    images = [iio.imread(item) for item in montage_paths]
    columns = len(heights)
    rows = int(math.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(18, 4.0 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, columns)
    for ax, image, item in zip(axes.flat, images, montage_paths):
        ax.imshow(image)
        ax.set_title(item.stem.replace("_depth_montage", "").replace("_montage", ""), fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    fig.suptitle(
        f"{scene_name} {modality}: {len(radii)} radii x {len(heights)} heights",
        fontsize=16,
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)


@hydra.main(version_base="1.1", config_name="camera_sweep", config_path="./isaacgymenvs/config")
def launch(cfg: DictConfig):
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    cfg.task.task.scene_config_path = cfg.scene.scene_list
    cfg.task.experiment_name = "camera_sweep"
    cfg.scene.num_tasks = 1

    asset_path = Path(os.environ["ASSET_PATH"])
    scene_ref = str(cfg.scene.scene_list[0])
    (
        center,
        table_min,
        table_max,
        scene_asset,
        support_label,
        support_z_levels,
    ) = _load_support_geometry(asset_path, scene_ref)
    task_index = int(cfg.camera_sweep.task_index)
    object_metadata = _load_object_metadata(asset_path, scene_ref, task_index)

    output_dir = Path(str(cfg.camera_sweep.output_dir)).expanduser().resolve()
    raw_dir = output_dir / "raw"
    depth_float_dir = output_dir / "depth_meters"
    depth_mm_dir = output_dir / "depth_millimeters"
    montage_dir = output_dir / "montages"
    depth_montage_dir = output_dir / "depth_montages"
    target_mask_dir = output_dir / "target_masks"
    target_pointcloud_dir = output_dir / "target_pointclouds"
    target_pointcloud_montage_dir = output_dir / "target_pointcloud_montages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    depth_float_dir.mkdir(parents=True, exist_ok=True)
    depth_mm_dir.mkdir(parents=True, exist_ok=True)
    montage_dir.mkdir(parents=True, exist_ok=True)
    depth_montage_dir.mkdir(parents=True, exist_ok=True)
    target_mask_dir.mkdir(parents=True, exist_ok=True)
    target_pointcloud_dir.mkdir(parents=True, exist_ok=True)
    target_pointcloud_montage_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))

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
    env = task.envs[0]
    camera_handle = task.cameras[0][0]
    task_slot = int(task.get_task_idx())
    target_object_index = int(task.task_obj_index[0][task_slot].item())
    target_object_label = str(task.task_obj_label[0][task_slot])
    task.set_target_color()
    target_description = (
        f"obj_{target_object_index}/"
        f"{object_metadata[target_object_index]['category']}"
    )
    task.gym.refresh_actor_root_state_tensor(task.sim)
    target_object_root_pose = (
        task._obj_state[0, target_object_index, :7].detach().cpu().numpy()
    )
    target_object_center_local = (
        task.obj_ref_point[0, target_object_index].detach().cpu().numpy()
    )
    target_rotation = _quaternion_xyzw_to_matrix(target_object_root_pose[3:7])
    target_object_center_world = (
        target_object_root_pose[:3]
        + np.asarray(target_object_center_local, dtype=np.float64) @ target_rotation.T
    )

    robot_cfg = cfg.task.env.robot
    robot_type = str(robot_cfg.get("type", "unknown"))
    hand_name = str(robot_cfg.get("hand_name", ""))
    if robot_type == "dro_hand":
        actor_label = f"floating {hand_name} hand (no arm)"
        hand_actor = task.robots[0]
        body_count = task.gym.get_actor_rigid_body_count(env, hand_actor)
        hand_color = gymapi.Vec3(0.95, 0.34, 0.08)
        for body_index in range(body_count):
            task.gym.set_rigid_body_color(
                env,
                hand_actor,
                body_index,
                gymapi.MESH_VISUAL,
                hand_color,
            )
    else:
        actor_label = robot_type

    radii = [float(value) for value in cfg.camera_sweep.radii]
    heights = [float(value) for value in cfg.camera_sweep.heights]
    angles = [float(value) for value in cfg.camera_sweep.azimuth_deg]
    depth_display_max_m = float(cfg.camera_sweep.depth_display_max_m)
    pointcloud_depth_min_m = float(cfg.camera_sweep.pointcloud_depth_min_m)
    pointcloud_depth_max_m = float(cfg.camera_sweep.pointcloud_depth_max_m)
    object_segment_base = int(getattr(task, "_object_actor_offset", 3)) + 1
    for item in object_metadata:
        item["segmentation_id"] = object_segment_base + item["index"]
        item["is_target"] = item["index"] == target_object_index
    goal_segmentation_id = object_segment_base + target_object_index
    supported_objects = [
        item
        for item in object_metadata
        if not item["label"].endswith("on_floor") or item["is_target"]
    ]
    records = []
    montage_paths = []
    depth_montage_paths = []
    target_pointcloud_montage_paths = []
    pointcloud_groups = {}
    infeasible_combinations = []

    for radius in radii:
        for height in heights:
            positions, images, depths = [], [], []
            group_pointclouds = []
            horizontal_radius = None
            group_name = f"r{_safe_name(radius)}_h{_safe_name(height)}"
            if radius < abs(height - center[2]):
                reason = (
                    f"vertical separation {abs(height-center[2]):.6f} m "
                    f"exceeds sphere radius {radius:.6f} m"
                )
                infeasible_combinations.append(
                    {
                        "radius_m": radius,
                        "height_m": height,
                        "support_center_height_m": float(center[2]),
                        "reason": reason,
                    }
                )
                montage_path = montage_dir / f"{group_name}_montage.png"
                depth_montage_path = depth_montage_dir / f"{group_name}_depth_montage.png"
                _save_infeasible_montage(
                    montage_path,
                    center,
                    radius,
                    height,
                    actor_label,
                    "RGB",
                    support_label,
                )
                _save_infeasible_montage(
                    depth_montage_path,
                    center,
                    radius,
                    height,
                    actor_label,
                    "DEPTH",
                    support_label,
                )
                montage_paths.append(montage_path)
                depth_montage_paths.append(depth_montage_path)
                target_pointcloud_montage_path = (
                    target_pointcloud_montage_dir
                    / f"{group_name}_target_pointcloud_montage.png"
                )
                _save_infeasible_montage(
                    target_pointcloud_montage_path,
                    center,
                    radius,
                    height,
                    actor_label,
                    "TARGET POINT CLOUD",
                    support_label,
                )
                target_pointcloud_montage_paths.append(
                    target_pointcloud_montage_path
                )
                print(f"skipped {group_name}: {reason}")
                continue

            group_visibility = {item["name"]: 0 for item in supported_objects}
            for angle in angles:
                position, horizontal_radius = _camera_position(center, radius, height, angle)
                task.gym.set_camera_location(
                    camera_handle,
                    env,
                    gymapi.Vec3(*position.tolist()),
                    gymapi.Vec3(*center.tolist()),
                )
                image, depth_m, segmentation = _render_camera(task, camera_handle)
                image_path = raw_dir / f"{group_name}_az{angle:+.0f}.png"
                depth_float_path = depth_float_dir / f"{group_name}_az{angle:+.0f}.npy"
                depth_mm_path = depth_mm_dir / f"{group_name}_az{angle:+.0f}.png"
                target_mask_path = target_mask_dir / f"{group_name}_az{angle:+.0f}.png"
                target_pointcloud_path = (
                    target_pointcloud_dir / f"{group_name}_az{angle:+.0f}.npz"
                )
                iio.imwrite(image_path, image)
                _save_depth_files(depth_float_path, depth_mm_path, depth_m)
                target_points_world, target_valid_mask = _unproject_target_points(
                    task,
                    env,
                    camera_handle,
                    depth_m,
                    segmentation,
                    goal_segmentation_id,
                    pointcloud_depth_min_m,
                    pointcloud_depth_max_m,
                )
                target_points_object = _world_to_object_centered(
                    target_points_world,
                    target_object_root_pose,
                    target_object_center_local,
                )
                iio.imwrite(
                    target_mask_path,
                    ((segmentation == goal_segmentation_id) * 255).astype(np.uint8),
                )
                np.savez_compressed(
                    target_pointcloud_path,
                    points_world_m=target_points_world,
                    points_object_centered_m=target_points_object,
                    target_valid_pixel_mask=target_valid_mask,
                    object_root_pose_xyzw=target_object_root_pose,
                    object_center_local_m=target_object_center_local,
                    object_center_world_m=target_object_center_world,
                    camera_position_m=position,
                    look_at_m=center,
                    goal_segmentation_id=np.int32(goal_segmentation_id),
                )
                group_pointclouds.append(
                    {
                        "azimuth_deg": angle,
                        "points_object_centered_m": target_points_object,
                    }
                )
                positions.append(position)
                images.append(image)
                depths.append(depth_m)
                object_depth_pixels = {}
                valid_depth = np.isfinite(depth_m)
                for item in supported_objects:
                    count = int(
                        np.count_nonzero(
                            valid_depth & (segmentation == item["segmentation_id"])
                        )
                    )
                    object_depth_pixels[item["name"]] = count
                    group_visibility[item["name"]] += count
                records.append(
                    {
                        "radius_m": radius,
                        "height_m": height,
                        "azimuth_deg": angle,
                        "position_m": position.tolist(),
                        "target_m": center.tolist(),
                        "horizontal_radius_m": horizontal_radius,
                        "frame_valid": bool(float(image.std()) >= 1.0),
                        "frame_mean": float(image.mean()),
                        "frame_std": float(image.std()),
                        "depth_valid_fraction": float(np.mean(valid_depth)),
                        "depth_min_m": float(np.nanmin(depth_m)),
                        "depth_median_m": float(np.nanmedian(depth_m)),
                        "depth_max_m": float(np.nanmax(depth_m)),
                        "supported_object_depth_pixels": object_depth_pixels,
                        "target_object_depth_pixels": object_depth_pixels.get(
                            f"obj_{target_object_index}", 0
                        ),
                        "goal_segmentation_id": goal_segmentation_id,
                        "target_point_count": int(len(target_points_object)),
                        "target_mask_png": str(target_mask_path),
                        "target_pointcloud_npz": str(target_pointcloud_path),
                        "rgb_image": str(image_path),
                        "depth_meters_npy": str(depth_float_path),
                        "depth_millimeters_png": str(depth_mm_path),
                    }
                )

            montage_path = montage_dir / f"{group_name}_montage.png"
            _save_group_montage(
                montage_path,
                center,
                table_min,
                table_max,
                positions,
                images,
                radius,
                height,
                horizontal_radius,
                angles,
                actor_label,
                support_label,
                support_z_levels,
                target_description,
            )
            montage_paths.append(montage_path)
            print(f"saved {montage_path}")
            visible_objects = sum(count > 0 for count in group_visibility.values())
            depth_montage_path = depth_montage_dir / f"{group_name}_depth_montage.png"
            _save_depth_montage(
                depth_montage_path,
                center,
                table_min,
                table_max,
                positions,
                depths,
                radius,
                height,
                horizontal_radius,
                angles,
                actor_label,
                depth_display_max_m,
                visible_objects,
                len(supported_objects),
                support_label,
                support_z_levels,
                target_description,
                group_visibility.get(f"obj_{target_object_index}", 0),
            )
            depth_montage_paths.append(depth_montage_path)
            pointcloud_groups[(radius, height)] = group_pointclouds
            print(f"saved {depth_montage_path}")

    target_cloud_axis_limit = _target_cloud_limit(pointcloud_groups)
    target_pointcloud_montage_paths = []
    for radius in radii:
        for height in heights:
            group_name = f"r{_safe_name(radius)}_h{_safe_name(height)}"
            path = (
                target_pointcloud_montage_dir
                / f"{group_name}_target_pointcloud_montage.png"
            )
            group = pointcloud_groups.get((radius, height))
            if group is not None:
                _save_target_pointcloud_montage(
                    path,
                    group,
                    radius,
                    height,
                    target_description,
                    goal_segmentation_id,
                    pointcloud_depth_min_m,
                    pointcloud_depth_max_m,
                    target_cloud_axis_limit,
                )
            target_pointcloud_montage_paths.append(path)

    pointcloud_density = []
    for (radius, height), group in sorted(pointcloud_groups.items()):
        counts = [len(item["points_object_centered_m"]) for item in group]
        pointcloud_density.append(
            {
                "radius_m": radius,
                "height_m": height,
                "points_per_view": counts,
                "mean_points_per_view": float(np.mean(counts)),
                "total_points_five_views": int(np.sum(counts)),
                "visible_views": int(np.count_nonzero(np.asarray(counts) > 0)),
            }
        )
    _save_pointcloud_density_plot(
        output_dir / "target_pointcloud_density_vs_radius.png",
        pointcloud_groups,
        radii,
        heights,
        target_description,
    )

    metadata = {
        "scene": scene_ref,
        "scene_asset": str(scene_asset),
        "support_region_center_m": center.tolist(),
        "support_xy_min_m": table_min.tolist(),
        "support_xy_max_m": table_max.tolist(),
        "support_label": support_label,
        "support_z_levels_m": support_z_levels,
        "target_object_index": target_object_index,
        "target_object_name": f"obj_{target_object_index}",
        "target_object_category": object_metadata[target_object_index]["category"],
        "target_object_task_label": target_object_label,
        "target_object_color": "red (1.0, 0.0, 0.0)",
        "goal_segmentation_id": goal_segmentation_id,
        "goal_segmentation_rule": "goal_seg_id = task_obj_index + 4",
        "pointcloud_depth_range_m": [
            pointcloud_depth_min_m,
            pointcloud_depth_max_m,
        ],
        "pointcloud_rule": (
            "one 3D point per pixel satisfying depth range and "
            "segmentation == goal_segmentation_id"
        ),
        "target_object_root_pose_xyz_xyzw": target_object_root_pose.tolist(),
        "target_object_center_local_m": target_object_center_local.tolist(),
        "target_object_center_world_m": target_object_center_world.tolist(),
        "object_centered_transform": (
            "p_object_centered = R_object^T * (p_world - t_object) "
            "- object_ref_point"
        ),
        "target_pointcloud_axis_limit_m": target_cloud_axis_limit,
        "pointcloud_density": pointcloud_density,
        "task_name": str(cfg.task.name),
        "actor": actor_label,
        "robot_type": robot_type,
        "hand_name": hand_name or None,
        "robot_joint_names": list(getattr(task, "robot_joint_names", [])),
        "hand_initial_root_pose": (
            list(task._hand_float_roots[0][:7])
            if hasattr(task, "_hand_float_roots")
            else None
        ),
        "depth_units": (
            "meters in .npy; millimeters in uint16 PNG "
            "(0=invalid, values above 65.535m clipped)"
        ),
        "depth_display_max_m": depth_display_max_m,
        "object_metadata": object_metadata,
        "infeasible_combinations": infeasible_combinations,
        "records": records,
    }
    (output_dir / "camera_poses.json").write_text(json.dumps(metadata, indent=2))
    group_count = len(radii) * len(heights)
    _save_contact_sheet(
        output_dir / f"all_{group_count}_groups_rgb.png",
        montage_paths,
        str(cfg.scene.name),
        "RGB camera sweep",
        radii,
        heights,
    )
    _save_contact_sheet(
        output_dir / f"all_{group_count}_groups_depth.png",
        depth_montage_paths,
        str(cfg.scene.name),
        "metric depth sweep",
        radii,
        heights,
    )
    _save_contact_sheet(
        output_dir / f"all_{group_count}_groups_target_pointcloud.png",
        target_pointcloud_montage_paths,
        str(cfg.scene.name),
        "target-only object-centered partial point clouds",
        radii,
        heights,
    )
    print(f"saved sweep to {output_dir}")
    task.exit()
    # Isaac Gym Preview 4 can fail in native teardown after all outputs are complete.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    launch()
