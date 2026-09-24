#!/usr/bin/env python3
"""Run the v6 environment refiner using only geometry visible in scene_pc.

The original v6 refiner assumes that every scene partial point cloud contains
at least 32 points in a narrow band around the object's support height, then
fits a signed tabletop plane. Formal 111-camera sampling includes grazing and
underside views for which that plane is legitimately unobservable. In those
views, this adapter disables only the unavailable signed-plane term while
retaining the visible-scene point-clearance energy. It does not use the full
scene mesh or any hidden environment geometry.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("refine_contactdiffusion_environment_six.py")
SPEC = importlib.util.spec_from_file_location("contactdiff_environment_v6", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ORIGINAL_PREPARE = MODULE.prepare_environment


def prepare_visible_environment(
    scene: np.ndarray,
    object_pc: np.ndarray,
    *,
    crop_margin: float,
    voxel: float,
    maximum: int,
) -> tuple[np.ndarray, dict]:
    try:
        info = ORIGINAL_PREPARE(
            scene,
            object_pc,
            crop_margin=crop_margin,
            voxel=voxel,
            maximum=maximum,
        )
        info[1]["support_plane_observed"] = True
        info[1]["environment_geometry_source"] = "selected_view_partial_scene"
        return info
    except RuntimeError as error:
        if "support plane" not in str(error):
            raise

    lower = object_pc.min(axis=0) - float(crop_margin)
    upper = object_pc.max(axis=0) + float(crop_margin)
    mask = np.all((scene >= lower[None]) & (scene <= upper[None]), axis=1)
    local = np.ascontiguousarray(scene[mask], dtype=np.float32)
    if len(local) == 0:
        raise RuntimeError("No visible scene points remain after local crop")
    sampled = MODULE.voxel_downsample(local, voxel, maximum)

    # A very low plane makes ReLU(support_z + clearance - z) identically zero
    # for this metric-scale scene. The remaining term is exactly the unsigned
    # clearance penalty to the selected view's visible scene points.
    disabled_support_z = float(min(scene[:, 2].min(), object_pc[:, 2].min()) - 1000.0)
    return sampled, {
        "input_scene_points": int(len(scene)),
        "cropped_scene_points": int(len(local)),
        "optimized_scene_points": int(len(sampled)),
        "voxel_size_m": float(voxel),
        "support_plane_z_m": disabled_support_z,
        "table_xy_min": lower[:2].tolist(),
        "table_xy_max": upper[:2].tolist(),
        "support_plane_observed": False,
        "environment_geometry_source": "selected_view_partial_scene",
        "support_plane_fallback": "disabled_unobserved_signed_plane_term",
    }


MODULE.prepare_environment = prepare_visible_environment

if __name__ == "__main__":
    MODULE.main()
