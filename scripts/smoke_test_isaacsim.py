#!/usr/bin/env python3
"""Launch Isaac Sim headlessly, create a physics world, step it, and exit."""

from __future__ import annotations

import json
import os
import sys


portable_root = os.environ.get("CONTACTDIFF_OMNI_PORTABLE_ROOT")
if portable_root and "--portable-root" not in sys.argv:
    sys.argv.extend(["--portable-root", portable_root])

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": True,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
        "fast_shutdown": True,
    }
)

try:
    from omni.isaac.core import World

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240.0)
    world.scene.add_default_ground_plane()
    world.reset()
    for _ in range(8):
        world.step(render=False)
    print(
        json.dumps(
            {
                "isaacsim_launch": "ok",
                "headless": True,
                "physics_steps": 8,
                "physics_dt": 1.0 / 240.0,
            },
            indent=2,
        ),
        flush=True,
    )
finally:
    simulation_app.close()
