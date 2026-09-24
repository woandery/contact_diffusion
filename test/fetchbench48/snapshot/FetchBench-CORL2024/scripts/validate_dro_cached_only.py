"""Run the unchanged validator, forbidding implicit candidate regeneration."""
import runpy
import subprocess
from pathlib import Path

import isaacgym  # noqa: F401 -- must precede torch/task imports
from isaacgymenvs.tasks.fetch.fetch_ptd_dro_render import FetchPtdDRORender

original = FetchPtdDRORender._run_dro


def cached_only(self, goal_pc):
    assert self.cfg['solution']['physics_only']
    run = subprocess.run

    def forbidden(*args, **kwargs):
        raise RuntimeError('Cached-only validation: candidate metadata mismatch; regeneration forbidden')

    subprocess.run = forbidden
    try:
        return original(self, goal_pc)
    finally:
        subprocess.run = run


FetchPtdDRORender._run_dro = cached_only
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'InfiniGym/isaacgymenvs/validate_dro_lift.py'), run_name='__main__')
