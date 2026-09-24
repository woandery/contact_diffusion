"""Physically close and lift D(R,O) hand candidates in a FetchBench scene."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import isaacgym  # noqa: F401 - import before task modules
from omegaconf import DictConfig, OmegaConf

import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


@hydra.main(version_base="1.1", config_name="config", config_path="./config")
def launch(cfg: DictConfig) -> None:
    allowed = {"FetchPtdDRORenderBarrett", "FetchPtdDRORenderShadow"}
    if cfg.task.name not in allowed:
        raise ValueError(f"validate_dro_lift.py requires one of {sorted(allowed)}")

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    cfg.task.task.scene_config_path = cfg.scene.scene_list
    cfg.task.experiment_name = (
        f"{cfg.scene.name}_{cfg.task.name}_lift_task"
        f"{int(cfg.task.solution.task_index):03d}"
    )

    env = isaacgymenvs.make(
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
    task_index = int(cfg.task.solution.task_index)
    if task_index < 0 or task_index >= int(cfg.scene.num_tasks):
        raise IndexError(
            f"task_index={task_index} is outside [0, {int(cfg.scene.num_tasks) - 1}]"
        )

    env.reset_task(task_index)
    summary = env.validate_lifts()
    artifact_dir = Path(summary["artifact_dir"])
    (artifact_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    print(json.dumps(summary, indent=2))
    env.exit()


if __name__ == "__main__":
    launch()
