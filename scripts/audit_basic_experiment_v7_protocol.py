#!/usr/bin/env python3
"""Audit the v7 checkpoint and exact inheritance of the v6 execution route."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from scripts.audit_basic_experiment_v6_protocol import load_merged_config, sha256
from utils.basic_experiment_protocol import (
    FETCHBENCH_AR32K_V7_PROTOCOL_ID, MIXED_FULL_PARTIAL_AR56K_V6_PROTOCOL_ID,
)

V7_PATH = ROOT / "configs/basic_experiment_fetchbench_ar32k_eawq_o10i20_palm0_v7_protocol.yaml"
V6_PATH = ROOT / "configs/basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_protocol.yaml"
INHERITED = ("generation", "fk_optimization", "ranking", "hands", "isaac_gym", "execution")
TRAINING = {
    "object_observation_mode": "mixed_full_synthetic_real_partial",
    "real_partial_probability": 0.6,
    "synthetic_partial_probability": 0.2,
    "partial_normalization_reference": "full",
    "normalize": True, "num_points": 2048, "n_values": [2, 3, 5],
    "append_observation_type_feature": False,
}
SHA = "c55badbd2e1ce7bc9cda9b58832e68003b4b757ee02eedee47e01e697c7ec491"


def equal(actual, expected, field):
    if actual != expected:
        raise ValueError(f"{field}: {actual!r} != {expected!r}")


def audit(checkpoint=None):
    v7 = yaml.safe_load(V7_PATH.read_text())
    v6 = yaml.safe_load(V6_PATH.read_text())
    equal(v7["protocol_id"], FETCHBENCH_AR32K_V7_PROTOCOL_ID, "protocol_id")
    equal(v7["supersedes"], MIXED_FULL_PARTIAL_AR56K_V6_PROTOCOL_ID, "supersedes")
    equal(v7["base_protocol_id"], v6["base_protocol_id"], "base_protocol_id")
    for section in INHERITED:
        equal(v7[section], v6[section], section)
    changed_training = {
        "source", "checkpoint_training_observation", "checkpoint_training_full_probability",
        "checkpoint_training_partial_probability",
        "checkpoint_training_real_partial_probability",
        "checkpoint_training_synthetic_partial_probability",
        "checkpoint_training_real_partial_origin",
    }
    for key in set(v6["contact_source"]) | set(v7["contact_source"]):
        if key not in changed_training:
            equal(v7["contact_source"].get(key), v6["contact_source"].get(key), key)
    spec = v7["model_checkpoint"]
    equal(spec["step"], 32000, "step")
    equal(spec["sha256"], SHA, "sha256")
    equal(spec["repository_path"], "weights/v7/best_val.pt", "repository_path")
    path = Path(checkpoint).resolve() if checkpoint else ROOT / spec["repository_path"]
    equal(path.stat().st_size, spec["size_bytes"], "checkpoint size")
    equal(sha256(path), SHA, "checkpoint hash")
    config_path = ROOT / spec["training_config"]
    equal(sha256(config_path), spec["training_config_sha256"], "training config hash")
    cfg = load_merged_config(config_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    equal(ckpt["step"], 32000, "embedded step")
    equal(ckpt["model_type"], "autoregressive_contact_diffusion", "model type")
    for key, value in TRAINING.items():
        equal(cfg["dataset"].get(key), value, f"config.dataset.{key}")
        equal(ckpt["config"]["dataset"].get(key), value, f"checkpoint.dataset.{key}")
    for key, value in {"max_steps": 32000, "lr": 5e-5, "batch_size": 192}.items():
        equal(cfg["train"][key], value, f"config.train.{key}")
        equal(ckpt["config"]["train"][key], value, f"checkpoint.train.{key}")
    equal(ckpt["config"]["model"]["object_input_dim"], 3, "object input dimension")
    equal(ckpt["config"]["diffusion"]["prediction_type"], "v_prediction", "prediction")
    if not all(torch.isfinite(value).all().item() for value in ckpt["model"].values()):
        raise ValueError("Nonfinite model state")
    return {
        "status": "pass", "protocol_id": v7["protocol_id"],
        "checkpoint": str(path), "checkpoint_sha256": SHA, "checkpoint_step": 32000,
        "training_mixture": {"fetchbench_camera_partial": 0.6, "synthetic_partial": 0.2, "full": 0.2},
        "inference_observation": v7["contact_source"]["condition_observation"],
        "v6_sections_exact": list(INHERITED),
        "scope": "configuration_and_weight_audit_not_physics_success_evaluation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
