#!/usr/bin/env python3
"""Fail-fast audit for the frozen mixed-full/partial AR basic experiment v6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.basic_experiment_protocol import (  # noqa: E402
    MIXED_FULL_PARTIAL_AR56K_V6_PROTOCOL_ID,
    PARTIAL_AR64K_V5_PROTOCOL_ID,
    PROTOCOL_ID,
)


PROTOCOL_PATH = (
    ROOT
    / "configs/basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_protocol.yaml"
)
V5_PROTOCOL_PATH = (
    ROOT
    / "configs/basic_experiment_partial_ar64k_eawq_o10i20_palm0_v5_protocol.yaml"
)
EXPECTED_CHECKPOINT = Path(
    "/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion/outputs/"
    "contact_ar_success_k128_mixedfullpartial_normfull_from8k_lr2e4_gb768_56k_4x4090/"
    "model/checkpoints/step_00056000.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "05bd542bb0c10a74ed8dafd7586a58ceedb17097dd08eb344011bb91e98516b4"
)
EXPECTED_REPOSITORY_CHECKPOINT = ROOT / "weights/v6/step_00056000.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drift: {actual!r} != {expected!r}")


def merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_merged_config(path: Path) -> dict:
    current = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_name = current.pop("base_config", None)
    if base_name is None:
        return current
    base_path = Path(str(base_name))
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return merge_dicts(load_merged_config(base_path), current)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional accessible copy of the remote checkpoint for SHA256 audit.",
    )
    args = parser.parse_args()

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    v5 = yaml.safe_load(V5_PROTOCOL_PATH.read_text(encoding="utf-8"))
    require_equal(
        protocol.get("protocol_id"),
        MIXED_FULL_PARTIAL_AR56K_V6_PROTOCOL_ID,
        "protocol_id",
    )
    require_equal(protocol.get("supersedes"), PARTIAL_AR64K_V5_PROTOCOL_ID, "supersedes")
    require_equal(protocol.get("base_protocol_id"), PROTOCOL_ID, "base_protocol_id")

    checkpoint = protocol.get("model_checkpoint", {})
    require_equal(checkpoint.get("path"), str(EXPECTED_CHECKPOINT), "checkpoint.path")
    require_equal(
        checkpoint.get("repository_path"),
        "weights/v6/step_00056000.pt",
        "checkpoint.repository_path",
    )
    require_equal(checkpoint.get("step"), 56000, "checkpoint.step")
    require_equal(checkpoint.get("sha256"), EXPECTED_CHECKPOINT_SHA256, "checkpoint.sha256")
    require_equal(checkpoint.get("size_bytes"), 86709034, "checkpoint.size_bytes")
    require_equal(
        checkpoint.get("model_type"),
        "autoregressive_contact_diffusion",
        "checkpoint.model_type",
    )

    source = protocol.get("contact_source", {})
    expected_source = {
        "checkpoint_training_observation": "mixed_full_synthetic_partial",
        "checkpoint_training_full_probability": 0.5,
        "checkpoint_training_partial_probability": 0.5,
        "condition_observation": "full_object_pc",
        "condition_shape": [2048, 3],
        "synthetic_partial_crop_applied_at_inference": False,
        "training_supports_inference_observation": True,
        "normalization_reference": "full_2048_object_pc_centered_unit_radius",
        "representation": "free_xyz_then_nearest_full_2048_object_point",
        "generation_order": "sequential_generated_prefix",
        "diffusion_steps_per_point": 50,
        "sampler": "ddim",
        "model_project_to_surface": False,
        "fk_adapter_project_to_full_surface": True,
        "partial_seed_formula": "none_not_applicable",
    }
    for key, expected in expected_source.items():
        require_equal(source.get(key), expected, f"contact_source.{key}")

    for section in (
        "generation",
        "fk_optimization",
        "ranking",
        "hands",
        "isaac_gym",
        "execution",
    ):
        require_equal(protocol.get(section), v5.get(section), f"v5-inherited {section}")

    train_config_path = ROOT / str(checkpoint["training_config"])
    require_equal(
        sha256(train_config_path),
        checkpoint["training_config_sha256"],
        "training_config_sha256",
    )
    train_config = load_merged_config(train_config_path)
    dataset = train_config["dataset"]
    require_equal(dataset["object_observation_mode"], "mixed_full_synthetic_partial", "training observation")
    require_equal(dataset["partial_probability"], 0.5, "training partial probability")
    require_equal(dataset["synthetic_partial_keep_ratio"], 0.5, "training keep ratio")
    require_equal(dataset["partial_normalization_reference"], "full", "training normalization")
    require_equal(dataset["num_points"], 2048, "training point count")
    require_equal(dataset["n_values"], [2, 3, 5], "training contact counts")
    require_equal(train_config["model"]["object_input_dim"], 3, "training object input")
    require_equal(train_config["diffusion"]["prediction_type"], "v_prediction", "diffusion target")
    require_equal(train_config["train"]["max_steps"], 56000, "new training steps")
    require_equal(train_config["train"]["lr"], 0.0002, "training learning rate")
    require_equal(train_config["train"]["batch_size"], 192, "per-GPU batch")

    runner_path = (
        ROOT
        / "scripts/run_basic_experiment_v6_mixed_full_partial_ar56k_ood10_8gpu.sh"
    )
    runner = runner_path.read_text(encoding="utf-8")
    for token in (
        "step_00056000.pt",
        "CONTACT_AR_BASIC_SAMPLES=32",
        "CONTACT_AR_BASIC_OBSERVATION=full",
        EXPECTED_CHECKPOINT_SHA256,
    ):
        if token not in runner:
            raise ValueError(f"v6 runner missing frozen token: {token}")

    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint
        else EXPECTED_REPOSITORY_CHECKPOINT
    )
    checkpoint_audit = {
        "path": str(checkpoint_path),
        "accessible": checkpoint_path.is_file(),
        "sha256": None,
    }
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"v6 checkpoint is missing: {checkpoint_path}")
    checkpoint_audit["sha256"] = sha256(checkpoint_path)
    require_equal(
        checkpoint_audit["sha256"],
        EXPECTED_CHECKPOINT_SHA256,
        "accessible checkpoint SHA256",
    )

    report = {
        "status": "pass",
        "protocol_id": MIXED_FULL_PARTIAL_AR56K_V6_PROTOCOL_ID,
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "checkpoint": checkpoint_audit,
        "generation": protocol["generation"],
        "contact_source": source,
        "v5_inherited_sections_exact": [
            "generation",
            "fk_optimization",
            "ranking",
            "hands",
            "isaac_gym",
            "execution",
        ],
        "v6_physx_results_present": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
