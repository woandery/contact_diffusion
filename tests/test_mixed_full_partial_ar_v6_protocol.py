from pathlib import Path

import yaml

from utils.basic_experiment_protocol import (
    MIXED_FULL_PARTIAL_AR56K_V6_PROTOCOL_ID,
    PARTIAL_AR64K_V5_PROTOCOL_ID,
    PROTOCOL_ID,
    SUPPORTED_PROTOCOL_IDS,
)


ROOT = Path(__file__).resolve().parents[1]
V6_PATH = (
    ROOT
    / "configs/basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_protocol.yaml"
)
V5_PATH = (
    ROOT
    / "configs/basic_experiment_partial_ar64k_eawq_o10i20_palm0_v5_protocol.yaml"
)
SMOKE_PATH = (
    ROOT
    / "configs/basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_smoke1_protocol.yaml"
)


def test_v6_pins_mixed_checkpoint_and_full_cloud_inference():
    protocol = yaml.safe_load(V6_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == MIXED_FULL_PARTIAL_AR56K_V6_PROTOCOL_ID
    assert protocol["supersedes"] == PARTIAL_AR64K_V5_PROTOCOL_ID
    assert protocol["base_protocol_id"] == PROTOCOL_ID
    assert MIXED_FULL_PARTIAL_AR56K_V6_PROTOCOL_ID in SUPPORTED_PROTOCOL_IDS

    checkpoint = protocol["model_checkpoint"]
    assert checkpoint["step"] == 56000
    assert checkpoint["repository_path"] == "weights/v6/step_00056000.pt"
    assert checkpoint["sha256"] == (
        "05bd542bb0c10a74ed8dafd7586a58ceedb17097dd08eb344011bb91e98516b4"
    )
    assert checkpoint["size_bytes"] == 86709034
    assert (ROOT / checkpoint["repository_path"]).is_file()

    source = protocol["contact_source"]
    assert source["checkpoint_training_observation"] == (
        "mixed_full_synthetic_partial"
    )
    assert source["checkpoint_training_full_probability"] == 0.5
    assert source["checkpoint_training_partial_probability"] == 0.5
    assert source["condition_observation"] == "full_object_pc"
    assert source["synthetic_partial_crop_applied_at_inference"] is False
    assert source["training_supports_inference_observation"] is True
    assert source["partial_seed_formula"] == "none_not_applicable"


def test_v6_inherits_v5_budget_fk_ranking_and_physics_exactly():
    v6 = yaml.safe_load(V6_PATH.read_text(encoding="utf-8"))
    v5 = yaml.safe_load(V5_PATH.read_text(encoding="utf-8"))
    for section in (
        "generation",
        "fk_optimization",
        "ranking",
        "hands",
        "isaac_gym",
        "execution",
    ):
        assert v6[section] == v5[section]
    assert v6["generation"] == {
        "contact_sets_per_object": 32,
        "particles_per_contact_set": 32,
        "optimization_steps": 400,
        "retained_per_contact_set": 32,
    }


def test_v6_runner_freezes_full_observation_and_32_sets():
    runner = (
        ROOT
        / "scripts/run_basic_experiment_v6_mixed_full_partial_ar56k_ood10_8gpu.sh"
    ).read_text(encoding="utf-8")
    assert "step_00056000.pt" in runner
    assert "CONTACT_AR_BASIC_SAMPLES=32" in runner
    assert "CONTACT_AR_BASIC_OBSERVATION=full" in runner
    assert "basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_protocol.yaml" in runner


def test_v6_local_smoke_reduces_only_dataset_scope_and_uses_cpu_physx():
    formal = yaml.safe_load(V6_PATH.read_text(encoding="utf-8"))
    smoke = yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8"))
    assert smoke["base_protocol_id"] == formal["base_protocol_id"]
    assert smoke["model_checkpoint"]["step"] == formal["model_checkpoint"]["step"]
    assert smoke["model_checkpoint"]["sha256"] == formal["model_checkpoint"]["sha256"]
    assert smoke["generation"] == {
        "contact_sets_per_object": 1,
        "particles_per_contact_set": 32,
        "optimization_steps": 400,
        "retained_per_contact_set": 32,
    }
    for hand in ("Barrett", "shadow_hand"):
        assert smoke["hands"][hand]["closure_adapter"] == formal["hands"][hand][
            "closure_adapter"
        ]
    for key in (
        "name",
        "version",
        "computation_mode",
        "ranking_only",
        "scope",
        "feasibility_gate",
        "distal_metric",
        "full_hand_metric",
        "normalization",
        "fusion",
        "tie_break",
        "success_labels_used",
        "final_retained_per_contact_set",
    ):
        assert smoke["ranking"][key] == formal["ranking"][key]

    runner = (ROOT / "scripts/run_basic_experiment_v6_local_smoke.sh").read_text(
        encoding="utf-8"
    )
    assert "--inference-object-observation full" in runner
    assert "--particles 32 --optimization-steps 400 --diffusion-steps 50" in runner
    assert "--cpu-physics" in runner
    assert "--direction-seconds 1.0 --direction-order cedex" in runner
