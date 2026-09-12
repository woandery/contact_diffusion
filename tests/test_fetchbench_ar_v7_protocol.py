import pytest
import yaml
from scripts.audit_basic_experiment_v7_protocol import INHERITED, V6_PATH, V7_PATH
from scripts.infer_local_contactdiffusion_grasp import training_observation_supports
from utils.basic_experiment_protocol import FETCHBENCH_AR32K_V7_PROTOCOL_ID, SUPPORTED_PROTOCOL_IDS


def test_v7_preserves_v6_execution():
    v6 = yaml.safe_load(V6_PATH.read_text())
    v7 = yaml.safe_load(V7_PATH.read_text())
    assert v7["protocol_id"] == FETCHBENCH_AR32K_V7_PROTOCOL_ID
    assert v7["protocol_id"] in SUPPORTED_PROTOCOL_IDS
    for section in INHERITED:
        assert v7[section] == v6[section]
    assert v7["contact_source"]["synthetic_partial_crop_applied_at_inference"] is False
    assert v7["contact_source"]["condition_observation"] == "full_object_pc"


@pytest.mark.parametrize("real,synthetic,full_supported", [
    (0.6, 0.2, True), (0.8, 0.2, False), (1.0, 0.0, False), (0.0, 0.0, True),
])
def test_three_way_mixture_support(real, synthetic, full_supported):
    cfg = {"object_observation_mode": "mixed_full_synthetic_real_partial",
           "real_partial_probability": real, "synthetic_partial_probability": synthetic}
    assert training_observation_supports(cfg, "full") is full_supported
    assert training_observation_supports(cfg, "synthetic_partial") is (synthetic > 0)
    assert training_observation_supports(cfg, "real_partial") is (real > 0)


@pytest.mark.parametrize("real,synthetic", [(-0.1, 0.2), (0.6, 0.6), (float("nan"), 0.2)])
def test_invalid_mixture_rejected(real, synthetic):
    with pytest.raises(ValueError):
        training_observation_supports({
            "object_observation_mode": "mixed_full_synthetic_real_partial",
            "real_partial_probability": real, "synthetic_partial_probability": synthetic,
        }, "full")
