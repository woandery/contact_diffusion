from pathlib import Path

import numpy as np
import yaml

from scripts.summarize_contact_set_guidance_ab import aggregate, exact_mcnemar_p
from scripts.summarize_contact_energy_diagnostic import (
    Arm,
    aggregate as aggregate_energy_arm,
    compare as compare_energy_arms,
)
from utils.basic_experiment_protocol import PROTOCOL_ID
from utils.contact_set_guidance import (
    MATCHER_VERSION,
    contact_set_signature,
    matched_random_surface_contacts,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_surface() -> np.ndarray:
    rng = np.random.default_rng(11)
    points = rng.normal(size=(512, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= np.asarray([0.05, 0.035, 0.025])
    return points.astype(np.float32)


def test_matched_random_surface_control_is_deterministic_unique_and_label_free():
    surface = synthetic_surface()
    reference = surface[[3, 17, 91, 203, 411]]
    first = matched_random_surface_contacts(
        surface, reference, seed=1234, candidate_count=256
    )
    second = matched_random_surface_contacts(
        surface, reference, seed=1234, candidate_count=256
    )

    assert np.array_equal(first["contacts"], second["contacts"])
    assert first["indices"] == second["indices"]
    assert len(set(first["indices"])) == len(reference)
    assert np.array_equal(first["contacts"], surface[first["indices"]])
    distances = np.linalg.norm(
        first["contacts"][:, None, :] - reference[None, :, :], axis=2
    )
    assert not np.all(distances.min(axis=1) < 1.0e-9)
    diagnostics = first["diagnostics"]
    assert diagnostics["matcher_version"] == MATCHER_VERSION
    assert diagnostics["centroid_direction_matched"] is False
    assert diagnostics["success_labels_used"] is False
    assert np.isfinite(diagnostics["score"])


def test_contact_signature_is_rotation_and_scale_normalized():
    surface = synthetic_surface()
    contacts = surface[[0, 1, 2, 3, 4]]
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    left = contact_set_signature(contacts, surface)
    right = contact_set_signature(
        3.5 * contacts @ rotation.T + 0.7,
        3.5 * surface @ rotation.T + 0.7,
    )
    assert np.allclose(left.pairwise_distances, right.pairwise_distances)
    assert np.allclose(left.radial_distances, right.radial_distances)
    assert np.isclose(left.centroid_radius, right.centroid_radius)


def test_guidance_protocol_is_an_all_particle_v4_paired_extension():
    protocol = yaml.safe_load(
        (ROOT / "configs/contact_set_guidance_ab_palm0_v4_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["base_protocol_id"] == PROTOCOL_ID
    assert protocol["generation"] == {
        "contact_sets_per_object": 64,
        "particles_per_contact_set": 32,
        "optimization_steps": 400,
        "retained_per_contact_set": 32,
    }
    assert protocol["contact_target_ab"]["pairing"]["fk_initialization"] == (
        "identical and derived from the paired diffusion set"
    )
    assert protocol["fk_optimization"]["palm_distance_weight"] == 0.0
    assert protocol["isaac_gym"]["active_env_count"] == 512
    assert protocol["isaac_gym"]["envs_per_row"] == 23
    assert protocol["isaac_gym"]["batches_per_object_hand_variant"] == 4


def test_mcnemar_exact_probability_is_stable_for_large_experiments():
    value = exact_mcnemar_p(12000, 12500)
    assert 0.0 <= value <= 1.0


def test_paired_summary_reports_particle_and_contact_set_outcomes():
    keys = [
        ("barrett", "apple", sample, particle)
        for sample in range(2)
        for particle in range(2)
    ]
    outcomes_a = {
        key: {"valid_simulation": True, "final_success": key[2] == 0}
        for key in keys
    }
    outcomes_b = {
        key: {
            "valid_simulation": True,
            "final_success": key[2] == 1 and key[3] == 0,
        }
        for key in keys
    }
    report = aggregate(
        keys,
        outcomes_a,
        outcomes_b,
        bootstrap_seed=3,
        bootstrap_draws=20,
    )["final"]
    assert report["diffusion_particle_successes"] == 2
    assert report["matched_random_particle_successes"] == 1
    assert report["diffusion_oracle32_sets"] == 1
    assert report["matched_random_oracle32_sets"] == 1
    assert report["oracle32_diffusion_only_sets"] == 1
    assert report["oracle32_random_only_sets"] == 1


def test_h100_runner_keeps_the_frozen_v4_budget_and_layout():
    runner = (ROOT / "scripts/run_contact_set_guidance_ab_4h100.sh").read_text(
        encoding="utf-8"
    )
    assert "--samples-per-object 64" in runner
    assert "--particles 32 --optimization-steps 400 --diffusion-steps 50" in runner
    assert "--initialization-contact-source diffusion" in runner
    assert "--candidate-mode all" in runner
    assert "--max-samples-per-object 512" in runner
    assert "--device-id 0 --envs-per-row 23" in runner
    assert "task_index+=gpu_count" in runner


def _synthetic_energy_arm(name: str, successes: set[tuple[int, int]]) -> Arm:
    candidates = {}
    outcomes = {}
    metadata = {}
    for sample in range(2):
        set_key = ("barrett", "apple", sample)
        metadata[set_key] = {
            "sample_seed": sample,
            "source_diffusion_contacts_sha256": f"source-{sample}",
            "target_contacts_sha256": f"source-{sample}",
            "initialization_state_sha256": f"initial-{sample}",
            "particle_indices": (0, 1),
        }
        for particle in range(2):
            key = (*set_key, particle)
            candidates[key] = {
                "rank": particle,
                "contact_chamfer_m": 0.001 * (particle + 1),
                "assigned_contact_error_m": 0.002 * (particle + 1),
            }
            success = (sample, particle) in successes
            outcomes[key] = {
                "rank": particle,
                "final": success,
                "strict": success,
                "valid": True,
            }
    return Arm(
        name=name,
        root=ROOT,
        contact_weight=100.0,
        target_mode="diffusion",
        metadata=metadata,
        candidates=candidates,
        outcomes=outcomes,
    )


def test_contact_energy_summary_reports_multik_and_paired_delta():
    reference = _synthetic_energy_arm("w100", {(0, 0)})
    improved = _synthetic_energy_arm("w200", {(0, 0), (1, 1)})
    keys = sorted(reference.outcomes)

    aggregate_report = aggregate_energy_arm(improved, keys)
    assert aggregate_report["final"]["particle_success_rate"] == 0.5
    assert aggregate_report["final"]["success_at_k"]["1"]["sets"] == 1
    assert aggregate_report["final"]["success_at_k"]["2"]["sets"] == 2
    assert aggregate_report["fk_contact_attainment"]["contact_chamfer_m"][
        "median"
    ] == 0.0015

    paired = compare_energy_arms(
        improved, reference, keys, seed=7, draws=20
    )["final"]
    assert paired["particle_delta_pp"] == 25.0
    assert paired["success_at_k"]["2"]["delta_pp"] == 50.0


def test_contact_energy_runner_freezes_contacts_and_sweeps_one_fk_term():
    runner = (ROOT / "scripts/run_contact_energy_diagnostic_4gpu.sh").read_text(
        encoding="utf-8"
    )
    assert (
        "new_variants=(contact_w000 contact_w025 contact_w100_4090 contact_w200)"
        in runner
    )
    assert "--source-diffusion-candidates" in runner
    assert "--initialization-contact-source diffusion" in runner
    assert '--contact-weight "${contact_weights[${variant}]}"' in runner
    assert "validating_320_gpu_physx_batches" in runner
