import numpy as np
import torch
from types import SimpleNamespace

from datasets.contact_dataset import ContactDatasetV0, contact_collate_fn
from utils import (
    PointCloudMultiContactGraspEvaluator,
    estimate_point_cloud_normals,
    match_predicted_contacts_to_point_cloud,
)
from utils.validation_grasp_quality import evaluate_generated_contact_batch


SQRT3_OVER_2 = np.sqrt(3.0) / 2.0
CLOUD = np.array(
    [
        [1.0, 0, 0],
        [-1.0, 0, 0],
        [-0.5, SQRT3_OVER_2, 0],
        [-0.5, -SQRT3_OVER_2, 0],
        [0, 1.0, 0],
        [0, -1.0, 0],
        [0, 0, 1.0],
        [0, 0, -1.0],
    ]
)
NORMALS = CLOUD.copy()


def evaluate(indices, **kwargs):
    defaults = dict(
        friction_coef=0.8,
        num_cone_faces=16,
        soft_fingers=True,
        finger_radius=0.2,
        center_of_mass=np.zeros(3),
    )
    defaults.update(kwargs)
    return PointCloudMultiContactGraspEvaluator.evaluate(
        CLOUD, np.asarray(indices), NORMALS, **defaults
    )


def test_exact_selected_indices_supply_contacts_and_normals():
    indices = np.array([0, 2, 3])
    result = evaluate(indices)
    assert result["valid"]
    assert result["force_closure"]
    assert result["epsilon"] > 0
    assert np.array_equal(result["selected_indices"], indices)
    assert np.allclose(result["contact_points"], CLOUD[indices])
    assert np.allclose(result["contact_normals"], NORMALS[indices])
    assert result["surface_validation"] == "exact_point_cloud_indices"
    assert result["normal_convention"] == "outward_provided_unverified"


def test_two_and_five_indexed_contacts():
    two = evaluate([0, 1])
    five = evaluate([0, 1, 4, 5, 6])
    assert two["valid"] and two["num_contacts"] == 2
    assert five["valid"] and five["num_contacts"] == 5
    assert five["grasp_matrix"].shape == (6, 5 * (16 + 2))


def test_point_cloud_mean_is_explicit_fallback_for_center_of_mass():
    result = PointCloudMultiContactGraspEvaluator.evaluate(
        CLOUD,
        np.array([0, 1]),
        NORMALS,
        friction_coef=0.8,
        soft_fingers=True,
        finger_radius=0.2,
    )
    assert result["valid"]
    assert result["center_of_mass_source"] == "point_cloud_mean"
    assert np.allclose(result["center_of_mass"], CLOUD.mean(axis=0))


def test_invalid_normals_indices_and_duplicate_coordinates_are_rejected():
    bad_normals = NORMALS.copy()
    bad_normals[0] *= 2
    bad_normal = PointCloudMultiContactGraspEvaluator.evaluate(
        CLOUD, np.array([0, 1]), bad_normals
    )
    duplicate_index = PointCloudMultiContactGraspEvaluator.evaluate(
        CLOUD, np.array([0, 0]), NORMALS
    )
    duplicated_cloud = np.vstack((CLOUD, CLOUD[0]))
    duplicated_normals = np.vstack((NORMALS, NORMALS[0]))
    duplicate_point = PointCloudMultiContactGraspEvaluator.evaluate(
        duplicated_cloud, np.array([0, len(CLOUD)]), duplicated_normals
    )
    assert not bad_normal["valid"] and "unit" in bad_normal["failure_reason"]
    assert not duplicate_index["valid"] and "duplicate indices" in duplicate_index["failure_reason"]
    assert not duplicate_point["valid"] and "duplicate contact" in duplicate_point["failure_reason"]


def test_point_cloud_robust_is_reproducible():
    kwargs = dict(
        num_samples=16,
        normal_sigma=0.02,
        friction_coef=0.8,
        friction_sigma=0.03,
        seed=31,
        num_cone_faces=12,
        soft_fingers=True,
        finger_radius=0.2,
        center_of_mass=np.zeros(3),
    )
    first = PointCloudMultiContactGraspEvaluator.evaluate_robust(
        CLOUD, np.array([0, 2, 3]), NORMALS, **kwargs
    )
    second = PointCloudMultiContactGraspEvaluator.evaluate_robust(
        CLOUD, np.array([0, 2, 3]), NORMALS, **kwargs
    )
    assert first == second
    assert first["num_valid_samples"] == 16
    assert np.isclose(first["epsilon_std"] ** 2, first["epsilon_variance"])


def test_normal_precomputation_orients_complete_sphere_outward():
    count = 600
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (np.arange(count) + 0.5) / count
    radius = np.sqrt(1.0 - z * z)
    theta = golden_angle * np.arange(count)
    sphere = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
    estimated = estimate_point_cloud_normals(sphere, k_neighbors=24)
    alignment = np.sum(estimated * sphere, axis=1)
    assert np.allclose(np.linalg.norm(estimated, axis=1), 1.0)
    assert np.percentile(alignment, 5) > 0.98


def test_dataset_loads_and_collates_precomputed_normals(tmp_path):
    sample_dir = tmp_path / "train" / "n2"
    sample_dir.mkdir(parents=True)
    count = 2048
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    z = np.linspace(-0.999, 0.999, count)
    radius = np.sqrt(1.0 - z * z)
    points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z)).astype(
        np.float32
    )
    indices = np.array([0, 1024], dtype=np.int64)
    np.savez(
        sample_dir / "sample_00000000.npz",
        object_pc=points,
        object_normals=points,
        contacts=points[indices],
        num_contacts=np.array(2),
        selected_indices=indices,
        object_name=np.array("sphere"),
        robot_name=np.array("test_hand"),
    )
    dataset = ContactDatasetV0(str(tmp_path), "train", 2)
    item = dataset[0]
    batch = contact_collate_fn([item, item])
    assert item["object_normals"].shape == (2048, 3)
    assert batch["object_normals"].shape == (2, 2048, 3)
    assert torch.allclose(item["object_normals"], item["object_pc"], atol=1e-6)


def test_prediction_matching_is_unique_and_distance_gated():
    predictions = np.array([[0.99, 0, 0], [0.98, 0.01, 0]])
    matched = match_predicted_contacts_to_point_cloud(
        predictions, CLOUD, max_projection_distance=2.0
    )
    rejected = match_predicted_contacts_to_point_cloud(
        predictions + 10.0, CLOUD, max_projection_distance=0.1
    )
    assert matched["valid"]
    assert len(np.unique(matched["predicted_indices"])) == 2
    assert not rejected["valid"]
    assert "distance gate" in rejected["failure_reason"]


class FixedSampleModel:
    def __init__(self, contacts):
        self.contacts = torch.as_tensor(contacts, dtype=torch.float32)

    def sample(self, object_pc, num_contacts, **_kwargs):
        return self.contacts.to(object_pc.device).expand(len(object_pc), -1, -1)


def test_validation_metrics_score_generated_not_ground_truth_indices():
    generated = CLOUD[[0, 2, 3]] + np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
    config = SimpleNamespace(
        samples_per_batch=1,
        dc=3,
        num_steps=2,
        sampler="ddim",
        max_projection_distance=0.05,
        max_projection_distance_factor=2.0,
        require_precomputed_normals=True,
        friction_coef=0.8,
        num_cone_faces=16,
        soft_fingers=True,
        finger_radius=0.2,
        torque_scaling=None,
        fixed_total_force_budget=True,
    )
    cloud_batch = torch.from_numpy(CLOUD.astype(np.float32)).unsqueeze(0)
    normal_batch = torch.from_numpy(NORMALS.astype(np.float32)).unsqueeze(0)
    metrics = evaluate_generated_contact_batch(
        FixedSampleModel(generated),
        cloud_batch,
        cloud_batch,
        3,
        config,
        object_normals=normal_batch,
        seed=4,
    )
    assert metrics["grasp_projection_valid_rate"] == 1.0
    assert metrics["grasp_quality_valid_rate"] == 1.0
    assert metrics["grasp_force_closure_rate"] == 1.0
    assert metrics["grasp_epsilon_mean"] > 0
    assert metrics["grasp_projection_distance_max"] > 0


def test_validation_invalid_prediction_contributes_zero_quality():
    config = SimpleNamespace(
        samples_per_batch=1,
        dc=3,
        num_steps=2,
        sampler="ddim",
        max_projection_distance=0.01,
        max_projection_distance_factor=2.0,
        require_precomputed_normals=True,
    )
    cloud_batch = torch.from_numpy(CLOUD.astype(np.float32)).unsqueeze(0)
    normal_batch = torch.from_numpy(NORMALS.astype(np.float32)).unsqueeze(0)
    metrics = evaluate_generated_contact_batch(
        FixedSampleModel(np.full((3, 3), 10.0)),
        cloud_batch,
        cloud_batch,
        3,
        config,
        object_normals=normal_batch,
    )
    assert metrics["grasp_projection_valid_rate"] == 0.0
    assert metrics["grasp_force_closure_rate"] == 0.0
    assert metrics["grasp_epsilon_mean"] == 0.0
