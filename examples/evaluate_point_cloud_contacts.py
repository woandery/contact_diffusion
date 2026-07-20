"""Evaluate diffusion contacts using exact point-cloud indices and normals."""

import numpy as np

from utils import (
    PointCloudMultiContactGraspEvaluator,
    estimate_point_cloud_normals,
)


def precompute_normals(object_pc, output_path):
    """Run once for a complete closed cloud, then store with the dataset."""
    outward_normals = estimate_point_cloud_normals(object_pc, k_neighbors=30)
    np.save(output_path, outward_normals.astype(np.float32))


def evaluate_sample(sample):
    """Convert one dataset/dataloader sample into scalar diffusion labels."""
    object_pc = np.asarray(sample["object_pc"])
    object_normals = np.asarray(sample["object_normals"])
    selected_indices = np.asarray(sample["selected_indices"])

    quality = PointCloudMultiContactGraspEvaluator.evaluate(
        object_pc,
        selected_indices,
        object_normals,
        friction_coef=0.5,
        num_cone_faces=8,
        soft_fingers=True,
        # Prefer a physical COM from object metadata when it is available.
        center_of_mass=sample.get("center_of_mass"),
    )
    robust = PointCloudMultiContactGraspEvaluator.evaluate_robust(
        object_pc,
        selected_indices,
        object_normals,
        friction_coef=0.5,
        friction_sigma=0.05,
        normal_sigma=0.02,
        soft_fingers=True,
        num_samples=100,
        seed=7,
        center_of_mass=sample.get("center_of_mass"),
    )
    return {
        "valid": float(quality["valid"]),
        "force_closure_label": float(quality["force_closure"]),
        "epsilon_label": float(quality["epsilon"] if quality["valid"] else 0.0),
        "robust_epsilon_label": robust["mean_epsilon"],
        "robust_force_closure_label": robust["force_closure_rate"],
    }


if __name__ == "__main__":
    # Replace these paths with one preprocessed dataset sample.
    with np.load("sample_with_normals.npz", allow_pickle=False) as data:
        labels = evaluate_sample(
            {
                "object_pc": data["object_pc"],
                "object_normals": data["object_normals"],
                "selected_indices": data["selected_indices"],
            }
        )
    print(labels)
