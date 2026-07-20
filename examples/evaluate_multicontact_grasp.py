"""Minimal Dex-Net + ContactDiffusion multi-contact quality example.

Run with the Dex-Net source directory on PYTHONPATH and replace the two file
arguments with the matching mesh and SDF for one object.
"""

import argparse

import numpy as np
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile

from dexnet.grasping import GraspableObject3D
from utils.multi_contact_grasp_quality import MultiContactGraspEvaluator


def scalar_training_label(result):
    """A simple [0, 1) regression label; keep FC as a separate class label."""
    if not result["valid"]:
        return 0.0
    return float(result["epsilon"] / (1.0 + result["epsilon"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("sdf")
    args = parser.parse_args()

    obj = GraspableObject3D(SdfFile(args.sdf).read(), ObjFile(args.mesh).read())
    surface_points, _ = obj.sdf.surface_points(grid_basis=False)

    # In training/evaluation, replace these with diffusion output in object frame.
    diffusion_outputs = {
        2: np.asarray(surface_points[:2]),
        3: np.asarray(surface_points[:3]),
        5: np.asarray(surface_points[:5]),
    }
    for num_contacts, points in diffusion_outputs.items():
        quality = MultiContactGraspEvaluator.evaluate(
            obj,
            points,
            friction_coef=0.5,
            num_cone_faces=8,
            soft_fingers=True,
        )
        robust = MultiContactGraspEvaluator.evaluate_robust(
            obj,
            points,
            friction_coef=0.5,
            num_cone_faces=8,
            soft_fingers=True,
            num_samples=100,
            # Nonzero point noise requires a trusted projection callback.
            point_sigma=0.0,
            normal_sigma=0.02,
            friction_sigma=0.05,
            seed=7,
        )
        labels = {
            "force_closure": float(quality["force_closure"]),
            "epsilon": scalar_training_label(quality),
            "robust_epsilon": robust["mean_epsilon"],
            "robust_force_closure": robust["force_closure_rate"],
        }
        print(f"n={num_contacts}: valid={quality['valid']} labels={labels}")


if __name__ == "__main__":
    main()
