"""Quality metrics for an already generated set of 3-D contacts.

This module follows Dex-Net's point-contact conventions without constructing a
``ParallelJawPtGrasp3D``.  Public arrays are in the object frame.  Returned
``contact_normals`` are outward-facing, while friction-cone forces use the
opposite (inward-facing) normal.  ``forces`` and ``torques`` are shaped
``(3, M)`` and ``grasp_matrix`` is shaped ``(6, M)``.

Ferrari-Canny epsilon is the radius of the origin-centred Euclidean ball in the
convex hull of the primitive wrenches.  This is the usual global L1 contact
force budget: adding contacts adds choices, not force budget.  Target-wrench
metrics optionally use per-contact limits.  With ``fixed_total_force_budget``,
``force_limit`` is instead treated as a total limit and divided by N.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial import cKDTree


Array = np.ndarray
ProjectionFn = Callable[[Any, Array], Array]


class MultiContactGraspEvaluator:
    """Evaluate geometric and quasistatic quality for any N >= 2 contacts.

    The object is expected to be a Dex-Net ``GraspableObject3D`` or a compatible
    object exposing ``sdf`` and ``mesh.center_of_mass``.  Its SDF must provide
    ``transform_pt_obj_to_grid``, ``on_surface``, ``surface_normal``, and
    ``transform_pt_grid_to_obj``.
    """

    @staticmethod
    def evaluate(
        obj: Any,
        contact_points: Array,
        contact_normals: Optional[Array] = None,
        friction_coef: float = 0.5,
        num_cone_faces: int = 8,
        soft_fingers: bool = False,
        finger_radius: float = 0.005,
        torque_scaling: Optional[float] = None,
        target_wrench: Optional[Array] = None,
        force_limit: Optional[float] = None,
        *,
        fixed_total_force_budget: bool = False,
        project_contacts: bool = False,
        surface_projection_fn: Optional[ProjectionFn] = None,
        normal_tolerance: float = 1e-3,
        normal_alignment_tolerance: float = 0.5,
        duplicate_tolerance: float = 1e-9,
        hull_tolerance: float = 1e-9,
        wrench_tolerance: float = 1e-5,
    ) -> Dict[str, Any]:
        """Evaluate a set of generated contact points.

        Parameters
        ----------
        contact_points, contact_normals:
            ``(N, 3)`` or ``(3, N)`` arrays in the object frame.  Supplied
            normals must be finite and unit length.  They are checked against
            the SDF normal, flipped if inward, and returned as outward normals.
        torque_scaling:
            Multiplier applied to every torque coordinate.  If omitted, it is
            ``1 / bounding_box_diagonal`` in object-frame units.
        target_wrench:
            Optional 6-vector to synthesize from positive primitive wrenches.
            It uses the same scaled torque coordinates as ``grasp_matrix``.
        force_limit:
            Per-contact L1 coefficient limit for target-wrench metrics.  When
            ``fixed_total_force_budget=True``, this is a total force limit and
            each contact receives ``force_limit / N``.
        project_contacts, surface_projection_fn:
            Projection is performed only through this explicit callback, whose
            signature is ``fn(obj, point) -> projected_point``.  No private SDF
            or nearest-neighbour guess is used.
        """
        inferred_num_contacts = _infer_num_contacts(contact_points)
        try:
            points = _as_n_by_3(contact_points, "contact_points")
        except ValueError as exc:
            return _invalid_result(inferred_num_contacts, str(exc))
        n_contacts = points.shape[0]
        if n_contacts < 2:
            return _invalid_result(n_contacts, "at least two contacts are required")

        try:
            mu = _finite_nonnegative_scalar(friction_coef, "friction_coef")
            radius = _finite_nonnegative_scalar(finger_radius, "finger_radius")
            num_faces_value = float(num_cone_faces)
            if (
                not np.isfinite(num_faces_value)
                or not num_faces_value.is_integer()
                or num_faces_value < 3
            ):
                raise ValueError("num_cone_faces must be an integer >= 3")
            num_faces = int(num_faces_value)
            if (
                duplicate_tolerance < 0
                or hull_tolerance < 0
                or normal_tolerance < 0
                or not 0 <= normal_alignment_tolerance <= 1
            ):
                raise ValueError("numeric tolerances must be non-negative")
            supplied_normals = None
            if contact_normals is not None:
                supplied_normals = _as_n_by_3(contact_normals, "contact_normals")
                if supplied_normals.shape != points.shape:
                    raise ValueError(
                        "contact_normals must contain exactly one normal per contact"
                    )
                lengths = np.linalg.norm(supplied_normals, axis=1)
                if not np.allclose(lengths, 1.0, atol=normal_tolerance, rtol=0.0):
                    raise ValueError("contact_normals must be unit length")
            if target_wrench is not None:
                target = np.asarray(target_wrench, dtype=np.float64).reshape(-1)
                if target.shape != (6,) or not np.all(np.isfinite(target)):
                    raise ValueError("target_wrench must be a finite 6-vector")
            else:
                target = None
            if force_limit is not None:
                force_limit_value = _finite_nonnegative_scalar(force_limit, "force_limit")
                if force_limit_value == 0:
                    raise ValueError("force_limit must be positive")
            else:
                force_limit_value = None
        except (TypeError, ValueError) as exc:
            return _invalid_result(n_contacts, str(exc))

        if project_contacts and surface_projection_fn is None:
            return _invalid_result(
                n_contacts,
                "project_contacts=True requires an explicit surface_projection_fn",
            )

        resolved_points = points.copy()
        outward_normals = np.zeros_like(points)
        validity = []
        for i, point in enumerate(points):
            item = {
                "index": i,
                "valid": False,
                "on_surface": False,
                "projected": False,
                "input_normal_orientation": None,
                "failure_reason": None,
            }
            try:
                candidate = point
                on_surface, sdf_value = _surface_status(obj, candidate)
                item["sdf_value"] = float(sdf_value)
                if not on_surface and project_contacts:
                    candidate = np.asarray(
                        surface_projection_fn(obj, candidate.copy()), dtype=np.float64
                    ).reshape(-1)
                    if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
                        raise ValueError("surface projection returned an invalid point")
                    on_surface, sdf_value = _surface_status(obj, candidate)
                    item["projected"] = True
                    item["sdf_value"] = float(sdf_value)
                item["on_surface"] = bool(on_surface)
                if not on_surface:
                    raise ValueError("contact point is not on the SDF surface")

                sdf_outward = _sdf_outward_normal(obj, candidate)
                if supplied_normals is None:
                    outward = sdf_outward
                    item["input_normal_orientation"] = "queried_from_sdf"
                else:
                    given = supplied_normals[i] / np.linalg.norm(supplied_normals[i])
                    alignment = float(np.dot(given, sdf_outward))
                    item["normal_alignment"] = alignment
                    if abs(alignment) < normal_alignment_tolerance:
                        raise ValueError("provided normal is inconsistent with SDF normal")
                    if alignment < 0:
                        outward = -given
                        item["input_normal_orientation"] = "inward_flipped_to_outward"
                    else:
                        outward = given
                        item["input_normal_orientation"] = "outward"
                resolved_points[i] = candidate
                outward_normals[i] = outward / np.linalg.norm(outward)
                item["valid"] = True
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                item["failure_reason"] = str(exc)
            validity.append(item)

        invalid_items = [v for v in validity if not v["valid"]]
        if invalid_items:
            indices = [v["index"] for v in invalid_items]
            return _invalid_result(
                n_contacts,
                "invalid contact(s): %s" % indices,
                contact_validity=validity,
                contact_normals=outward_normals,
            )

        duplicate_pairs = []
        for i in range(n_contacts):
            for j in range(i + 1, n_contacts):
                if np.linalg.norm(resolved_points[i] - resolved_points[j]) <= duplicate_tolerance:
                    duplicate_pairs.append((i, j))
        if duplicate_pairs:
            for i, j in duplicate_pairs:
                validity[i]["valid"] = False
                validity[j]["valid"] = False
                validity[i]["failure_reason"] = "duplicate contact"
                validity[j]["failure_reason"] = "duplicate contact"
            return _invalid_result(
                n_contacts,
                "duplicate contact points: %s" % duplicate_pairs,
                contact_validity=validity,
                contact_normals=outward_normals,
            )

        try:
            center = _center_of_mass(obj)
            characteristic_length = _object_characteristic_length(obj)
        except (AttributeError, TypeError, ValueError) as exc:
            return _invalid_result(
                n_contacts,
                str(exc),
                contact_validity=validity,
                contact_normals=outward_normals,
            )

        return _evaluate_resolved_contacts(
            resolved_points,
            outward_normals,
            validity,
            center,
            characteristic_length,
            mu,
            num_faces,
            soft_fingers,
            radius,
            torque_scaling,
            target,
            force_limit_value,
            fixed_total_force_budget,
            hull_tolerance,
            wrench_tolerance,
            normal_convention="outward",
            center_of_mass_source="mesh.center_of_mass",
        )

    @staticmethod
    def evaluate_robust(
        obj: Any,
        contact_points: Array,
        contact_normals: Optional[Array] = None,
        friction_coef: float = 0.5,
        num_cone_faces: int = 8,
        soft_fingers: bool = False,
        finger_radius: float = 0.005,
        torque_scaling: Optional[float] = None,
        target_wrench: Optional[Array] = None,
        force_limit: Optional[float] = None,
        *,
        num_samples: int = 100,
        point_sigma: Any = 0.0,
        normal_sigma: Any = 0.0,
        friction_sigma: float = 0.0,
        seed: Optional[int] = None,
        fixed_total_force_budget: bool = False,
        project_contacts: bool = False,
        surface_projection_fn: Optional[ProjectionFn] = None,
        **evaluate_kwargs: Any,
    ) -> Dict[str, Any]:
        """Monte Carlo quality under independent point/normal/friction noise.

        Point and normal sigmas may be scalars or 3-vectors.  Every perturbed
        sample goes through :meth:`evaluate`, including surface validation and
        optional explicit projection.  Sample variance uses ``ddof=1`` and is
        reported separately from its square root.
        """
        num_samples_value = float(num_samples)
        if (
            not np.isfinite(num_samples_value)
            or not num_samples_value.is_integer()
            or num_samples_value <= 0
        ):
            raise ValueError("num_samples must be a positive integer")
        base_points = _as_n_by_3(contact_points, "contact_points")
        if contact_normals is None:
            base_normals = None
        else:
            base_normals = _as_n_by_3(contact_normals, "contact_normals")
            if base_normals.shape != base_points.shape:
                raise ValueError("contact_normals shape does not match contact_points")
        point_scale = _sigma_vector(point_sigma, "point_sigma")
        normal_scale = _sigma_vector(normal_sigma, "normal_sigma")
        friction_scale = _finite_nonnegative_scalar(friction_sigma, "friction_sigma")

        rng = np.random.default_rng(seed)
        epsilons = []
        closures = []
        for _ in range(int(num_samples)):
            sampled_points = base_points + rng.normal(size=base_points.shape) * point_scale
            sampled_normals = None
            points_already_projected = False
            if base_normals is not None:
                sampled_normals = base_normals + rng.normal(size=base_normals.shape) * normal_scale
                lengths = np.linalg.norm(sampled_normals, axis=1, keepdims=True)
                if np.any(lengths <= 1e-12):
                    continue
                sampled_normals = sampled_normals / lengths
            sampled_mu = max(float(rng.normal(friction_coef, friction_scale)), 0.0)

            # When normals were not supplied, derive them from each perturbed
            # (and possibly projected) surface point before adding normal noise.
            if base_normals is None and np.any(normal_scale > 0):
                geometry = MultiContactGraspEvaluator.evaluate(
                    obj,
                    sampled_points,
                    None,
                    friction_coef=sampled_mu,
                    num_cone_faces=num_cone_faces,
                    soft_fingers=soft_fingers,
                    finger_radius=finger_radius,
                    torque_scaling=torque_scaling,
                    fixed_total_force_budget=fixed_total_force_budget,
                    project_contacts=project_contacts,
                    surface_projection_fn=surface_projection_fn,
                    **evaluate_kwargs,
                )
                if not geometry["valid"]:
                    continue
                sampled_points = geometry["contact_points"]
                points_already_projected = True
                sampled_normals = geometry["contact_normals"] + (
                    rng.normal(size=base_points.shape) * normal_scale
                )
                lengths = np.linalg.norm(sampled_normals, axis=1, keepdims=True)
                if np.any(lengths <= 1e-12):
                    continue
                sampled_normals = sampled_normals / lengths

            result = MultiContactGraspEvaluator.evaluate(
                obj,
                sampled_points,
                sampled_normals,
                friction_coef=sampled_mu,
                num_cone_faces=num_cone_faces,
                soft_fingers=soft_fingers,
                finger_radius=finger_radius,
                torque_scaling=torque_scaling,
                target_wrench=target_wrench,
                force_limit=force_limit,
                fixed_total_force_budget=fixed_total_force_budget,
                project_contacts=project_contacts and not points_already_projected,
                surface_projection_fn=(
                    surface_projection_fn if not points_already_projected else None
                ),
                **evaluate_kwargs,
            )
            if result["valid"]:
                epsilons.append(result["epsilon"])
                closures.append(float(result["force_closure"]))

        values = np.asarray(epsilons, dtype=np.float64)
        if values.size:
            mean = float(np.mean(values))
            variance = float(np.var(values, ddof=1)) if values.size > 1 else 0.0
            std = float(np.sqrt(variance))
            closure_rate = float(np.mean(closures))
            percentile_10 = float(np.percentile(values, 10))
        else:
            mean = variance = std = closure_rate = percentile_10 = 0.0
        return {
            "mean_epsilon": mean,
            "epsilon_variance": variance,
            "epsilon_std": std,
            "force_closure_rate": closure_rate,
            "epsilon_percentile_10": percentile_10,
            "num_valid_samples": int(values.size),
            "num_samples": int(num_samples),
        }


class PointCloudMultiContactGraspEvaluator:
    """Evaluate contacts selected directly from an oriented object point cloud.

    This backend never snaps arbitrary coordinates to a nearest point.  Contact
    coordinates are always gathered as ``object_points[selected_indices]``.
    ``object_normals`` must contain a precomputed outward unit normal for every
    point.  Their outward orientation cannot be proven from an unoriented point
    cloud, so it remains an explicit caller/data-preprocessing contract.
    """

    @staticmethod
    def evaluate(
        object_points: Array,
        selected_indices: Array,
        object_normals: Array,
        friction_coef: float = 0.5,
        num_cone_faces: int = 8,
        soft_fingers: bool = False,
        finger_radius: float = 0.005,
        torque_scaling: Optional[float] = None,
        center_of_mass: Optional[Array] = None,
        target_wrench: Optional[Array] = None,
        force_limit: Optional[float] = None,
        *,
        fixed_total_force_budget: bool = False,
        normal_tolerance: float = 1e-3,
        duplicate_tolerance: float = 1e-9,
        hull_tolerance: float = 1e-9,
        wrench_tolerance: float = 1e-5,
    ) -> Dict[str, Any]:
        """Evaluate indexed point-cloud contacts in the point-cloud frame.

        ``center_of_mass`` should be supplied when known.  If omitted, the
        point-cloud mean is used as a geometric-centre approximation and the
        result records ``center_of_mass_source='point_cloud_mean'``.  This can
        be biased for partial/single-view clouds.  Default torque scaling is
        the reciprocal point-cloud bounding-box diagonal.
        """
        try:
            cloud = _as_n_by_3(object_points, "object_points")
            normals = _as_n_by_3(object_normals, "object_normals")
            if normals.shape != cloud.shape:
                raise ValueError("object_normals must have the same shape as object_points")
            raw_indices = np.asarray(selected_indices)
            indices = raw_indices.reshape(-1)
            if raw_indices.ndim != 1 or indices.size < 2:
                raise ValueError("selected_indices must be a 1-D array with N >= 2")
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("selected_indices must contain integers")
            indices = indices.astype(np.int64, copy=False)
            if np.any(indices < 0) or np.any(indices >= len(cloud)):
                raise ValueError("selected_indices contains an out-of-range index")
            if len(np.unique(indices)) != len(indices):
                raise ValueError("selected_indices contains duplicate indices")
            lengths = np.linalg.norm(normals, axis=1)
            if not np.allclose(lengths, 1.0, atol=normal_tolerance, rtol=0.0):
                raise ValueError("object_normals must be outward unit vectors")
            mu = _finite_nonnegative_scalar(friction_coef, "friction_coef")
            radius = _finite_nonnegative_scalar(finger_radius, "finger_radius")
            num_faces_value = float(num_cone_faces)
            if (
                not np.isfinite(num_faces_value)
                or not num_faces_value.is_integer()
                or num_faces_value < 3
            ):
                raise ValueError("num_cone_faces must be an integer >= 3")
            num_faces = int(num_faces_value)
            points = cloud[indices].copy()
            contact_normals = normals[indices].copy()
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    if np.linalg.norm(points[i] - points[j]) <= duplicate_tolerance:
                        raise ValueError(f"duplicate contact points: ({i}, {j})")
            if center_of_mass is None:
                center = np.mean(cloud, axis=0)
                center_source = "point_cloud_mean"
            else:
                center = np.asarray(center_of_mass, dtype=np.float64).reshape(-1)
                if center.shape != (3,) or not np.all(np.isfinite(center)):
                    raise ValueError("center_of_mass must be a finite 3-vector")
                center_source = "provided"
            characteristic_length = float(
                np.linalg.norm(np.max(cloud, axis=0) - np.min(cloud, axis=0))
            )
            if not np.isfinite(characteristic_length) or characteristic_length <= 1e-12:
                raise ValueError("point-cloud bounding-box diagonal must be positive")
            if target_wrench is None:
                target = None
            else:
                target = np.asarray(target_wrench, dtype=np.float64).reshape(-1)
                if target.shape != (6,) or not np.all(np.isfinite(target)):
                    raise ValueError("target_wrench must be a finite 6-vector")
            if force_limit is None:
                force_limit_value = None
            else:
                force_limit_value = _finite_nonnegative_scalar(force_limit, "force_limit")
                if force_limit_value == 0:
                    raise ValueError("force_limit must be positive")
        except (TypeError, ValueError) as exc:
            return _invalid_result(_infer_index_count(selected_indices), str(exc))

        validity = [
            {
                "index": i,
                "point_cloud_index": int(point_index),
                "valid": True,
                "on_surface": True,
                "surface_evidence": "selected_index",
                "projected": False,
                "input_normal_orientation": "outward_provided_unverified",
                "failure_reason": None,
            }
            for i, point_index in enumerate(indices)
        ]
        result = _evaluate_resolved_contacts(
            points,
            contact_normals,
            validity,
            center,
            characteristic_length,
            mu,
            num_faces,
            soft_fingers,
            radius,
            torque_scaling,
            target,
            force_limit_value,
            fixed_total_force_budget,
            hull_tolerance,
            wrench_tolerance,
            normal_convention="outward_provided_unverified",
            center_of_mass_source=center_source,
        )
        if result["valid"]:
            result["selected_indices"] = indices.copy()
            result["surface_validation"] = "exact_point_cloud_indices"
        return result

    @staticmethod
    def evaluate_robust(
        object_points: Array,
        selected_indices: Array,
        object_normals: Array,
        *,
        num_samples: int = 100,
        normal_sigma: Any = 0.0,
        friction_coef: float = 0.5,
        friction_sigma: float = 0.0,
        seed: Optional[int] = None,
        **evaluate_kwargs: Any,
    ) -> Dict[str, Any]:
        """Robust indexed evaluation with normal and friction uncertainty.

        Contact positions remain exactly indexed point-cloud points.  Positional
        noise is intentionally absent because a point cloud supplies no reliable
        continuous surface projection.
        """
        count_value = float(num_samples)
        if not np.isfinite(count_value) or not count_value.is_integer() or count_value <= 0:
            raise ValueError("num_samples must be a positive integer")
        cloud = _as_n_by_3(object_points, "object_points")
        normals = _as_n_by_3(object_normals, "object_normals")
        if normals.shape != cloud.shape:
            raise ValueError("object_normals must have the same shape as object_points")
        indices = np.asarray(selected_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("selected_indices must be a 1-D integer array")
        indices = indices.astype(np.int64, copy=False)
        normal_scale = _sigma_vector(normal_sigma, "normal_sigma")
        friction_scale = _finite_nonnegative_scalar(friction_sigma, "friction_sigma")
        rng = np.random.default_rng(seed)
        epsilons = []
        closures = []
        for _ in range(int(num_samples)):
            sampled_normals = normals.copy()
            perturbed = sampled_normals[indices] + (
                rng.normal(size=(len(indices), 3)) * normal_scale
            )
            lengths = np.linalg.norm(perturbed, axis=1, keepdims=True)
            if np.any(lengths <= 1e-12):
                continue
            sampled_normals[indices] = perturbed / lengths
            sampled_mu = max(float(rng.normal(friction_coef, friction_scale)), 0.0)
            result = PointCloudMultiContactGraspEvaluator.evaluate(
                cloud,
                indices,
                sampled_normals,
                friction_coef=sampled_mu,
                **evaluate_kwargs,
            )
            if result["valid"]:
                epsilons.append(result["epsilon"])
                closures.append(float(result["force_closure"]))
        return _robust_summary(epsilons, closures, int(num_samples))


def estimate_point_cloud_normals(
    object_points: Array,
    k_neighbors: int = 30,
    viewpoint: Optional[Array] = None,
) -> Array:
    """Estimate oriented normals once for later indexed evaluation.

    Local PCA supplies the unoriented normal.  With ``viewpoint``, normals are
    oriented toward that viewpoint (appropriate for a visible single-view
    surface).  Otherwise they are oriented away from the point-cloud mean,
    which assumes a reasonably complete closed-surface cloud and is unreliable
    for strongly concave or partial clouds.  Persist the returned ``(P, 3)``
    array alongside ``object_pc`` rather than recomputing it per training step.
    """
    points = _as_n_by_3(object_points, "object_points")
    if int(k_neighbors) != k_neighbors or not 3 <= int(k_neighbors) <= len(points):
        raise ValueError("k_neighbors must be an integer in [3, num_points]")
    if viewpoint is None:
        orientation_origins = np.broadcast_to(np.mean(points, axis=0), points.shape)
        orient_toward = False
    else:
        view = np.asarray(viewpoint, dtype=np.float64).reshape(-1)
        if view.shape != (3,) or not np.all(np.isfinite(view)):
            raise ValueError("viewpoint must be a finite 3-vector")
        orientation_origins = np.broadcast_to(view, points.shape)
        orient_toward = True
    _, neighborhoods = cKDTree(points).query(points, k=int(k_neighbors))
    normals = np.empty_like(points)
    for i, neighbors in enumerate(neighborhoods):
        local = points[neighbors]
        centered = local - np.mean(local, axis=0)
        covariance = centered.T.dot(centered) / float(len(local))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[1] <= 1e-15:
            raise ValueError(f"degenerate normal-estimation neighborhood at index {i}")
        normal = eigenvectors[:, 0]
        reference = orientation_origins[i] - points[i]
        desired_positive = np.dot(normal, reference) > 0
        if desired_positive != orient_toward:
            normal = -normal
        normals[i] = normal / np.linalg.norm(normal)
    return normals


def match_predicted_contacts_to_point_cloud(
    predicted_points: Array,
    object_points: Array,
    *,
    max_projection_distance: Optional[float] = None,
    max_distance_factor: float = 2.0,
) -> Dict[str, Any]:
    """Uniquely match continuous predictions to sampled surface points.

    Hungarian assignment prevents two contacts from silently collapsing onto
    one point.  If ``max_projection_distance`` is omitted, the gate is
    ``max_distance_factor`` times the median nearest-neighbour spacing of the
    object cloud.  The caller should report the returned distances because the
    downstream quality describes the projected, not raw, contact set.
    """
    try:
        predicted = _as_n_by_3(predicted_points, "predicted_points")
        cloud = _as_n_by_3(object_points, "object_points")
        if len(predicted) < 2:
            raise ValueError("at least two predicted contacts are required")
        if len(predicted) > len(cloud):
            raise ValueError("more predicted contacts than object points")
        if max_projection_distance is None:
            if not np.isfinite(max_distance_factor) or max_distance_factor <= 0:
                raise ValueError("max_distance_factor must be finite and positive")
            neighbor_distances, _ = cKDTree(cloud).query(cloud, k=2)
            spacing = float(np.median(neighbor_distances[:, 1]))
            if not np.isfinite(spacing) or spacing <= 1e-12:
                raise ValueError("cannot infer spacing from a degenerate point cloud")
            threshold = float(max_distance_factor * spacing)
        else:
            threshold = float(max_projection_distance)
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("max_projection_distance must be finite and positive")
        pairwise = np.linalg.norm(
            predicted[:, None, :] - cloud[None, :, :], axis=2
        )
        contact_rows, cloud_columns = linear_sum_assignment(pairwise)
        indices = np.empty(len(predicted), dtype=np.int64)
        distances = np.empty(len(predicted), dtype=np.float64)
        indices[contact_rows] = cloud_columns
        distances[contact_rows] = pairwise[contact_rows, cloud_columns]
        valid = bool(np.all(distances <= threshold))
        return {
            "valid": valid,
            "predicted_indices": indices,
            "projected_contact_points": cloud[indices],
            "projection_distances": distances,
            "max_projection_distance": threshold,
            "point_cloud_spacing": (
                spacing if max_projection_distance is None else None
            ),
            "failure_reason": None if valid else "prediction exceeds projection distance gate",
        }
    except (TypeError, ValueError) as exc:
        return {
            "valid": False,
            "predicted_indices": np.empty(0, dtype=np.int64),
            "projected_contact_points": np.empty((0, 3)),
            "projection_distances": np.empty(0),
            "max_projection_distance": max_projection_distance,
            "point_cloud_spacing": None,
            "failure_reason": str(exc),
        }


def _as_n_by_3(values: Array, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array")
    if array.shape[1] == 3:
        result = array.copy()
    elif array.shape[0] == 3:
        result = array.T.copy()
    else:
        raise ValueError(f"{name} must have shape (N, 3) or (3, N)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _infer_num_contacts(values: Array) -> int:
    try:
        array = np.asarray(values)
    except Exception:
        return 0
    if array.ndim != 2:
        return 0
    if array.shape[1] == 3:
        return int(array.shape[0])
    if array.shape[0] == 3:
        return int(array.shape[1])
    return 0


def _infer_index_count(values: Array) -> int:
    try:
        array = np.asarray(values)
    except Exception:
        return 0
    return int(array.size) if array.ndim == 1 else 0


def _finite_nonnegative_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _surface_status(obj: Any, point: Array) -> Tuple[bool, float]:
    sdf = obj.sdf
    grid = sdf.transform_pt_obj_to_grid(point)
    if hasattr(sdf, "is_out_of_bounds") and sdf.is_out_of_bounds(grid):
        return False, float("inf")
    status = sdf.on_surface(grid)
    if isinstance(status, tuple):
        on_surface, value = status[:2]
    else:
        on_surface, value = bool(status), sdf[grid]
    return bool(on_surface), float(np.asarray(value).reshape(-1)[0])


def _sdf_outward_normal(obj: Any, point: Array) -> Array:
    sdf = obj.sdf
    grid = sdf.transform_pt_obj_to_grid(point)
    normal_grid = np.asarray(sdf.surface_normal(grid), dtype=np.float64).reshape(-1)
    if normal_grid.shape != (3,) or not np.all(np.isfinite(normal_grid)):
        raise ValueError("SDF returned an invalid surface normal")
    normal = np.asarray(
        sdf.transform_pt_grid_to_obj(normal_grid, direction=True), dtype=np.float64
    ).reshape(-1)
    length = np.linalg.norm(normal)
    if normal.shape != (3,) or not np.isfinite(length) or length <= 1e-12:
        raise ValueError("SDF returned a zero or invalid surface normal")
    return normal / length


def _center_of_mass(obj: Any) -> Array:
    center = np.asarray(obj.mesh.center_of_mass, dtype=np.float64).reshape(-1)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("obj.mesh.center_of_mass must be a finite 3-vector")
    return center


def _object_characteristic_length(obj: Any) -> float:
    minimum, maximum = obj.mesh.bounding_box()
    minimum = np.asarray(minimum, dtype=np.float64).reshape(-1)
    maximum = np.asarray(maximum, dtype=np.float64).reshape(-1)
    diagonal = float(np.linalg.norm(maximum - minimum))
    if not np.isfinite(diagonal) or diagonal <= 1e-12:
        raise ValueError("object bounding-box diagonal must be positive")
    return diagonal


def _resolve_torque_scaling(characteristic_length: float, requested: Optional[float]) -> float:
    if requested is None:
        return 1.0 / characteristic_length
    scaling = float(requested)
    if not np.isfinite(scaling) or scaling <= 0:
        raise ValueError("torque_scaling must be finite and positive")
    return scaling


def _evaluate_resolved_contacts(
    points: Array,
    outward_normals: Array,
    validity,
    center: Array,
    characteristic_length: float,
    friction_coef: float,
    num_faces: int,
    soft_fingers: bool,
    finger_radius: float,
    requested_torque_scaling: Optional[float],
    target: Optional[Array],
    force_limit_value: Optional[float],
    fixed_total_force_budget: bool,
    hull_tolerance: float,
    wrench_tolerance: float,
    *,
    normal_convention: str,
    center_of_mass_source: str,
) -> Dict[str, Any]:
    """Shared validated-contact to wrench-space evaluation path."""
    n_contacts = points.shape[0]
    try:
        scaling = _resolve_torque_scaling(
            characteristic_length, requested_torque_scaling
        )
        forces, raw_torques, contact_ids = _contact_primitives(
            points,
            outward_normals,
            center,
            friction_coef,
            num_faces,
            soft_fingers,
            finger_radius,
        )
        scaled_torques = scaling * raw_torques
        grasp_matrix = np.vstack((forces, scaled_torques))
        epsilon, force_closure, hull_info = _ferrari_canny(
            grasp_matrix, hull_tolerance
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return _invalid_result(
            n_contacts,
            str(exc),
            contact_validity=validity,
            contact_normals=outward_normals,
        )

    per_contact_limit = None
    total_limit = None
    partial_closure = None
    wrench_resistance = None
    target_residual = None
    required_force_l1 = None
    required_force_l2 = None
    if target is not None:
        if force_limit_value is None:
            total_limit = 1.0
            per_contact_limit = 1.0
            group_limits = None
        elif fixed_total_force_budget:
            total_limit = force_limit_value
            per_contact_limit = force_limit_value / n_contacts
            group_limits = np.full(n_contacts, per_contact_limit)
        else:
            per_contact_limit = force_limit_value
            total_limit = force_limit_value * n_contacts
            group_limits = np.full(n_contacts, per_contact_limit)
        solution, target_residual = _solve_target_wrench(
            grasp_matrix, target, contact_ids, group_limits
        )
        partial_closure = bool(target_residual <= wrench_tolerance)
        if partial_closure:
            required_force_l1 = float(np.sum(solution))
            required_force_l2 = float(np.linalg.norm(solution))
            resistance_limit = (
                total_limit
                if fixed_total_force_budget or force_limit_value is None
                else force_limit_value
            )
            wrench_resistance = float(
                1.0 / (required_force_l2 + 1e-12)
                - 1.0 / (2.0 * resistance_limit)
            )
        else:
            wrench_resistance = 0.0

    return {
        "valid": True,
        "num_contacts": n_contacts,
        "force_closure": bool(force_closure),
        "epsilon": float(epsilon),
        "partial_closure": partial_closure,
        "wrench_resistance": wrench_resistance,
        "contact_validity": validity,
        "contact_points": points,
        "contact_normals": outward_normals,
        "normal_convention": normal_convention,
        "force_normal_convention": "inward",
        "center_of_mass": center,
        "center_of_mass_source": center_of_mass_source,
        "forces": forces,
        "torques": scaled_torques,
        "raw_torques": raw_torques,
        "grasp_matrix": grasp_matrix,
        "primitive_contact_indices": contact_ids,
        "torque_scaling": float(scaling),
        "object_characteristic_length": float(characteristic_length),
        "force_budget": {
            "epsilon_budget": "global_L1_unit_budget",
            "fixed_total_force_budget": bool(fixed_total_force_budget),
            "per_contact_force_limit": per_contact_limit,
            "total_force_limit": total_limit,
        },
        "target_wrench_residual": target_residual,
        "required_contact_force_l1": required_force_l1,
        "required_contact_force_l2": required_force_l2,
        "wrench_rank": hull_info["rank"],
        "hull_min_signed_distance": hull_info["min_signed_distance"],
        "failure_reason": None,
    }


def _tangent_basis(inward: Array) -> Tuple[Array, Array]:
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(inward)))] = 1.0
    tangent_1 = np.cross(inward, axis)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(inward, tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)
    return tangent_1, tangent_2


def _contact_primitives(
    points: Array,
    outward_normals: Array,
    center: Array,
    friction_coef: float,
    num_faces: int,
    soft_fingers: bool,
    finger_radius: float,
) -> Tuple[Array, Array, Array]:
    force_columns = []
    torque_columns = []
    contact_ids = []
    for i, (point, outward) in enumerate(zip(points, outward_normals)):
        inward = -outward
        tangent_1, tangent_2 = _tangent_basis(inward)
        for j in range(num_faces):
            theta = 2.0 * np.pi * j / float(num_faces)
            force = inward + friction_coef * (
                np.cos(theta) * tangent_1 + np.sin(theta) * tangent_2
            )
            force_columns.append(force)
            torque_columns.append(np.cross(point - center, force))
            contact_ids.append(i)
        if soft_fingers:
            torsional_magnitude = np.pi * finger_radius**2 * friction_coef
            for sign in (-1.0, 1.0):
                force_columns.append(np.zeros(3))
                torque_columns.append(sign * torsional_magnitude * inward)
                contact_ids.append(i)
    return (
        np.asarray(force_columns, dtype=np.float64).T,
        np.asarray(torque_columns, dtype=np.float64).T,
        np.asarray(contact_ids, dtype=np.int64),
    )


def _ferrari_canny(grasp_matrix: Array, tolerance: float):
    rank = int(np.linalg.matrix_rank(grasp_matrix, tol=max(tolerance, 1e-12)))
    info = {"rank": rank, "min_signed_distance": 0.0}
    if rank < 6 or grasp_matrix.shape[1] < 7 or not np.all(np.isfinite(grasp_matrix)):
        return 0.0, False, info
    try:
        hull = ConvexHull(grasp_matrix.T)
    except (QhullError, ValueError):
        return 0.0, False, info
    equations = np.asarray(hull.equations, dtype=np.float64)
    normals = equations[:, :-1]
    offsets = equations[:, -1]
    lengths = np.linalg.norm(normals, axis=1)
    if (
        equations.ndim != 2
        or equations.shape[1] != 7
        or not np.all(np.isfinite(equations))
        or np.any(lengths <= 0)
    ):
        return 0.0, False, info
    signed_distances = -offsets / lengths
    minimum = float(np.min(signed_distances))
    info["min_signed_distance"] = minimum
    strictly_inside = bool(np.all(signed_distances > tolerance))
    if not strictly_inside:
        return 0.0, False, info
    return minimum, True, info


def _solve_target_wrench(
    grasp_matrix: Array,
    target: Array,
    contact_ids: Array,
    group_limits: Optional[Array],
) -> Tuple[Array, float]:
    num_variables = grasp_matrix.shape[1]

    def objective(weights):
        residual = grasp_matrix.dot(weights) - target
        return 0.5 * residual.dot(residual) + 5e-13 * weights.dot(weights)

    def jacobian(weights):
        return grasp_matrix.T.dot(grasp_matrix.dot(weights) - target) + 1e-12 * weights

    constraints = []
    if group_limits is None:
        constraints.append(
            {"type": "ineq", "fun": lambda w: 1.0 - np.sum(w), "jac": lambda w: -np.ones_like(w)}
        )
    else:
        for contact_index, limit in enumerate(group_limits):
            mask = (contact_ids == contact_index).astype(np.float64)
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, m=mask, lim=limit: lim - m.dot(w),
                    "jac": lambda w, m=mask: -m,
                }
            )
    solution = minimize(
        objective,
        np.zeros(num_variables),
        jac=jacobian,
        bounds=[(0.0, None)] * num_variables,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    weights = np.maximum(np.asarray(solution.x, dtype=np.float64), 0.0)
    residual = float(np.linalg.norm(grasp_matrix.dot(weights) - target))
    return weights, residual


def _sigma_vector(value: Any, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(3, float(array))
    else:
        array = array.reshape(-1)
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be a finite non-negative scalar or 3-vector")
    return array


def _robust_summary(epsilons, closures, num_samples: int) -> Dict[str, Any]:
    values = np.asarray(epsilons, dtype=np.float64)
    if values.size:
        mean = float(np.mean(values))
        variance = float(np.var(values, ddof=1)) if values.size > 1 else 0.0
        std = float(np.sqrt(variance))
        closure_rate = float(np.mean(closures))
        percentile_10 = float(np.percentile(values, 10))
    else:
        mean = variance = std = closure_rate = percentile_10 = 0.0
    return {
        "mean_epsilon": mean,
        "epsilon_variance": variance,
        "epsilon_std": std,
        "force_closure_rate": closure_rate,
        "epsilon_percentile_10": percentile_10,
        "num_valid_samples": int(values.size),
        "num_samples": int(num_samples),
    }


def _invalid_result(
    num_contacts: int,
    reason: str,
    *,
    contact_validity=None,
    contact_normals=None,
) -> Dict[str, Any]:
    return {
        "valid": False,
        "num_contacts": int(num_contacts),
        "force_closure": False,
        "epsilon": 0.0,
        "partial_closure": None,
        "wrench_resistance": None,
        "contact_validity": [] if contact_validity is None else contact_validity,
        "contact_normals": np.empty((0, 3)) if contact_normals is None else contact_normals,
        "forces": np.empty((3, 0)),
        "torques": np.empty((3, 0)),
        "grasp_matrix": np.empty((6, 0)),
        "failure_reason": str(reason),
    }


__all__ = [
    "MultiContactGraspEvaluator",
    "PointCloudMultiContactGraspEvaluator",
    "estimate_point_cloud_normals",
    "match_predicted_contacts_to_point_cloud",
]
