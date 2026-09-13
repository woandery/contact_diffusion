"""Deterministic controls for contact-set guidance experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MATCHER_VERSION = "surface-random-shape-matched-v1"


@dataclass(frozen=True)
class ContactSetSignature:
    """Rotation-free geometric statistics used to match a random control."""

    pairwise_distances: np.ndarray
    radial_distances: np.ndarray
    centroid_radius: float
    scale: float


def _as_points(value: np.ndarray, *, name: str, minimum: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < minimum:
        raise ValueError(f"{name} must have shape [N, 3] with N >= {minimum}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    return points


def contact_set_signature(
    contacts: np.ndarray, object_points: np.ndarray
) -> ContactSetSignature:
    """Return orientation-free spread statistics in object-relative units.

    The signature deliberately matches spread and radial extent without
    matching the direction of the contact centroid.  A random control is thus
    not handed the semantic surface region selected by ContactDiffusion.
    """

    object_points = _as_points(object_points, name="object_points", minimum=4)
    contacts = _as_points(contacts, name="contacts", minimum=2)
    object_center = object_points.mean(axis=0)
    scale = float(np.linalg.norm(object_points.max(axis=0) - object_points.min(axis=0)))
    scale = max(scale, 1.0e-9)
    first, second = np.triu_indices(contacts.shape[0], k=1)
    pairwise = np.linalg.norm(contacts[first] - contacts[second], axis=1) / scale
    radial = np.linalg.norm(contacts - object_center, axis=1) / scale
    centroid_radius = float(np.linalg.norm(contacts.mean(axis=0) - object_center) / scale)
    return ContactSetSignature(
        pairwise_distances=np.sort(pairwise),
        radial_distances=np.sort(radial),
        centroid_radius=centroid_radius,
        scale=scale,
    )


def _sample_unique_rows(
    rng: np.random.Generator, row_count: int, width: int, upper: int
) -> np.ndarray:
    indices = rng.integers(0, upper, size=(row_count, width), dtype=np.int64)
    duplicate = np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0, axis=1)
    while bool(duplicate.any()):
        count = int(duplicate.sum())
        indices[duplicate] = rng.integers(
            0, upper, size=(count, width), dtype=np.int64
        )
        duplicate = np.any(
            np.diff(np.sort(indices, axis=1), axis=1) == 0, axis=1
        )
    return indices


def matched_random_surface_contacts(
    object_points: np.ndarray,
    reference_contacts: np.ndarray,
    *,
    seed: int,
    candidate_count: int = 4096,
) -> dict:
    """Select a deterministic random surface set matched to reference shape.

    Candidate sets contain unique rows from the full XYZ object cloud.  The
    search matches sorted pairwise distances, sorted object-centric radii, and
    centroid-radius magnitude.  It never matches centroid direction and never
    reads FK or simulator outcomes.
    """

    object_points = _as_points(object_points, name="object_points", minimum=4)
    reference_contacts = _as_points(
        reference_contacts, name="reference_contacts", minimum=2
    )
    contact_count = int(reference_contacts.shape[0])
    if object_points.shape[0] < contact_count:
        raise ValueError("object_points has fewer rows than the contact set")
    if int(candidate_count) <= 0:
        raise ValueError("candidate_count must be positive")

    reference = contact_set_signature(reference_contacts, object_points)
    rng = np.random.default_rng(int(seed))
    indices = _sample_unique_rows(
        rng,
        int(candidate_count),
        contact_count,
        int(object_points.shape[0]),
    )
    candidates = object_points[indices]
    object_center = object_points.mean(axis=0)
    scale = reference.scale

    first, second = np.triu_indices(contact_count, k=1)
    pairwise = np.linalg.norm(
        candidates[:, first, :] - candidates[:, second, :], axis=2
    ) / scale
    pairwise.sort(axis=1)
    radial = np.linalg.norm(candidates - object_center[None, None, :], axis=2) / scale
    radial.sort(axis=1)
    centroid_radius = np.linalg.norm(
        candidates.mean(axis=1) - object_center[None, :], axis=1
    ) / scale

    pairwise_mse = np.mean(
        (pairwise - reference.pairwise_distances[None, :]) ** 2, axis=1
    )
    radial_mse = np.mean(
        (radial - reference.radial_distances[None, :]) ** 2, axis=1
    )
    centroid_error_sq = (centroid_radius - reference.centroid_radius) ** 2
    # Equal weights operate on dimensionless, object-normalized statistics.
    score = pairwise_mse + radial_mse + centroid_error_sq
    # A projected diffusion target is itself usually a subset of object_points.
    # Exclude an accidental reproduction of that same unordered surface set;
    # otherwise a rare B sample would cease to be a meaningful control.
    reference_distance = np.linalg.norm(
        candidates[:, :, None, :] - reference_contacts[None, None, :, :], axis=3
    )
    equality_tolerance = max(scale * 1.0e-7, 1.0e-9)
    same_unordered_set = np.logical_and(
        reference_distance.min(axis=2).max(axis=1) <= equality_tolerance,
        reference_distance.min(axis=1).max(axis=1) <= equality_tolerance,
    )
    score[same_unordered_set] = np.inf
    if not bool(np.isfinite(score).any()):
        raise ValueError(
            "all matched-random candidates reproduce the reference contact set"
        )
    best = int(np.argmin(score))
    chosen = candidates[best].astype(np.float32, copy=False)
    chosen_signature = contact_set_signature(chosen, object_points)

    return {
        "contacts": chosen,
        "indices": indices[best].tolist(),
        "diagnostics": {
            "matcher_version": MATCHER_VERSION,
            "seed": int(seed),
            "candidate_count": int(candidate_count),
            "object_point_count": int(object_points.shape[0]),
            "contact_count": contact_count,
            "excluded_reference_set_candidates": int(same_unordered_set.sum()),
            "object_scale_m": float(scale),
            "score": float(score[best]),
            "pairwise_mse": float(pairwise_mse[best]),
            "radial_mse": float(radial_mse[best]),
            "centroid_radius_error_sq": float(centroid_error_sq[best]),
            "reference_pairwise_distances_normalized": (
                reference.pairwise_distances.tolist()
            ),
            "selected_pairwise_distances_normalized": (
                chosen_signature.pairwise_distances.tolist()
            ),
            "reference_radial_distances_normalized": (
                reference.radial_distances.tolist()
            ),
            "selected_radial_distances_normalized": (
                chosen_signature.radial_distances.tolist()
            ),
            "reference_centroid_radius_normalized": float(
                reference.centroid_radius
            ),
            "selected_centroid_radius_normalized": float(
                chosen_signature.centroid_radius
            ),
            "centroid_direction_matched": False,
            "success_labels_used": False,
        },
    }
