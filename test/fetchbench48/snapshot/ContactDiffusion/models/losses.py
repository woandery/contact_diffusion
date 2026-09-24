"""Losses for unordered contact diffusion."""

from __future__ import annotations

import itertools
from typing import Optional

import torch
import torch.nn.functional as F


def noise_mse_loss(eps_pred: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(eps_pred, eps)


def chamfer_loss_contacts(pred_contacts: torch.Tensor, target_contacts: torch.Tensor) -> torch.Tensor:
    return chamfer_loss_contacts_per_sample(pred_contacts, target_contacts).mean()


def chamfer_loss_contacts_per_sample(
    pred_contacts: torch.Tensor, target_contacts: torch.Tensor
) -> torch.Tensor:
    pred_xyz = pred_contacts[..., :3]
    target_xyz = target_contacts[..., :3]
    dist = torch.cdist(pred_xyz, target_xyz, p=2).pow(2)
    pred_to_target = dist.min(dim=2)[0].mean(dim=1)
    target_to_pred = dist.min(dim=1)[0].mean(dim=1)
    return pred_to_target + target_to_pred


def _metric_scale_mm(
    reference: torch.Tensor,
    normalization_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return one metric-space millimetre multiplier per batch element.

    Coordinates are assumed to already be in metres when no normalization
    scale is supplied.  Unit-radius datasets pass their per-sample radius in
    metres so that reported metrics are comparable across object sizes.
    """

    batch_size = int(reference.shape[0])
    if normalization_scale is None:
        scale_m = reference.new_ones(batch_size)
    else:
        scale_m = torch.as_tensor(
            normalization_scale,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(-1)
        if scale_m.numel() == 1 and batch_size != 1:
            scale_m = scale_m.expand(batch_size)
        if scale_m.numel() != batch_size:
            raise ValueError(
                "normalization_scale must contain one value per batch element; "
                f"got {scale_m.numel()} for batch size {batch_size}"
            )
    return scale_m * 1000.0


def symmetric_chamfer_mm_per_sample(
    pred_contacts: torch.Tensor,
    target_contacts: torch.Tensor,
    normalization_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unsquared symmetric Chamfer distance in millimetres per sample."""

    pred_xyz = pred_contacts[..., :3]
    target_xyz = target_contacts[..., :3]
    distances = torch.cdist(pred_xyz, target_xyz, p=2)
    pred_to_target = distances.min(dim=2).values.mean(dim=1)
    target_to_pred = distances.min(dim=1).values.mean(dim=1)
    scale_mm = _metric_scale_mm(pred_xyz, normalization_scale)
    return 0.5 * (pred_to_target + target_to_pred) * scale_mm


def point_to_plane_mm_per_sample(
    pred_contacts: torch.Tensor,
    object_pc: torch.Tensor,
    object_normals: torch.Tensor,
    normalization_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Nearest-surface point-to-plane distance in millimetres per sample.

    Nearest neighbours are selected in XYZ space.  The residual is then
    projected onto the corresponding unit surface normal, so tangential error
    is intentionally not penalized by this metric.
    """

    pred_xyz = pred_contacts[..., :3]
    object_xyz = object_pc[..., :3]
    if object_normals.shape != object_xyz.shape:
        raise ValueError(
            "object_normals must align with object_pc XYZ; "
            f"got {object_normals.shape} and {object_xyz.shape}"
        )
    nearest_index = torch.cdist(pred_xyz, object_xyz, p=2).argmin(dim=2)
    gather_index = nearest_index.unsqueeze(-1).expand(-1, -1, 3)
    nearest_points = torch.gather(object_xyz, dim=1, index=gather_index)
    unit_normals = F.normalize(object_normals, dim=2)
    nearest_normals = torch.gather(unit_normals, dim=1, index=gather_index)
    plane_distance = ((pred_xyz - nearest_points) * nearest_normals).sum(dim=2).abs()
    scale_mm = _metric_scale_mm(pred_xyz, normalization_scale)
    return plane_distance.mean(dim=1) * scale_mm


def surface_loss_contacts(
    pred_contacts: torch.Tensor, object_pc: torch.Tensor, squared: bool = True
) -> torch.Tensor:
    pred_xyz = pred_contacts[..., :3]
    obj_xyz = object_pc[..., :3]
    dist = torch.cdist(pred_xyz, obj_xyz, p=2)
    min_dist = dist.min(dim=2)[0]
    if squared:
        min_dist = min_dist.pow(2)
    return min_dist.mean()


def pairwise_contact_distances(contacts: torch.Tensor) -> torch.Tensor:
    n = contacts.shape[1]
    if n < 2:
        return contacts.new_zeros((contacts.shape[0], 0))
    dist = torch.cdist(contacts[..., :3], contacts[..., :3], p=2)
    pair_mask = torch.triu(torch.ones(n, n, device=contacts.device, dtype=torch.bool), diagonal=1)
    return dist[:, pair_mask]


def permutation_aligned_relative_geometry_mm_per_sample(
    pred_contacts: torch.Tensor,
    target_contacts: torch.Tensor,
    normalization_scale: Optional[torch.Tensor] = None,
    huber_delta_mm: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare within-set geometry after exact one-to-one point alignment.

    For the small contact sets used here (N=3 or N=5), all permutations are
    enumerated and the target ordering with minimum mean point distance is
    selected. The differentiable loss then compares every pairwise edge length
    in physical millimetres. This penalizes hybrid sets whose individual points
    come from incompatible grasp modes while remaining permutation invariant.

    Returns the per-sample Smooth-L1 loss and the per-sample mean absolute edge
    error in millimetres.
    """

    pred_xyz = pred_contacts[..., :3]
    target_xyz = target_contacts[..., :3]
    if pred_xyz.shape != target_xyz.shape:
        raise ValueError(
            "pred_contacts and target_contacts must have identical [B, N, C] shape; "
            f"got {pred_contacts.shape} and {target_contacts.shape}"
        )
    batch_size, num_contacts, _ = pred_xyz.shape
    if num_contacts < 2:
        zero = pred_xyz.sum(dim=(1, 2)) * 0.0
        return zero, zero.detach()
    if num_contacts > 8:
        raise ValueError(
            "Exact permutation alignment is intended for small contact sets (N<=8); "
            f"got N={num_contacts}"
        )

    permutations = torch.tensor(
        list(itertools.permutations(range(num_contacts))),
        device=target_xyz.device,
        dtype=torch.long,
    )
    # [B, P, N, 3].  The discrete alignment is intentionally selected without
    # gradients; gradients flow through the predicted pairwise distances below.
    permuted_target = target_xyz[:, permutations]
    assignment_cost = torch.linalg.vector_norm(
        pred_xyz[:, None] - permuted_target, dim=-1
    ).mean(dim=-1)
    best_permutation = assignment_cost.detach().argmin(dim=1)
    aligned_target = permuted_target[
        torch.arange(batch_size, device=target_xyz.device), best_permutation
    ]

    pred_edges = pairwise_contact_distances(pred_xyz)
    target_edges = pairwise_contact_distances(aligned_target)
    scale_mm = _metric_scale_mm(pred_xyz, normalization_scale).unsqueeze(1)
    edge_error_mm = (pred_edges - target_edges) * scale_mm
    loss = F.smooth_l1_loss(
        edge_error_mm,
        torch.zeros_like(edge_error_mm),
        beta=float(huber_delta_mm),
        reduction="none",
    ).mean(dim=1)
    mean_abs_error = edge_error_mm.abs().mean(dim=1)
    return loss, mean_abs_error.detach()


def diversity_loss_contacts(
    pred_contacts: torch.Tensor,
    sigma: float = 0.03,
    margin: Optional[float] = None,
) -> torch.Tensor:
    n = pred_contacts.shape[1]
    if n < 2:
        return pred_contacts.new_tensor(0.0)
    pair_dist = pairwise_contact_distances(pred_contacts)
    if margin is None:
        loss = torch.exp(-pair_dist.pow(2) / (sigma**2))
    else:
        loss = torch.relu(margin - pair_dist)
    return loss.mean()


def compute_contact_losses(
    eps_pred: torch.Tensor,
    eps: torch.Tensor,
    c0_pred: torch.Tensor,
    c0: torch.Tensor,
    object_pc: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    lambda_noise: float = 1.0,
    lambda_set: float = 1.0,
    lambda_surface: float = 0.1,
    lambda_div: float = 0.1,
    set_loss_type: str = "chamfer",
    diversity_sigma: float = 0.03,
    lambda_chamfer: Optional[float] = None,
    lambda_diversity: Optional[float] = None,
    chamfer_max_timestep: Optional[int] = None,
    lambda_point_to_plane: float = 0.0,
    lambda_chamfer_mm: float = 0.0,
    point_to_plane_max_timestep: Optional[int] = None,
    lambda_relative_geometry: float = 0.0,
    relative_geometry_max_timestep: Optional[int] = None,
    relative_geometry_huber_delta_mm: float = 5.0,
    object_normals: Optional[torch.Tensor] = None,
    normalization_scale: Optional[torch.Tensor] = None,
) -> tuple[dict, dict]:
    losses = {}
    stats = {}
    if lambda_chamfer is not None:
        lambda_set = lambda_chamfer
    if lambda_diversity is not None:
        lambda_div = lambda_diversity

    metric_scale = None
    if normalization_scale is not None:
        metric_scale = torch.as_tensor(
            normalization_scale,
            device=c0.device,
            dtype=c0.dtype,
        ).reshape(-1)
        if metric_scale.numel() == 1 and c0.shape[0] != 1:
            metric_scale = metric_scale.expand(c0.shape[0])
        if metric_scale.numel() != c0.shape[0]:
            raise ValueError(
                "normalization_scale must contain one value per batch element"
            )

    losses["noise"] = (lambda_noise, noise_mse_loss(eps_pred, eps))

    batch_mask = torch.ones(c0.shape[0], device=c0.device, dtype=torch.bool)
    chamfer_mask = batch_mask
    if chamfer_max_timestep is not None:
        if timesteps is None:
            raise ValueError("chamfer_max_timestep requires timesteps")
        chamfer_mask = timesteps.le(int(chamfer_max_timestep))
        stats["chamfer_batch_fraction"] = chamfer_mask.float().mean().detach()

    # chamfer_mm is the primary interpretable metric.  It is unsquared,
    # symmetric, averaged across directions, and restored to physical units.
    if chamfer_mask.any():
        per_sample_chamfer_mm = symmetric_chamfer_mm_per_sample(
            c0_pred[chamfer_mask],
            c0[chamfer_mask],
            None if metric_scale is None else metric_scale[chamfer_mask],
        )
        stats["chamfer_mm"] = per_sample_chamfer_mm.mean().detach()
        stats["chamfer_mm_min"] = per_sample_chamfer_mm.min().detach()
        stats["chamfer_mm_max"] = per_sample_chamfer_mm.max().detach()
        if lambda_chamfer_mm != 0:
            losses["chamfer_mm"] = (
                float(lambda_chamfer_mm),
                per_sample_chamfer_mm.mean(),
            )
    else:
        stats["chamfer_mm"] = (c0_pred.sum() * 0.0).detach()
        if lambda_chamfer_mm != 0:
            losses["chamfer_mm"] = (
                float(lambda_chamfer_mm),
                c0_pred.sum() * 0.0,
            )

    if lambda_set != 0:
        c0_pred_for_set = c0_pred[chamfer_mask]
        c0_for_set = c0[chamfer_mask]
        if not chamfer_mask.any():
            set_loss = c0_pred.sum() * 0.0
            losses["chamfer"] = (lambda_set, set_loss)
            stats["chamfer"] = set_loss.detach()
        else:
            if set_loss_type != "chamfer":
                raise ValueError("Standalone ContactDiffusion currently supports set_loss_type='chamfer'.")
            per_sample_chamfer = chamfer_loss_contacts_per_sample(c0_pred_for_set, c0_for_set)
            set_loss = per_sample_chamfer.mean()
            losses["chamfer"] = (lambda_set, set_loss)
            stats["chamfer"] = set_loss.detach()
            stats["chamfer_min"] = per_sample_chamfer.min().detach()
            stats["chamfer_max"] = per_sample_chamfer.max().detach()
            stats["chamfer_mean"] = set_loss.detach()

    if lambda_point_to_plane != 0:
        if object_normals is None:
            raise ValueError(
                "lambda_point_to_plane is non-zero but object_normals were not provided"
            )
        plane_mask = batch_mask
        if point_to_plane_max_timestep is not None:
            if timesteps is None:
                raise ValueError("point_to_plane_max_timestep requires timesteps")
            plane_mask = timesteps.le(int(point_to_plane_max_timestep))
        stats["point_to_plane_batch_fraction"] = plane_mask.float().mean().detach()
        if plane_mask.any():
            per_sample_plane_mm = point_to_plane_mm_per_sample(
                c0_pred[plane_mask],
                object_pc[plane_mask],
                object_normals[plane_mask],
                None if metric_scale is None else metric_scale[plane_mask],
            )
            point_to_plane_loss = per_sample_plane_mm.mean()
        else:
            point_to_plane_loss = c0_pred.sum() * 0.0
        losses["point_to_plane_mm"] = (
            float(lambda_point_to_plane),
            point_to_plane_loss,
        )
        stats["point_to_plane_mm"] = point_to_plane_loss.detach()

    if lambda_relative_geometry != 0:
        relative_mask = batch_mask
        if relative_geometry_max_timestep is not None:
            if timesteps is None:
                raise ValueError("relative_geometry_max_timestep requires timesteps")
            relative_mask = timesteps.le(int(relative_geometry_max_timestep))
        stats["relative_geometry_batch_fraction"] = relative_mask.float().mean().detach()
        if relative_mask.any():
            relative_loss, relative_abs = (
                permutation_aligned_relative_geometry_mm_per_sample(
                    c0_pred[relative_mask],
                    c0[relative_mask],
                    None if metric_scale is None else metric_scale[relative_mask],
                    huber_delta_mm=relative_geometry_huber_delta_mm,
                )
            )
            relative_loss = relative_loss.mean()
            relative_abs = relative_abs.mean()
        else:
            relative_loss = c0_pred.sum() * 0.0
            relative_abs = relative_loss.detach()
        losses["relative_geometry_mm"] = (
            float(lambda_relative_geometry),
            relative_loss,
        )
        stats["relative_geometry_abs_mm"] = relative_abs.detach()

    if lambda_surface != 0:
        losses["surface"] = (lambda_surface, surface_loss_contacts(c0_pred, object_pc))
    if lambda_div != 0:
        losses["diversity"] = (
            lambda_div,
            diversity_loss_contacts(c0_pred, sigma=diversity_sigma),
        )
    return losses, stats
