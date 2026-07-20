"""Losses for unordered contact diffusion."""

from __future__ import annotations

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
) -> tuple[dict, dict]:
    losses = {}
    stats = {}
    if lambda_chamfer is not None:
        lambda_set = lambda_chamfer
    if lambda_diversity is not None:
        lambda_div = lambda_diversity

    losses["noise"] = (lambda_noise, noise_mse_loss(eps_pred, eps))

    if lambda_set != 0:
        c0_pred_for_set = c0_pred
        c0_for_set = c0
        if chamfer_max_timestep is not None:
            if timesteps is None:
                raise ValueError("chamfer_max_timestep requires timesteps")
            low_t_mask = timesteps.le(int(chamfer_max_timestep))
            stats["chamfer_batch_fraction"] = low_t_mask.float().mean().detach()
            if low_t_mask.any():
                c0_pred_for_set = c0_pred[low_t_mask]
                c0_for_set = c0[low_t_mask]
            else:
                set_loss = c0_pred.sum() * 0.0
                losses["chamfer"] = (lambda_set, set_loss)
                stats["chamfer"] = set_loss.detach()
                return losses, stats

        if set_loss_type != "chamfer":
            raise ValueError("Standalone ContactDiffusion currently supports set_loss_type='chamfer'.")
        per_sample_chamfer = chamfer_loss_contacts_per_sample(c0_pred_for_set, c0_for_set)
        set_loss = per_sample_chamfer.mean()
        losses["chamfer"] = (lambda_set, set_loss)
        stats["chamfer"] = set_loss.detach()
        stats["chamfer_min"] = per_sample_chamfer.min().detach()
        stats["chamfer_max"] = per_sample_chamfer.max().detach()
        stats["chamfer_mean"] = set_loss.detach()

    if lambda_surface != 0:
        losses["surface"] = (lambda_surface, surface_loss_contacts(c0_pred, object_pc))
    if lambda_div != 0:
        losses["diversity"] = (
            lambda_div,
            diversity_loss_contacts(c0_pred, sigma=diversity_sigma),
        )
    return losses, stats
