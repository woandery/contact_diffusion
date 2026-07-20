"""DDPM/DDIM utilities for contact set diffusion."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from models.denoiser import ContactSetDenoiser
from models.losses import compute_contact_losses


def gather_scheduler_values(values: torch.Tensor, timesteps: torch.Tensor, ndim: int):
    out = values.to(device=timesteps.device)[timesteps]
    while out.ndim < ndim:
        out = out.unsqueeze(-1)
    return out


def predict_x0_from_eps(
    contacts_t: torch.Tensor,
    eps_pred: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = gather_scheduler_values(alphas_cumprod, timesteps, contacts_t.ndim)
    return (contacts_t - (1.0 - alpha_bar).sqrt() * eps_pred) / alpha_bar.sqrt()


def random_permute_contact_set(contacts: torch.Tensor) -> torch.Tensor:
    perm = torch.randperm(contacts.shape[1], device=contacts.device)
    return contacts[:, perm, :]


def project_contacts_to_surface(contacts: torch.Tensor, object_pc: torch.Tensor) -> torch.Tensor:
    xyz = contacts[..., :3]
    dist = torch.cdist(xyz, object_pc[..., :3], p=2)
    idx = dist.argmin(dim=2)
    nn_xyz = torch.gather(
        object_pc[..., :3],
        dim=1,
        index=idx.unsqueeze(-1).expand(-1, -1, 3),
    )
    if contacts.shape[-1] == 3:
        return nn_xyz
    return torch.cat([nn_xyz, contacts[..., 3:]], dim=-1)


class ContactDiffusion(nn.Module):
    def __init__(
        self,
        denoiser: ContactSetDenoiser,
        num_diffusion_iters: int = 1000,
        num_diffusion_iters_eval: int = 50,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        random_permute_contacts: bool = True,
        loss_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.num_diffusion_iters = int(num_diffusion_iters)
        self.num_diffusion_iters_eval = int(num_diffusion_iters_eval)
        self.random_permute_contacts = bool(random_permute_contacts)
        self.loss_cfg = loss_cfg or {}
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type="epsilon",
        )

    @classmethod
    def from_config(cls, cfg):
        from types import SimpleNamespace

        denoiser_cfg = SimpleNamespace(
            dc=cfg.model.dc,
            d_model=cfg.model.d_model,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            ffn_dim=getattr(cfg.model, "ffn_dim", None),
            dropout=getattr(cfg.model, "dropout", 0.0),
            num_diffusion_iters=cfg.diffusion.num_timesteps,
            n_values=list(cfg.dataset.n_values),
            use_n_embedding=cfg.model.use_n_embedding,
            use_object_cross_attention=cfg.model.use_object_cross_attention,
            object_input_dim=getattr(cfg.model, "object_input_dim", 3),
            object_encoder_type=getattr(cfg.model, "object_encoder_type", "simple_pointnet"),
            object_num_tokens=getattr(cfg.model, "object_num_tokens", 64),
            activation=getattr(cfg.model, "activation", "GELU"),
        )
        denoiser = ContactSetDenoiser.from_config(denoiser_cfg)
        loss_cfg = {
            "lambda_noise": cfg.loss.lambda_noise,
            "lambda_chamfer": cfg.loss.lambda_chamfer,
            "lambda_surface": cfg.loss.lambda_surface,
            "lambda_diversity": cfg.loss.lambda_diversity,
            "set_loss_type": getattr(cfg.loss, "set_loss_type", "chamfer"),
            "diversity_sigma": getattr(cfg.loss, "diversity_sigma", 0.01),
            "chamfer_max_timestep": getattr(cfg.loss, "chamfer_max_timestep", None),
        }
        return cls(
            denoiser=denoiser,
            num_diffusion_iters=cfg.diffusion.num_timesteps,
            num_diffusion_iters_eval=getattr(cfg.sampling, "num_steps", 50),
            beta_schedule=getattr(cfg.diffusion, "beta_schedule", "squaredcos_cap_v2"),
            beta_start=getattr(cfg.diffusion, "beta_start", 1e-4),
            beta_end=getattr(cfg.diffusion, "beta_end", 0.02),
            clip_sample=getattr(cfg.diffusion, "clip_sample", True),
            random_permute_contacts=getattr(cfg.train, "random_permute_contacts", True),
            loss_cfg=loss_cfg,
        )

    def forward(
        self,
        contacts_t: torch.Tensor,
        timesteps: torch.Tensor,
        object_pc: torch.Tensor,
        num_contacts: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        return self.denoiser(contacts_t, timesteps, object_pc, num_contacts)

    def training_step(
        self,
        object_pc: torch.Tensor,
        contacts: torch.Tensor,
        num_contacts: Union[int, torch.Tensor, None] = None,
    ):
        if num_contacts is None:
            num_contacts = contacts.shape[1]
        c0 = random_permute_contact_set(contacts) if self.random_permute_contacts else contacts
        batch_size = c0.shape[0]
        device = c0.device
        eps = torch.randn_like(c0)
        timesteps = torch.randint(0, self.num_diffusion_iters, (batch_size,), device=device).long()
        contacts_t = self.noise_scheduler.add_noise(c0, eps, timesteps)
        eps_pred = self.denoiser(contacts_t, timesteps, object_pc, num_contacts)
        c0_pred = predict_x0_from_eps(
            contacts_t,
            eps_pred,
            timesteps,
            self.noise_scheduler.alphas_cumprod.to(device),
        )
        losses, stats = compute_contact_losses(
            eps_pred=eps_pred,
            eps=eps,
            c0_pred=c0_pred,
            c0=c0,
            object_pc=object_pc,
            timesteps=timesteps,
            **self.loss_cfg,
        )
        return {
            "contacts_t": contacts_t,
            "contacts_pred": c0_pred,
            "eps_pred": eps_pred,
            "eps": eps,
            "timesteps": timesteps,
        }, losses, stats

    @torch.no_grad()
    def sample(
        self,
        object_pc: torch.Tensor,
        num_contacts: int,
        dc: Optional[int] = None,
        num_steps: Optional[int] = None,
        sampler: str = "ddim",
        project_to_surface: bool = False,
    ) -> torch.Tensor:
        return sample_contacts(
            self.denoiser,
            object_pc=object_pc,
            num_contacts=num_contacts,
            dc=dc or self.denoiser.dc,
            num_steps=num_steps or self.num_diffusion_iters_eval,
            sampler=sampler,
            project_to_surface=project_to_surface,
            beta_schedule=self.noise_scheduler.config.beta_schedule,
            beta_start=getattr(self.noise_scheduler.config, "beta_start", 1e-4),
            beta_end=getattr(self.noise_scheduler.config, "beta_end", 0.02),
            num_train_timesteps=self.num_diffusion_iters,
        )


@torch.no_grad()
def sample_contacts(
    model: nn.Module,
    object_pc: torch.Tensor,
    num_contacts: int,
    dc: int = 3,
    num_steps: int = 50,
    sampler: str = "ddim",
    project_to_surface: bool = False,
    beta_schedule: str = "squaredcos_cap_v2",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    device = object_pc.device
    batch_size = object_pc.shape[0]
    contacts_t = torch.randn(batch_size, num_contacts, dc, device=device)
    scheduler_cls = DDIMScheduler if sampler == "ddim" else DDPMScheduler
    scheduler = scheduler_cls(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(num_steps)
    for timestep in scheduler.timesteps:
        t = torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)
        eps_pred = model(contacts_t, t, object_pc, num_contacts)
        contacts_t = scheduler.step(eps_pred, timestep, contacts_t).prev_sample
    if project_to_surface:
        contacts_t = project_contacts_to_surface(contacts_t, object_pc)
    return contacts_t
