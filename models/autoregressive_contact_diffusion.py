"""Autoregressive free-XYZ diffusion for sparse contact sets.

The set distribution is factorized without discrete modes or surface anchors:

    p(S | O) = p(x_1 | O) * prod_i p(x_i | O, {x_j}_{j<i})

GT contacts are randomly permuted during training, so the clean prefix is an
unordered set rather than a fixed finger/contact identity.  At inference every
new contact starts from Gaussian noise and remains a continuous XYZ value.
"""

from __future__ import annotations

import itertools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from models.denoiser import ContactSetDenoiser
from models.diffusion import predict_x0_from_model_output


class AutoregressiveContactDenoiser(ContactSetDenoiser):
    """V4-compatible denoiser that predicts only the final, noisy token."""

    def __init__(self, *args, max_contacts: int = 8, **kwargs):
        if kwargs.get("use_set_token", False):
            raise ValueError("AutoregressiveContactDenoiser does not use a set token.")
        super().__init__(*args, **kwargs)
        self.max_contacts = int(max_contacts)
        self.role_embedding = nn.Embedding(2, self.d_model)
        self.prefix_count_embedding = nn.Embedding(self.max_contacts, self.d_model)
        self.object_global_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # A V4 checkpoint is a meaningful initialization before these two new
        # conditioning paths have learned anything.
        nn.init.zeros_(self.role_embedding.weight)
        nn.init.zeros_(self.prefix_count_embedding.weight)

    def forward(
        self,
        noisy_next: torch.Tensor,
        timesteps: torch.Tensor,
        object_pc: Optional[torch.Tensor],
        prefix: Optional[torch.Tensor] = None,
        total_contacts: Optional[int] = None,
        object_features: Optional[tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
        prefix_timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noisy_next.ndim != 3 or noisy_next.shape[1] != 1:
            raise ValueError(f"Expected noisy_next [B,1,C], got {noisy_next.shape}")
        batch_size = noisy_next.shape[0]
        if prefix is None:
            prefix = noisy_next.new_zeros((batch_size, 0, noisy_next.shape[-1]))
        if prefix.ndim != 3 or prefix.shape[0] != batch_size:
            raise ValueError(f"Expected prefix [B,K,C], got {prefix.shape}")

        prefix_length = int(prefix.shape[1])
        if prefix_length >= self.max_contacts:
            raise ValueError(
                f"Prefix length {prefix_length} exceeds max_contacts={self.max_contacts}"
            )
        total_contacts = int(total_contacts or (prefix_length + 1))

        contact_xyz = torch.cat([prefix, noisy_next], dim=1)
        z = self.contact_in(contact_xyz)

        time_feature = self.time_encoder(timesteps)
        z[:, -1, :] = z[:, -1, :] + time_feature
        if prefix_length > 0:
            if prefix_timesteps is None:
                prefix_timesteps = torch.zeros(
                    batch_size, prefix_length, device=z.device, dtype=torch.long
                )
            if prefix_timesteps.shape != (batch_size, prefix_length):
                raise ValueError(
                    "Expected prefix_timesteps "
                    f"{(batch_size, prefix_length)}, got {prefix_timesteps.shape}"
                )
            prefix_time_feature = self.time_encoder(prefix_timesteps.reshape(-1))
            z[:, :prefix_length, :] = z[:, :prefix_length, :] + prefix_time_feature.reshape(
                batch_size, prefix_length, self.d_model
            )

        role_ids = torch.zeros(
            batch_size, prefix_length + 1, device=z.device, dtype=torch.long
        )
        role_ids[:, -1] = 1
        z = z + self.role_embedding(role_ids)
        z = z + self.prefix_count_embedding.weight[prefix_length][None, None, :]

        n_tensor = torch.full(
            (batch_size,), total_contacts, device=z.device, dtype=torch.long
        )
        z = z + self._n_embedding(n_tensor)[:, None, :]

        if object_features is None:
            object_tokens, object_xyz = self.encode_object_geometry(object_pc=object_pc)
        else:
            object_tokens, object_xyz = object_features
        if object_tokens is not None:
            # ATISS-style global condition: the first/noisy contact receives a
            # stable object summary even when its noisy XYZ is far from the
            # surface and relative-position attention is not yet informative.
            z = z + self.object_global_proj(object_tokens.mean(dim=1))[:, None, :]
        if not self.use_object_cross_attention:
            object_tokens, object_xyz = None, None
        for block in self.blocks:
            z = block(
                z,
                object_tokens,
                contact_xyz=contact_xyz[..., :3],
                object_xyz=object_xyz,
            )
        return self.contact_out(self.norm(z[:, -1:, :]))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return values.new_zeros(())
    return values[mask].mean()


class AutoregressiveContactDiffusion(nn.Module):
    """Sequential conditional diffusion with continuous XYZ outputs."""

    def __init__(
        self,
        denoiser: AutoregressiveContactDenoiser,
        num_diffusion_iters: int = 1000,
        num_diffusion_iters_eval: int = 50,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        prefix_noise_std: float = 0.0,
        prefix_noise_max_timestep: int = 0,
        stabilization_timestep: int = 0,
        symmetric_all_first: bool = True,
        loss_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.num_diffusion_iters = int(num_diffusion_iters)
        self.num_diffusion_iters_eval = int(num_diffusion_iters_eval)
        self.prediction_type = "sample" if prediction_type == "x0" else str(prediction_type)
        self.prefix_noise_std = float(prefix_noise_std)
        self.prefix_noise_max_timestep = int(prefix_noise_max_timestep)
        self.stabilization_timestep = int(stabilization_timestep)
        self.symmetric_all_first = bool(symmetric_all_first)
        self.loss_cfg = loss_cfg or {}
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            beta_schedule=str(beta_schedule),
            clip_sample=bool(clip_sample),
            prediction_type=self.prediction_type,
        )

    @classmethod
    def from_config(cls, cfg):
        denoiser = AutoregressiveContactDenoiser(
            dc=int(cfg.model.dc),
            d_model=int(cfg.model.d_model),
            num_layers=int(cfg.model.num_layers),
            num_heads=int(cfg.model.num_heads),
            ffn_dim=getattr(cfg.model, "ffn_dim", None),
            dropout=float(getattr(cfg.model, "dropout", 0.0)),
            num_diffusion_iters=int(cfg.diffusion.num_timesteps),
            n_values=list(cfg.dataset.n_values),
            use_n_embedding=bool(cfg.model.use_n_embedding),
            use_object_cross_attention=bool(cfg.model.use_object_cross_attention),
            object_input_dim=int(getattr(cfg.model, "object_input_dim", 3)),
            object_encoder_type=str(
                getattr(cfg.model, "object_encoder_type", "simple_pointnet")
            ),
            object_num_tokens=int(getattr(cfg.model, "object_num_tokens", 64)),
            pointnet_local_npoints=getattr(cfg.model, "pointnet_local_npoints", None),
            pointnet_local_radii=getattr(cfg.model, "pointnet_local_radii", None),
            pointnet_local_nsamples=getattr(cfg.model, "pointnet_local_nsamples", None),
            include_object_xyz_in_tokens=bool(
                getattr(cfg.model, "include_object_xyz_in_tokens", False)
            ),
            use_set_token=False,
            activation=str(getattr(cfg.model, "activation", "GELU")),
            max_contacts=int(getattr(cfg.model, "max_contacts", max(cfg.dataset.n_values) + 1)),
        )
        loss_cfg = {
            "lambda_noise": float(getattr(cfg.loss, "lambda_noise", 1.0)),
            "lambda_xyz_mm": float(getattr(cfg.loss, "lambda_xyz_mm", 0.0)),
            "lambda_relative_mm": float(getattr(cfg.loss, "lambda_relative_mm", 0.0)),
            "lambda_set_mm": float(getattr(cfg.loss, "lambda_set_mm", 0.0)),
            "lambda_surface_mm": float(getattr(cfg.loss, "lambda_surface_mm", 0.0)),
            "geometry_max_timestep": int(
                getattr(cfg.loss, "geometry_max_timestep", cfg.diffusion.num_timesteps - 1)
            ),
            "huber_delta_mm": float(getattr(cfg.loss, "huber_delta_mm", 2.0)),
        }
        return cls(
            denoiser=denoiser,
            num_diffusion_iters=int(cfg.diffusion.num_timesteps),
            num_diffusion_iters_eval=int(getattr(cfg.sampling, "num_steps", 50)),
            beta_schedule=str(getattr(cfg.diffusion, "beta_schedule", "squaredcos_cap_v2")),
            beta_start=float(getattr(cfg.diffusion, "beta_start", 1e-4)),
            beta_end=float(getattr(cfg.diffusion, "beta_end", 0.02)),
            clip_sample=bool(getattr(cfg.diffusion, "clip_sample", True)),
            prediction_type=str(getattr(cfg.diffusion, "prediction_type", "epsilon")),
            prefix_noise_std=float(getattr(cfg.train, "prefix_noise_std", 0.0)),
            prefix_noise_max_timestep=int(
                getattr(cfg.train, "prefix_noise_max_timestep", 0)
            ),
            stabilization_timestep=int(
                getattr(cfg.sampling, "stabilization_timestep", 0)
            ),
            symmetric_all_first=bool(getattr(cfg.train, "symmetric_all_first", True)),
            loss_cfg=loss_cfg,
        )

    @staticmethod
    def _coordinate_to_mm(normalization_scale: Optional[torch.Tensor], reference) -> torch.Tensor:
        if normalization_scale is None:
            return reference.new_full((reference.shape[0],), 1000.0)
        scale = normalization_scale.to(device=reference.device, dtype=reference.dtype)
        if scale.ndim > 1:
            scale = scale.reshape(scale.shape[0], -1)[:, 0]
        return scale * 1000.0

    def forward(
        self,
        object_pc: torch.Tensor,
        contacts: torch.Tensor,
        object_normals: Optional[torch.Tensor] = None,
        normalization_scale: Optional[torch.Tensor] = None,
        prefix_length: Optional[int] = None,
        timesteps: Optional[torch.Tensor] = None,
        trajectory: bool = False,
        rollout_generated_probability: float = 0.0,
    ):
        if trajectory:
            return self.trajectory_training_step(
                object_pc,
                contacts,
                object_normals=object_normals,
                normalization_scale=normalization_scale,
                timesteps=timesteps,
                rollout_generated_probability=rollout_generated_probability,
            )
        return self.training_step(
            object_pc,
            contacts,
            object_normals=object_normals,
            normalization_scale=normalization_scale,
            prefix_length=prefix_length,
            timesteps=timesteps,
        )

    def training_step(
        self,
        object_pc: torch.Tensor,
        contacts: torch.Tensor,
        object_normals: Optional[torch.Tensor] = None,
        normalization_scale: Optional[torch.Tensor] = None,
        prefix_length: Optional[int] = None,
        timesteps: Optional[torch.Tensor] = None,
    ):
        del object_normals
        original_batch_size, num_contacts, _ = contacts.shape
        device = contacts.device

        # Start with a per-example random ordering, then rotate it N times.
        # Consequently every GT point is used exactly once at every sequence
        # position in this batch (in particular, every point is a first point).
        permutations = torch.rand(
            original_batch_size, num_contacts, device=device
        ).argsort(dim=1)
        base_order = torch.gather(
            contacts,
            1,
            permutations.unsqueeze(-1).expand(-1, -1, contacts.shape[-1]),
        )
        if self.symmetric_all_first:
            positions = (
                torch.arange(num_contacts, device=device)[None, :, None]
                + torch.arange(num_contacts, device=device)[None, None, :]
            ) % num_contacts
            positions = positions.expand(original_batch_size, -1, -1)
            ordered = torch.gather(
                base_order[:, None].expand(-1, num_contacts, -1, -1),
                2,
                positions[..., None].expand(-1, -1, -1, contacts.shape[-1]),
            ).reshape(original_batch_size * num_contacts, num_contacts, contacts.shape[-1])
            object_features = self.denoiser.encode_object_geometry(object_pc=object_pc)
            object_features = (
                object_features[0].repeat_interleave(num_contacts, dim=0),
                None
                if object_features[1] is None
                else object_features[1].repeat_interleave(num_contacts, dim=0),
            )
            expanded_object_pc = object_pc.repeat_interleave(num_contacts, dim=0)
            if normalization_scale is not None:
                normalization_scale = normalization_scale.repeat_interleave(
                    num_contacts, dim=0
                )
        else:
            ordered = base_order
            expanded_object_pc = object_pc
            object_features = None
        batch_size = ordered.shape[0]
        if prefix_length is None:
            prefix_length = int(torch.randint(0, num_contacts, (), device=device).item())
        if not 0 <= int(prefix_length) < num_contacts:
            raise ValueError(f"Invalid prefix_length={prefix_length} for N={num_contacts}")
        prefix_length = int(prefix_length)
        clean_prefix = ordered[:, :prefix_length]
        target = ordered[:, prefix_length : prefix_length + 1]
        model_prefix = clean_prefix
        prefix_timesteps = None
        if prefix_length > 0:
            prefix_timesteps = torch.zeros(
                batch_size, prefix_length, device=device, dtype=torch.long
            )
            if self.training and self.prefix_noise_max_timestep > 0:
                prefix_timesteps = torch.randint(
                    0,
                    self.prefix_noise_max_timestep + 1,
                    (batch_size, prefix_length),
                    device=device,
                )
                alpha_prefix = self.noise_scheduler.alphas_cumprod.to(device)[
                    prefix_timesteps
                ][..., None]
                model_prefix = (
                    alpha_prefix.sqrt() * clean_prefix
                    + (1.0 - alpha_prefix).sqrt() * torch.randn_like(clean_prefix)
                )
            elif self.training and self.prefix_noise_std > 0.0:
                model_prefix = clean_prefix + self.prefix_noise_std * torch.randn_like(
                    clean_prefix
                )

        noise = torch.randn_like(target)
        if timesteps is None:
            timesteps = torch.randint(
                0, self.num_diffusion_iters, (batch_size,), device=device
            ).long()
        elif timesteps.shape[0] == original_batch_size and batch_size != original_batch_size:
            timesteps = timesteps.repeat_interleave(num_contacts, dim=0)
        elif timesteps.shape[0] != batch_size:
            raise ValueError(
                f"Expected {original_batch_size} or {batch_size} timesteps, got {timesteps.shape[0]}"
            )
        target_t = self.noise_scheduler.add_noise(target, noise, timesteps)
        model_output = self.denoiser(
            target_t,
            timesteps,
            expanded_object_pc,
            prefix=model_prefix,
            total_contacts=num_contacts,
            object_features=object_features,
            prefix_timesteps=prefix_timesteps,
        )
        if self.prediction_type == "epsilon":
            diffusion_target = noise
        elif self.prediction_type == "sample":
            diffusion_target = target
        elif self.prediction_type == "v_prediction":
            diffusion_target = self.noise_scheduler.get_velocity(target, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type={self.prediction_type!r}")
        target_pred = predict_x0_from_model_output(
            target_t,
            model_output,
            timesteps,
            self.noise_scheduler.alphas_cumprod.to(device),
            self.prediction_type,
        )

        noise_loss = F.mse_loss(model_output, diffusion_target)
        mm_scale = self._coordinate_to_mm(normalization_scale, target)
        point_error_mm = torch.linalg.vector_norm(target_pred - target, dim=-1).squeeze(1) * mm_scale
        geometry_mask = timesteps.le(int(self.loss_cfg["geometry_max_timestep"]))
        xyz_loss = _masked_mean(
            F.huber_loss(
                point_error_mm,
                torch.zeros_like(point_error_mm),
                delta=float(self.loss_cfg["huber_delta_mm"]),
                reduction="none",
            ),
            geometry_mask,
        )

        relative_loss = target.new_zeros(())
        if prefix_length > 0:
            true_dist = torch.linalg.vector_norm(target - clean_prefix, dim=-1)
            pred_dist = torch.linalg.vector_norm(target_pred - clean_prefix, dim=-1)
            relative_error_mm = (pred_dist - true_dist).abs() * mm_scale[:, None]
            per_sample_relative = F.huber_loss(
                relative_error_mm,
                torch.zeros_like(relative_error_mm),
                delta=float(self.loss_cfg["huber_delta_mm"]),
                reduction="none",
            ).mean(dim=1)
            relative_loss = _masked_mean(per_sample_relative, geometry_mask)

        nearest_surface_mm = torch.cdist(
            target_pred[..., :3], expanded_object_pc[..., :3]
        ).amin(dim=(1, 2)) * mm_scale
        surface_loss = _masked_mean(
            F.huber_loss(
                nearest_surface_mm,
                torch.zeros_like(nearest_surface_mm),
                delta=float(self.loss_cfg["huber_delta_mm"]),
                reduction="none",
            ),
            geometry_mask,
        )
        total_loss = (
            float(self.loss_cfg["lambda_noise"]) * noise_loss
            + float(self.loss_cfg["lambda_xyz_mm"]) * xyz_loss
            + float(self.loss_cfg["lambda_relative_mm"]) * relative_loss
            + float(self.loss_cfg["lambda_surface_mm"]) * surface_loss
        )
        losses = {
            "total": total_loss,
            "noise": noise_loss,
            "xyz_mm": xyz_loss,
            "relative_mm": relative_loss,
            "surface_mm": surface_loss,
        }
        stats = {
            "point_error_mm": point_error_mm.mean().detach(),
            "nearest_surface_mm": nearest_surface_mm.mean().detach(),
            "prefix_length": target.new_tensor(float(prefix_length)),
            "low_t_fraction": geometry_mask.float().mean().detach(),
        }
        outputs = {
            "contacts_t": target_t,
            "contacts_pred": target_pred,
            "model_output": model_output,
            "target": target,
            "timesteps": timesteps,
            "prefix": clean_prefix,
        }
        return outputs, losses, stats

    def trajectory_training_step(
        self,
        object_pc: torch.Tensor,
        contacts: torch.Tensor,
        object_normals: Optional[torch.Tensor] = None,
        normalization_scale: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        rollout_generated_probability: float = 0.0,
    ):
        """Train every autoregressive position from one coherent source GT set.

        Previous predictions may replace teacher-forced prefix tokens, but are
        detached before reuse.  A final one-to-one set loss always targets the
        same source set rather than selecting a different empirical mode.
        """

        del object_normals
        original_batch_size, num_contacts, _ = contacts.shape
        device = contacts.device
        permutations = torch.rand(
            original_batch_size, num_contacts, device=device
        ).argsort(dim=1)
        base_order = torch.gather(
            contacts,
            1,
            permutations.unsqueeze(-1).expand(-1, -1, contacts.shape[-1]),
        )
        if self.symmetric_all_first:
            positions = (
                torch.arange(num_contacts, device=device)[None, :, None]
                + torch.arange(num_contacts, device=device)[None, None, :]
            ) % num_contacts
            positions = positions.expand(original_batch_size, -1, -1)
            ordered = torch.gather(
                base_order[:, None].expand(-1, num_contacts, -1, -1),
                2,
                positions[..., None].expand(-1, -1, -1, contacts.shape[-1]),
            ).reshape(
                original_batch_size * num_contacts,
                num_contacts,
                contacts.shape[-1],
            )
            object_features = self.denoiser.encode_object_geometry(object_pc=object_pc)
            object_features = (
                object_features[0].repeat_interleave(num_contacts, dim=0),
                None
                if object_features[1] is None
                else object_features[1].repeat_interleave(num_contacts, dim=0),
            )
            expanded_object_pc = object_pc.repeat_interleave(num_contacts, dim=0)
            if normalization_scale is not None:
                normalization_scale = normalization_scale.repeat_interleave(
                    num_contacts, dim=0
                )
            if timesteps is not None:
                timesteps = timesteps.repeat_interleave(num_contacts, dim=0)
        else:
            ordered = base_order
            object_features = None
            expanded_object_pc = object_pc

        batch_size = ordered.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.num_diffusion_iters,
                (batch_size, num_contacts),
                device=device,
            ).long()
        if timesteps.shape != (batch_size, num_contacts):
            raise ValueError(
                f"Expected trajectory timesteps {(batch_size, num_contacts)}, "
                f"got {tuple(timesteps.shape)}"
            )

        mm_scale = self._coordinate_to_mm(normalization_scale, ordered)
        model_prefix = ordered.new_zeros((batch_size, 0, ordered.shape[-1]))
        predictions = []
        noise_losses = []
        point_errors = []
        surface_errors = []
        geometry_masks = []
        generated_tokens = []
        max_geometry_t = int(self.loss_cfg["geometry_max_timestep"])
        rollout_probability = float(rollout_generated_probability)

        for position in range(num_contacts):
            target = ordered[:, position : position + 1]
            position_timesteps = timesteps[:, position]
            noise = torch.randn_like(target)
            target_t = self.noise_scheduler.add_noise(
                target, noise, position_timesteps
            )
            prefix_timesteps = torch.zeros(
                batch_size,
                position,
                device=device,
                dtype=torch.long,
            )
            model_output = self.denoiser(
                target_t,
                position_timesteps,
                expanded_object_pc,
                prefix=model_prefix,
                total_contacts=num_contacts,
                object_features=object_features,
                prefix_timesteps=prefix_timesteps,
            )
            if self.prediction_type == "epsilon":
                diffusion_target = noise
            elif self.prediction_type == "sample":
                diffusion_target = target
            elif self.prediction_type == "v_prediction":
                diffusion_target = self.noise_scheduler.get_velocity(
                    target, noise, position_timesteps
                )
            else:
                raise ValueError(
                    f"Unsupported prediction_type={self.prediction_type!r}"
                )
            target_pred = predict_x0_from_model_output(
                target_t,
                model_output,
                position_timesteps,
                self.noise_scheduler.alphas_cumprod.to(device),
                self.prediction_type,
            )
            predictions.append(target_pred)
            noise_losses.append(F.mse_loss(model_output, diffusion_target))
            point_errors.append(
                torch.linalg.vector_norm(target_pred - target, dim=-1).squeeze(1)
                * mm_scale
            )
            surface_errors.append(
                torch.cdist(target_pred[..., :3], expanded_object_pc[..., :3]).amin(
                    dim=(1, 2)
                )
                * mm_scale
            )
            geometry_masks.append(position_timesteps.le(max_geometry_t))

            if self.training and rollout_probability > 0.0:
                use_generated = torch.rand(
                    batch_size, 1, 1, device=device
                ).lt(rollout_probability)
                next_prefix = torch.where(
                    use_generated, target_pred.detach(), target
                )
                generated_tokens.append(use_generated.float().mean())
            else:
                next_prefix = target
                generated_tokens.append(target.new_zeros(()))
            model_prefix = torch.cat([model_prefix, next_prefix], dim=1)

        predicted_set = torch.cat(predictions, dim=1)
        point_error_mm = torch.stack(point_errors, dim=1)
        surface_error_mm = torch.stack(surface_errors, dim=1)
        geometry_mask = torch.stack(geometry_masks, dim=1)
        xyz_loss = _masked_mean(
            F.huber_loss(
                point_error_mm,
                torch.zeros_like(point_error_mm),
                delta=float(self.loss_cfg["huber_delta_mm"]),
                reduction="none",
            ),
            geometry_mask,
        )
        surface_loss = _masked_mean(
            F.huber_loss(
                surface_error_mm,
                torch.zeros_like(surface_error_mm),
                delta=float(self.loss_cfg["huber_delta_mm"]),
                reduction="none",
            ),
            geometry_mask,
        )

        pair_mask = torch.triu(
            torch.ones(
                num_contacts, num_contacts, device=device, dtype=torch.bool
            ),
            diagonal=1,
        )
        predicted_pair = torch.cdist(predicted_set, predicted_set)
        target_pair = torch.cdist(ordered, ordered)
        pair_error_mm = (
            (predicted_pair - target_pair).abs() * mm_scale[:, None, None]
        )[:, pair_mask]
        pair_geometry_mask = geometry_mask.all(dim=1)
        relative_per_sample = F.huber_loss(
            pair_error_mm,
            torch.zeros_like(pair_error_mm),
            delta=float(self.loss_cfg["huber_delta_mm"]),
            reduction="none",
        ).mean(dim=1)
        relative_loss = _masked_mean(relative_per_sample, pair_geometry_mask)

        set_error = predicted_set.new_full((batch_size,), float("inf"))
        for permutation in itertools.permutations(range(num_contacts)):
            candidate = torch.linalg.vector_norm(
                predicted_set - ordered[:, permutation, :], dim=-1
            ).mean(dim=1)
            set_error = torch.minimum(set_error, candidate)
        set_error_mm = set_error * mm_scale
        set_loss = _masked_mean(
            F.huber_loss(
                set_error_mm,
                torch.zeros_like(set_error_mm),
                delta=float(self.loss_cfg["huber_delta_mm"]),
                reduction="none",
            ),
            pair_geometry_mask,
        )
        noise_loss = torch.stack(noise_losses).mean()
        total_loss = (
            float(self.loss_cfg["lambda_noise"]) * noise_loss
            + float(self.loss_cfg["lambda_xyz_mm"]) * xyz_loss
            + float(self.loss_cfg["lambda_relative_mm"]) * relative_loss
            + float(self.loss_cfg["lambda_set_mm"]) * set_loss
            + float(self.loss_cfg["lambda_surface_mm"]) * surface_loss
        )
        losses = {
            "total": total_loss,
            "noise": noise_loss,
            "xyz_mm": xyz_loss,
            "relative_mm": relative_loss,
            "set_mm": set_loss,
            "surface_mm": surface_loss,
        }
        stats = {
            "point_error_mm": point_error_mm.mean().detach(),
            "nearest_surface_mm": surface_error_mm.mean().detach(),
            "set_error_mm": set_error_mm.mean().detach(),
            "relative_error_mm": pair_error_mm.mean().detach(),
            "rollout_generated_fraction": torch.stack(generated_tokens).mean().detach(),
            "low_t_fraction": geometry_mask.float().mean().detach(),
        }
        for position in range(num_contacts):
            stats[f"position_{position + 1}_error_mm"] = point_error_mm[
                :, position
            ].mean().detach()
        outputs = {
            "contacts_pred": predicted_set,
            "contacts_target": ordered,
            "timesteps": timesteps,
        }
        return outputs, losses, stats

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
        if project_to_surface:
            raise ValueError(
                "Autoregressive free-XYZ sampling intentionally disables surface projection."
            )
        device = object_pc.device
        batch_size = object_pc.shape[0]
        dc = int(dc or self.denoiser.dc)
        scheduler_cls = DDIMScheduler if sampler == "ddim" else DDPMScheduler
        scheduler = scheduler_cls(
            num_train_timesteps=self.num_diffusion_iters,
            beta_start=float(getattr(self.noise_scheduler.config, "beta_start", 1e-4)),
            beta_end=float(getattr(self.noise_scheduler.config, "beta_end", 0.02)),
            beta_schedule=str(self.noise_scheduler.config.beta_schedule),
            clip_sample=bool(self.noise_scheduler.config.clip_sample),
            prediction_type=self.prediction_type,
        )
        scheduler.set_timesteps(int(num_steps or self.num_diffusion_iters_eval), device=device)

        prefix = object_pc.new_zeros((batch_size, 0, dc))
        object_features = self.denoiser.encode_object_geometry(object_pc=object_pc)
        for _ in range(int(num_contacts)):
            current = torch.randn(batch_size, 1, dc, device=device)
            model_prefix = prefix
            prefix_timesteps = None
            if prefix.shape[1] > 0:
                prefix_timesteps = torch.full(
                    (batch_size, prefix.shape[1]),
                    self.stabilization_timestep,
                    device=device,
                    dtype=torch.long,
                )
                if self.stabilization_timestep > 0:
                    alpha_prefix = self.noise_scheduler.alphas_cumprod.to(device)[
                        self.stabilization_timestep
                    ]
                    # Diffusion Forcing uses deterministic scaling for clean
                    # context during DDIM stabilization.
                    model_prefix = alpha_prefix.sqrt() * prefix
            for timestep in scheduler.timesteps:
                t = torch.full(
                    (batch_size,), int(timestep), device=device, dtype=torch.long
                )
                model_output = self.denoiser(
                    current,
                    t,
                    object_pc,
                    prefix=model_prefix,
                    total_contacts=int(num_contacts),
                    object_features=object_features,
                    prefix_timesteps=prefix_timesteps,
                )
                current = scheduler.step(model_output, timestep, current).prev_sample
            prefix = torch.cat([prefix, current], dim=1)
        return prefix
