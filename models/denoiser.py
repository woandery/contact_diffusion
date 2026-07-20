"""Transformer denoiser for unordered contact point sets.

The default object encoder is a lightweight local-token PointNet-style MLP so
the project can run without CUDA PointNet++ extensions.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

try:
    from grasp_gen.models.pointnet.pointnet2_modules import PointnetSAModule
except (ImportError, OSError) as exc:
    sibling_graspgen = Path(__file__).resolve().parents[2] / "GraspGen"
    if sibling_graspgen.exists() and str(sibling_graspgen) not in sys.path:
        sys.path.insert(0, str(sibling_graspgen))
    try:
        from grasp_gen.models.pointnet.pointnet2_modules import PointnetSAModule
    except (ImportError, OSError):
        PointnetSAModule = None
        _POINTNET_IMPORT_ERROR = exc
    else:
        _POINTNET_IMPORT_ERROR = None
else:
    _POINTNET_IMPORT_ERROR = None


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size = x.shape[0]
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb.reshape(batch_size, -1)


class SimplePointCloudEncoder(nn.Module):
    """Small local-token point-cloud encoder.

    It preserves a deterministic subset of spatial tokens instead of collapsing
    the whole object to one global feature.  This keeps the cross-attention path
    intact while avoiding any dependency on PointNet++ CUDA ops.
    """

    def __init__(self, output_embedding_dim: int, input_dim: int = 3, num_tokens: int = 64):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_embedding_dim),
        )

    def forward(self, pc: torch.Tensor) -> dict[str, torch.Tensor]:
        if pc.ndim != 3:
            raise ValueError(f"Expected object point cloud [B, P, C], got {pc.shape}")
        num_tokens = min(self.num_tokens, pc.shape[1])
        idx = torch.linspace(0, pc.shape[1] - 1, num_tokens, device=pc.device).round().long()
        xyz = pc[:, idx, :3]
        return {"tokens": self.point_mlp(pc[:, idx]), "xyz": xyz}


class LocalPointNetPlusPlusEncoder(nn.Module):
    """PointNet++ local-token encoder from the GraspGen contact diffusion model."""

    def __init__(
        self,
        output_embedding_dim: int,
        input_dim: int = 3,
        npoints: Optional[list[int]] = None,
        radii: Optional[list[float]] = None,
        nsamples: Optional[list[int]] = None,
    ):
        super().__init__()
        if PointnetSAModule is None:
            raise ImportError(
                "object_encoder_type='pointnet' requires GraspGen's PointNet++ extension. "
                "Install/activate GraspGen pointnet2_ops, or use object_encoder_type='simple_pointnet'."
            ) from _POINTNET_IMPORT_ERROR

        npoints = list([256, 64, 32] if npoints is None else npoints)
        radii = list([0.02, 0.04, 0.08] if radii is None else radii)
        nsamples = list([64, 128, 64] if nsamples is None else nsamples)
        if not (len(npoints) == len(radii) == len(nsamples) == 3):
            raise ValueError("PointNet++ local encoder expects three npoints/radii/nsamples values.")

        feature_dim = max(0, int(input_dim) - 3)
        mlps = [
            [feature_dim, 64, 128],
            [128, 128, 256],
            [256, 256, output_embedding_dim],
        ]
        self.sa_modules = nn.ModuleList(
            [
                PointnetSAModule(
                    npoint=int(npoints[0]),
                    radius=float(radii[0]),
                    nsample=int(nsamples[0]),
                    mlp=mlps[0],
                    use_xyz=True,
                ),
                PointnetSAModule(
                    npoint=int(npoints[1]),
                    radius=float(radii[1]),
                    nsample=int(nsamples[1]),
                    mlp=mlps[1],
                    use_xyz=True,
                ),
                PointnetSAModule(
                    npoint=int(npoints[2]),
                    radius=float(radii[2]),
                    nsample=int(nsamples[2]),
                    mlp=mlps[2],
                    use_xyz=True,
                ),
            ]
        )

    def forward(self, pc: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.shape[-1] > 3 else None
        for module in self.sa_modules:
            xyz, _, features, _ = module(xyz, features)
        return {"tokens": features.transpose(1, 2).contiguous(), "xyz": xyz}


class ContactDenoisingBlock(nn.Module):
    """Transformer block that preserves contact token cardinality."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "GELU",
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * d_model

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.num_heads = int(num_heads)
        self.relative_position_bias = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_heads),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        contact_tokens: torch.Tensor,
        object_tokens: Optional[torch.Tensor],
        contact_xyz: Optional[torch.Tensor] = None,
        object_xyz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = contact_tokens
        self_out, _ = self.self_attn(z, z, z, need_weights=False)
        z = self.norm1(z + self.dropout(self_out))

        if object_tokens is not None:
            attn_mask = None
            if contact_xyz is not None and object_xyz is not None:
                delta = object_xyz[:, None, :, :] - contact_xyz[:, :, None, :]
                bias = self.relative_position_bias(delta)
                attn_mask = bias.permute(0, 3, 1, 2).reshape(
                    -1, contact_tokens.shape[1], object_tokens.shape[1]
                )
            cross_out, _ = self.cross_attn(
                query=z,
                key=object_tokens,
                value=object_tokens,
                attn_mask=attn_mask,
                need_weights=False,
            )
            z = self.norm2(z + self.dropout(cross_out))
        else:
            z = self.norm2(z)

        z = self.norm3(z + self.dropout(self.ffn(z)))
        return z


class ContactSetDenoiser(nn.Module):
    """Denoise unordered contact sets conditioned on object point clouds."""

    def __init__(
        self,
        dc: int = 3,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_diffusion_iters: int = 1000,
        n_values: Optional[list[int]] = None,
        use_n_embedding: bool = True,
        use_object_cross_attention: bool = True,
        object_encoder: Optional[nn.Module] = None,
        object_feature_dim: Optional[int] = None,
        object_input_dim: int = 3,
        object_encoder_type: str = "simple_pointnet",
        object_num_tokens: int = 64,
        pointnet_local_npoints: Optional[list[int]] = None,
        pointnet_local_radii: Optional[list[float]] = None,
        pointnet_local_nsamples: Optional[list[int]] = None,
        activation: str = "GELU",
    ):
        super().__init__()
        if n_values is None:
            n_values = [2, 3, 5]

        self.dc = int(dc)
        self.d_model = int(d_model)
        self.num_diffusion_iters = int(num_diffusion_iters)
        self.n_values = [int(n) for n in n_values]
        self.use_n_embedding = bool(use_n_embedding)
        self.use_object_cross_attention = bool(use_object_cross_attention)

        self.contact_in = nn.Linear(self.dc, self.d_model)
        self.contact_out = nn.Linear(self.d_model, self.dc)
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(self.d_model),
            nn.Linear(self.d_model, self.d_model * 4),
            nn.Mish(),
            nn.Linear(self.d_model * 4, self.d_model),
        )

        if self.use_n_embedding:
            self.n_to_idx = {int(n): i for i, n in enumerate(self.n_values)}
            self.n_embedding = nn.Embedding(len(self.n_values), self.d_model)
            self.n_fallback = nn.Sequential(
                SinusoidalPosEmb(self.d_model),
                nn.Linear(self.d_model, self.d_model),
            )
        else:
            self.n_to_idx = {}
            self.n_embedding = None
            self.n_fallback = None

        if object_encoder is None:
            if object_encoder_type == "simple_pointnet":
                self.object_encoder = SimplePointCloudEncoder(
                    output_embedding_dim=self.d_model,
                    input_dim=int(object_input_dim),
                    num_tokens=int(object_num_tokens),
                )
            elif object_encoder_type in ("pointnet", "pointnet++", "pointnet_local"):
                self.object_encoder = LocalPointNetPlusPlusEncoder(
                    output_embedding_dim=self.d_model,
                    input_dim=int(object_input_dim),
                    npoints=pointnet_local_npoints,
                    radii=pointnet_local_radii,
                    nsamples=pointnet_local_nsamples,
                )
            else:
                raise ValueError(
                    "Unsupported object_encoder_type="
                    f"{object_encoder_type!r}; expected 'simple_pointnet' or 'pointnet'."
                )
            object_feature_dim = self.d_model
        else:
            self.object_encoder = object_encoder

        if object_feature_dim is None:
            object_feature_dim = self.d_model
        self.object_proj = (
            nn.Identity()
            if int(object_feature_dim) == self.d_model
            else nn.Linear(int(object_feature_dim), self.d_model)
        )

        self.blocks = nn.ModuleList(
            [
                ContactDenoisingBlock(
                    d_model=self.d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)

    @classmethod
    def from_config(cls, cfg):
        return cls(
            dc=cfg.dc,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            ffn_dim=getattr(cfg, "ffn_dim", None),
            dropout=getattr(cfg, "dropout", 0.0),
            num_diffusion_iters=cfg.num_diffusion_iters,
            n_values=list(cfg.n_values),
            use_n_embedding=cfg.use_n_embedding,
            use_object_cross_attention=cfg.use_object_cross_attention,
            object_input_dim=getattr(cfg, "object_input_dim", 3),
            object_encoder_type=getattr(cfg, "object_encoder_type", "simple_pointnet"),
            object_num_tokens=getattr(cfg, "object_num_tokens", 64),
            pointnet_local_npoints=getattr(cfg, "pointnet_local_npoints", None),
            pointnet_local_radii=getattr(cfg, "pointnet_local_radii", None),
            pointnet_local_nsamples=getattr(cfg, "pointnet_local_nsamples", None),
            activation=getattr(cfg, "activation", "GELU"),
        )

    def _num_contacts_tensor(
        self, num_contacts: Union[int, torch.Tensor], batch_size: int, device
    ) -> torch.Tensor:
        if torch.is_tensor(num_contacts):
            n_tensor = num_contacts.to(device=device).long()
            if n_tensor.ndim == 0:
                n_tensor = n_tensor[None].expand(batch_size)
        else:
            n_tensor = torch.full((batch_size,), int(num_contacts), device=device, dtype=torch.long)
        return n_tensor

    def _n_embedding(self, n_tensor: torch.Tensor) -> torch.Tensor:
        if not self.use_n_embedding:
            return torch.zeros(n_tensor.shape[0], self.d_model, device=n_tensor.device)

        idx = torch.full_like(n_tensor, -1)
        for n_value, n_idx in self.n_to_idx.items():
            idx = torch.where(idx.eq(-1) & n_tensor.eq(n_value), n_idx, idx)

        known = idx.ge(0)
        emb = torch.zeros(n_tensor.shape[0], self.d_model, device=n_tensor.device)
        if known.any():
            emb[known] = self.n_embedding(idx[known])
        if (~known).any():
            emb[~known] = self.n_fallback(n_tensor[~known].float())
        return emb

    def encode_object_geometry(
        self,
        object_pc: Optional[torch.Tensor] = None,
        object_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if object_tokens is not None:
            if object_tokens.ndim == 2:
                object_tokens = object_tokens.unsqueeze(1)
            return self.object_proj(object_tokens), None
        if object_pc is None:
            return None, None

        encoded = self.object_encoder(object_pc)
        if isinstance(encoded, dict):
            tokens = encoded.get("tokens")
            xyz = encoded.get("xyz")
            if tokens is None:
                raise ValueError("Object encoder output is missing 'tokens'.")
        else:
            tokens = encoded
            xyz = None

        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        elif tokens.ndim == 3 and tokens.shape[1] == self.d_model:
            tokens = tokens.transpose(1, 2)
        return self.object_proj(tokens), xyz

    def forward(
        self,
        contacts_t: torch.Tensor,
        timesteps: torch.Tensor,
        object_pc: Optional[torch.Tensor] = None,
        num_contacts: Union[int, torch.Tensor, None] = None,
        object_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if contacts_t.ndim != 3:
            raise ValueError(f"Expected contacts_t [B, n, dc], got {contacts_t.shape}")
        if contacts_t.shape[-1] != self.dc:
            raise ValueError(f"Expected dc={self.dc}, got {contacts_t.shape[-1]}")

        batch_size, n, _ = contacts_t.shape
        device = contacts_t.device
        if num_contacts is None:
            num_contacts = n
        n_tensor = self._num_contacts_tensor(num_contacts, batch_size, device)

        if torch.is_tensor(timesteps) and timesteps.ndim == 0:
            timesteps = timesteps[None].expand(batch_size)
        timesteps = timesteps.to(device=device).long()

        z = self.contact_in(contacts_t)
        z = z + self.time_encoder(timesteps).unsqueeze(1)
        z = z + self._n_embedding(n_tensor).unsqueeze(1)

        obj_tokens, obj_xyz = self.encode_object_geometry(object_pc, object_tokens)
        if not self.use_object_cross_attention:
            if obj_tokens is not None:
                z = z + obj_tokens.mean(dim=1, keepdim=True)
            obj_tokens = None

        for block in self.blocks:
            z = block(
                z,
                obj_tokens,
                contact_xyz=contacts_t[..., :3],
                object_xyz=obj_xyz,
            )
        return self.contact_out(self.norm(z))
