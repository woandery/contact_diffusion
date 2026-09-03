"""Transformer denoiser for unordered contact point sets.

The default object encoder is a lightweight local-token PointNet-style MLP so
the project can run without CUDA PointNet++ extensions.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    from pointnet2_ops.pointnet2_modules import PointnetSAModule
except (ImportError, OSError) as exc:
    PointnetSAModule = None
    _POINTNET_IMPORT_ERROR = exc
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
    """PointNet++ local-token encoder for object point clouds."""

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
                "object_encoder_type='pointnet' requires the PointNet++ extension. "
                "Run scripts/install_pointnet2_ops.sh from this repository, "
                "or use object_encoder_type='simple_pointnet'."
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
            out = module(xyz, features)
            if len(out) == 2:
                xyz, features = out
            elif len(out) == 4:
                xyz, _, features, _ = out
            else:
                raise RuntimeError(f"Unexpected PointNet++ SA output length: {len(out)}")
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
        include_object_xyz_in_tokens: bool = False,
        use_set_token: bool = False,
        use_set_token_film: bool = False,
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
        self.include_object_xyz_in_tokens = bool(include_object_xyz_in_tokens)
        self.use_set_token = bool(use_set_token)
        self.use_set_token_film = bool(use_set_token_film)
        if self.use_set_token_film and not self.use_set_token:
            raise ValueError("use_set_token_film requires use_set_token=True")

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
        self.object_xyz_proj = (
            nn.Linear(3, self.d_model)
            if self.include_object_xyz_in_tokens
            else None
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
        if self.use_set_token:
            # The token is dynamic, not just a fixed learned constant: at every
            # denoising step it summarizes both the current noisy contact set
            # and the conditioned object.  It then participates in every
            # self/cross-attention block as a shared grasp-mode state.
            self.set_token_base = nn.Parameter(torch.empty(1, 1, self.d_model))
            nn.init.normal_(self.set_token_base, mean=0.0, std=0.02)
            self.set_token_init = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
            if self.use_set_token_film:
                self.set_token_film_norms = nn.ModuleList(
                    [nn.LayerNorm(self.d_model) for _ in range(int(num_layers))]
                )
                self.set_token_film = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(self.d_model),
                            nn.SiLU(),
                            nn.Linear(self.d_model, 2 * self.d_model),
                        )
                        for _ in range(int(num_layers))
                    ]
                )
                # Preserve the warm-started network at initialization.  The
                # new shared conditioning path is learned progressively.
                for film in self.set_token_film:
                    nn.init.zeros_(film[-1].weight)
                    nn.init.zeros_(film[-1].bias)
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
            include_object_xyz_in_tokens=getattr(
                cfg, "include_object_xyz_in_tokens", False
            ),
            use_set_token=getattr(cfg, "use_set_token", False),
            use_set_token_film=getattr(cfg, "use_set_token_film", False),
            activation=getattr(cfg, "activation", "GELU"),
        )

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
        tokens = self.object_proj(tokens)
        if self.object_xyz_proj is not None and xyz is not None:
            tokens = tokens + self.object_xyz_proj(xyz)
        return tokens, xyz

    def forward(
        self,
        contacts_t: torch.Tensor,
        timesteps: torch.Tensor,
        object_pc: Optional[torch.Tensor] = None,
        object_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if contacts_t.ndim != 3:
            raise ValueError(f"Expected contacts_t [B, n, dc], got {contacts_t.shape}")
        if contacts_t.shape[-1] != self.dc:
            raise ValueError(f"Expected dc={self.dc}, got {contacts_t.shape[-1]}")

        batch_size, n, _ = contacts_t.shape
        device = contacts_t.device
        n_tensor = torch.full((batch_size,), n, device=device, dtype=torch.long)

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

        contact_xyz = contacts_t[..., :3]
        if self.use_set_token:
            set_context = z.mean(dim=1, keepdim=True)
            if obj_tokens is not None:
                set_context = set_context + obj_tokens.mean(dim=1, keepdim=True)
            set_token = self.set_token_base.expand(batch_size, -1, -1)
            set_token = set_token + self.set_token_init(set_context)
            z = torch.cat([set_token, z], dim=1)
            # A centroid is used only to define the SET query's relative bias
            # to object surface tokens. Contact token coordinates are unchanged.
            set_xyz = contact_xyz.mean(dim=1, keepdim=True)
            contact_xyz = torch.cat([set_xyz, contact_xyz], dim=1)

        for block_index, block in enumerate(self.blocks):
            z = block(
                z,
                obj_tokens,
                contact_xyz=contact_xyz,
                object_xyz=obj_xyz,
            )
            if self.use_set_token_film:
                mode = z[:, :1]
                contacts = z[:, 1:]
                shift, scale = self.set_token_film[block_index](mode).chunk(2, dim=-1)
                contacts = contacts + shift + scale * self.set_token_film_norms[
                    block_index
                ](contacts)
                z = torch.cat([mode, contacts], dim=1)
        if self.use_set_token:
            z = z[:, 1:]
        return self.contact_out(self.norm(z))
