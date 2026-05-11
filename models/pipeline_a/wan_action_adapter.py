"""
Wan2.1-T2V-1.3B Action Adapter

Injects LAOF latent actions z_{1:T} into Wan2.1's DiT via two forward hooks:

  1. Global conditioning  → added to condition_embedder.time_embedder output
                            dim: time_embed_dim = 1536
  2. Per-frame bias       → added to patch_embedding output (video tokens)
                            dim: hidden_dim = 1536

Architecture of Wan2.1-T2V-1.3B (diffusers):
  patch_embedding      : Conv3d(16, 1536, k=(1,2,2))
  condition_embedder   :
    time_embedder      : Linear(256→1536→1536)
    time_proj          : Linear(1536→9216)
    text_embedder      : Linear(4096→1536→1536)
  blocks[i]            : norm1 / attn1(self) / attn2(cross) / norm2 / ffn / norm3
  norm_out + proj_out
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


# ── ActionAdapter ─────────────────────────────────────────────────────────────

class WanActionAdapter(nn.Module):
    """
    Lightweight MLP that converts z_{1:T} → (global_cond, frame_bias).

    Args:
        la_dim            : latent action dimension (128 from LAOF Stage-2)
        hidden_dim        : Wan2.1-1.3B hidden size = 1536
        temporal_compression: how many frames per latent frame (Wan VAE: 4)
        mlp_hidden        : internal MLP hidden size
    
    Stability (vs「Adapter 一上来把 DiT feature 打散」):
        - LayerNorm(z)（每帧语义空间归一）
        - 注入线性层权重/偏置初始化为 **0**：step0 等价于不改变 backbone；
          训练中再慢慢学偏移。（旧 ckpt：最后一层若为随机初始化仍可加载，
          optimizer 会从当前权重接着训）
    """

    def __init__(
        self,
        la_dim: int = 128,
        hidden_dim: int = 1536,
        temporal_compression: int = 4,
        mlp_hidden: int = 512,
        zero_init_injectors: bool = True,
        use_z_layernorm: bool = True,
    ):
        super().__init__()
        self.temporal_compression = temporal_compression
        self.hidden_dim = hidden_dim
        self.use_z_layernorm = use_z_layernorm

        self.z_norm = nn.LayerNorm(la_dim) if use_z_layernorm else nn.Identity()

        # shared temporal encoder: la_dim → mlp_hidden
        self.temporal_enc = nn.Sequential(
            nn.Linear(la_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
        )

        # global head: mean-pooled → time_embed_dim (1536)
        self.global_head = nn.Linear(mlp_hidden, hidden_dim)

        # per-frame head: (T_lat, mlp_hidden) → (T_lat, hidden_dim)
        self.frame_head = nn.Linear(mlp_hidden, hidden_dim)

        if zero_init_injectors:
            # Small-random init instead of strict zero: breaks the zero-output local optimum
            # while keeping early-training perturbation small enough not to destabilise DiT.
            # std=0.01 gives gc magnitude ~0.01*sqrt(mlp_hidden)≈0.45, well within safe range.
            nn.init.normal_(self.global_head.weight, std=0.01)
            nn.init.zeros_(self.global_head.bias)
            nn.init.normal_(self.frame_head.weight, std=0.01)
            nn.init.zeros_(self.frame_head.bias)

    def temporal_pool(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Pool z_{1:T} (B, T, la_dim) → z_lat (B, T_lat, la_dim)."""
        B, T, D = z_seq.shape
        k = self.temporal_compression
        T_lat = max(1, T // k)
        # reshape and average over k-step windows
        T_trim = T_lat * k
        z_trim = z_seq[:, :T_trim, :]          # (B, T_trim, D)
        z_lat  = z_trim.view(B, T_lat, k, D).mean(dim=2)   # (B, T_lat, D)
        return z_lat

    def forward(self, z_seq: torch.Tensor):
        """
        z_seq : (B, T_orig, la_dim)
        Returns:
            global_cond : (B, hidden_dim=1536)  — added to time_embedder output
            frame_bias  : (B, T_lat, hidden_dim) — added to patch_embedding output
        """
        z_lat       = self.temporal_pool(z_seq)      # (B, T_lat, la_dim)
        z_lat       = self.z_norm(z_lat)
        h           = self.temporal_enc(z_lat)         # (B, T_lat, mlp_hidden)
        global_cond = self.global_head(h.mean(1))      # (B, hidden_dim)
        frame_bias  = self.frame_head(h)               # (B, T_lat, hidden_dim)
        return global_cond, frame_bias


# ── Thread-local hook state ────────────────────────────────────────────────────

_wan_global_cond: Optional[torch.Tensor] = None   # (B, 1536)
_wan_frame_bias:  Optional[torch.Tensor] = None   # (B, T_lat, 1536)


def set_wan_global_cond(x: Optional[torch.Tensor]):
    global _wan_global_cond
    _wan_global_cond = x


def set_wan_frame_bias(x: Optional[torch.Tensor]):
    global _wan_frame_bias
    _wan_frame_bias = x


# ── Hook helpers ─────────────────────────────────────────────────────────────

def _make_time_hook():
    def _hook(module, input, output):
        global _wan_global_cond
        if _wan_global_cond is None:
            return output
        gc = _wan_global_cond.to(dtype=output.dtype, device=output.device)
        if gc.shape == output.shape:
            return output + gc
        return output
    return _hook


def _make_patch_hook():
    def _hook(module, input, output):
        global _wan_frame_bias
        if _wan_frame_bias is None:
            return output
        B, C, T, H, W = output.shape
        fb = _wan_frame_bias.to(dtype=output.dtype, device=output.device)
        T_lat = fb.shape[1]
        if T_lat != T:
            import torch.nn.functional as F
            fb = F.interpolate(
                fb.permute(0, 2, 1).unsqueeze(-1),
                size=(T, 1), mode="nearest"
            ).squeeze(-1).permute(0, 2, 1)
        bias = fb.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
        return output + bias
    return _hook


# ── Hook registration for diffusers WanTransformer3DModel ────────────────────

def register_wan_time_emb_hook(transformer):
    """For diffusers WanTransformer3DModel."""
    return transformer.condition_embedder.time_embedder.register_forward_hook(_make_time_hook())


def register_wan_patch_emb_hook(transformer):
    """For diffusers WanTransformer3DModel."""
    return transformer.patch_embedding.register_forward_hook(_make_patch_hook())


# ── Hook registration for original WanModel (wan library) ────────────────────

def register_wan_orig_time_hook(model):
    """For original wan.modules.model.WanModel — hooks on model.time_embedding."""
    return model.time_embedding.register_forward_hook(_make_time_hook())


def register_wan_orig_patch_hook(model):
    """For original wan.modules.model.WanModel — hooks on model.patch_embedding."""
    return model.patch_embedding.register_forward_hook(_make_patch_hook())
