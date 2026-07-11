"""
MusicDiT — Diffusion Transformer for music latents (flow matching).

Input:  (B, VAE_LATENT_DIM, VAE_LATENT_LEN) noisy latent + timestep + tag list
Output: (B, VAE_LATENT_DIM, VAE_LATENT_LEN) predicted velocity

Training objective (linear flow matching):
  z_t     = (1-t)*z_noise + t*z_data
  target  = z_data - z_noise              (constant velocity along the linear path)
  loss    = MSE(model(z_t, t, tags), target)

Inference: Euler ODE from z~N(0,I) at t=0 toward data at t=1, with CFG.

Design notes (why this is efficient for the 32×645 latent):
  - Attention via F.scaled_dot_product_attention → flash kernel on A100, never
    materializes the 645×645 score matrix (the dominant memory cost at this L).
  - Tag conditioning is fully vectorized (one embedding lookup + scatter-mean),
    so there are no per-sample host→device syncs in the training hot loop.
  - adaLN-Zero everywhere (blocks + final layer): every block is identity at init,
    which is what makes deep DiTs train stably without warmup tricks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ── Helpers ──────────────────────────────────────────────────────────────────

def sinusoidal_embedding(pos, dim):
    """pos: (N,) float → (N, dim) sinusoidal features."""
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=pos.device) / half
    )
    x = pos[:, None].float() * freqs[None]          # (N, half)
    return torch.cat([x.sin(), x.cos()], dim=-1)    # (N, dim)


def modulate(x, shift, scale):
    """adaLN modulation. x: (B, L, D)  shift/scale: (B, D)."""
    return x * (1 + scale[:, None]) + shift[:, None]


# ── Conditioning embedders ───────────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """t ∈ [0,1]  (B,)  →  (B, D).  t is scaled ×1000 so the sinusoids spread out."""
    def __init__(self):
        super().__init__()
        H, D = config.DIT_T_EMBED_DIM, config.DIT_D_MODEL
        self.mlp = nn.Sequential(nn.Linear(H, D), nn.SiLU(), nn.Linear(D, D))
        self.H = H

    def forward(self, t):
        return self.mlp(sinusoidal_embedding(t * 1000.0, self.H))


class TagEmbedder(nn.Module):
    """
    Multi-label tag conditioning (vectorized).

    Each active tag → a learned Td-vector; mean-pooled per sample → projected to D.
    A learnable `null` vector is used for (a) empty tag lists and (b) CFG dropout.
    """
    def __init__(self):
        super().__init__()
        D, Td = config.DIT_D_MODEL, config.DIT_TAG_DIM
        self.embed = nn.Embedding(config.DIT_VOCAB_SIZE, Td)
        self.proj  = nn.Linear(Td, D)
        self.null  = nn.Parameter(torch.zeros(D))

    def forward(self, tag_lists, drop_mask=None):
        """
        tag_lists : list[list[int]] of length B
        drop_mask : (B,) bool — True → replace with null (CFG training dropout)
        returns   : (B, D)
        """
        device = self.null.device
        B      = len(tag_lists)
        Td     = config.DIT_TAG_DIM

        counts = torch.tensor([len(t) for t in tag_lists], device=device)   # (B,)
        flat   = [tag for tags in tag_lists for tag in tags]                # all tags, flat

        summed = torch.zeros(B, Td, device=device)
        if flat:
            flat_idx  = torch.tensor(flat, dtype=torch.long, device=device)
            batch_idx = torch.repeat_interleave(torch.arange(B, device=device), counts)
            summed.index_add_(0, batch_idx, self.embed(flat_idx))           # scatter-sum
        mean = summed / counts.clamp(min=1).unsqueeze(1)                    # scatter-mean
        out  = self.proj(mean)                                             # (B, D)

        # Empty tag lists (count==0) and CFG-dropped samples → the null vector.
        replace = (counts == 0)
        if drop_mask is not None:
            replace = replace | drop_mask.to(device)
        return torch.where(replace.unsqueeze(1), self.null, out)


# ── Transformer block ────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head self-attention using the fused flash kernel (SDPA)."""
    def __init__(self):
        super().__init__()
        D, H     = config.DIT_D_MODEL, config.DIT_HEADS
        self.H   = H
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.H, D // self.H)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each (B, H, L, Dh)
        y = F.scaled_dot_product_attention(q, k, v)   # flash → (B, H, L, Dh)
        y = y.transpose(1, 2).reshape(B, L, D)
        return self.out(y)


class DiTBlock(nn.Module):
    """adaLN-Zero DiT block: attention + FFN, both gated by the conditioning."""
    def __init__(self):
        super().__init__()
        D = config.DIT_D_MODEL
        self.norm1 = nn.LayerNorm(D, elementwise_affine=False)
        self.attn  = Attention()
        self.norm2 = nn.LayerNorm(D, elementwise_affine=False)
        self.ffn   = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        # 6 modulation vectors from cond; zero-init → block is identity at init.
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond):
        sh_a, sc_a, g_a, sh_f, sc_f, g_f = self.adaLN(cond).chunk(6, dim=-1)
        x = x + g_a[:, None] * self.attn(modulate(self.norm1(x), sh_a, sc_a))
        x = x + g_f[:, None] * self.ffn(modulate(self.norm2(x), sh_f, sc_f))
        return x


class FinalLayer(nn.Module):
    """Canonical DiT output head: cond-modulated LayerNorm → zero-init projection."""
    def __init__(self, out_dim):
        super().__init__()
        D = config.DIT_D_MODEL
        self.norm  = nn.LayerNorm(D, elementwise_affine=False)
        self.proj  = nn.Linear(D, out_dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        for m in (self.proj, self.adaLN[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, cond):
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


# ── MusicDiT ─────────────────────────────────────────────────────────────────

class MusicDiT(nn.Module):
    """
    forward(z, t, tag_lists, drop_mask) → predicted velocity (B, Ld, L)
    sample(tag_lists)                   → generated latents  (B, Ld, L)
    """
    def __init__(self):
        super().__init__()
        D  = config.DIT_D_MODEL
        Ld = config.VAE_LATENT_DIM   # 32
        L  = config.VAE_LATENT_LEN   # 645

        self.in_proj   = nn.Linear(Ld, D)
        self.t_embed   = TimestepEmbedder()
        self.tag_embed = TagEmbedder()
        self.blocks    = nn.ModuleList([DiTBlock() for _ in range(config.DIT_LAYERS)])
        self.final     = FinalLayer(Ld)

        # Fixed sinusoidal position embedding for the 645 latent frames (no params).
        self.register_buffer("pos_embed", sinusoidal_embedding(torch.arange(L).float(), D))

    def forward(self, z, t, tag_lists, drop_mask=None):
        """
        z : (B, Ld, L)  t : (B,)  tag_lists : list[list[int]]  drop_mask : (B,) bool | None
        returns velocity (B, Ld, L)
        """
        x    = self.in_proj(z.permute(0, 2, 1)) + self.pos_embed          # (B, L, D)
        cond = self.t_embed(t) + self.tag_embed(tag_lists, drop_mask)     # (B, D)
        for block in self.blocks:
            x = block(x, cond)
        return self.final(x, cond).permute(0, 2, 1)                       # (B, Ld, L)

    @torch.no_grad()
    def sample(self, tag_lists, steps=None, cfg_scale=None, device=None):
        """Euler ODE sampling with CFG (cond + uncond batched in one pass)."""
        steps     = steps     or config.EULER_STEPS
        cfg_scale = cfg_scale or config.CFG_SCALE
        device    = device    or next(self.parameters()).device
        B         = len(tag_lists)
        Ld, L     = config.VAE_LATENT_DIM, config.VAE_LATENT_LEN
        null_tags = [[] for _ in range(B)]

        z  = torch.randn(B, Ld, L, device=device)
        dt = 1.0 / steps
        self.eval()
        for i in range(steps):
            t     = torch.full((B,), i / steps, device=device)
            z2    = torch.cat([z, z], dim=0)
            t2    = torch.cat([t, t], dim=0)
            v2    = self(z2, t2, tag_lists + null_tags)
            v_cond, v_uncond = v2.chunk(2, dim=0)
            z     = z + dt * (v_uncond + cfg_scale * (v_cond - v_uncond))
        return z
