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
import torch.utils.checkpoint as cp

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


def _text_attend_mask(feats, drop_mask, text_drop, device):
    """
    (B, T+1) bool attend mask for CrossAttention: real T5 token positions are attendable only
    for samples that (a) actually have a caption and (b) are not being CFG/text-dropped; the
    appended null-token column (see CrossAttention) is attendable ONLY for those being dropped.
    Exactly one of "real tokens" or "the null token" is ever attendable per sample — never both,
    never neither — so every sample has somewhere valid to attend.
    """
    pad     = feats["text_mask"].to(device)                 # (B, T) True = real (non-pad) token
    invalid = feats["text_valid"].to(device) < 0.5          # no caption available for this track
    td      = drop_mask if text_drop is None else text_drop
    drop    = invalid | td.to(device)
    real_attendable = pad & (~drop[:, None])                 # (B, T)
    null_attendable = drop[:, None]                          # (B, 1)
    return torch.cat([real_attendable, null_attendable], dim=1)   # (B, T+1)


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


class FeatureEmbedder(nn.Module):
    """
    Musical-feature conditioning (extract_features.py).

    Global path : tempo (continuous, sinusoidal) + key (12) + mode (2) -> (B, D),
                  summed into the adaLN conditioning vector alongside tags/timestep.
    Chroma path : per-frame (13, L) chroma+energy -> (B, L, D), ADDED TO THE INPUT
                  TOKENS. This is the high-bit signal: it tells the model what
                  harmony/dynamics happen at each frame, which the ~3 tags cannot.

    Both paths honour `drop_mask` (CFG dropout) and a per-sample `valid` flag, so
    tracks with no extracted features fall back to a learned null and still train.
    """
    def __init__(self):
        super().__init__()
        D, Fd = config.DIT_D_MODEL, config.DIT_FEAT_DIM

        if config.DIT_USE_GLOBAL_FEATS:
            self.key_embed  = nn.Embedding(12, Fd)
            self.mode_embed = nn.Embedding(2,  Fd)
            self.global_mlp = nn.Sequential(nn.Linear(Fd, D), nn.SiLU(), nn.Linear(D, D))
            self.global_null = nn.Parameter(torch.zeros(D))
            self.Fd = Fd
            # zero-init the output, same rationale as chroma_proj below: a fine-tune
            # from a feature-less checkpoint must START at that checkpoint's exact
            # behaviour, then learn to use the new conditioning.
            nn.init.zeros_(self.global_mlp[-1].weight)
            nn.init.zeros_(self.global_mlp[-1].bias)

        # Chroma and texture share one injection pattern:  out = gate * proj(x)
        #   - proj is NORMAL-init and gate is ZERO-init (adaLN-Zero: gate the block to zero, never
        #     its contents). Zero-initing BOTH is a deadlock — d(out)/d(proj) is proportional to
        #     gate (=0) and d(out)/d(gate) is proportional to proj(x) (=0), so neither can ever
        #     receive gradient. An entire run trained with texture_gate and |texture_proj| both
        #     pinned at exactly 0.000, contributing nothing at all.
        #   - the zero gate makes a fine-tune from a feature-less checkpoint start at EXACTLY that
        #     checkpoint's behaviour, then dial the contribution up on its own.
        #   - inputs are standardized upstream (compute_feature_stats.py + JamendoDataset), which
        #     is what keeps proj(x) from injecting a constant: see DIT_STANDARDIZE_FEATS.
        #
        # NOTE: there is deliberately NO LayerNorm on these paths. It was added to texture on the
        # theory that an unbounded projection destabilizes the backbone — a theory since disproven
        # (texture was measured INERT for the entire run it supposedly destabilized). LayerNorm is
        # actively HARMFUL here: it normalizes each token to a fixed norm of sqrt(D), which throws
        # away exactly the magnitude information these channels exist to carry (chroma's frame
        # energy, texture's flatness/onset strength). Standardized inputs + a gate bound the
        # injection without destroying its scale.
        if config.DIT_USE_CHROMA:
            self.chroma_proj = nn.Linear(config.DIT_CHROMA_IN, D)
            self.chroma_null = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.chroma_proj.weight, std=0.02)
            nn.init.zeros_(self.chroma_proj.bias)
            self.chroma_gate = nn.Parameter(torch.zeros(1))

        if config.DIT_USE_TEXTURE:
            # DELIBERATELY a separate projection rather than widening chroma_proj from 13 to 17
            # inputs: widening changes Linear(13,512)'s shape, which forces a re-init of the
            # chroma weights too.
            self.texture_proj = nn.Linear(config.DIT_TEXTURE_IN, D)
            self.texture_null = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.texture_proj.weight, std=0.02)
            nn.init.zeros_(self.texture_proj.bias)
            self.texture_gate = nn.Parameter(torch.zeros(1))

    def global_cond(self, feats, drop_mask=None):
        """-> (B, D) to be added to the adaLN conditioning vector."""
        device = self.global_null.device
        tempo = feats["tempo"].to(device).float()
        tempo = (tempo.clamp(config.DIT_TEMPO_MIN, config.DIT_TEMPO_MAX) - config.DIT_TEMPO_MIN) \
                / (config.DIT_TEMPO_MAX - config.DIT_TEMPO_MIN)          # -> [0,1]
        h = sinusoidal_embedding(tempo * 1000.0, self.Fd) \
            + self.key_embed(feats["key"].to(device)) \
            + self.mode_embed(feats["mode"].to(device))
        out = self.global_mlp(h)                                          # (B, D)

        replace = (feats["valid"].to(device) < 0.5)
        if drop_mask is not None:
            replace = replace | drop_mask.to(device)
        return torch.where(replace.unsqueeze(1), self.global_null, out)

    def chroma_tokens(self, feats, drop_mask=None):
        """-> (B, L, D) to be added to the input tokens. WHICH NOTES, per frame."""
        device = self.chroma_null.device
        c = feats["chroma"].to(device).float().permute(0, 2, 1)           # (B, L, 13)
        out = self.chroma_gate * self.chroma_proj(c)                      # (B, L, D)

        replace = (feats["valid"].to(device) < 0.5)
        if drop_mask is not None:
            replace = replace | drop_mask.to(device)
        return torch.where(replace[:, None, None], self.chroma_null, out)

    def texture_tokens(self, feats, drop_mask=None):
        """-> (B, L, D). Onset / percussive / centroid / flatness, per frame. HOW NOISY."""
        device = self.texture_null.device
        t = feats["texture"].to(device).float().permute(0, 2, 1)          # (B, L, 4)
        out = self.texture_gate * self.texture_proj(t)                    # (B, L, D)

        replace = (feats["valid"].to(device) < 0.5)
        if drop_mask is not None:
            replace = replace | drop_mask.to(device)
        return torch.where(replace[:, None, None], self.texture_null, out)


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


class CrossAttention(nn.Module):
    """
    Latent tokens (queries) attend to the T5 text token sequence (keys/values) — THE mechanism
    that makes text the primary, high-bandwidth conditioning signal (self-attention + adaLN's
    pooled `cond` vector cannot express "attend to THIS word for THIS frame"; cross-attention can).

    One extra learned NULL token is appended to the key/value sequence. The attend mask (built
    once per forward pass by MusicDiT._text_attend_mask) routes CFG-dropped or caption-less
    samples to attend ONLY to that null token — a genuine learned "no text" state, the same idea
    as chroma_null/texture_null, adapted for a variable-length KV sequence instead of a single
    additive vector.

    q/kv/out are normal-init (default nn.Linear init); only the *gate* in DiTBlock is zero-init.
    This is deliberate — the double-zero-init deadlock (both projection AND gate at zero) already
    cost one full training run on the texture path (see FeatureEmbedder's docstring); the fix
    there was proj=normal-init + gate=zero-init, applied here from the start.
    """
    def __init__(self):
        super().__init__()
        D, H = config.DIT_D_MODEL, config.DIT_HEADS
        self.H = H
        self.q   = nn.Linear(D, D)
        self.kv  = nn.Linear(config.TEXT_DIM, 2 * D)
        self.out = nn.Linear(D, D)
        self.null_token = nn.Parameter(torch.zeros(1, 1, config.TEXT_DIM))

    def forward(self, x, text_emb, attend_mask):
        """
        x          : (B, L, D)          latent tokens, the queries
        text_emb   : (B, T, TEXT_DIM)   T5 hidden states (T = config.TEXT_MAX_TOKENS)
        attend_mask: (B, T+1) bool      True = attendable; already includes the null-token column
        """
        B, L, D = x.shape
        T = text_emb.shape[1]
        kv_in = torch.cat([text_emb, self.null_token.expand(B, 1, -1)], dim=1)  # (B, T+1, TEXT_DIM)

        q  = self.q(x).reshape(B, L, self.H, D // self.H).transpose(1, 2)         # (B,H,L,Dh)
        kv = self.kv(kv_in).reshape(B, T + 1, 2, self.H, D // self.H)
        k, v = kv.permute(2, 0, 3, 1, 4)                                         # (B,H,T+1,Dh)

        # Boolean mask (True = attend), not a float -inf bias: SDPA's fused/efficient backends
        # accept bool masks directly, while an arbitrary float additive bias forces the
        # unoptimized "math" fallback, which explicitly materializes the (B,H,L,T+1) score matrix.
        # Verified equal to the float-bias gradient (cross_gate.grad matched to 6 decimal places)
        # via an A/B dummy-tensor test with the final layer's zero-init broken (a fresh model's
        # zero-init final projection otherwise makes EVERY upstream gradient exactly zero on the
        # first pass regardless of masking approach, which produced a false "gradient is broken"
        # reading the first time this was checked).
        mask = attend_mask[:, None, None, :]                                      # (B,1,1,T+1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)               # (B,H,L,Dh)
        return self.out(y.transpose(1, 2).reshape(B, L, D))


class DiTBlock(nn.Module):
    """adaLN-Zero DiT block: self-attn + cross-attn (text) + FFN.

    Self-attn and FFN are modulated by the pooled `cond` vector (adaLN-Zero, 6-way chunk,
    unchanged from the tags/chroma-only architecture). Cross-attention is a THIRD, independently
    gated additive term — plain LayerNorm (no adaLN modulation) before it, a single zero-init
    scalar gate after, exactly the chroma_gate/texture_gate pattern that is already proven stable
    (no LayerNorm on the injected content itself — see FeatureEmbedder's note on why that was
    actively harmful for texture).
    """
    def __init__(self):
        super().__init__()
        D = config.DIT_D_MODEL
        self.norm1 = nn.LayerNorm(D, elementwise_affine=False)
        self.attn  = Attention()

        self.use_text = config.DIT_USE_TEXT
        if self.use_text:
            self.norm_c     = nn.LayerNorm(D, elementwise_affine=False)
            self.cross_attn = CrossAttention()
            self.cross_gate = nn.Parameter(torch.zeros(1))

        self.norm2 = nn.LayerNorm(D, elementwise_affine=False)
        self.ffn   = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        # 6 modulation vectors from cond; zero-init → block is identity at init.
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond, text_emb=None, text_attend=None):
        sh_a, sc_a, g_a, sh_f, sc_f, g_f = self.adaLN(cond).chunk(6, dim=-1)
        x = x + g_a[:, None] * self.attn(modulate(self.norm1(x), sh_a, sc_a))
        if self.use_text and text_emb is not None:
            x = x + self.cross_gate * self.cross_attn(self.norm_c(x), text_emb, text_attend)
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

        self.use_text  = config.DIT_USE_TEXT
        self.use_feats = (config.DIT_USE_GLOBAL_FEATS or config.DIT_USE_CHROMA
                          or config.DIT_USE_TEXTURE)
        if self.use_feats:
            self.feat_embed = FeatureEmbedder()

        # Fixed sinusoidal position embedding for the 645 latent frames (no params).
        self.register_buffer("pos_embed", sinusoidal_embedding(torch.arange(L).float(), D))

    def forward(self, z, t, tag_lists, drop_mask=None, feats=None, chroma_drop=None,
                text_drop=None):
        """
        z : (B, Ld, L)  t : (B,)  tag_lists : list[list[int]]  drop_mask : (B,) bool | None
        feats : dict from JamendoDataset(load_features=..., load_text=...) | None — may hold
            chroma/texture AND/OR text_emb/text_mask/text_valid, folded into one dict (see
            src/dataset.py); each key is used only if present, so either conditioning source can
            be loaded independently.
        chroma_drop : (B,) bool | None — drops ONLY chroma+texture, independently of drop_mask.
        text_drop   : (B,) bool | None — drops ONLY text, independently of drop_mask
            (config.DIT_TEXT_DROPOUT). Both default to drop_mask when not given, so CFG's
            unconditional branch drops everything by default, but each secondary signal can ALSO
            be dropped on its own — the model sees p(z|tags), p(z|tags,text),
            p(z|tags,text,chroma), etc., not just "everything or nothing".
        returns velocity (B, Ld, L)
        """
        x    = self.in_proj(z.permute(0, 2, 1)) + self.pos_embed          # (B, L, D)
        cond = self.t_embed(t) + self.tag_embed(tag_lists, drop_mask)     # (B, D)

        text_emb = text_attend = None
        if self.use_text and feats is not None and "text_emb" in feats:
            text_emb    = feats["text_emb"].to(x.device).float()          # (B, T, TEXT_DIM)
            text_attend = _text_attend_mask(feats, drop_mask, text_drop, x.device)

        if self.use_feats and feats is not None:
            if config.DIT_USE_GLOBAL_FEATS and "tempo" in feats:
                cond = cond + self.feat_embed.global_cond(feats, drop_mask)
            cd = drop_mask if chroma_drop is None else chroma_drop
            if config.DIT_USE_CHROMA and "chroma" in feats:
                x = x + self.feat_embed.chroma_tokens(feats, cd)          # WHICH NOTES, per frame
            if config.DIT_USE_TEXTURE and "texture" in feats:
                # Dropped on the SAME mask as chroma, so CFG's unconditional branch stays
                # genuinely unconditional (otherwise guidance would leak content through it).
                x = x + self.feat_embed.texture_tokens(feats, cd)         # HOW NOISY, per frame

        # Gradient checkpointing, in PAIRS of blocks rather than individually. Checkpointing every
        # single block maximizes memory savings but also maximizes recompute cost (one extra
        # forward per checkpoint segment): at batch 768 with per-block checkpointing, the GPU sat
        # at 100% utilization (fully compute-bound, not a data-loading stall) but was slower than
        # needed -- confirmed via nvidia-smi, not a data-loading bottleneck since 100% GPU util
        # means it was compute-bound the whole time.
        # Grouping 2 blocks per checkpoint halves the number of recompute segments (6 vs 12) while
        # still avoiding storing most of the 12-layer activation graph -- memory was at 53GB of an
        # 80GB budget (target ~60GB) with full per-block checkpointing, i.e. real room to trade
        # some of that savings back for speed.
        checkpointing = self.training and torch.is_grad_enabled()
        group = config.DIT_CKPT_GROUP_SIZE
        blocks = self.blocks
        if checkpointing:
            def run_group(x_in, blk_group, cond, text_emb, text_attend):
                for blk in blk_group:
                    x_in = blk(x_in, cond, text_emb=text_emb, text_attend=text_attend)
                return x_in
            for i in range(0, len(blocks), group):
                blk_group = blocks[i:i + group]
                x = cp.checkpoint(run_group, x, blk_group, cond, text_emb, text_attend,
                                  use_reentrant=False)
        else:
            for block in blocks:
                x = block(x, cond, text_emb=text_emb, text_attend=text_attend)
        return self.final(x, cond).permute(0, 2, 1)                       # (B, Ld, L)

    @torch.no_grad()
    def sample(self, tag_lists, steps=None, cfg_scale=None, device=None, feats=None,
               solver=None):
        """
        ODE sampling with CFG (cond + uncond batched in one pass).

        feats: optional dict — text_emb/text_mask (the primary, text-primary redesign path) and/or
        musical features (tempo/key/chroma/texture, the secondary reference-track path). Text_emb
        is generated at call time from the live user prompt via a frozen T5 encoder (see
        generate.py) and passed in through the SAME dict as any reference-track features. The
        conditional half of the CFG batch sees everything present; the unconditional half is
        force-dropped, so guidance pushes *toward* the requested text/musical content.

        solver: "euler" (1st order) or "heun" (2nd order). Heun takes an Euler step, then
        re-evaluates the velocity at the predicted endpoint and averages the two — halving
        the integration error for 2x the function evaluations per step, so at half the steps
        it is the same cost and strictly more accurate.
        """
        steps     = steps     or config.EULER_STEPS
        cfg_scale = cfg_scale or config.CFG_SCALE
        solver    = solver    or config.ODE_SOLVER
        device    = device    or next(self.parameters()).device
        B         = len(tag_lists)
        Ld, L     = config.VAE_LATENT_DIM, config.VAE_LATENT_LEN
        null_tags = [[] for _ in range(B)]

        feats2 = drop2 = None
        if feats is not None and self.use_feats:
            feats2 = {k: torch.cat([v, v], dim=0) for k, v in feats.items()}
            # first half conditional (keep feats), second half unconditional (drop feats)
            drop2 = torch.cat([torch.zeros(B, dtype=torch.bool),
                               torch.ones(B,  dtype=torch.bool)]).to(device)

        def velocity(z, t_scalar):
            t  = torch.full((B,), t_scalar, device=device)
            z2 = torch.cat([z, z], dim=0)
            t2 = torch.cat([t, t], dim=0)
            # drop2 force-drops everything (tags, globals, chroma, texture, text) on the uncond half.
            v2 = self(z2, t2, tag_lists + null_tags, drop_mask=drop2, feats=feats2,
                      chroma_drop=drop2, text_drop=drop2)
            v_cond, v_uncond = v2.chunk(2, dim=0)
            return v_uncond + cfg_scale * (v_cond - v_uncond)

        z  = torch.randn(B, Ld, L, device=device)
        dt = 1.0 / steps
        self.eval()
        for i in range(steps):
            t = i / steps
            v = velocity(z, t)
            if solver == "heun" and i < steps - 1:
                z_pred = z + dt * v                       # Euler predictor
                v2_    = velocity(z_pred, t + dt)         # velocity at the predicted endpoint
                z = z + dt * 0.5 * (v + v2_)              # trapezoidal corrector
            else:
                z = z + dt * v
        return z
