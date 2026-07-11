"""
Smoke test for the spectral VAE rebuild.

Verifies shapes, loss computation, backward pass, VRAM, and step timing —
without touching real audio data or starting a training run.

Run from project root:
    python smoke_test.py
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.losses import (MultiScaleMelLoss, adaptive_weight,
                        adv_d_loss, adv_g_loss, feat_match_loss)
from src.vae import MultiSTFTDiscriminator, WaveformVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS = "✓"
FAIL = "✗  FAIL"


def check(name, cond, detail=""):
    mark = PASS if cond else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {mark}  {name}{suffix}")
    if not cond:
        sys.exit(1)


# ---------------------------------------------------------------------------
print("=== Spectral VAE Smoke Test ===\n")

vae  = WaveformVAE().to(device)
disc = MultiSTFTDiscriminator().to(device)
mel  = MultiScaleMelLoss().to(device)
vae.train(); disc.train()

n_vae  = sum(p.numel() for p in vae.parameters()) / 1e6
n_disc = sum(p.numel() for p in disc.parameters()) / 1e6
print(f"  VAE  params : {n_vae:.1f} M")
print(f"  Disc params : {n_disc:.1f} M\n")

# ---------------------------------------------------------------------------
# [1] Shape check — 3-second crop (training regime)
# ---------------------------------------------------------------------------
print("[1] Shape check — 3s crop (VAE_TRAIN_SAMPLES)")
B_crop = 2
x_crop = torch.randn(B_crop, 1, config.VAE_TRAIN_SAMPLES, device=device)

lat_len_crop = config.VAE_TRAIN_SAMPLES // config.VAE_DEC_HOP // config.VAE_DEC_FRAME_UP

with torch.no_grad():
    recon_c, mu_c, lv_c = vae(x_crop)

check("recon shape == input",
      recon_c.shape == x_crop.shape,
      f"{tuple(recon_c.shape)}")
check(f"mu shape == ({B_crop}, {config.VAE_LATENT_DIM}, {lat_len_crop})",
      mu_c.shape == (B_crop, config.VAE_LATENT_DIM, lat_len_crop),
      str(tuple(mu_c.shape)))
check("logvar shape == mu shape",
      lv_c.shape == mu_c.shape)
print()

# ---------------------------------------------------------------------------
# [2] Shape check — full 30-second clip (cache-time regime)
# ---------------------------------------------------------------------------
print("[2] Shape check — full clip (CHUNK_SAMPLES)")
x_full = torch.randn(1, 1, config.CHUNK_SAMPLES, device=device)

with torch.no_grad():
    recon_f, mu_f, lv_f = vae(x_full)

check("recon shape == input",
      recon_f.shape == x_full.shape,
      f"{tuple(recon_f.shape)}")
check(f"mu shape == (1, {config.VAE_LATENT_DIM}, {config.VAE_LATENT_LEN})",
      mu_f.shape == (1, config.VAE_LATENT_DIM, config.VAE_LATENT_LEN),
      str(tuple(mu_f.shape)))
print()

# ---------------------------------------------------------------------------
# [3] Discriminator output format
# ---------------------------------------------------------------------------
print("[3] MS-STFT discriminator forward")
x_d = torch.randn(2, 1, config.VAE_TRAIN_SAMPLES, device=device)

with torch.no_grad():
    d_out = disc(x_d)

check(f"n_discs == {len(config.STFTD_FFTS)}",
      len(d_out) == len(config.STFTD_FFTS),
      str(len(d_out)))
check("each disc returns (logit, feats) tuple",
      all(isinstance(o, tuple) and len(o) == 2 for o in d_out))
check("logit is 4-D (B, 1, F, T)",
      d_out[0][0].ndim == 4,
      str(tuple(d_out[0][0].shape)))
check("feats are non-empty lists",
      all(len(o[1]) > 0 for o in d_out))
print()

# ---------------------------------------------------------------------------
# [4] Full G-step: losses + backward (bf16 autocast)
# ---------------------------------------------------------------------------
print("[4] G-step losses + backward (bf16 autocast)")
x_b   = torch.randn(2, 1, config.VAE_TRAIN_SAMPLES, device=device)
opt_g = torch.optim.AdamW(vae.parameters(),  lr=1e-4)
opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    recon_b, mu_b, lv_b = vae(x_b)

l_mel  = mel(x_b, recon_b)
l_wave = F.l1_loss(recon_b.float(), x_b)
l_kl   = 0.5 * (mu_b.float().pow(2) + lv_b.float().exp() - lv_b.float() - 1).mean()
recon_loss = l_mel + config.VAE_WAVE_WEIGHT * l_wave + config.VAE_KL_WEIGHT * l_kl

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    d_fake    = disc(recon_b)
    d_real_fm = disc(x_b.detach())

l_adv  = adv_g_loss(d_fake)
l_feat = feat_match_loss(d_real_fm, d_fake)
lam    = adaptive_weight(recon_loss, l_adv, vae.last_layer, config.VAE_LAMBDA_MAX)
g_loss = recon_loss + config.VAE_ADV_WEIGHT * lam * l_adv + config.VAE_FEAT_WEIGHT * l_feat

opt_g.zero_grad()
g_loss.backward()
opt_g.step()

check("mel  loss is finite",  l_mel.item()  == l_mel.item(),  f"{l_mel.item():.4f}")
check("wave loss is finite",  l_wave.item() == l_wave.item(), f"{l_wave.item():.4f}")
check("kl   loss is finite",  l_kl.item()   == l_kl.item(),   f"{l_kl.item():.5f}")
check("adv_g is finite",      l_adv.item()  == l_adv.item(),  f"{l_adv.item():.4f}")
check("feat  is finite",      l_feat.item() == l_feat.item(), f"{l_feat.item():.4f}")
check("lambda > 0",           lam.item() > 0,                 f"λ={lam.item():.3f}")
print()

# ---------------------------------------------------------------------------
# [5] Full D-step: backward
# ---------------------------------------------------------------------------
print("[5] D-step losses + backward")

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    d_real2 = disc(x_b)
    d_fake2 = disc(recon_b.detach())

l_d = adv_d_loss(d_real2, d_fake2)
opt_d.zero_grad()
l_d.backward()
opt_d.step()

check("adv_d is finite", l_d.item() == l_d.item(), f"{l_d.item():.4f}")
print()

# ---------------------------------------------------------------------------
# [6] VRAM
# ---------------------------------------------------------------------------
print("[6] VRAM")
vram_peak = torch.cuda.max_memory_allocated() / 1e9
check(f"peak VRAM < 40 GB", vram_peak < 40.0, f"{vram_peak:.2f} GB used")
print()

# ---------------------------------------------------------------------------
# [7] Step timing (10 steps, B=2, 3s crop, bf16)
# ---------------------------------------------------------------------------
print("[7] Step timing (10 steps, B=2, 3s crop, bf16)")
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
for _ in range(10):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        r, m, lv = vae(x_b)
    loss = mel(x_b, r) + F.l1_loss(r.float(), x_b)
    opt_g.zero_grad()
    loss.backward()
    opt_g.step()
torch.cuda.synchronize()
t1 = time.perf_counter()

ms_step = (t1 - t0) / 10 * 1000
steps_hr = 3_600_000 / ms_step
vram_steps = torch.cuda.max_memory_allocated() / 1e9

print(f"  {ms_step:.0f} ms/step  →  {steps_hr:.0f} steps/hr")
print(f"  Peak VRAM during timing: {vram_steps:.2f} GB")
check("step time VRAM < 40 GB", vram_steps < 40.0, f"{vram_steps:.2f} GB")
print()

# ---------------------------------------------------------------------------
print("=== All checks passed ===")
