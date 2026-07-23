"""
Diagnose WHY DiT-generated audio is muddy while VAE reconstruction is decent.

Central hypothesis: the VAE was trained with a very low KL weight (1e-4), which
maximizes reconstruction fidelity but leaves the latent space poorly regularized
toward N(0,I). Such a latent space can be "fragile" for generation: the decoder
only ever saw the exact latents its encoder produced, so a generated latent that
is even slightly off-manifold decodes to artifacts (often HF noise -> "mud"),
even though the DiT's training loss looks fine.

Three tests, each isolating a different suspect:

  1. LATENT DISTRIBUTION  — how far from N(0,I) is the real latent? (kurtosis,
     inter-channel correlation, temporal correlation). Far-from-Gaussian or
     highly-correlated latents are harder for a diffusion/flow model AND signal
     a weakly-regularized space.

  2. DECODER ROBUSTNESS   — take a REAL latent, add Gaussian noise at increasing
     levels, decode, measure degradation. If small perturbations already cause
     large SNR drop / HF blow-up, the decoder is fragile and the latent space is
     the bottleneck (not the DiT). Calibrated against the DiT's own error level.

  3. LATENT SMOOTHNESS    — decode the midpoint of two real latents. A smooth,
     generative-friendly space decodes interpolations to plausible audio; a
     fragile space decodes them to mush. Directly relevant because a flow model
     spends most of its trajectory in between real data points.

Usage:  python diagnose_latent_space.py
"""

import glob
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from src.dataset import JamendoDataset, collate_fn
from src.vae import WaveformVAE

VAE_CKPT = "checkpoints/vae/vae_step0200000.pt"
N_CLIPS  = 16
device   = torch.device("cuda")

run_dir = os.path.join("results", "latent_diag", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)


def snr_db(ref, x):
    return 10.0 * torch.log10(ref.pow(2).mean() / ((ref - x).pow(2).mean() + 1e-12)).item()


def hf_ratio(audio, sr=config.SAMPLE_RATE, cutoff=4000):
    win = torch.hann_window(1024, device=audio.device)
    S = torch.stft(audio.float(), 1024, 256, 1024, win, return_complex=True).abs().pow(2)
    freqs = torch.linspace(0, sr / 2, S.shape[0], device=audio.device)
    return (S[freqs > cutoff].sum() / S.sum().clamp(min=1e-12)).item()


# ── Load VAE + a batch of real audio ────────────────────────────────────────
vae = WaveformVAE().to(device)
vae.load_state_dict(torch.load(VAE_CKPT, map_location=device, weights_only=False)["vae"])
vae.eval(); vae.requires_grad_(False)

ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH)
dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)
torch.manual_seed(0)
clips = []
for clip, tags in dl:
    clips.append(clip)
    if len(clips) == N_CLIPS:
        break
audio = torch.cat(clips, dim=0).to(device)                      # (N, 1, 660480)

with torch.no_grad():
    mu, logvar = vae.encode(audio)                              # (N, 32, 645)
    z = mu.float()

log = open(os.path.join(run_dir, "report.txt"), "w")
def out(s=""):
    print(s); log.write(s + "\n")


# ── TEST 1: latent distribution vs N(0,I) ───────────────────────────────────
out("=" * 70)
out("TEST 1 — LATENT DISTRIBUTION (real encoded latents vs. N(0,I) prior)")
out("=" * 70)
flat = z.permute(1, 0, 2).reshape(z.shape[1], -1)              # (32, N*645)
per_ch_mean = flat.mean(1)
per_ch_std  = flat.std(1)
# excess kurtosis (0 for Gaussian; high = spiky/heavy-tailed)
zc = (flat - per_ch_mean[:, None]) / per_ch_std[:, None]
per_ch_kurt = (zc.pow(4).mean(1) - 3.0)
out(f"per-channel mean   : min {per_ch_mean.min():+.3f}  max {per_ch_mean.max():+.3f}  (N(0,1) -> 0)")
out(f"per-channel std    : min {per_ch_std.min():.3f}  max {per_ch_std.max():.3f}  (ratio {per_ch_std.max()/per_ch_std.min():.2f})")
out(f"excess kurtosis    : min {per_ch_kurt.min():+.2f}  max {per_ch_kurt.max():+.2f}  mean {per_ch_kurt.mean():+.2f}  (Gaussian -> 0)")

# inter-channel correlation (off-diagonal magnitude)
C = torch.corrcoef(flat)                                        # (32, 32)
off = C[~torch.eye(32, dtype=bool, device=C.device)].abs()
out(f"inter-channel corr : mean |rho| {off.mean():.3f}  max |rho| {off.max():.3f}  (independent -> 0)")

# temporal correlation: adjacent-frame correlation per channel
z_t0 = z[:, :, :-1].reshape(z.shape[1], -1)
z_t1 = z[:, :, 1:].reshape(z.shape[1], -1)
temporal_corr = F.cosine_similarity(
    (z_t0 - z_t0.mean(1, keepdim=True)),
    (z_t1 - z_t1.mean(1, keepdim=True)), dim=1).mean()
out(f"adjacent-frame corr: {temporal_corr:.3f}  (white/independent -> 0, smooth -> high)")
out()
out("Reading: latents far from unit-variance / high-kurtosis / correlated are")
out("harder for flow matching and indicate a weakly-regularized (KL=1e-4) space.")
out()


# ── TEST 2: decoder robustness to latent perturbation ───────────────────────
out("=" * 70)
out("TEST 2 — DECODER ROBUSTNESS (decode real latent + noise; is it fragile?)")
out("=" * 70)
with torch.no_grad():
    clean = vae.decode(z).float()
clean_hf = np.mean([hf_ratio(clean[i, 0]) for i in range(N_CLIPS)])
out(f"clean decode HF-ratio: {clean_hf:.4f}   (real audio HF-ratio ~0.0198)")
out()
out(f"{'noise sigma':>14} | {'SNR vs clean':>13} | {'HF-ratio':>9} | {'HF vs clean':>11}")
per_ch_std_col = per_ch_std.view(1, -1, 1)
for sigma in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
    torch.manual_seed(1)
    noise = torch.randn_like(z) * per_ch_std_col * sigma       # noise scaled per-channel
    with torch.no_grad():
        pert = vae.decode(z + noise).float()
    snr = np.mean([snr_db(clean[i, 0], pert[i, 0]) for i in range(N_CLIPS)])
    hf  = np.mean([hf_ratio(pert[i, 0]) for i in range(N_CLIPS)])
    out(f"{sigma:>14.2f} | {snr:>10.2f} dB | {hf:>9.4f} | {hf/clean_hf:>10.2f}x")
out()
out("Reading: if a small sigma (0.1-0.3 x per-channel std) already tanks SNR and")
out("inflates HF-ratio, the decoder is fragile -> generated latents that are even")
out("slightly off will decode muddy REGARDLESS of DiT training. The DiT's own")
out("generated latents sit at std-ratio ~0.91 (i.e. a real, non-trivial offset).")
out()


# ── TEST 3: latent-space smoothness (interpolation) ─────────────────────────
out("=" * 70)
out("TEST 3 — LATENT SMOOTHNESS (decode midpoint between two real latents)")
out("=" * 70)
with torch.no_grad():
    idx_a = torch.arange(0, N_CLIPS // 2)
    idx_b = torch.arange(N_CLIPS // 2, N_CLIPS)
    za, zb = z[idx_a], z[idx_b]
    dec_a = vae.decode(za).float()
    for alpha in [0.0, 0.25, 0.5]:
        zmid = (1 - alpha) * za + alpha * zb
        dec_mid = vae.decode(zmid).float()
        hf = np.mean([hf_ratio(dec_mid[i, 0]) for i in range(len(idx_a))])
        # SNR of interpolate vs endpoint A (only meaningful at alpha=0 as sanity, ~inf)
        tag = "endpoint (sanity)" if alpha == 0 else f"interp alpha={alpha}"
        out(f"{tag:>20}: HF-ratio {hf:.4f}  ({hf/clean_hf:.2f}x clean)")
out()
out("Reading: if interpolated latents decode with much higher HF-ratio than")
out("endpoints, the space is non-convex/fragile between data points — exactly")
out("the region a flow model traverses. Plausible interpolations -> space is OK.")
out()

log.close()
print(f"\nFull report saved to {run_dir}/report.txt")
