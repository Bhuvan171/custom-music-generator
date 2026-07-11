"""
Single-batch overfit sanity test for the MusicDiT (flow matching).

Mirrors overfit_test.py (the VAE test), but for the generative stage:
memorize a tiny FIXED batch of (latent, tags) pairs and verify the DiT can
reproduce it END-TO-END — noise → sampled latent → decoded audio.

Pipeline is tested RAW (no latent scaling), exactly as cache_latents.py /
train_dit.py consume latents today. The latent mean/std are measured and
reported as a diagnostic only (not applied), so we learn whether latent scale
matters for the real run without altering the tested path.

Metrics:
  - fm_loss    : flow-matching MSE on the fixed batch. Read as a FALLING trend.
  - latent_cos : cosine between a SAMPLED latent (from noise, cfg=1) and the true
                 latent. The core signal — an overfit conditional flow maps all
                 noise to the memorized target, so this climbs toward ~1.0.
  - SNR        : end-to-end, after decoding the sampled latent through the VAE.
                 The VAE's own recon SNR is the ceiling the DiT cannot beat.

PASS at final latent_cos > 0.9 (DiT memorized the conditional mapping).

Results (metrics CSV + real/vae_recon/dit_gen WAVs + spectrograms + summary) are
written to a timestamped dir under config.DIT_OVERFIT_DIR.
"""

import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import config
from src.dataset import JamendoDataset, collate_fn
from src.dit import MusicDiT
from src.vae import WaveformVAE

VAE_CKPT     = "checkpoints/vae/vae_step0200000.pt"
BATCH        = 4        # small batch → clean, fast overfit signal
STEPS        = 3000     # flow matching memorizes a tiny batch quickly
LR           = 5e-4     # high LR to overfit fast (DiT is more LR-sensitive than the conv VAE)
SAMPLE_EVERY = 500      # full Euler sampling is the expensive eval; run it periodically
LOG_EVERY    = 100
CFG          = 1.0      # pure conditional — cleanest memorization check (uncond path unused)

device = torch.device("cuda")

run_dir = os.path.join(config.DIT_OVERFIT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)


def snr_db(x, y):
    # time-domain signal-to-noise ratio in dB (higher is better) — same as overfit_test.py
    sig   = x.pow(2).mean()
    noise = (x - y).pow(2).mean()
    return 10.0 * torch.log10(sig / (noise + 1e-12))


def latent_cosine(a, b):
    # mean per-sample cosine similarity between flattened latents (B, ...)
    af = a.flatten(1)
    bf = b.flatten(1)
    return F.cosine_similarity(af, bf, dim=1).mean()


# ── Fixed batch: BATCH clips WITH non-empty tags ─────────────────────────────
ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH)
dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)

audio_list, tag_lists = [], []
for clip, tags in dl:
    if tags[0]:                       # skip empty-tag clips → each condition is informative
        audio_list.append(clip)       # (1, 1, 660480)
        tag_lists.append(tags[0])
    if len(audio_list) == BATCH:
        break

audio = torch.cat(audio_list, dim=0).to(device)   # (BATCH, 1, 660480) — fixed forever
print(f"Fixed batch: {len(tag_lists)} clips, tag counts = {[len(t) for t in tag_lists]}")

# ── Frozen VAE → fixed target latents (raw mu, no scaling) ───────────────────
vae  = WaveformVAE().to(device)
ckpt = torch.load(VAE_CKPT, map_location=device, weights_only=False)
vae.load_state_dict(ckpt["vae"])
vae.eval()
vae.requires_grad_(False)

with torch.no_grad():
    z1, _ = vae.encode(audio)         # (BATCH, 32, 645) — fixed target, RAW
z1 = z1.float()

# Diagnostic only — NOT applied anywhere
lat_mean = z1.mean().item()
lat_std  = z1.std().item()
ch_std   = z1.std(dim=(0, 2))         # per-channel std
print(f"Loaded VAE from {VAE_CKPT}")
print(f"[diagnostic] latent mean={lat_mean:+.4f}  std={lat_std:.4f}  "
      f"per-channel std range=[{ch_std.min():.3f}, {ch_std.max():.3f}]")

# ── DiT + optimizer ──────────────────────────────────────────────────────────
dit = MusicDiT().to(device)
opt = torch.optim.AdamW(dit.parameters(), lr=LR)

print(f"DiT params: {sum(p.numel() for p in dit.parameters()) / 1e6:.1f}M")
print(f"Overfitting on 1 batch  (B={BATCH}, {STEPS} steps, lr={LR}, cfg={CFG})")
print(f"Results dir: {run_dir}")
print(f"{'step':>6}  {'fm_loss':>9}  {'lat_cos':>8}  {'lat_mse':>9}")

rows = []   # {step, fm_loss, latent_cos, latent_mse}
for step in range(1, STEPS + 1):
    # fp32 (no autocast): tiny sanity test prioritises numerical robustness.
    t        = torch.rand(BATCH, device=device)
    z0       = torch.randn_like(z1)
    z_t      = (1 - t[:, None, None]) * z0 + t[:, None, None] * z1
    v_target = z1 - z0

    v_pred = dit(z_t, t, tag_lists)                 # no CFG dropout during overfit
    loss   = F.mse_loss(v_pred, v_target)

    opt.zero_grad()
    loss.backward()
    clip_grad_norm_(dit.parameters(), 1.0)
    opt.step()

    if torch.isnan(loss):
        print(f"  NaN in loss at step {step}")
        break

    if step % LOG_EVERY == 0:
        lat_cos = lat_mse = None
        if step % SAMPLE_EVERY == 0:
            dit.eval()
            z_gen   = dit.sample(tag_lists, steps=config.EULER_STEPS, cfg_scale=CFG, device=device)
            dit.train()
            lat_cos = latent_cosine(z_gen.float(), z1).item()
            lat_mse = F.mse_loss(z_gen.float(), z1).item()
        rows.append({"step": step, "fm_loss": loss.item(),
                     "latent_cos": lat_cos, "latent_mse": lat_mse})
        cos_s = f"{lat_cos:>8.4f}" if lat_cos is not None else f"{'-':>8}"
        mse_s = f"{lat_mse:>9.4f}" if lat_mse is not None else f"{'-':>9}"
        print(f"{step:>6}  {loss.item():>9.5f}  {cos_s}  {mse_s}", flush=True)

# ── Final eval: sample → decode → SNR ────────────────────────────────────────
dit.eval()
with torch.no_grad():
    z_gen     = dit.sample(tag_lists, steps=config.EULER_STEPS, cfg_scale=CFG, device=device)
    audio_gen = vae.decode(z_gen.float()).float()   # DiT-generated audio
    audio_vae = vae.decode(z1).float()              # VAE-recon reference (the ceiling)

final_cos       = latent_cosine(z_gen.float(), z1).item()
dit_gen_snr     = snr_db(audio.float(), audio_gen).item()
vae_recon_snr   = snr_db(audio.float(), audio_vae).item()
peak_vram       = torch.cuda.max_memory_allocated() / 1e9
passed          = final_cos > 0.9

# ── Save everything ──────────────────────────────────────────────────────────
with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["step", "fm_loss", "latent_cos", "latent_mse"])
    w.writeheader()
    w.writerows(rows)

for i in range(audio.shape[0]):
    real_np = audio[i, 0].cpu().numpy()
    vae_np  = audio_vae[i, 0].cpu().numpy()
    gen_np  = audio_gen[i, 0].cpu().numpy()
    sf.write(os.path.join(run_dir, f"clip{i}_real.wav"),      real_np, config.SAMPLE_RATE)
    sf.write(os.path.join(run_dir, f"clip{i}_vae_recon.wav"), vae_np,  config.SAMPLE_RATE)
    sf.write(os.path.join(run_dir, f"clip{i}_dit_gen.wav"),   gen_np,  config.SAMPLE_RATE)

    # 3-panel log-magnitude spectrogram: real / VAE-recon / DiT-gen (shared colour scale)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    fig.suptitle(f"clip {i}  |  latent_cos {final_cos:.3f}  |  "
                 f"DiT-gen SNR {dit_gen_snr:.2f} dB  (VAE ceiling {vae_recon_snr:.2f} dB)", fontsize=11)
    win  = torch.hann_window(1024)
    sigs = [audio[i, 0].cpu(), audio_vae[i, 0].cpu(), audio_gen[i, 0].cpu()]
    Ss   = [torch.stft(s.float(), 1024, 256, 1024, win, return_complex=True).abs()
              .clamp(min=1e-7).log10() for s in sigs]
    vmin = min(S.min().item() for S in Ss)
    vmax = max(S.max().item() for S in Ss)
    for ax, S, label in zip(axes, Ss, ["real", "vae_recon", "dit_gen"]):
        ax.imshow(S.numpy(), origin="lower", aspect="auto",
                  extent=[0, S.shape[1] / (config.SAMPLE_RATE / 256),
                          0, config.SAMPLE_RATE // 2 / 1000],
                  vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(label)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("freq (kHz)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"clip{i}_spec.png"), dpi=120)
    plt.close(fig)

# Convergence plot: fm_loss + latent_cos vs step
if rows:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    steps_x  = [r["step"] for r in rows]
    cos_pts  = [(r["step"], r["latent_cos"]) for r in rows if r["latent_cos"] is not None]
    ax1.plot([s for s, _ in cos_pts], [c for _, c in cos_pts],
             "o-", label="latent_cos", color="steelblue")
    ax1.axhline(0.9, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.6, label="pass gate (0.9)")
    ax1.set_ylabel("latent cosine", color="steelblue")
    ax1.set_xlabel("step")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(steps_x, [r["fm_loss"] for r in rows], label="fm_loss", color="tomato", alpha=0.8)
    ax2.set_ylabel("flow-matching MSE")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(f"DiT overfit convergence  B={BATCH}  steps={STEPS}  lr={LR}")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "convergence.png"), dpi=120)
    plt.close(fig)

with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(f"batch={BATCH}  steps={STEPS}  lr={LR}  cfg={CFG}  shape={list(z1.shape)}\n")
    f.write(f"[diagnostic] latent_mean={lat_mean:+.4f}  latent_std={lat_std:.4f}  "
            f"per_channel_std=[{ch_std.min():.3f}, {ch_std.max():.3f}]\n")
    if rows:
        f.write(f"final_fm_loss={rows[-1]['fm_loss']:.5f}\n")
    f.write(f"final_latent_cos={final_cos:.4f}\n")
    f.write(f"dit_gen_snr_db={dit_gen_snr:.2f}\n")
    f.write(f"vae_recon_snr_ceiling_db={vae_recon_snr:.2f}\n")
    f.write(f"peak_vram_gb={peak_vram:.2f}\n")
    f.write(f"result={'PASS' if passed else 'FAIL'}\n")

print()
print(f"[diagnostic] latent std={lat_std:.4f}  (raw pipeline; not applied)")
print(f"final latent_cos: {final_cos:.4f}")
print(f"DiT-gen SNR: {dit_gen_snr:.2f} dB   (VAE-recon ceiling: {vae_recon_snr:.2f} dB)")
print(f"peak VRAM: {peak_vram:.1f} GB  (budget: 40 GB)")
print(f"Results saved to {run_dir}/  (metrics.csv, summary.txt, convergence.png, "
      f"clip*_real/vae_recon/dit_gen.wav, clip*_spec.png)")
if passed:
    print(f"✓ PASS — latent_cos {final_cos:.3f}. DiT memorized the conditional mapping and "
          f"reproduces the batch end-to-end.")
else:
    print(f"✗ FAIL — latent_cos {final_cos:.3f} (< 0.9). DiT did not memorize the batch — "
          f"check architecture / lr / steps (and the reported latent_std).")
