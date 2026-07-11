"""
Single-batch overfit sanity test for the spectral VAE.

Memorizes a fixed batch with deterministic decode (mu, no sampling) and reports
reconstruction quality two ways:
  - mel : the multi-scale log-mel training loss (same loss train_vae.py uses).
          Read it as a FALLING TREND, not an absolute target.
  - SNR : time-domain signal-to-noise ratio in dB — the interpretable metric.
          On a memorized batch a healthy encoder+decoder climbs past the +6 dB
          gate (~step 1400) toward the ~7-8 dB pre-GAN ceiling; that proves the
          encode→decode path fits data and gradients flow. Note the WaveNeXt
          decoder converges slower than the old conv decoder (it learns its
          synthesis basis), so this needs ~2x the steps the old test used.

This is a pure-reconstruction test: no discriminator, no KL, deterministic mu
decode. Transient sharpness / crisp drums come from the GAN in full training,
not from here. PASS at SNR > 6 dB (just under the ~7-8 dB pre-GAN ceiling).

This is a SMALL-batch sanity check (B=4): a few clips memorized fast give a clean,
high SNR signal. A big batch only makes memorization harder and the signal weaker,
so the < 10 GB VRAM target belongs to the real training run (train_vae.py), NOT
here — at B=4 this peaks ~3.7 GB.

Results for each run (metrics CSV + real/recon WAVs + summary) are written to a
timestamped directory under config.OVERFIT_DIR so you can compare runs side by side.
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
from src.losses import MultiScaleMelLoss
from src.vae import WaveformVAE

BATCH = 4           # small batch → clean, fast overfit signal (peaks ~3.7 GB).
                    # The < 10 GB VRAM target is for train_vae.py, not this sanity test.
STEPS = 8000        # extended run to find the true SNR ceiling before changing the latent.
                    # WaveNeXt learns its synthesis basis from scratch; prior 2000-step runs
                    # were still climbing — this tells us where it actually plateaus.
LR    = 1e-3        # high LR to overfit the single batch fast
LOG_EVERY = 100

device = torch.device("cuda")

# Per-run results directory (timestamped so successive runs don't clobber).
run_dir = os.path.join(config.OVERFIT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)

ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn, num_workers=2)

batch, _ = next(iter(dl))
real = batch.to(device)   # (BATCH, 1, 660480) — fixed forever

vae = WaveformVAE().to(device)
opt = torch.optim.AdamW(vae.parameters(), lr=LR)
mel = MultiScaleMelLoss().to(device)


def snr_db(x, y):
    # time-domain signal-to-noise ratio in dB (higher is better)
    sig   = x.pow(2).mean()
    noise = (x - y).pow(2).mean()
    return 10.0 * torch.log10(sig / (noise + 1e-12))


print(f"VAE params: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M")
print(f"Overfitting on 1 batch of shape {list(real.shape)}  (B={BATCH}, {STEPS} steps, lr={LR})")
print(f"Results dir: {run_dir}")
print(f"{'step':>6}  {'mel':>8}  {'wave':>8}  {'SNR(dB)':>8}")

rows = []   # collected metrics, written to metrics.csv at the end
for step in range(1, STEPS + 1):
    # fp32 (no autocast): this tiny sanity test prioritises numerical robustness
    # over speed. Real training uses bf16 at a 5× lower LR.
    # Deterministic decode (mu, no sampling) — tests pure recon capacity.
    mu, logvar = vae.encode(real)
    recon = vae.decode(mu)

    l_mel  = mel(real, recon)
    l_wave = F.l1_loss(recon, real)               # time-domain term anchors phase
    loss = l_mel + config.VAE_WAVE_WEIGHT * l_wave
    opt.zero_grad()
    loss.backward()
    clip_grad_norm_(vae.parameters(), 1.0)
    opt.step()

    if torch.isnan(loss):
        print(f"  NaN in loss at step {step}")
        break

    if step % LOG_EVERY == 0:
        snr = snr_db(real.float(), recon.float()).item()
        rows.append({"step": step, "mel": l_mel.item(), "wave": l_wave.item(), "snr": snr})
        print(f"{step:>6}  {l_mel.item():>8.4f}  {l_wave.item():>8.4f}  {snr:>8.2f}", flush=True)

# Final deterministic recon: measure SNR and save WAVs to listen to.
vae.eval()
with torch.no_grad():
    mu, _ = vae.encode(real)
    recon = vae.decode(mu).float()
final_snr = snr_db(real.float(), recon).item()
peak_vram = torch.cuda.max_memory_allocated() / 1e9
passed = final_snr > 6.0

# ── Save everything to the run dir ────────────────────────────────────────────
with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["step", "mel", "wave", "snr"])
    w.writeheader()
    w.writerows(rows)

for i in range(real.shape[0]):
    r_np = real[i, 0].cpu().numpy()
    p_np = recon[i, 0].cpu().numpy()
    sf.write(os.path.join(run_dir, f"clip{i}_real.wav"),  r_np, config.SAMPLE_RATE)
    sf.write(os.path.join(run_dir, f"clip{i}_recon.wav"), p_np, config.SAMPLE_RATE)

    # Real vs recon spectrogram — log-magnitude, same colour scale per clip
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    fig.suptitle(f"clip {i}  |  SNR {final_snr:.2f} dB  |  step {STEPS}", fontsize=11)
    win = torch.hann_window(1024)
    for ax, sig, label in zip(axes, [real[i, 0].cpu(), recon[i, 0].cpu()], ["real", "recon"]):
        S = torch.stft(sig.float(), 1024, 256, 1024, win, return_complex=True).abs().clamp(min=1e-7).log10()
        ax.imshow(S.numpy(), origin="lower", aspect="auto",
                  extent=[0, S.shape[1] / (config.SAMPLE_RATE / 256),
                          0, config.SAMPLE_RATE // 2 / 1000],
                  vmin=S.min().item(), vmax=S.max().item(), cmap="magma")
        ax.set_title(label)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("freq (kHz)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"clip{i}_spec.png"), dpi=120)
    plt.close(fig)

# SNR curve plot
if rows:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    steps_x = [r["step"] for r in rows]
    ax1.plot(steps_x, [r["snr"] for r in rows], label="SNR (dB)", color="steelblue")
    ax1.axhline(6.0, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.6, label="pass gate (6 dB)")
    ax1.set_ylabel("SNR (dB)", color="steelblue")
    ax1.set_xlabel("step")
    ax2 = ax1.twinx()
    ax2.plot(steps_x, [r["mel"] for r in rows],  label="mel loss",  color="tomato",  alpha=0.8)
    ax2.plot(steps_x, [r["wave"] for r in rows], label="wave loss", color="orange",  alpha=0.8)
    ax2.set_ylabel("loss")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title(f"Overfit convergence  B={BATCH}  steps={STEPS}  lr={LR}")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "convergence.png"), dpi=120)
    plt.close(fig)

with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(f"batch={BATCH}  steps={STEPS}  lr={LR}  shape={list(real.shape)}\n")
    f.write(f"final_snr_db={final_snr:.2f}\n")
    if rows:
        f.write(f"final_mel={rows[-1]['mel']:.4f}\n")
        f.write(f"final_wave={rows[-1]['wave']:.4f}\n")
    f.write(f"peak_vram_gb={peak_vram:.2f}\n")
    f.write(f"result={'PASS' if passed else 'FAIL'}\n")

print()
print(f"peak VRAM: {peak_vram:.1f} GB  (budget: 40 GB)")
print(f"Results saved to {run_dir}/  (metrics.csv, summary.txt, convergence.png, clip*_real/recon.wav, clip*_spec.png)")
if passed:
    print(f"✓ PASS — SNR {final_snr:.2f} dB. Encoder+decoder reconstruct correctly.")
    print("  (Pre-GAN regression ceiling is ~7-8 dB; transient sharpness / crisp drums")
    print("   come from the discriminator in full training, not from this test.)")
else:
    print(f"✗ FAIL — SNR {final_snr:.2f} dB. Below pre-GAN expectation — check setup.")
