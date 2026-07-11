"""
Reconstruction diagnostics — figure out WHY the VAE output sounds bad.

Loads the latest checkpoint, reconstructs a few real clips, and reports several
INDEPENDENT differentiators so we don't guess:

  1. Per-frequency-band energy ratio recon/real (dB)
        lows very negative          -> THIN  (decoder not reproducing low end)
        highs positive, lows negative -> TINNY (bright/harsh spectral balance)
  2. Spectral centroid (brightness): recon brighter than real -> tinny
  3. Overall SNR (deterministic mu vs sampled z): is the latent NOISE hurting?
  4. Spectrogram images real vs recon (visual confirmation)
  5. Transient zoom on the loudest onset (drum smearing)
  6. real/recon WAVs to listen

Run from project root:  python diagnose_recon.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

import config
from src.dataset import JamendoDataset, collate_fn
from src.vae import WaveformVAE

device = torch.device("cuda")
OUT = "samples/diag"
os.makedirs(OUT, exist_ok=True)
SR = config.SAMPLE_RATE
N_FFT, HOP = 2048, 512
BANDS = [("sub-bass", 20, 60), ("bass", 60, 250), ("low-mid", 250, 500),
         ("mid", 500, 2000), ("high-mid", 2000, 4000), ("presence", 4000, 6000),
         ("brilliance", 6000, SR // 2)]


def stft_mag(x):
    win = torch.hann_window(N_FFT, device=x.device)
    return torch.stft(x, N_FFT, HOP, N_FFT, win, return_complex=True).abs()  # (freq, time)


def band_energy(mag, lo, hi):
    res = SR / N_FFT
    a, b = int(lo / res), int(hi / res) + 1
    return (mag[a:b] ** 2).sum().item()


def spectral_centroid(mag):
    freqs = torch.linspace(0, SR / 2, mag.shape[0], device=mag.device).unsqueeze(1)
    return ((freqs * mag).sum() / (mag.sum() + 1e-9)).item()


def snr_db(x, y):
    return 10 * np.log10((x ** 2).mean() / (((x - y) ** 2).mean() + 1e-12))


# ── load latest checkpoint ───────────────────────────────────────────────
ckpts = sorted(glob.glob(os.path.join(config.CKPT_DIR, "vae_step*.pt")))
assert ckpts, f"no checkpoints in {config.CKPT_DIR}"
ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
vae = WaveformVAE().to(device)
vae.load_state_dict(ckpt["vae"])
vae.eval()
print(f"loaded {ckpts[-1]} (step {ckpt['step']})")

# ── a few real clips ─────────────────────────────────────────────────────
ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH)
dl = DataLoader(ds, batch_size=3, shuffle=True, collate_fn=collate_fn, num_workers=2)
batch, _ = next(iter(dl))
real = batch.to(device)   # (3, 1, T)

with torch.no_grad():
    mu, logvar = vae.encode(real)
    recon_mu = vae.decode(mu).float()                 # deterministic
    recon_s  = vae.decode(vae.reparameterize(mu, logvar)).float()  # sampled

print("\n=== per-band energy ratio  recon/real (dB) ===")
print("  negative => recon MISSING that band (thin);  positive => recon TOO HOT (tinny)")
for i in range(real.shape[0]):
    xr = real[i, 0].float()
    yr = recon_mu[i, 0]
    ys = recon_s[i, 0]
    Mr, My = stft_mag(xr), stft_mag(yr)

    snr_mu = snr_db(xr.cpu().numpy(), yr.cpu().numpy())
    snr_sa = snr_db(xr.cpu().numpy(), ys.cpu().numpy())
    cen_r, cen_y = spectral_centroid(Mr), spectral_centroid(My)
    print(f"\nclip {i} | SNR(mu) {snr_mu:+.1f} dB | SNR(sampled) {snr_sa:+.1f} dB "
          f"| centroid real {cen_r:.0f}Hz recon {cen_y:.0f}Hz "
          f"({'BRIGHTER/tinny' if cen_y > cen_r * 1.1 else 'ok'})")
    for name, lo, hi in BANDS:
        er, ey = band_energy(Mr, lo, hi), band_energy(My, lo, hi)
        ratio = 10 * np.log10((ey + 1e-12) / (er + 1e-12))
        bar = "#" * int(max(0, min(24, 12 + ratio)))
        print(f"  {name:11s} {lo:5d}-{hi:5d}Hz : {ratio:+6.1f} dB  |{bar}")

    # spectrogram real vs recon
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for a, (M, t) in zip(ax, [(Mr, "real"), (My, "recon")]):
        db = 20 * torch.log10(M.cpu() + 1e-7).numpy()
        a.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=-60, vmax=20,
                 extent=[0, xr.shape[-1] / SR, 0, SR / 2 / 1000])
        a.set_title(f"clip{i} {t}"); a.set_xlabel("s"); a.set_ylabel("kHz")
    plt.tight_layout(); plt.savefig(f"{OUT}/clip{i}_spec.png", dpi=110); plt.close()

    # transient zoom around the loudest onset
    c = int(np.argmax(xr.abs().cpu().numpy()))
    w = int(0.03 * SR)
    s, e = max(0, c - w), min(xr.shape[-1], c + w)
    plt.figure(figsize=(12, 4))
    plt.plot(xr.cpu().numpy()[s:e], label="real", lw=1)
    plt.plot(yr.cpu().numpy()[s:e], label="recon", lw=1, alpha=0.8)
    plt.title(f"clip{i} transient @ {c / SR:.2f}s (±30ms)"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/clip{i}_transient.png", dpi=110); plt.close()

    sf.write(f"{OUT}/clip{i}_real.wav",  xr.cpu().numpy(), SR)
    sf.write(f"{OUT}/clip{i}_recon.wav", yr.cpu().numpy(), SR)

print(f"\npeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(f"wrote spectrograms, transient plots, WAVs -> {OUT}/")
print("\nHOW TO READ:")
print("  lows strongly negative                 -> THIN  (no low end; add MPD won't fix — capacity/RF)")
print("  highs >0 while lows <0  /  centroid up  -> TINNY (bright balance; MSD-only artifact)")
print("  SNR(sampled) << SNR(mu)                 -> latent NOISE is degrading output (KL too loose)")
print("  transient plot rounded vs real spike    -> transient smearing (needs MPD / less compression)")
