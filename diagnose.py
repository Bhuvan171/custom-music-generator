"""
Combined diagnostic for the widened VAE (base_ch=64):
  1. Overfit 4 clips, deterministic decode, 1000 steps (== overfit_test).
  2. Decompose the final STFT loss by FFT size and linear-vs-log term, so we
     know how much of the floor is genuine error vs the over-sensitive
     log-magnitude term at large FFT.
Run from project root: python diagnose.py
"""
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import config
from src.dataset import JamendoDataset, collate_fn
from src.losses import MultiResSTFTLoss
from src.vae import WaveformVAE

device = torch.device("cuda")
ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH)
dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2)
batch, _ = next(iter(dl))
real = batch.to(device)

vae = WaveformVAE().to(device)
opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
stft = MultiResSTFTLoss().to(device)

print(f"VAE params: {sum(p.numel() for p in vae.parameters())/1e6:.1f}M  (base_ch={config.VAE_BASE_CH})")
print("overfit 4 clips, deterministic decode, 1000 steps")
for step in range(1, 1001):
    mu, logvar = vae.encode(real)
    recon = vae.decode(mu)
    loss = stft(real, recon)
    opt.zero_grad(); loss.backward()
    clip_grad_norm_(vae.parameters(), 1.0)
    opt.step()
    if step % 100 == 0:
        print(f"  step {step:4d}  stft {loss.item():.4f}", flush=True)

print(f"\nfinal floor: {loss.item():.4f}")

# ---- decompose ----
print("\nPer-FFT-size breakdown of the final loss (linear L1 | log L1):")
x = real.squeeze(1).float()
y = recon.detach().squeeze(1).float()
tot_lin = tot_log = 0.0
for n in config.STFT_FFT_SIZES:
    win = torch.hann_window(n, device=x.device)
    Sx = torch.stft(x, n, n // 4, n, win, return_complex=True).abs().clamp(config.STFT_EPS)
    Sy = torch.stft(y, n, n // 4, n, win, return_complex=True).abs().clamp(config.STFT_EPS)
    lin = F.l1_loss(Sx, Sy).item()
    log = F.l1_loss(Sx.log(), Sy.log()).item()
    tot_lin += lin; tot_log += log
    print(f"  n={n:5d}   lin {lin:.4f}   log {log:.4f}   (sum {lin+log:.4f})")
print(f"  TOTAL    lin {tot_lin/len(config.STFT_FFT_SIZES):.4f}   "
      f"log {tot_log/len(config.STFT_FFT_SIZES):.4f}")
