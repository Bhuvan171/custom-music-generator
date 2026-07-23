"""
The muddiness puzzle, sharpened:
  - Random latent NOISE makes decoded audio MORE muffled (lower HF)  [test 2 above]
  - Yet DiT-generated audio is HARSHER than real (HF-ratio 0.022-0.030 vs 0.0198)
  => the DiT's error is NOT random noise; it's a STRUCTURED offset that adds HF.

Two inference-side suspects for structured HF excess, both free to test (no retrain):
  A. CFG scale too high. v = v_unc + s*(v_cond - v_unc) EXTRAPOLATES; large s is a
     known cause of over-sharpening / harsh HF artifacts (well documented for images).
     Current config uses s=4.0.
  B. Too few ODE steps. 50 Euler steps may under-integrate, leaving the sample
     off the data manifold in a way that decodes to artifacts.

This sweeps both against the real-latent reference and reports std-ratio + HF-ratio.

Usage:  python diagnose_sampling.py --checkpoint checkpoints/dit/dit_step0115000.pt
"""

import argparse
import glob
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.dataset import JamendoDataset, collate_fn
from src.dit import MusicDiT
from src.vae import WaveformVAE

VAE_CKPT = "checkpoints/vae/vae_step0200000.pt"
N_CLIPS  = 8
device   = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()
ckpt_path = args.checkpoint or sorted(glob.glob("checkpoints/dit/dit_step*.pt"))[-1]


def hf_ratio(audio, sr=config.SAMPLE_RATE, cutoff=4000):
    win = torch.hann_window(1024, device=audio.device)
    S = torch.stft(audio.float(), 1024, 256, 1024, win, return_complex=True).abs().pow(2)
    freqs = torch.linspace(0, sr / 2, S.shape[0], device=audio.device)
    return (S[freqs > cutoff].sum() / S.sum().clamp(min=1e-12)).item()


vae = WaveformVAE().to(device)
vae.load_state_dict(torch.load(VAE_CKPT, map_location=device, weights_only=False)["vae"])
vae.eval(); vae.requires_grad_(False)

dit = MusicDiT().to(device)
ck = torch.load(ckpt_path, map_location=device, weights_only=False)
dit.load_state_dict(ck["ema"]); dit.eval()
print(f"DiT EMA from {ckpt_path} (step {ck['step']})")

with open(config.LATENT_STATS_PATH) as f:
    st = json.load(f)
lmean = torch.tensor(st["mean"], device=device).view(1, -1, 1)
lstd  = torch.tensor(st["std"],  device=device).view(1, -1, 1)

# reference clips + real latent stats
ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH)
dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)
torch.manual_seed(0)
audio_list, tag_lists = [], []
for clip, tags in dl:
    if tags[0]:
        audio_list.append(clip); tag_lists.append(tags[0])
    if len(audio_list) == N_CLIPS:
        break
audio = torch.cat(audio_list, dim=0).to(device)
with torch.no_grad():
    mu, _ = vae.encode(audio)
real_std = mu.float().std().item()
real_hf  = np.mean([hf_ratio(audio[i, 0]) for i in range(N_CLIPS)])
print(f"REAL: latent std {real_std:.3f}   audio HF-ratio {real_hf:.4f}\n")


def gen(cfg_scale, steps):
    torch.manual_seed(0)
    with torch.no_grad():
        z = dit.sample(tag_lists, steps=steps, cfg_scale=cfg_scale, device=device)
        z = z.float() * lstd + lmean
        a = vae.decode(z).float()
    return z.std().item() / real_std, np.mean([hf_ratio(a[i, 0]) for i in range(N_CLIPS)])


print("SUSPECT A — CFG scale sweep (steps=50):")
print(f"{'cfg':>5} | {'std-ratio':>9} | {'HF-ratio':>9} | {'HF vs real':>10}")
for s in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
    sr, hf = gen(s, 50)
    print(f"{s:>5.1f} | {sr:>9.3f} | {hf:>9.4f} | {hf/real_hf:>9.2f}x")

print("\nSUSPECT B — ODE steps sweep (cfg=3.0):")
print(f"{'steps':>5} | {'std-ratio':>9} | {'HF-ratio':>9} | {'HF vs real':>10}")
for n in [25, 50, 100, 250]:
    sr, hf = gen(3.0, n)
    print(f"{n:>5d} | {sr:>9.3f} | {hf:>9.4f} | {hf/real_hf:>9.2f}x")

print("\nReading: the config that brings std-ratio -> 1.0 AND HF-ratio -> real's")
print("0.0198 (~1.0x) is the best inference setting. If low CFG fixes HF excess,")
print("the muddiness was largely an over-guidance artifact, not a training failure.")
