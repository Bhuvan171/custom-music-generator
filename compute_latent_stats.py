"""
Compute per-channel mean/std over the full cached latent set, for normalizing
DiT training targets (see config.LATENT_STATS_PATH). Run once after
cache_latents.py, before train_dit.py.

Usage:
  python compute_latent_stats.py
"""

import glob
import json
import os

import torch

import config

files = sorted(glob.glob(os.path.join(config.LATENTS_DIR, "*.pt")))
print(f"Computing latent stats over {len(files)} cached files...")

lat = torch.stack([torch.load(f, weights_only=True) for f in files])  # (N, 32, 645)
mean = lat.mean(dim=(0, 2))  # (32,)
std  = lat.std(dim=(0, 2))   # (32,)

os.makedirs(os.path.dirname(config.LATENT_STATS_PATH), exist_ok=True)
with open(config.LATENT_STATS_PATH, "w") as f:
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

print(f"global std before norm: {lat.std().item():.4f}")
print(f"per-channel std range: [{std.min():.4f}, {std.max():.4f}]")
print(f"Saved to {config.LATENT_STATS_PATH}")
