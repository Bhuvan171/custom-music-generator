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

# Stats must describe the latents the DiT actually trains on. When feature conditioning
# is active, training is restricted to feature-aligned (deterministically cropped) tracks,
# so stale random-crop latents must not skew the mean/std used to normalize them.
if os.path.isdir(config.FEATURES_DIR):
    have = {f for f in os.listdir(config.FEATURES_DIR) if f.endswith(".pt")}
    if have:
        before = len(files)
        files = [f for f in files if os.path.basename(f) in have]
        print(f"Restricting to {len(files)} feature-aligned latents (of {before} on disk).")

if not files:
    raise SystemExit(f"No latents found in {config.LATENTS_DIR} (matching {config.FEATURES_DIR})")

print(f"Computing latent stats over {len(files)} cached files...")

# STREAMING sum / sum-of-squares rather than torch.stack over every latent, so memory is O(1)
# instead of O(N) on a shared box that has already lost a training run to a host-RAM OOM kill.
# Honest note on the size: a (32,645) fp32 latent is only 82.5 KB, so even 53.6k of them is ~4.4 GB
# and the stacked version would in fact have fit. The alarming 646 KB/file on disk was a
# torch.save-a-view bug (see cache_latents.py), not real tensor size — do not use file size as a
# proxy for memory here. Streaming is kept because it costs nothing and stops mattering at scale.
# (compute_feature_stats.py uses this same pattern.)
n_frames = 0
csum = csq = None
for i, f in enumerate(files):
    x = torch.load(f, weights_only=True).float()      # (32, 645)
    if csum is None:
        csum = torch.zeros(x.shape[0], dtype=torch.float64)
        csq  = torch.zeros(x.shape[0], dtype=torch.float64)
    csum += x.sum(dim=1).double()
    csq  += x.pow(2).sum(dim=1).double()
    n_frames += x.shape[1]
    if (i + 1) % 5000 == 0:
        print(f"  {i + 1}/{len(files)}", flush=True)

mean = (csum / n_frames)                                  # (32,)
var  = (csq / n_frames) - mean.pow(2)
std  = var.clamp(min=1e-12).sqrt()
# The old code reported lat.std() (a single global scalar) for the log line below; recover it from
# the per-channel moments rather than re-reading every file.
global_std = ((csq.sum() / (n_frames * len(mean))) - (csum.sum() / (n_frames * len(mean))) ** 2).clamp(min=1e-12).sqrt()
mean, std = mean.float(), std.float()

os.makedirs(os.path.dirname(config.LATENT_STATS_PATH), exist_ok=True)
with open(config.LATENT_STATS_PATH, "w") as f:
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

print(f"global std before norm: {global_std.item():.4f}")
print(f"per-channel std range: [{std.min():.4f}, {std.max():.4f}]")
print(f"Saved to {config.LATENT_STATS_PATH}")
