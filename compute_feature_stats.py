"""
Compute per-channel mean/std for the chroma and texture conditioning channels, for
standardizing them before they reach the DiT (see config.FEATURE_STATS_PATH).
Run after extract_features.py, before train_dit.py.

WHY THIS EXISTS — the DC-bias defect.
The raw channels have large non-zero means (chroma 0.399, percussive 0.503, flatness 0.243).
A projection of such a signal decomposes as

    proj(c) = W @ mean  +  W @ (c - mean)
              ^^^^^^^^^    ^^^^^^^^^^^^^^
              CONSTANT     the actual per-frame information
              added to
              all 645
              tokens

and the constant term carries exactly zero per-frame information while consuming conditioning
bandwidth and dragging the backbone's input distribution away from what it was pretrained on.
Measured on checkpoint dit_ft/dit_step0008000.pt: of the 5.35 per-token chroma injection norm,
4.48 was that constant and only 2.93 was real harmony — ~70% of the signal was a bias the model
could have learned for free from its own weights. Standardizing to zero mean makes W @ mean vanish.

Stats are computed over the TRAINING features only (config.FEATURES_DIR). The held-out set
(config.FEATURES_VAL_DIR) is deliberately excluded and then normalized with these same numbers —
using val statistics would leak the val distribution into training.

Usage:
  python compute_feature_stats.py
"""

import glob
import json
import os

import torch

import config

files = sorted(glob.glob(os.path.join(config.FEATURES_DIR, "*.pt")))
if not files:
    raise SystemExit(f"No features found in {config.FEATURES_DIR} — run extract_features.py first.")

print(f"Computing feature stats over {len(files)} training feature files...")

# Streaming sum / sum-of-squares: (N, C, 645) fp32 for 10k tracks would be ~350 MB for chroma
# alone, and this needs to keep working when the set grows.
acc = {}
n_frames = 0
stale = 0
for i, f in enumerate(files):
    d = torch.load(f, weights_only=True)
    if d.get("version") != config.FEATURE_VERSION:
        stale += 1
        continue
    for key in ("chroma", "texture"):
        x = d[key].float()                       # (C, L)
        if key not in acc:
            acc[key] = {"sum": torch.zeros(x.shape[0]), "sq": torch.zeros(x.shape[0])}
        acc[key]["sum"] += x.sum(dim=1)
        acc[key]["sq"]  += x.pow(2).sum(dim=1)
    n_frames += d["chroma"].shape[1]
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(files)}", flush=True)

if stale:
    raise SystemExit(
        f"{stale} of {len(files)} feature files are version != {config.FEATURE_VERSION}.\n"
        f"Re-run: python extract_features.py --limit <N>   (it re-extracts stale files automatically)"
    )

stats = {}
for key, a in acc.items():
    mean = a["sum"] / n_frames
    var  = (a["sq"] / n_frames) - mean.pow(2)
    std  = var.clamp(min=1e-8).sqrt()
    stats[f"{key}_mean"] = mean.tolist()
    stats[f"{key}_std"]  = std.tolist()
    print(f"  {key:8s} mean [{mean.min():.4f}, {mean.max():.4f}]  std [{std.min():.4f}, {std.max():.4f}]")

os.makedirs(os.path.dirname(config.FEATURE_STATS_PATH), exist_ok=True)
with open(config.FEATURE_STATS_PATH, "w") as f:
    json.dump(stats, f)
print(f"\nSaved to {config.FEATURE_STATS_PATH}  ({n_frames:,} frames)")
