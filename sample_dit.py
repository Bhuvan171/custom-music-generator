"""
Generate real audio from a DiT checkpoint to actually LISTEN to output quality.

Loss curves can plateau while still describing a model that generates
reasonable audio (flow-matching MSE has a nonzero floor). This is the
end-to-end check: sample latents from noise (EMA weights, CFG), decode
through the frozen VAE, save real vs generated pairs.

Usage:
  python sample_dit.py --checkpoint checkpoints/dit/dit_step0070000.pt
"""

import argparse
import glob
import json
import os
from datetime import datetime

import soundfile as sf
import torch
from torch.utils.data import DataLoader

import config
from src.dataset import JamendoDataset, collate_fn
from src.dit import MusicDiT
from src.vae import WaveformVAE

VAE_CKPT = "checkpoints/vae/vae_step0200000.pt"
N_CLIPS  = 6

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="DiT checkpoint (default: latest in checkpoints/dit/)")
parser.add_argument("--cfg-scale", type=float, default=config.CFG_SCALE)
args = parser.parse_args()

ckpt_path = args.checkpoint
if ckpt_path is None:
    ckpts = sorted(glob.glob("checkpoints/dit/dit_step*.pt"))
    ckpt_path = ckpts[-1]

device = torch.device("cuda")

vae = WaveformVAE().to(device)
vae.load_state_dict(torch.load(VAE_CKPT, map_location=device, weights_only=False)["vae"])
vae.eval()
vae.requires_grad_(False)

dit = MusicDiT().to(device)
dit_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
dit.load_state_dict(dit_ckpt["ema"])   # EMA weights — smoother than raw training weights
dit.eval()
print(f"Loaded DiT EMA weights from {ckpt_path}  (step {dit_ckpt['step']})")

# DiT was trained on normalized latents; denormalize back to the VAE's native scale before decoding.
with open(config.LATENT_STATS_PATH) as f:
    _stats = json.load(f)
latent_mean = torch.tensor(_stats["mean"], device=device).view(1, -1, 1)
latent_std  = torch.tensor(_stats["std"],  device=device).view(1, -1, 1)

# Real clips with non-empty tags, for meaningful conditioning + a real reference to compare against.
ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH)
dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)

audio_list, tag_lists = [], []
for clip, tags in dl:
    if tags[0]:
        audio_list.append(clip)
        tag_lists.append(tags[0])
    if len(audio_list) == N_CLIPS:
        break
audio = torch.cat(audio_list, dim=0).to(device)

with torch.no_grad():
    z_gen     = dit.sample(tag_lists, steps=config.EULER_STEPS, cfg_scale=args.cfg_scale, device=device)
    z_gen     = z_gen.float() * latent_std + latent_mean   # back to raw VAE latent scale
    audio_gen = vae.decode(z_gen).float()

run_dir = os.path.join(config.DIT_SAMPLE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)
for i in range(N_CLIPS):
    sf.write(os.path.join(run_dir, f"clip{i}_real.wav"), audio[i, 0].cpu().numpy(), config.SAMPLE_RATE)
    sf.write(os.path.join(run_dir, f"clip{i}_dit_gen.wav"), audio_gen[i, 0].cpu().numpy(), config.SAMPLE_RATE)
    print(f"  clip{i}: tags={tag_lists[i]}")

print(f"\nSaved {N_CLIPS} real/generated pairs to {run_dir}")
