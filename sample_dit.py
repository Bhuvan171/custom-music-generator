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
from src.dataset import JamendoDataset, collate_fn, _track_id_from_path
from src.dit import MusicDiT
from src.vae import WaveformVAE

_ft_ckpts = sorted(glob.glob(os.path.join(config.FT_DEC_CKPT_DIR, "dec_step*.pt")))
VAE_CKPT  = _ft_ckpts[-1] if _ft_ckpts else "checkpoints/vae/vae_step0200000.pt"
N_CLIPS  = 6

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="DiT checkpoint (default: latest in checkpoints/dit/)")
parser.add_argument("--cfg-scale", type=float, default=config.CFG_SCALE)
parser.add_argument("--heldout", action="store_true",
                    help="Sample tracks the model was NEVER TRAINED ON (config.FEATURES_VAL_DIR). "
                         "Without this you are listening to training data — the model has seen these "
                         "clips ~100 times and the audio will flatter it, exactly as the train-set-only "
                         "metrics did (held-out GAP is 86%% of ceiling vs 102%% on train).")
args = parser.parse_args()

ckpt_path = args.checkpoint
if ckpt_path is None:
    # Prefer the fine-tuned (chroma-conditioned) checkpoints if any exist.
    ckpts = sorted(glob.glob(os.path.join(config.DIT_FT_CKPT_DIR, "dit_step*.pt"))) \
            or sorted(glob.glob("checkpoints/dit/dit_step*.pt"))
    if not ckpts:
        raise SystemExit("No DiT checkpoints found.")
    ckpt_path = ckpts[-1]

device = torch.device("cuda")

vae = WaveformVAE().to(device)
_missing, _unexpected = vae.load_state_dict(
    torch.load(VAE_CKPT, map_location=device, weights_only=False)["vae"], strict=False)
if _missing or _unexpected:
    print(f"  Partial VAE load from {VAE_CKPT} (head architecture differs): "
          f"missing={_missing} unexpected={_unexpected}")
print(f"VAE checkpoint: {VAE_CKPT}")
vae.eval()
vae.requires_grad_(False)

dit = MusicDiT().to(device)
dit_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
# strict=False: pre-chroma checkpoints have no feat_embed weights (they stay zero-init = no-op).
dit.load_state_dict(dit_ckpt["ema"], strict=False)   # EMA — smoother than raw training weights
dit.eval()
print(f"Loaded DiT EMA weights from {ckpt_path}  (step {dit_ckpt['step']})")

# DiT was trained on normalized latents; denormalize back to the VAE's native scale before decoding.
with open(config.LATENT_STATS_PATH) as f:
    _stats = json.load(f)
latent_mean = torch.tensor(_stats["mean"], device=device).view(1, -1, 1)
latent_std  = torch.tensor(_stats["std"],  device=device).view(1, -1, 1)

# Real clips with non-empty tags, for meaningful conditioning + a real reference to compare against.
# deterministic_crop matches how latents/features were built, so the loaded features
# describe exactly this audio.
use_feats = config.DIT_USE_GLOBAL_FEATS or config.DIT_USE_CHROMA or config.DIT_USE_TEXTURE
feats_dir = config.FEATURES_VAL_DIR if args.heldout else config.FEATURES_DIR
print(f"Sampling {'HELD-OUT (never trained on)' if args.heldout else 'TRAINING'} tracks "
      f"from {feats_dir}  |  CFG {args.cfg_scale}")
ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH,
                    deterministic_crop=True, load_features=use_feats,
                    features_dir=feats_dir)
# SEEDED. An unseeded shuffle picks different tracks on every invocation, so two runs of
# this script produce files that look comparable (clip0_*) but describe different songs.
# Metadata is checked BEFORE eds[idx] decodes a 30s FLAC — the held-out set is ~2k of 55.6k
# tracks, so a load-then-check loop would decode ~450 clips to find 6.
_g = torch.Generator()
_g.manual_seed(config.DIT_EVAL_SEED)
audio_list, tag_lists, feat_list, track_ids = [], [], [], []
for idx in torch.randperm(len(ds), generator=_g).tolist():
    tid = _track_id_from_path(ds.files[idx])
    if not ds.track_tags.get(tid):
        continue
    if use_feats and not os.path.exists(os.path.join(feats_dir, f"{tid}.pt")):
        continue
    item = ds[idx]
    if use_feats:
        feat_list.append({k: v.unsqueeze(0) for k, v in item[2].items()})
    audio_list.append(item[0].unsqueeze(0))
    tag_lists.append(item[1])
    track_ids.append(tid)
    if len(audio_list) == N_CLIPS:
        break
audio = torch.cat(audio_list, dim=0).to(device)

feats = None
if use_feats:
    feats = {k: torch.cat([f[k] for f in feat_list], dim=0).to(device) for k in feat_list[0]}

with torch.no_grad():
    # The VAE reconstruction is the CEILING: the best this pipeline can possibly sound, since the
    # DiT's output must decode through the same VAE. gen-vs-ceiling is the only fair comparison —
    # gen-vs-real also charges the DiT for every VAE artifact.
    mu, _         = vae.encode(audio)
    audio_ceiling = vae.decode(mu.float()).float()
    z_gen     = dit.sample(tag_lists, steps=config.EULER_STEPS, cfg_scale=args.cfg_scale,
                           device=device, feats=feats)
    z_gen     = z_gen.float() * latent_std + latent_mean   # back to raw VAE latent scale
    audio_gen = vae.decode(z_gen).float()

run_dir = os.path.join(config.DIT_SAMPLE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)


def write(path, x):
    """
    Write a clip, peak-normalizing only if it would otherwise clip. Returns the original peak.

    sf.write() defaults to PCM_16 for .wav, which HARD-CLIPS anything outside [-1, 1] — and
    generated audio routinely leaves that range. Measured over the 16 seeded eval clips at
    CFG 1.5: peak 2.45, and 8 of 16 clips clipped (clip0 peaked at 1.98, which is why it
    sounded "radically different" from everything else). Hard clipping a music signal
    manufactures broadband harmonic distortion — i.e. it INVENTS the exact noisy/muddy
    artifact we have been trying to diagnose. Only 0.078% of samples were affected, so this
    is not the main quality problem, but it is free to fix and it was corrupting what we heard.
    """
    x = x.cpu().numpy()
    peak = float(abs(x).max())
    sf.write(path, x / peak if peak > 1.0 else x, config.SAMPLE_RATE)
    return peak


inv_vocab = {v: k for k, v in ds.vocab.items()}
for i in range(N_CLIPS):
    write(os.path.join(run_dir, f"clip{i}_1real.wav"), audio[i, 0])
    write(os.path.join(run_dir, f"clip{i}_2vae_ceiling.wav"), audio_ceiling[i, 0])
    p_gen = write(os.path.join(run_dir, f"clip{i}_3dit_gen.wav"), audio_gen[i, 0])
    flag = f"  [peak {p_gen:.2f} -> normalized; would have CLIPPED]" if p_gen > 1.0 else ""
    names = ", ".join(inv_vocab.get(t, str(t)) for t in tag_lists[i])
    print(f"  clip{i} ({track_ids[i]}): {names}{flag}")

print(f"\nSaved {N_CLIPS} x (real / vae_ceiling / dit_gen) to {run_dir}")
print("Listen in that order. dit_gen vs vae_ceiling is the DiT's real gap —")
print("real vs vae_ceiling is the VAE's, and the DiT cannot fix that half.")
