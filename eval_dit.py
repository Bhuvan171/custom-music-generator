"""
Evaluate DiT generation quality beyond the training loss curve.

"Muddy but structured" audio from an undertrained flow-matching model has a
specific signature: the model averages over the many plausible velocities for
a given (z_t, t, tags), so its samples regress toward the mean latent path.
The tell is GENERATED LATENT VARIANCE COLLAPSE relative to real latents —
this shows up as lost high-frequency / transient detail after decoding, i.e.
exactly "muddy". This script checks for that directly, plus the usual
loss-curve / spectrogram / SNR-vs-VAE-ceiling views, across whichever DiT
checkpoints are currently on disk.

Usage:
  python eval_dit.py
"""

import csv
import glob
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from src.dataset import JamendoDataset, collate_fn
from src.dit import MusicDiT
from src.vae import WaveformVAE

_ft_ckpts  = sorted(glob.glob(os.path.join(config.FT_DEC_CKPT_DIR, "dec_step*.pt")))
VAE_CKPT   = _ft_ckpts[-1] if _ft_ckpts else "checkpoints/vae/vae_step0200000.pt"
N_CLIPS    = 6      # fixed real clips used for generation / SNR / spectrograms
N_REF      = 300    # cached latents sampled to estimate the "real" variance reference
SEED       = 42
device     = torch.device("cuda")

run_dir = os.path.join("results", "eval_dit", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)


def snr_db(x, y):
    sig   = x.pow(2).mean()
    noise = (x - y).pow(2).mean()
    return 10.0 * torch.log10(sig / (noise + 1e-12))


def latent_cosine(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean()


def hf_energy_ratio(audio, sr=config.SAMPLE_RATE, cutoff_hz=4000):
    # fraction of spectral energy above cutoff_hz — proxy for "brightness" vs "mud"
    win = torch.hann_window(1024, device=audio.device)
    S   = torch.stft(audio.float(), 1024, 256, 1024, win, return_complex=True).abs().pow(2)
    freqs = torch.linspace(0, sr / 2, S.shape[0], device=audio.device)
    hf_mask = freqs > cutoff_hz
    return (S[hf_mask].sum() / S.sum().clamp(min=1e-12)).item()


# ── Reference: real latent variance from a random sample of the cache ───────
latent_files = sorted(glob.glob(os.path.join(config.LATENTS_DIR, "*.pt")))
rng = np.random.RandomState(SEED)
sample_files = rng.choice(latent_files, size=min(N_REF, len(latent_files)), replace=False)
real_latents = torch.stack([torch.load(f, weights_only=True) for f in sample_files])  # (N_REF, 32, 645)
real_std_global  = real_latents.std().item()
real_std_perchan = real_latents.std(dim=(0, 2))  # (32,)
print(f"Real latent reference: global std={real_std_global:.4f}  (from {len(sample_files)} cached latents)")

# DiT is trained on normalized latents; denormalize its output back to VAE-native scale
# before decoding or comparing against real (raw) latents.
with open(config.LATENT_STATS_PATH) as f:
    _stats = json.load(f)
latent_mean = torch.tensor(_stats["mean"], device=device).view(1, -1, 1)
latent_std  = torch.tensor(_stats["std"],  device=device).view(1, -1, 1)

# ── Frozen VAE ────────────────────────────────────────────────────────────
vae = WaveformVAE().to(device)
_missing, _unexpected = vae.load_state_dict(
    torch.load(VAE_CKPT, map_location=device, weights_only=False)["vae"], strict=False)
if _missing or _unexpected:
    print(f"  Partial VAE load from {VAE_CKPT} (head architecture differs): "
          f"missing={_missing} unexpected={_unexpected}")
print(f"VAE checkpoint: {VAE_CKPT}")
vae.eval()
vae.requires_grad_(False)

# ── Fixed real clips (with tags) for generation / SNR / spectrograms ────────
# deterministic_crop matches how latents/features were built, so the loaded features
# describe exactly this audio (a random crop would misalign the chroma).
use_feats = config.DIT_USE_GLOBAL_FEATS or config.DIT_USE_CHROMA or config.DIT_USE_TEXTURE
ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH,
                    deterministic_crop=True, load_features=use_feats)
dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)
torch.manual_seed(SEED)
audio_list, tag_lists, feat_list = [], [], []
for batch in dl:
    if use_feats:
        clip, tags, f = batch
        if not tags[0] or f["valid"][0] < 0.5:
            continue
        feat_list.append(f)
    else:
        clip, tags = batch
        if not tags[0]:
            continue
    audio_list.append(clip)
    tag_lists.append(tags[0])
    if len(audio_list) == N_CLIPS:
        break
audio = torch.cat(audio_list, dim=0).to(device)

feats = None
if use_feats:
    feats = {k: torch.cat([f[k] for f in feat_list], dim=0).to(device) for k in feat_list[0]}

with torch.no_grad():
    z1, _         = vae.encode(audio)
    audio_vae_rec = vae.decode(z1.float())
vae_ceiling_snr = snr_db(audio.float(), audio_vae_rec.float()).item()
vae_ceiling_hf  = np.mean([hf_energy_ratio(audio_vae_rec[i, 0]) for i in range(N_CLIPS)])
real_hf         = np.mean([hf_energy_ratio(audio[i, 0]) for i in range(N_CLIPS)])
print(f"VAE reconstruction ceiling: SNR={vae_ceiling_snr:.2f} dB  HF-ratio={vae_ceiling_hf:.4f}  (real HF-ratio={real_hf:.4f})")

# ── Evaluate every available DiT checkpoint ─────────────────────────────────
ckpts = sorted(glob.glob("checkpoints/dit/dit_step*.pt"))
rows = []
dit = MusicDiT().to(device)
latest_gen_audio = None
for ckpt_path in ckpts:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # strict=False: pre-chroma checkpoints have no feat_embed weights. Those stay zero-init,
    # i.e. a no-op — so an old checkpoint evaluates exactly as it did before, and stays
    # directly comparable to the fine-tuned ones.
    dit.load_state_dict(ckpt["ema"], strict=False)
    dit.eval()
    step = ckpt["step"]

    torch.manual_seed(SEED)   # identical initial noise across checkpoints -> fair trend comparison
    with torch.no_grad():
        z_gen     = dit.sample(tag_lists, steps=config.EULER_STEPS, cfg_scale=config.CFG_SCALE,
                               device=device, feats=feats)
        z_gen     = z_gen.float() * latent_std + latent_mean   # back to raw VAE latent scale
        audio_gen = vae.decode(z_gen).float()

    gen_std_global  = z_gen.std().item()
    gen_std_perchan = z_gen.std(dim=(0, 2)).cpu()
    lat_cos = latent_cosine(z_gen.float(), z1).item()
    lat_mse = F.mse_loss(z_gen.float(), z1).item()
    snr_gen = snr_db(audio.float(), audio_gen).item()
    hf_gen  = np.mean([hf_energy_ratio(audio_gen[i, 0]) for i in range(N_CLIPS)])

    rows.append({
        "step": step,
        "latent_cos": lat_cos,
        "latent_mse": lat_mse,
        "gen_std": gen_std_global,
        "real_std_ref": real_std_global,
        "std_ratio": gen_std_global / real_std_global,
        "snr_gen_db": snr_gen,
        "vae_ceiling_snr_db": vae_ceiling_snr,
        "hf_ratio_gen": hf_gen,
        "hf_ratio_real": real_hf,
    })
    print(f"step {step:>7d} | latent_cos {lat_cos:.4f} | std_ratio {gen_std_global/real_std_global:.3f} "
          f"| SNR {snr_gen:.2f}dB (ceiling {vae_ceiling_snr:.2f}dB) | HF-ratio {hf_gen:.4f} (real {real_hf:.4f})")

    if ckpt_path == ckpts[-1]:
        latest_gen_audio = audio_gen
        latest_step = step
        latest_perchan = gen_std_perchan

# ── Save metrics CSV ─────────────────────────────────────────────────────────
with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ── Plot 1: DiT training loss curve (the plateau) ────────────────────────────
steps_l, losses = [], []
with open(config.DIT_STATS_CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        steps_l.append(int(row["step"]))
        losses.append(float(row["loss"]))
losses_arr = np.array(losses)
roll = np.convolve(losses_arr, np.ones(50) / 50, mode="valid")
plt.figure(figsize=(9, 4))
plt.plot(steps_l, losses_arr, alpha=0.25, label="raw loss")
plt.plot(steps_l[49:], roll, label="rolling mean (50 steps)", linewidth=2)
plt.xlabel("step"); plt.ylabel("flow-matching MSE loss"); plt.legend()
plt.title("DiT training loss"); plt.tight_layout()
plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=120); plt.close()

# ── Plot 2: quality metrics trend across available checkpoints ──────────────
ckpt_steps = [r["step"] for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(ckpt_steps, [r["latent_cos"] for r in rows], "o-")
axes[0].set_title("latent cosine sim (gen vs real)"); axes[0].set_xlabel("step"); axes[0].set_ylim(0, 1)
axes[1].plot(ckpt_steps, [r["std_ratio"] for r in rows], "o-")
axes[1].axhline(1.0, color="gray", linestyle="--", label="real variance")
axes[1].set_title("generated / real latent std ratio\n(<1 = variance collapse = 'muddy')")
axes[1].set_xlabel("step"); axes[1].legend()
axes[2].plot(ckpt_steps, [r["snr_gen_db"] for r in rows], "o-", label="dit_gen SNR")
axes[2].axhline(vae_ceiling_snr, color="gray", linestyle="--", label="VAE ceiling")
axes[2].set_title("SNR vs real clip"); axes[2].set_xlabel("step"); axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "metrics_trend.png"), dpi=120); plt.close()

# ── Plot 3: per-channel latent std, real vs generated (latest checkpoint) ───
plt.figure(figsize=(10, 4))
plt.plot(real_std_perchan.numpy(), label="real latents (population)", linewidth=2)
plt.plot(latest_perchan.numpy(), label=f"DiT-generated (step {latest_step})", linewidth=2)
plt.xlabel("latent channel"); plt.ylabel("std"); plt.legend()
plt.title("Per-channel latent std: variance collapse check")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "latent_variance.png"), dpi=120); plt.close()

# ── Plot 4: spectrograms, real / vae_recon / dit_gen (latest checkpoint) ────
i = 0
win  = torch.hann_window(1024)
sigs = [audio[i, 0].cpu(), audio_vae_rec[i, 0].cpu(), latest_gen_audio[i, 0].cpu()]
Ss   = [torch.stft(s.float(), 1024, 256, 1024, win, return_complex=True).abs()
          .clamp(min=1e-7).log10() for s in sigs]
vmin = min(S.min().item() for S in Ss); vmax = max(S.max().item() for S in Ss)
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
fig.suptitle(f"clip 0  (step {latest_step})  |  tags={tag_lists[i]}")
for ax, S, label in zip(axes, Ss, ["real", "vae_recon (ceiling)", "dit_gen"]):
    ax.imshow(S.numpy(), origin="lower", aspect="auto",
              extent=[0, S.shape[1] / (config.SAMPLE_RATE / 256), 0, config.SAMPLE_RATE // 2 / 1000],
              vmin=vmin, vmax=vmax, cmap="magma")
    ax.set_title(label); ax.set_xlabel("time (s)"); ax.set_ylabel("freq (kHz)")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "spectrograms.png"), dpi=120); plt.close()

# ── Save the audio itself for listening ──────────────────────────────────────
def write_audio(path, x):
    """
    Peak-normalize only if the clip would otherwise clip. sf.write() defaults to PCM_16 for
    .wav and HARD-CLIPS outside [-1, 1]; measured, 8 of 16 generated eval clips exceeded it
    (peak 2.45). Hard clipping manufactures broadband distortion — it invents the very
    noise artifact this script exists to measure.
    """
    x = x.cpu().numpy()
    peak = float(abs(x).max())
    sf.write(path, x / peak if peak > 1.0 else x, config.SAMPLE_RATE)


for i in range(N_CLIPS):
    write_audio(os.path.join(run_dir, f"clip{i}_real.wav"), audio[i, 0])
    write_audio(os.path.join(run_dir, f"clip{i}_vae_recon.wav"), audio_vae_rec[i, 0])
    write_audio(os.path.join(run_dir, f"clip{i}_dit_gen.wav"), latest_gen_audio[i, 0])

print(f"\nSaved metrics, plots, and audio to {run_dir}")
