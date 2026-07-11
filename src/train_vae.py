"""
VAE-GAN training loop.

Usage:
  python src/train_vae.py
  python src/train_vae.py --resume checkpoints/vae/vae_step0010000.pt
  python src/train_vae.py --resume checkpoints/vae/vae_step0010000.pt --no-adv
"""

import argparse
import csv
import glob
import os
import random
import sys

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

# train_vae.py lives in src/, but config.py is in the project root one level up.
# Add the root to the path so `import config` / `from src....` work when run as
# `python src/train_vae.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.dataset import JamendoDataset, collate_fn
from src.losses import (MultiScaleMelLoss, adaptive_weight, adv_d_loss,
                        adv_g_loss, feat_match_loss)
from src.vae import MultiSTFTDiscriminator, WaveformVAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(step, vae, disc, opt_g, opt_d):
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    path = os.path.join(config.CKPT_DIR, f"vae_step{step:07d}.pt")
    torch.save({
        "step": step,
        "vae":  vae.state_dict(),
        "disc": disc.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }, path)
    # Keep only the last VAE_KEEP_LAST checkpoints
    ckpts = sorted(glob.glob(os.path.join(config.CKPT_DIR, "vae_step*.pt")))
    for old in ckpts[:-config.VAE_KEEP_LAST]:
        os.remove(old)
    return path


def save_audio_samples(step, vae, fixed_clips, device):
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    vae.eval()
    with torch.no_grad():
        for i, clip in enumerate(fixed_clips):
            real = clip.unsqueeze(0).to(device)   # (1, 1, T)
            recon, _, _ = vae(real)
            real_np  = real[0, 0].cpu().float().numpy()
            recon_np = recon[0, 0].cpu().float().numpy()
            sf.write(
                os.path.join(config.SAMPLE_DIR, f"step{step:07d}_{i}_real.wav"),
                real_np, config.SAMPLE_RATE,
            )
            sf.write(
                os.path.join(config.SAMPLE_DIR, f"step{step:07d}_{i}_recon.wav"),
                recon_np, config.SAMPLE_RATE,
            )
    vae.train()


def log_row(writer, step, row):
    writer.writerow({"step": step, **row})


def collapse_check(step, adv_d_val):
    if adv_d_val is not None and adv_d_val < 0.05:
        print(f"\n⚠  COLLAPSE WARNING step {step}: adv_d={adv_d_val:.4f} — discriminator has won")
        print("   Consider: --no-adv restart from last checkpoint, or lower VAE_ADV_WEIGHT in config.py\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Path to VAE checkpoint to resume from")
    parser.add_argument("--no-adv", action="store_true", help="Disable adversarial loss (STFT+KL only)")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Dataset & dataloader
    ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH)
    dl = DataLoader(
        ds,
        batch_size=config.VAE_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.VAE_NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Models
    vae  = WaveformVAE().to(device)
    disc = MultiSTFTDiscriminator().to(device)

    # Optimizers
    opt_g = torch.optim.AdamW(vae.parameters(),  lr=config.VAE_LR, betas=(0.5, 0.9), eps=1e-6)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=config.VAE_LR, betas=(0.5, 0.9), eps=1e-6)

    # Loss
    mel_loss = MultiScaleMelLoss().to(device)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["vae"])
        disc.load_state_dict(ckpt["disc"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        step = ckpt["step"]
        print(f"Resumed from step {step}")

    # Grab 3 fixed clips for audio samples (taken once from first batch)
    fixed_clips = None

    # CSV logging
    csv_fields = ["step", "mel", "wave", "kl", "adv_g", "adv_d", "feat", "Dr", "Df", "lam", "gN", "dN"]
    csv_exists = os.path.exists(config.STATS_CSV) and args.resume
    stats_file = open(config.STATS_CSV, "a" if csv_exists else "w", newline="")
    csv_writer = csv.DictWriter(stats_file, fieldnames=csv_fields)
    if not csv_exists:
        csv_writer.writeheader()


    print(f"VAE params:  {sum(p.numel() for p in vae.parameters())/1e6:.1f}M")
    print(f"Disc params: {sum(p.numel() for p in disc.parameters())/1e6:.1f}M")
    print(f"ADV starts at step {config.VAE_ADV_START}"
          + (" (disabled via --no-adv)" if args.no_adv else ""))
    print(f"Checkpoints every {config.VAE_CKPT_EVERY} steps → {config.CKPT_DIR}/")
    print(f"Audio samples every {config.VAE_SAMPLE_EVERY} steps → {config.SAMPLE_DIR}/")
    print()

    vae.train()
    disc.train()

    while step < config.VAE_TOTAL_STEPS:
        for batch, _ in dl:
            if step >= config.VAE_TOTAL_STEPS:
                break

            real = batch.to(device)   # (B, 1, 660480)

            # Collect fixed clips once (full length) for sample saving
            if fixed_clips is None:
                fixed_clips = [real[i].cpu() for i in range(min(3, real.shape[0]))]

            # Train on a short random crop — the VAE is convolutional, so it learns
            # local audio structure from short segments (≪ VRAM, faster steps).
            # Full 30s clips are still encoded later at cache time.
            if real.shape[-1] > config.VAE_TRAIN_SAMPLES:
                s = random.randint(0, real.shape[-1] - config.VAE_TRAIN_SAMPLES)
                real = real[..., s:s + config.VAE_TRAIN_SAMPLES]

            # ── Generator step ────────────────────────────────────────────────
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                recon, mu, logvar = vae(real)

            l_mel   = mel_loss(real, recon)
            l_wave  = F.l1_loss(recon.float(), real)     # time-domain term anchors phase
            l_kl    = 0.5 * (mu.float().pow(2) + logvar.float().exp() - logvar.float() - 1).mean()
            kl_w    = config.VAE_KL_WEIGHT * min(1.0, step / config.VAE_KL_WARMUP)
            recon_loss = l_mel + config.VAE_WAVE_WEIGHT * l_wave + kl_w * l_kl

            use_adv = (not args.no_adv) and (step >= config.VAE_ADV_START)
            lam = l_adv = l_feat = l_dval = d_r = d_f = 0.0

            if use_adv:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    d_fake = disc(recon)
                    d_real_fm = disc(real.detach())

                l_adv  = adv_g_loss(d_fake)
                l_feat = feat_match_loss(d_real_fm, d_fake)
                lam    = adaptive_weight(
                    recon_loss, l_adv, vae.last_layer, config.VAE_LAMBDA_MAX
                )
                g_loss = recon_loss + config.VAE_ADV_WEIGHT * lam * l_adv + config.VAE_FEAT_WEIGHT * l_feat
            else:
                g_loss = recon_loss

            opt_g.zero_grad()
            g_loss.backward()
            gN = clip_grad_norm_(vae.parameters(), config.VAE_GRAD_CLIP).item()
            opt_g.step()

            # ── Discriminator step ────────────────────────────────────────────
            dN = 0.0
            if use_adv:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    d_real = disc(real)
                    d_fake_det = disc(recon.detach())

                l_dval = adv_d_loss(d_real, d_fake_det)
                opt_d.zero_grad()
                l_dval.backward()
                dN = clip_grad_norm_(disc.parameters(), config.VAE_GRAD_CLIP).item()
                opt_d.step()

                # Average discriminator logits for health monitoring
                d_r = sum(logit.mean().item() for logit, _ in d_real)  / len(d_real)
                d_f = sum(logit.mean().item() for logit, _ in d_fake_det) / len(d_fake_det)

            step += 1

            # ── Logging ───────────────────────────────────────────────────────
            if step % config.VAE_LOG_EVERY == 0:
                mel_v   = l_mel.item()
                wave_v  = l_wave.item()
                kl_v    = l_kl.item()
                adv_g_v = l_adv.item() if use_adv else 0.0
                adv_d_v = l_dval.item() if use_adv else 0.0
                feat_v  = l_feat.item() if use_adv else 0.0
                lam_v   = lam.item() if use_adv else 0.0

                vram = torch.cuda.max_memory_allocated() / 1e9   # peak so far; cap is 40 GB
                if use_adv:
                    line = (f"step {step:7d} | mel {mel_v:.4f} | wave {wave_v:.4f} | kl {kl_v:.5f} | "
                            f"adv_g {adv_g_v:.4f} | adv_d {adv_d_v:.4f} | feat {feat_v:.4f} | "
                            f"Dr {d_r:+.2f} Df {d_f:+.2f} | lam {lam_v:.3f} | gN {gN:.2f} dN {dN:.2f} | vram {vram:.1f}GB")
                else:
                    line = (f"step {step:7d} | mel {mel_v:.4f} | wave {wave_v:.4f} | kl {kl_v:.5f} | "
                            f"gN {gN:.2f} | vram {vram:.1f}GB")
                print(line)
                sys.stdout.flush()

                csv_writer.writerow({
                    "step": step,
                    "mel": mel_v, "wave": wave_v, "kl": kl_v,
                    "adv_g": adv_g_v, "adv_d": adv_d_v, "feat": feat_v,
                    "Dr": d_r, "Df": d_f,
                    "lam": lam_v, "gN": gN, "dN": dN,
                })
                stats_file.flush()

                collapse_check(step, adv_d_v if use_adv else None)

            # ── Checkpoints ───────────────────────────────────────────────────
            if step % config.VAE_CKPT_EVERY == 0:
                path = save_checkpoint(step, vae, disc, opt_g, opt_d)
                print(f"  → saved {path}")

            # ── Audio samples ─────────────────────────────────────────────────
            if step % config.VAE_SAMPLE_EVERY == 0 and fixed_clips is not None:
                save_audio_samples(step, vae, fixed_clips, device)
                print(f"  → audio samples saved to {config.SAMPLE_DIR}/")

    stats_file.close()
    save_checkpoint(step, vae, disc, opt_g, opt_d)
    print("Training complete.")


if __name__ == "__main__":
    main()
