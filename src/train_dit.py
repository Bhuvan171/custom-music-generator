"""
DiT training loop — flow matching on cached, per-channel-normalized VAE latents.

Prerequisites:
  1. src/cache_latents.py   — populates config.LATENTS_DIR with raw *.pt latents
  2. compute_latent_stats.py — writes config.LATENT_STATS_PATH (per-channel mean/std)

Targets are normalized to ~unit variance per channel before flow matching so the
z1 (data) distribution matches the z0 ~ N(0,1) noise prior scale-for-scale, and so
MSE loss doesn't over-weight the highest-variance latent channels. Generated
latents are in normalized space — denormalize (z * std + mean) before vae.decode().

Usage:
  python src/train_dit.py
  python src/train_dit.py --resume checkpoints/dit/dit_step0010000.pt
"""

import argparse
import copy
import csv
import glob
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.dataset import JamendoDataset, collate_fn
from src.dit import MusicDiT


# ── Helpers ──────────────────────────────────────────────────────────────────

def ema_update(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.lerp_(p, 1.0 - decay)


def save_checkpoint(step, model, ema_model, opt, scheduler):
    os.makedirs(config.DIT_CKPT_DIR, exist_ok=True)
    path = os.path.join(config.DIT_CKPT_DIR, f"dit_step{step:07d}.pt")
    torch.save({
        "step":      step,
        "dit":       model.state_dict(),
        "ema":       ema_model.state_dict(),
        "opt":       opt.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    ckpts = sorted(glob.glob(os.path.join(config.DIT_CKPT_DIR, "dit_step*.pt")))
    for old in ckpts[:-config.DIT_KEEP_LAST]:
        os.remove(old)
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Path to DiT checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Dataset: pre-cached latent .pt files from config.LATENTS_DIR
    ds = JamendoDataset(config.LATENTS_DIR, config.TSV_PATH, config.VOCAB_PATH, normalize_latents=True)
    if not any(f.endswith(".pt") for f in ds.files):
        print("ERROR: No .pt latent files found in", config.LATENTS_DIR)
        print("Run: python src/cache_latents.py --checkpoint <vae_checkpoint>")
        sys.exit(1)
    # Keep only .pt files (latents, not raw audio)
    ds.files = [f for f in ds.files if f.endswith(".pt")]
    print(f"DiT dataset: {len(ds.files)} latent files")

    dl = DataLoader(
        ds,
        batch_size=config.DIT_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.DIT_NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model     = MusicDiT().to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    # AdamW with default betas — flow matching is not adversarial, no GAN instability
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.DIT_LR,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
    )
    def lr_lambda(s):
        if s < config.DIT_WARMUP:
            return s / max(1, config.DIT_WARMUP)
        progress = (s - config.DIT_WARMUP) / max(1, config.DIT_TOTAL_STEPS - config.DIT_WARMUP)
        cosine   = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return config.DIT_LR_MIN_RATIO + (1 - config.DIT_LR_MIN_RATIO) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["dit"])
        ema_model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        print(f"Resumed from step {step}")

    csv_exists  = os.path.exists(config.DIT_STATS_CSV) and args.resume
    stats_file  = open(config.DIT_STATS_CSV, "a" if csv_exists else "w", newline="")
    csv_writer  = csv.DictWriter(stats_file, fieldnames=["step", "loss", "lr", "vram"])
    if not csv_exists:
        csv_writer.writeheader()

    print(f"DiT params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Latent: ({config.VAE_LATENT_DIM}, {config.VAE_LATENT_LEN})   "
          f"Tags: {config.DIT_VOCAB_SIZE}   D={config.DIT_D_MODEL}   L={config.DIT_LAYERS}")
    print()

    model.train()
    while step < config.DIT_TOTAL_STEPS:
        for latents, tag_lists in dl:
            if step >= config.DIT_TOTAL_STEPS:
                break

            # z1: clean latents (B, 32, 645)
            z1 = latents.to(device)
            B  = z1.shape[0]

            # ── Flow matching ─────────────────────────────────────────────────
            t        = torch.rand(B, device=device)                            # (B,) ∈ [0,1]
            z0       = torch.randn_like(z1)                                    # noise
            z_t      = (1 - t[:, None, None]) * z0 + t[:, None, None] * z1   # interpolated
            v_target = z1 - z0                                                 # velocity

            # CFG: randomly drop conditioning
            drop_mask = torch.rand(B) < config.DIT_CFG_DROPOUT               # (B,) bool

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_pred = model(z_t, t, tag_lists, drop_mask=drop_mask)
                loss   = F.mse_loss(v_pred.float(), v_target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            ema_update(ema_model, model, config.DIT_EMA_DECAY)

            step += 1

            if step % config.DIT_LOG_EVERY == 0:
                lr   = opt.param_groups[0]["lr"]
                vram = torch.cuda.max_memory_allocated() / 1e9
                print(f"step {step:7d} | loss {loss.item():.6f} | lr {lr:.2e} | vram {vram:.1f}GB",
                      flush=True)
                csv_writer.writerow({"step": step, "loss": loss.item(), "lr": lr, "vram": vram})
                stats_file.flush()

            if step % config.DIT_CKPT_EVERY == 0:
                path = save_checkpoint(step, model, ema_model, opt, scheduler)
                print(f"  → saved {path}")

    stats_file.close()
    save_checkpoint(step, model, ema_model, opt, scheduler)
    print("Training complete.")


if __name__ == "__main__":
    main()
