"""
Cache VAE latents for DiT training.

Encodes all FLAC files using the frozen VAE (mu only, no sampling noise)
and saves them as .pt files in config.LATENTS_DIR.

Run this ONCE after VAE training is complete:
  python src/cache_latents.py --checkpoint checkpoints/vae/vae_step0200000.pt

Each output file: {track_id}.pt  shape (VAE_LATENT_DIM, VAE_LATENT_LEN) = (32, 645)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.dataset import JamendoDataset, collate_fn
from src.vae import WaveformVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to trained VAE checkpoint")
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tracks that already have a cached .pt file")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load frozen VAE
    vae  = WaveformVAE().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    print(f"Loaded VAE from {args.checkpoint}")
    print(f"Latent shape per clip: ({config.VAE_LATENT_DIM}, {config.VAE_LATENT_LEN})")

    # Dataset: FLAC files only (skip any .pt files already there)
    ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH)
    ds.files = [f for f in ds.files if f.endswith(".flac")]

    os.makedirs(config.LATENTS_DIR, exist_ok=True)

    if args.skip_existing:
        existing = set(os.listdir(config.LATENTS_DIR))
        before   = len(ds.files)
        ds.files = [
            f for f in ds.files
            if f"track_{int(Path(f).stem.split('_')[0]):07d}.pt" not in existing
        ]
        print(f"Skipping {before - len(ds.files)} already-cached files.")

    print(f"Encoding {len(ds.files)} FLAC files → {config.LATENTS_DIR}/")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.VAE_NUM_WORKERS,
        pin_memory=True,
    )

    done = 0
    with torch.no_grad():
        for batch_idx, (audio, _) in enumerate(dl):
            audio = audio.to(device)   # (B, 1, 660480)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mu, _ = vae.encode(audio)   # (B, 32, 645)
            mu = mu.float().cpu()

            start = batch_idx * args.batch_size
            for i in range(mu.shape[0]):
                src  = ds.files[start + i]
                stem = Path(src).stem.split("_")[0]
                out  = os.path.join(config.LATENTS_DIR, f"track_{int(stem):07d}.pt")
                torch.save(mu[i], out)
                done += 1

            if (batch_idx + 1) % 100 == 0:
                vram = torch.cuda.max_memory_allocated() / 1e9
                print(f"  {done}/{len(ds.files)} encoded  (peak vram {vram:.2f} GB)", flush=True)

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nDone. {done} latents saved to {config.LATENTS_DIR}/")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Next: python src/train_dit.py")


if __name__ == "__main__":
    main()
