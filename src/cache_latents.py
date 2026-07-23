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
    # This job is FLAC-DECODE bound, not GPU bound. At the old defaults (batch 8, 4 workers) it
    # encoded 89 tracks/min with the A100 sitting at 0.68 GB and mostly idle — a 10.4-hour run for
    # 55,609 tracks. The GPU was never the constraint; the dataloader was.
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--workers", type=int, default=16,
                        help="FLAC-decode workers. This is the actual bottleneck. Keep well under "
                             "nproc if anything else CPU-heavy is running (e.g. extract_features "
                             "with 32 workers) or the two jobs just starve each other.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tracks that already have a cached .pt file")
    parser.add_argument("--deterministic", action="store_true",
                        help="Encode the FIRST 30s of each track instead of a random crop. "
                             "Required for frame-alignment with extract_features.py.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only encode the first N tracks. Iterates the SAME sorted FLAC list as "
                             "extract_features.py --limit N, so both cover an identical track set.")
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
    ds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH,
                        deterministic_crop=args.deterministic)
    ds.files = [f for f in ds.files if f.endswith(".flac")]
    if args.deterministic:
        print("Deterministic crop: encoding the first 30s of each track (feature-aligned).")
    if args.limit:
        ds.files = ds.files[: args.limit]
        print(f"Limit: first {len(ds.files)} tracks (must match extract_features.py --limit).")

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
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"batch {args.batch_size}, {args.workers} decode workers")

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
                # .clone() is LOAD-BEARING. mu[i] is a VIEW into the (B, 32, 645) batch tensor,
                # and torch.save serializes a view's ENTIRE underlying storage — so every file
                # silently contained the whole batch. Measured: 0.083 MB of real data written as
                # 0.662 MB at batch 8 (8x) and 5.285 MB at batch 64 (64x). That is 295 GB for the
                # full set instead of 4.6 GB, and since this job is disk-write bound it was also
                # the reason raising the batch size did not speed it up: bigger batch = bigger
                # bloat = more bytes per track. clone() materializes just this slice.
                torch.save(mu[i].clone(), out)
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
