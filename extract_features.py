"""
Extract musical conditioning features from each track's audio.

Why: the 195-way tag vocabulary carries only ~30-40 bits per track (mean 3.3 tags),
while a 30s latent needs orders of magnitude more information to be determined. A
flow model trained under that deficit predicts the conditional MEAN and produces
smeared ("muddy") audio. These features put musical content bits back:

  tempo (BPM), key, mode  -> global scalars, ~10-20 bits. Cheap but marginal alone.
  chroma (12 x 645)       -> per-frame harmony/melody contour. THE high-bit signal,
  rms    (1  x 645)       -> per-frame energy envelope (dynamics).

CRITICAL — crop alignment: features are extracted from the FIRST CHUNK_SAMPLES of
each track, matching JamendoDataset(deterministic_crop=True). The latents MUST be
re-cached with the same deterministic crop (cache_latents.py --deterministic), or
the chroma will describe a different slice of audio than the latent it conditions
and the signal becomes noise.

Output: data/features/track_XXXXXXX.pt  {tempo, key, mode, chroma (13,645) fp16}
        (chroma tensor is 12 chroma bins + 1 rms channel, stacked)

Usage:
  python extract_features.py                 # all tracks
  python extract_features.py --limit 5000    # subset (for a quick fine-tune test)
  python extract_features.py --workers 32
"""

import argparse
import os
import warnings
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

import config
from src.features import extract_all

def process(args):
    path, out_dir = args
    import librosa
    track_id = f"track_{int(Path(path).stem.split('_')[0]):07d}"
    out = os.path.join(out_dir, f"{track_id}.pt")
    # Existence is NOT enough, for three reasons:
    #  1. a job killed mid-torch.save leaves a truncated/0-byte file that a naive
    #     `if exists: skip` keeps forever, surfacing much later as an EOFError inside a
    #     DataLoader worker;
    #  2. files written before texture conditioning existed lack the "texture" key;
    #  3. files written before FEATURE_VERSION 2 have per-frame-normalized chroma, which is
    #     the exact defect this rewrite fixes — an "already exists" skip would keep them.
    # So: load it, and re-extract if it is broken OR stale.
    if os.path.exists(out):
        try:
            d = torch.load(out, weights_only=True)
            if "texture" in d and d.get("version") == config.FEATURE_VERSION:
                return "skip"
        except Exception:
            pass
        os.remove(out)
    try:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        # Match JamendoDataset(deterministic_crop=True) exactly: first CHUNK_SAMPLES, zero-pad if short.
        n = config.CHUNK_SAMPLES
        audio = audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))

        # ONE definition, in src/features.py, shared with generate.py (user reference tracks) and
        # train_dit's chroma_adherence metric. A reference processed even slightly differently
        # from the training data is out-of-distribution conditioning — worse than none at all.
        f = extract_all(audio, sr)

        # Atomic write: save to a temp file, then rename. os.replace is atomic on POSIX, so
        # an interrupted run can leave a stray .tmp but NEVER a half-written .pt that later
        # blows up a DataLoader worker.
        tmp = out + ".tmp"
        torch.save({
            "version": config.FEATURE_VERSION,
            "tempo":   f["tempo"],
            "key":     f["key"],
            "mode":    f["mode"],
            "chroma":  torch.from_numpy(f["chroma"]).half(),   # (13, 645) fp16
            "texture": torch.from_numpy(f["texture"]).half(),  # (4,  645) fp16
        }, tmp)
        os.replace(tmp, out)
        return "ok"
    except Exception as e:
        return f"fail {track_id}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process the first N tracks")
    ap.add_argument("--start", type=int, default=0,
                    help="skip the first N tracks. With --limit, selects the slice [start:start+limit] "
                         "of the SAME sorted FLAC list cache_latents.py --limit uses, so a held-out "
                         "slice is guaranteed disjoint from the training slice.")
    ap.add_argument("--out-dir", default=config.FEATURES_DIR,
                    help="where to write. Use config.FEATURES_VAL_DIR for a HELD-OUT eval set — it "
                         "must NOT go in FEATURES_DIR, or train_dit.py (train set = latents INTERSECT "
                         "features) would pull those tracks into training.")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob(os.path.join(config.STEMS_DIR, "**", "*.flac"), recursive=True))
    files = files[args.start:]
    if args.limit:
        files = files[: args.limit]
    print(f"Extracting features v{config.FEATURE_VERSION} for {len(files)} tracks "
          f"[{args.start}:{args.start + len(files)}] -> {args.out_dir}/  ({args.workers} workers)")

    ok = skip = fail = 0
    work = [(f, args.out_dir) for f in files]
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process, work, chunksize=16), 1):
            if r == "ok":     ok += 1
            elif r == "skip": skip += 1
            else:
                fail += 1
                if fail <= 5:
                    print("  ", r)
            if i % 2000 == 0:
                print(f"  {i}/{len(files)}  ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"\nDone. ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
