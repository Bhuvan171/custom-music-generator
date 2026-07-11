"""
Vocal stripping for the full MTG-Jamendo dataset.

- Reads all tracks from autotagging.tsv
- Batches B tracks through htdemucs (pads to max len in batch, trims after)
- Saves no_vocals stem as 22050 Hz mono FLAC (PCM_16, lossless)
- Deletes original mp3 after stem is confirmed on disk (keeps disk usage flat)
- Resumable: skips tracks whose .flac stem already exists
- Logs progress + ETA to both stdout and strip_full.log
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import librosa
import soundfile as sf

TSV_PATH   = "/home/bhuvan/6DGS/music-gen/mtg-jamendo-dataset/data/autotagging.tsv"
AUDIO_ROOT = "/home/bhuvan/6DGS/music-gen/data/jamendo_audio_low"
STEMS_ROOT = "/home/bhuvan/6DGS/music-gen/data/stems"
LOG_PATH   = "/home/bhuvan/6DGS/music-gen/strip_full.log"
TARGET_SR  = 22050
BATCH_SIZE = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger()

# ── Build track list ─────────────────────────────────────────────────────────
tracks = []
with open(TSV_PATH) as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        rel_path = parts[3]                                # e.g. "11/43411.mp3"
        stem     = rel_path.rsplit(".", 1)[0]              # "11/43411"
        src      = os.path.join(AUDIO_ROOT, stem + ".low.mp3")
        subdir, fname = os.path.split(stem)
        out      = os.path.join(STEMS_ROOT, subdir, fname + ".flac")
        tracks.append((parts[0], src, out))

total   = len(tracks)
pending = [(tid, src, out) for tid, src, out in tracks
           if os.path.exists(src) and not os.path.exists(out)]

log.info(f"Total tracks: {total} | Pending: {len(pending)}")

if not pending:
    log.info("Nothing to do — all stems already exist.")
    sys.exit(0)

# ── Load model ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")
log.info("Loading htdemucs model...")

from demucs.pretrained import get_model
from demucs.apply import apply_model

model = get_model("htdemucs")
model.to(device)
model.eval()
MODEL_SR   = model.samplerate          # 44100
vocals_idx = model.sources.index("vocals")
log.info(f"Model ready | SR={MODEL_SR} | sources={model.sources}")

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_stereo(path):
    """Return (2, T) float32 array at MODEL_SR."""
    y, _ = librosa.load(path, sr=MODEL_SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] > 2:
        y = y[:2]
    return y.astype(np.float32)

# ── Main loop ────────────────────────────────────────────────────────────────
completed = 0
t_start   = time.time()

for batch_start in range(0, len(pending), BATCH_SIZE):
    batch = pending[batch_start : batch_start + BATCH_SIZE]

    wavs, lengths, valid = [], [], []
    for tid, src, out in batch:
        try:
            w = load_stereo(src)
            wavs.append(w)
            lengths.append(w.shape[1])
            valid.append((tid, src, out))
        except Exception as e:
            log.warning(f"SKIP {tid}: {e}")

    if not wavs:
        continue

    # Pad batch to max length
    max_len = max(lengths)
    padded  = np.zeros((len(wavs), 2, max_len), dtype=np.float32)
    for i, w in enumerate(wavs):
        padded[i, :, :w.shape[1]] = w

    mix = torch.from_numpy(padded).to(device)  # (B, 2, T)

    with torch.no_grad():
        sources = apply_model(model, mix, device=device, progress=False)
        # (B, num_sources, 2, T)

    sources = sources.float().cpu().numpy()

    for i, (tid, src, out) in enumerate(valid):
        T       = lengths[i]
        stems_i = sources[i, :, :, :T]                        # (S, 2, T) trimmed
        no_vox  = stems_i.sum(0) - stems_i[vocals_idx]        # (2, T)
        mono    = no_vox.mean(0)                               # (T,)

        mono = librosa.resample(mono, orig_sr=MODEL_SR, target_sr=TARGET_SR)

        peak = np.abs(mono).max()
        if peak > 1.0:
            mono = mono / peak

        os.makedirs(os.path.dirname(out), exist_ok=True)
        sf.write(out, mono, TARGET_SR, subtype="PCM_16", format="FLAC")

        if os.path.exists(out):
            os.remove(src)

        completed += 1

    elapsed   = time.time() - t_start
    rate      = completed / elapsed
    remaining = len(pending) - (batch_start + len(batch))
    eta_h     = (remaining / rate) / 3600 if rate > 0 else float("inf")
    log.info(
        f"[{completed}/{len(pending)}] "
        f"{rate:.2f} tracks/s | ETA {eta_h:.1f}h"
    )

log.info(f"Done. {completed} stems saved to {STEMS_ROOT}")
