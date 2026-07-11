import os
import time
import numpy as np
import torch
import librosa
import soundfile as sf

AUDIO_ROOT = "/home/bhuvan/6DGS/music-gen/data/jamendo_audio_low"
OUT_DIR = "/home/bhuvan/6DGS/music-gen/data/test_stems"
TARGET_SR = 22050

TRACKS = [
    ("track_0043411", "11/43411.low.mp3"),
    ("track_0043412", "12/43412.low.mp3"),
    ("track_0043413", "13/43413.low.mp3"),
    ("track_0043415", "15/43415.low.mp3"),
    ("track_0043416", "16/43416.low.mp3"),
]

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading htdemucs model...")
from demucs.pretrained import get_model
from demucs.apply import apply_model

model = get_model("htdemucs")
model.to(device)
model.eval()
print(f"Model loaded. Sources: {model.sources}")
vocals_idx = model.sources.index("vocals")

for track_id, rel_path in TRACKS:
    src = os.path.join(AUDIO_ROOT, rel_path)
    t0 = time.time()

    # Load at model's native SR (44100), stereo
    y, _ = librosa.load(src, sr=model.samplerate, mono=False)  # (C, T) or (T,)
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] > 2:
        y = y[:2]

    mix = torch.from_numpy(y.copy()).unsqueeze(0).to(device)  # (1, 2, T)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            sources = apply_model(model, mix, device=device, progress=False)
            # sources: (1, num_sources, 2, T)

    sources = sources.squeeze(0).float().cpu().numpy()  # (num_sources, 2, T)

    no_vocals = sources.sum(0) - sources[vocals_idx]  # (2, T) sum of non-vocal stems
    vocals    = sources[vocals_idx]                    # (2, T)

    # Collapse stereo → mono
    no_vocals_mono = no_vocals.mean(0)  # (T,)
    vocals_mono    = vocals.mean(0)

    # Resample to 22050
    no_vocals_mono = librosa.resample(no_vocals_mono, orig_sr=model.samplerate, target_sr=TARGET_SR)
    vocals_mono    = librosa.resample(vocals_mono,    orig_sr=model.samplerate, target_sr=TARGET_SR)

    # Normalise peak to avoid clipping
    for arr in (no_vocals_mono, vocals_mono):
        peak = np.abs(arr).max()
        if peak > 1.0:
            arr /= peak

    # Original mix downsampled to 22050 mono for direct comparison
    original_mono = librosa.resample(y.mean(0), orig_sr=model.samplerate, target_sr=TARGET_SR)

    sf.write(os.path.join(OUT_DIR, f"{track_id}_original.wav"),  original_mono,  TARGET_SR)
    sf.write(os.path.join(OUT_DIR, f"{track_id}_no_vocals.wav"), no_vocals_mono, TARGET_SR)
    sf.write(os.path.join(OUT_DIR, f"{track_id}_vocals.wav"),    vocals_mono,    TARGET_SR)

    elapsed = time.time() - t0
    dur_s = len(no_vocals_mono) / TARGET_SR
    print(f"{track_id}: {dur_s:.1f}s audio in {elapsed:.1f}s ({dur_s/elapsed:.1f}x realtime)")

print(f"\nDone. Stems in {OUT_DIR}")
