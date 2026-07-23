"""
THE single definition of the chroma conditioning signal.

Both sides of the loop must use the same one:
  - extract_features.py  builds the chroma the model is CONDITIONED on
  - train_dit.chroma_adherence  extracts chroma from the GENERATED audio to check whether the
    model actually played what it was told

When these drift apart the metric silently compares different spaces. That happened: extraction
moved to norm=None + log compression while the metric still called librosa's default (norm=inf,
non-negative, every frame rescaled to max 1.0) and cosine-compared it against a standardized,
zero-mean target. The measured chroma ceiling fell from +0.512 to +0.313 purely from the mismatch —
the metric was reading worse because it was broken, not because the model was.

Keeping the definition here makes that class of drift impossible: change the recipe once and both
the conditioning and its own metric move together.
"""

import numpy as np

HOP  = 1024   # CHUNK_SAMPLES // HOP = 645 = VAE_LATENT_LEN -> frame-aligned with the latent
NFFT = 2048


def fit_frames(x, L):
    """Trim/pad a (C, T) feature to exactly L frames so it is latent-aligned."""
    x = x[:, :L]
    if x.shape[1] < L:
        x = np.pad(x, ((0, 0), (0, L - x.shape[1])))
    return x


def chroma_from_stft(S, sr, L):
    """
    Power spectrogram -> (12, L) energy-preserving chroma.

    norm=None is LOAD-BEARING. librosa's default (norm=inf) rescales EVERY frame so its max bin
    is exactly 1.0 — measured on a real track, the quietest and loudest frames both came back
    with max == 1.0000 despite a ~36,000x energy difference, and 11.6% of all training frames are
    near-silent. That fed the model 12 channels of amplified noise shaped like a confident chord
    wherever the music was quiet: actively wrong conditioning, not merely uninformative.

    Compression uses a PER-CLIP scale, not per-frame: relative loudness BETWEEN frames survives
    (silence -> ~0) while recording-level differences BETWEEN clips are normalized away. The 99th
    percentile rather than the max, so a single stray transient cannot set the scale.
    """
    raw = fit_frames(_chroma_stft(S, sr), L)
    return np.log1p(raw / (np.percentile(raw, 99) + 1e-8))


def _chroma_stft(S, sr):
    import librosa
    return librosa.feature.chroma_stft(S=S, sr=sr, norm=None)


def chroma_from_audio(audio, sr, L):
    """Waveform -> (12, L) chroma, by exactly the recipe extract_features.py conditions on."""
    import librosa
    S = np.abs(librosa.stft(audio, n_fft=NFFT, hop_length=HOP)) ** 2
    return chroma_from_stft(S, sr, L)


# Krumhansl-Schmuckler key profiles (major / minor).
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma_mean):
    """Correlate the mean chroma against rotated major/minor profiles -> (key, mode)."""
    best = (-2.0, 0, 1)   # (score, key, mode)  mode: 1=major 0=minor
    for k in range(12):
        for mode, profile in ((1, KS_MAJOR), (0, KS_MINOR)):
            r = np.corrcoef(chroma_mean, np.roll(profile, k))[0, 1]
            if np.isfinite(r) and r > best[0]:
                best = (r, k, mode)
    return best[1], best[2]


def extract_all(audio, sr, L=None):
    """
    Waveform -> the complete conditioning bundle the DiT was trained on:
        {tempo, key, mode, chroma (13, L), texture (4, L)}

    This is THE single definition, shared by extract_features.py (which builds the training
    set) and generate.py (which conditions on a user-supplied reference track). Keeping it in
    one place is not cosmetic: chroma extraction and the metric that scored it previously drifted
    apart and silently compared different spaces, collapsing the measured ceiling from +0.512 to
    +0.313. A reference track processed even slightly differently from the training data is
    out-of-distribution conditioning, which is worse than no conditioning at all.

    `audio` must already be mono and exactly CHUNK_SAMPLES long (see load_reference).
    """
    import librosa
    import config
    L = L or config.VAE_LATENT_LEN

    # ONE STFT, reused for every feature: per-call recomputation cost 2.57s/track vs 0.65s shared.
    S = np.abs(librosa.stft(audio, n_fft=NFFT, hop_length=HOP))

    onset = librosa.onset.onset_strength(S=librosa.power_to_db(S ** 2), sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(librosa.beat.tempo(onset_envelope=onset, sr=sr))[0])

    chroma = chroma_from_stft(S ** 2, sr, L)                                  # (12, L)
    prof   = chroma.mean(axis=1)
    key, mode = estimate_key(prof / (prof.max() + 1e-8))                      # SHAPE only
    rms    = fit_frames(librosa.feature.rms(S=S, frame_length=NFFT), L)
    rms    = np.log1p(rms / (np.percentile(rms, 99) + 1e-8))
    harmony = np.concatenate([chroma, rms], axis=0)                           # (13, L)

    # TEXTURE: chroma says WHICH NOTES, nothing about how noisy/percussive a frame is.
    # All four channels scaled to ~0..1 so none dominates the shared projection.
    H, P    = librosa.decompose.hpss(S)
    perc    = fit_frames((P / (H + P + 1e-8)).mean(0, keepdims=True), L)      # drum-like vs tonal
    onset_f = fit_frames(onset[None], L)
    cent    = fit_frames(librosa.feature.spectral_centroid(S=S, sr=sr), L)    # brightness
    flat    = fit_frames(librosa.feature.spectral_flatness(S=S), L)           # HOW NOISY per frame
    texture = np.concatenate([
        np.log1p(onset_f) / 5.0,        # onset peaks ~17 -> log1p ~2.9
        perc,                           # already 0..1
        cent / (sr / 2.0),              # Hz -> fraction of Nyquist
        flat ** 0.25,                   # values are tiny (2e-4..5e-2); spread them out
    ], axis=0).astype(np.float32)                                             # (4, L)

    return {"tempo": tempo, "key": int(key), "mode": int(mode),
            "chroma": harmony.astype(np.float32), "texture": texture}


def load_reference(path, offset=0.0):
    """
    Any audio file -> mono, SAMPLE_RATE, exactly CHUNK_SAMPLES (30s), matching how the training
    clips were cropped. Shorter files are zero-padded; `offset` (seconds) picks a later window.
    """
    import librosa
    import config
    audio, sr = librosa.load(path, sr=config.SAMPLE_RATE, mono=True,
                             offset=offset, duration=config.CLIP_DURATION + 1.0)
    n = config.CHUNK_SAMPLES
    audio = audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))
    return audio.astype(np.float32), config.SAMPLE_RATE
