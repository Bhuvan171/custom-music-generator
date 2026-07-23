# Project Transfer: Reference-Conditioned Latent Flow Matching for Music Generation

Full technical handoff. If you are picking this up cold, read this top to bottom before touching
code — several of the mistakes documented here were made twice because the first fix wasn't
written down anywhere durable.

**Status as of 2026-07-19: nothing is currently training. The last completed run
(`checkpoints/dit_ft/dit_step0030000.pt`) is healthy and is the one to generate from. A resume
from it (`dit_stats.csv` last row, step 30500) diverged and was not saved over — see "The
unresolved divergence" below.**

**STANDING RULE — never run installs or launch jobs directly. Give the exact command back to the
user and let THEM run it. This applies to `pip install`, training launches, captioning/generation
runs, smoke tests, everything — no exceptions, no "just this one quick check." The user runs
commands in their own terminal; the assistant's job is to write correct commands, not execute
them. This was violated repeatedly in the vLLM install/debug session on 2026-07-20 and made the
user furious — do not repeat it.**

---

## Table of contents

1. [What this is](#what-this-is)
2. [Pipeline, stage by stage](#pipeline-stage-by-stage)
3. [Every file, what it does](#every-file-what-it-does)
4. [Data & checkpoint inventory (current, verified)](#data--checkpoint-inventory-current-verified)
5. [Config reference](#config-reference)
6. [The measurement audit — hard-won lessons](#the-measurement-audit--hard-won-lessons)
7. [Open problems](#open-problems)
8. [Natural-language generation: what exists, what doesn't](#natural-language-generation-what-exists-what-doesnt)
9. [Planned but not built](#planned-but-not-built)
10. [How to run everything](#how-to-run-everything)
11. [Traps for whoever touches this next](#traps-for-whoever-touches-this-next)

---

## What this is

A from-scratch, two-stage music generator, trained entirely on a single shared A100 80GB, no
pretrained weights anywhere:

```
Stage 1: Waveform VAE      30s audio (660,480 samples) <-> latent (32, 645)
Stage 2: Music DiT         flow matching in that latent space, conditioned on
                             - 195-way multi-label tags (genre/instrument/mood)
                             - a REFERENCE TRACK's chroma (harmony), energy, and texture
                               (onset/percussiveness/brightness/noisiness), per-frame
```

Dataset: MTG-Jamendo, 55,609 tracks, `data/stems/**/*.flac`.

The generation entrypoint is `generate.py`. It is NOT text-to-music in the CLAP/MusicGen sense —
there is no text encoder. A "prompt" is (a) tags, optionally resolved from free text via a
sentence-embedding retrieval layer (`src/text_tags.py`), and (b) a reference audio file whose
harmonic/rhythmic/energy contour is imposed on the output at near-exact fidelity. See
[Natural-language generation](#natural-language-generation-what-exists-what-doesnt) for exactly
what this can and cannot do — this was the subject of the most recent design discussion and the
user was NOT satisfied with either the text control or the raw audio quality (see
[Open problems](#open-problems)).

---

## Pipeline, stage by stage

### Stage 1 — Waveform VAE (`src/vae.py`, `src/train_vae.py`, `src/losses.py`)

Fully frame-rate architecture — only the initial STFT and the final overlap-add touch sample rate.

```
Encoder: waveform -> STFT (n_fft=512, hop=256) -> sign-log1p(real,imag) -> Conv1d proj (384ch)
         -> 8x ConvNeXt1d -> stride-4 downsample -> Conv1d -> mu, logvar  (32, 645) each
Decoder: z (32,645) -> Conv1d proj -> 4x linear upsample -> Conv1d refine -> 8x ConvNeXt1d
         -> per-frame Linear(384->512) -> Hann-window overlap-add (50%) -> waveform
```

20.4M params (enc+dec). Discriminator: EnCodec-style multi-scale STFT (n_fft in {512,1024,2048}),
5.3M params. Losses: multi-scale log-mel L1 (weight 1.0), waveform L1 (5.0, anchors phase), KL
(1e-4, 10k-step warmup), hinge adversarial (2.0 x adaptive VQGAN-style weight), feature matching
(2.0). AdamW betas=(0.5,0.9), lr=2e-4, batch 16, 3s random crops. Adversarial loss activates at
step 5,000. **Trained for the full 200,000 steps. Frozen since.** Checkpoint:
`checkpoints/vae/vae_step0200000.pt`.

**This is the current quality ceiling — see [Open problems](#open-problems).** Measured on
ground-truth latents (best case, no DiT involved): reconstruction SNR 5.88 dB, spectral flatness
1.43x real (adds broadband noise), peak amplitude 2.00 (real: 1.00 — overshoots full scale), and
per-band energy loses **52% at 2-6kHz** and **37% at 0.5-2kHz** (the presence/clarity region)
while over-emphasizing bass. A probe (linear regressor: 32-dim latent -> real 2-6kHz envelope)
gets R²=0.518 — the information survives in the latent, the DECODER fails to use it. This is why
the recommended fix is a decoder-only fine-tune, not a full VAE retrain (see
[Planned but not built](#planned-but-not-built)).

### Stage 2 — Latent caching (`src/cache_latents.py`)

Encodes every FLAC through the frozen VAE (mu only, no sampling noise) once, so DiT training
never re-runs the encoder. `--deterministic` takes the FIRST 30s of each track (required for
frame-alignment with extracted features — a random crop silently misaligns chroma against the
latent it's meant to describe). Output: `data/latents/track_XXXXXXX.pt`, one `(32,645)` float32
tensor per track, ~82.5 KB "true" size.

**A serious bug lived here from the start of the project until 2026-07-17**: `torch.save(mu[i],
out)` saves a VIEW into the batch tensor, and `torch.save` serializes the view's entire
underlying storage. Every file on disk was 8x-64x larger than the real tensor (up to 5.285 MB
per 82.5 KB of actual data — verified). Fixed with `.clone()`. `data/latents/` dropped from 38 GB
to 4.7 GB across all 55,609 tracks. **If you ever see `data/latents/` above ~5 GB again, this bug
is back.**

### Stage 3 — Feature extraction (`extract_features.py`, `src/features.py`)

Computes the conditioning signal for the reference-track path. Per track, one shared STFT
(n_fft=2048, hop=1024 -> exactly 645 frames) produces:

- **tempo, key, mode** — global scalars (librosa beat tracking + Krumhansl-Schmuckler profile
  correlation)
- **chroma** `(13, 645)` fp16 — 12 pitch-class bins + 1 RMS-energy row, per frame
- **texture** `(4, 645)` fp16 — onset strength, percussive ratio (HPSS), spectral centroid,
  spectral flatness, per frame

`src/features.py` is the **single shared definition** used by `extract_features.py` (training
data), `generate.py` (user reference tracks), and `train_dit.chroma_adherence` (the metric that
scores chroma-following). This consolidation happened after chroma extraction and its own metric
drifted apart and silently collapsed the measured ceiling from +0.512 to +0.313 — see the
[measurement audit](#the-measurement-audit--hard-won-lessons).

`FEATURE_VERSION = 2` is stamped into every saved file; extraction re-processes any file whose
version doesn't match (or is missing the `texture` key), so a schema change can never be
silently skipped.

Two directories, **must stay separate**:
- `data/features/` — the 53,609-track TRAINING set
- `data/features_val/` — a 2,000-track HELD-OUT set, `files[10000:12000]` of the sorted FLAC
  list. Training is defined as `latents ∩ FEATURES_DIR` — putting held-out features in the
  training dir would pull those tracks into training with stale, misaligned latents.

### Stage 4 — Feature standardization (`compute_feature_stats.py`)

Per-channel mean/std over `data/features/` (training only — using held-out stats would leak the
val distribution). Saved to `data/feature_stats.json`. `JamendoDataset` applies `(x-mean)/std` to
chroma and texture before they reach the model. This exists specifically to kill a DC-bias
defect: raw channels have large non-zero means (chroma 0.399, percussive 0.503, flatness 0.243),
so an unstandardized `proj(c) = W@mean + W@(c-mean)` injects a CONSTANT vector into every one of
the 645 tokens. Measured before the fix: 84% of the injected signal was that constant, carrying
zero per-frame information. After standardization: 7.1%.

### Stage 5 — Latent normalization (`compute_latent_stats.py`)

Per-channel mean/std over the training-set latents (restricted to `latents ∩ features`, so stale
random-crop latents can't skew it). Saved to `data/latent_stats.json`. DiT trains on
`(z - mean)/std` so the flow-matching target's scale matches the `N(0,I)` noise prior and MSE
doesn't over-weight high-variance channels. **Recomputing this over a different track count
changes the space the model was trained in** — moved 13.9% median (18.3% max) when the training
set grew from 10k to 53.6k tracks; costs ~+0.077 flow-matching loss on resume, which is real but
small (see the divergence discussion below — this was NOT the cause of the big collapse).

Streams (running sum/sum-of-squares) rather than `torch.stack`-ing every file, purely for safety
margin on a shared machine — the actual per-file tensor is small enough (82.5 KB) that stacking
would have been fine; the streaming was motivated by an earlier (wrong) assumption about file
size that turned out to be measuring the latent-caching bloat bug, not real tensor size.

### Stage 6 — DiT training (`src/train_dit.py`, `src/dit.py`)

8-layer, d_model=512, 8-head DiT, adaLN-Zero throughout, **39.1M params** (~DiT-S).
`F.scaled_dot_product_attention` (flash kernel) so the 645x645 score matrix is never
materialized. Fixed sinusoidal position embedding over the 645 frames.

**Flow matching**: `t ~ sigmoid(N(0,1))` (logit-normal, concentrates on the hard middle of the
trajectory — SD3/Esser et al. 2024), `z_t = (1-t)z0 + t*z1`, target `v = z1-z0`,
`loss = MSE(model(z_t,t,cond), v)`.

**Conditioning** (`FeatureEmbedder` in `src/dit.py`):
- tags -> mean-pooled learned embedding -> `cond` (adaLN vector)
- tempo/key/mode -> sinusoidal+embedding -> `cond`
- **chroma** `(13,645)` -> `Linear(13,512)` -> ADDED TO INPUT TOKENS (per-frame, not global)
- **texture** `(4,645)` -> `Linear(4,512)` -> ADDED TO INPUT TOKENS (per-frame)

Both chroma and texture use the pattern `out = gate * proj(x)`: **normal-init proj + zero-init
scalar gate** (adaLN-Zero: gate the block to zero, never its contents). Zero-initing BOTH is a
deadlock — `d(out)/d(proj) ∝ gate` and `d(out)/d(gate) ∝ proj(x)`, both zero, neither can ever
move. This exact bug shipped once: an entire training run completed with texture fully inert
(gate and proj norm both pinned at 0.000) — see the audit.

**There is NO LayerNorm on the chroma/texture injection.** One was added to texture at one point,
justified by a theory (unbounded projection destabilizes the backbone) that was later disproven
by direct measurement (texture was inert during the run it supposedly destabilized). LayerNorm
was actively harmful: it normalizes every token to a fixed norm of √D, destroying the magnitude
information the flatness/onset channels exist to carry. Removing it is part of why FLAT recovered
from 0.70x to 1.08x — the LayerNorm-carrying version was the one that couldn't move flatness at
any CFG scale from 1.0 to 3.0. **Do not re-add it.**

**Training modes** (`--resume` vs `--finetune` — this distinction has caused real damage, read
carefully):
- `--resume PATH`: full resume — weights + optimizer state + step counter + LR schedule
  position. Use when continuing the SAME run (same architecture, same schedule).
- `--finetune PATH`: loads weights only (`strict=False`, so new zero-init modules like
  chroma/texture simply start from their init), resets step to 0, builds a **FRESH optimizer**.
  Use when the checkpoint predates a conditioning module that doesn't exist in it yet.
- **`--finetune` from an ALREADY-CONDITIONING-TRAINED checkpoint is dangerous.** A fresh Adam has
  `exp_avg_sq=0`, so its first updates are `lr*sign(grad)` on every parameter regardless of true
  gradient magnitude — survivable on a model still learning, destructive on a converged one
  (beta2=0.999 needs ~1000 steps of history to damp this, which is why the damage tracks the
  warmup ramp exactly). `train_dit.py` now WARNS and prints the correct `--resume` command if
  `chroma_gate` is non-zero at `--finetune` time (added 2026-07-19, see the audit).

**Divergence guards**:
- Outlier-batch guard: skip a step if `loss > 2.0x best_loss_ever_seen` (was 3.0x, and anchored
  to the running average rather than the best — both changes closed real gaps; see audit).
  Cannot see a slow multi-hundred-step slide by construction.
- Slow-slide guard: abort (no final checkpoint saved) if the running average stays above
  `1.35x` its best for 300 consecutive steps. This is what has fired on every divergence so far.

**Held-out evaluation** (`SNREvaluator` in `train_dit.py`): TWO instances now run per eval step —
one on `config.FEATURES_DIR` (train), one on `config.FEATURES_VAL_DIR` (held-out). Clip selection
records which track IDs were picked (`self.track_ids`), and `main()` explicitly checks the
held-out picks against the training set and prints `clean` or `CONTAMINATED`. This exists
because the ORIGINAL evaluator had no such check and it took most of a debugging session to
discover the eval set had silently been 16/16 identical to the training set the entire time.

Metrics computed: **chroma GAP** (matched-vs-shuffled adherence — see audit for why raw
adherence is meaningless), **STD ratio** (generated/real latent std — <1 = variance
collapse/smearing), **flatness ratio** (median of per-clip ratios vs the VAE's OWN
reconstruction, not real audio — >1 hissy, <1 too tonal/dull), **NOISY count** (clips >3x
flatness — a separate occasional failure the median hides), **SNR** (kept in the CSV, explicitly
documented as NOT a quality signal — see audit).

### Stage 7 — Sampling / inference (`MusicDiT.sample()` in `src/dit.py`)

CFG: conditional and unconditional batched into one pass, `v = v_uncond + cfg_scale*(v_cond -
v_uncond)`. **One `cfg_scale` applies uniformly to tags, chroma, AND texture together** — there
is no independent guidance strength per conditioning source. This is architecturally relevant to
the "output just copies the reference" complaint (see Open Problems): turning CFG down to
loosen adherence to the reference weakens tag-following equally.

Solver: 2nd-order Heun (predict with Euler, re-evaluate velocity at the predicted endpoint,
average) — halves integration error per step vs Euler at 2x the per-step cost, so at half the
steps it's free. Default 50 steps, `CFG_SCALE = 1.5`.

### Stage 8 — Generation entrypoints

- `sample_dit.py` — fixed seeded eval-style sampling, `--heldout` flag to sample from the
  held-out set instead of training tracks, peak-normalizes on write.
- `eval_dit.py` — full metrics + plots + audio across all available DiT checkpoints, for
  trend-watching across a training run.
- `generate.py` — the user-facing "make me music" entrypoint. `--prompt` (free text, resolved to
  tags via `src/text_tags.py`), `--tags` (exact tags), `--reference` (audio file, resolved from
  `references/` by bare filename), `--cfg-scale`, `--seed`, `--n`. See
  [Natural-language generation](#natural-language-generation-what-exists-what-doesnt).

---

## Every file, what it does

### Root

| File | Role |
|---|---|
| `config.py` | Every hyperparameter, path, and architectural constant. Read this first — comments document WHY values are what they are, including several "this used to be X, changed because Y broke" notes. |
| `extract_features.py` | Builds `data/features/` and (via `--start`/`--out-dir`) `data/features_val/`. Version-stamped, atomic writes. |
| `compute_feature_stats.py` | Per-channel chroma/texture mean+std over training features only. Streams. |
| `compute_latent_stats.py` | Per-channel latent mean+std over training latents only. Streams. |
| `generate.py` | User-facing generation: prompt/tags + optional reference -> audio. |
| `sample_dit.py` | Seeded eval-style sampling from a checkpoint; `--heldout`. |
| `eval_dit.py` | Full metric/plot/audio dump across all checkpoints in `checkpoints/dit/`. |
| `references/` | Drop-your-own-audio-here directory for `generate.py --reference`. Gitignored except its `README.md`. Currently contains a user-supplied `ivy.mp3`. |
| `README.md` | Public-facing project doc: architecture, results (train AND held-out numbers), the measurement audit, known issues, future work, paper citations. Rewritten 2026-07-17 to replace an earlier version that documented a pre-chroma, pre-audit state of the project. |
| `PROJECT.md` | File structure / hyperparameter table / setup steps (referenced by README, not audited in this session). |
| `project_context_transfer_v2.md` | LEGACY — an early architecture-planning doc from before this session's work. Superseded by this file for anything about current pipeline state. |
| `overfit_test.py`, `overfit_dit.py` | Sanity-check scripts cited in the README (VAE bottleneck test: 18.58 dB; DiT-with-perfect-conditioning test: 99% of VAE ceiling). Keep — they're load-bearing evidence for claims in the README. |
| `diagnose.py`, `diagnose_recon.py`, `diagnose_latent_space.py`, `diagnose_sampling.py`, `flac_demo.py`, `musicgen_baseline.py`, `plot_vae_stats.py`, `smoke_test.py`, `strip_full.py`, `strip_test.py` | One-off scripts from earlier stages (June data prep, VAE debugging). Data-prep ones (`strip_*`) are done and their inputs deleted; diagnostic ones are superseded by the metrics now built into `train_dit.py`. Candidates for deletion, never acted on. |

### `src/`

| File | Role |
|---|---|
| `vae.py` | `WaveformVAE` (encoder/decoder), `MultiSTFTDiscriminator`. |
| `train_vae.py` | VAE + discriminator training loop. Not touched this session. |
| `losses.py` | `MultiScaleMelLoss`, adversarial/feature-matching losses, VQGAN-style adaptive weight. |
| `dit.py` | `MusicDiT`, `FeatureEmbedder`, `TagEmbedder`, `DiTBlock`, `sample()`. The conditioning architecture lives here. |
| `train_dit.py` | The DiT training loop: flow matching, both evaluators, divergence guards, checkpointing, CSV logging. The most-edited file this session. |
| `dataset.py` | `JamendoDataset` — loads latents-or-audio + tags + optional features, with `features_dir` param and standardization. `_track_id_from_path` handles both raw-FLAC and cached-latent filename formats. |
| `features.py` | THE shared feature-extraction recipe (`extract_all`, `chroma_from_stft`, `load_reference`, `estimate_key`). Added 2026-07-18 to stop conditioning and its own metric from drifting apart. |
| `text_tags.py` | `TextTagMatcher` — free text -> tag vocabulary via sentence embeddings. Added 2026-07-19 (Route A of the natural-language discussion). |
| `cache_latents.py` | Encodes FLACs through the frozen VAE to `data/latents/`. |

---

## Data & checkpoint inventory (current, verified)

As of this writing (verify before trusting — this drifts):

```
data/stems/          55,609 FLACs, 356 GB   (raw audio, source of truth)
data/latents/         55,609 files, 4.7 GB  (post clone()-bug-fix; all deterministic-crop)
data/features/        53,609 files, 1.3 GB  (training set: files[0:10000] + files[12000:55609])
data/features_val/     2,000 files,  47 MB  (held-out: files[10000:12000] — verified 0 overlap)
data/latent_stats.json           (computed over the 53,609 training latents)
data/feature_stats.json          (computed over the 53,609 training features)
data/tag_embeddings.npz          (TextTagMatcher's cached tag embeddings)

checkpoints/vae/       3 files, 884 MB  — vae_step0196000/198000/200000.pt
                       200000 is THE frozen VAE used everywhere.
checkpoints/dit/       3 files, 1.8 GB  — dit_step0105000/110000/115000.pt
                       115000 is the pre-chroma, tags-only backbone (39.1M, 400k-step schedule).
checkpoints/dit_ft/    3 files, 1.8 GB  — dit_step0026000/28000/30000.pt
                       30000 is the CURRENT BEST MODEL. Fine-tuned from dit_step0115000 with
                       chroma+texture conditioning, on the FULL 53,609-track set, 30,000 steps,
                       completed cleanly (no divergence). This is what generate.py defaults to.
```

**Checkpoint contents**: `{"step", "dit", "ema", "opt", "scheduler"}`. Always sample from `ema`
(smoother than raw training weights). `strict=False` on load — absent keys mean zero-init
modules that behave as a no-op until trained.

**dit_stats.csv**: currently ends mid-divergence (step 30500, loss climbing 0.70->1.20) from a
`--resume` attempt off `dit_step0030000.pt` that was aborted by the slow-slide guard and NOT
saved over the healthy checkpoint. See [Open problems](#open-problems).

---

## Config reference

Full file is `config.py` with extensive inline rationale; the entries most likely to matter to
someone continuing this work:

```
VAE_LATENT_DIM=32  VAE_LATENT_LEN=645  CHUNK_SAMPLES=660480  SAMPLE_RATE=22050

DIT_D_MODEL=512  DIT_HEADS=8  DIT_LAYERS=8  (39.1M params total)
DIT_BATCH_SIZE=384                    # ~52-54GB VRAM on the shared A100
DIT_LR=6e-4  DIT_WARMUP=1000  DIT_TOTAL_STEPS=400000        # from-scratch schedule
DIT_FINETUNE_LR=3e-5  DIT_FINETUNE_FEAT_LR=5e-5              # fine-tune backbone vs new-module LR
DIT_FINETUNE_STEPS=60000              # raised from 30000 on 2026-07-19 so a --resume at step
                                       # 30000 lands mid-schedule (53.8% of peak) instead of on
                                       # the 5% floor (effectively frozen)
DIT_RESUME_WARMUP=1000                # re-warm LR after an optimizer-state DISCARD
DIT_LOSS_SPIKE_FACTOR=2.0             # outlier-batch guard, anchored to best_loss not loss_avg
DIT_DIVERGE_FACTOR=1.35               # slow-slide guard, 300-step sustained threshold
DIT_FT_CKPT_EVERY=1000  DIT_FT_KEEP_LAST=10   # fine-tune checkpoint cadence/retention

DIT_USE_GLOBAL_FEATS=True  DIT_USE_CHROMA=True  DIT_USE_TEXTURE=True
DIT_CHROMA_IN=13  DIT_TEXTURE_IN=4
DIT_STANDARDIZE_FEATS=True            # kills the DC-bias defect, do not disable
DIT_CHROMA_DROPOUT=0.3                # independent of CFG dropout; the "only value ever
                                       # observed stable" belief was DISPROVEN 2026-07-17 (a run
                                       # diverged at 0.3 too) — this is now just "seems fine",
                                       # not a load-bearing safety value
DIT_CFG_DROPOUT=0.2

FEATURE_VERSION=2                     # bump on any change to the chroma/texture recipe
DIT_EVAL_HELDOUT=True                 # the train-vs-held-out comparison, added 2026-07-16

EULER_STEPS=50  ODE_SOLVER="heun"  CFG_SCALE=1.5
```

---

## The measurement audit — hard-won lessons

Most debugging time in this project went into metrics that were wrong, not models that were
broken. In rough chronological order:

1. **The eval set was 100% training data, 16/16, structurally.** `SNREvaluator` skipped any clip
   without a feature file; only training tracks had feature files. So "GAP 98% of ceiling" was a
   memorization score for the entire early chroma-conditioning phase. Fixed by adding a genuinely
   disjoint held-out split and printing an explicit contamination check every run.

2. **Chroma extraction was energy-blind.** `librosa.chroma_stft` defaults to `norm=inf`, which
   rescales every frame to max=1.0. Measured: a silent frame and a fortissimo frame both read
   `1.0000`. 11.6% of training frames are near-silent; every one was fed to the model as a
   confident chord. Fixed with `norm=None` + per-clip log compression (`src/features.py`).

3. **~84% of the chroma/texture injection was a constant**, because raw channel means are
   non-zero and the projection was unstandardized — `proj(c) = W@mean + W@(c-mean)`, and the
   first term is identical across all 645 tokens, carrying zero information. Fixed by
   standardizing inputs (`compute_feature_stats.py` + dataset-level normalization).

4. **A zero-init deadlock silently killed the texture path for an entire training run.** Both
   `texture_proj` and `texture_gate` were zero-init; `d(out)/d(proj) ∝ gate=0` and
   `d(out)/d(gate) ∝ proj(x)=0`, so neither could ever move. Confirmed via checkpoint inspection
   (`texture_gate=0.000`, `‖texture_proj‖=0.000` after thousands of steps). Fixed: normal-init
   the projection, zero-init only the gate (canonical adaLN-Zero).

5. **A LayerNorm on the texture injection, added to fix a theory that was later disproven, was
   itself actively harmful** — it normalized every token to a fixed norm, destroying the
   magnitude information (flatness/onset strength) those channels exist to carry. Removed.

6. **Chroma extraction and the metric that scored it drifted apart.** The metric kept calling
   librosa's default (`norm=inf`) after extraction moved to `norm=None` + log compression, so it
   compared a non-negative vector against a standardized zero-mean one — measured cost: ceiling
   dropped from +0.512 to +0.313 from the mismatch alone. Fixed by consolidating both into
   `src/features.py`.

7. **The stats CSV silently shifted columns** when new fields were appended under a stale header
   (SNR was being read as `chroma_proj_norm` during an actual debugging session). Fixed: header
   match check that archives the old file on schema change.

8. **A `torch.save`-a-view bug bloated every cached latent 8x-64x** (documented above under Stage
   2). Found while investigating why a re-caching job was catastrophically slower than estimated
   despite raising batch size and worker count — the job was disk-write-bound on bloated writes,
   and a bigger batch made every write bigger, not faster.

9. **`--finetune` from an already-converged, conditioning-trained checkpoint destroys it via a
   fresh-Adam kick**, not the mechanism originally suspected (feature-module magnitude, chroma
   dropout, data misalignment — all individually measured and exonerated first). The fingerprint
   that broke the case open: loss was FLAT while LR<1e-5, climbed in exact lockstep with the
   warmup ramp, plateaued the moment LR maxed — while chroma GAP stayed high the whole time (the
   model wasn't forgetting anything, it was being shaken out of its minimum). A subsequent
   `--resume` at HALF the LR diverged FASTER, which at first looked contradictory but wasn't:
   both runs cross the same absolute LR threshold, just from different starting points and rates.

10. **Generated audio was being hard-clipped on write.** `sf.write` defaults to PCM_16 for
    `.wav`, hard-clipping outside [-1,1]. 8/16 generated clips exceeded it (peak 2.45) — and
    critically, the VAE reconstruction from GROUND-TRUTH latents also overshoots (peak 2.00), so
    this was never a DiT or CFG artifact. All write paths now peak-normalize.

**The throughline**: repeatedly, a plausible-sounding cause was proposed, believed, sometimes
acted on, and then falsified by a direct measurement that took 10-20 minutes to run. The
standing rule that emerged: **before proposing a fix, find the number that would prove it
wrong, and check that number first.**

---

## Open problems

### 1. Raw audio quality has artifacts, and it is confirmed to be the VAE decoder, not the DiT

Directly measured (2026-07-19) by comparing `dit_gen` against `vae_ceiling` (the same clips,
decoded from REAL ground-truth latents) in `samples/dit/20260717_080742/`:

```
clip   file          crest(dB)   flatness
 0     real             16.5       0.032
 0     vae_ceiling      18.6       0.054
 0     dit_gen          18.1       0.041
 5     real             11.8       0.007
 5     vae_ceiling      17.3       0.009
 5     dit_gen          18.2       0.011
```

The jump from `real` to `vae_ceiling` (best-case decode) is as large or larger than the jump from
`vae_ceiling` to `dit_gen` on almost every clip. **The DiT is not adding meaningfully more noise
than the decoder already has.** This confirms the −52%/−37% band-energy loss and 1.4x-1.7x
flatness inflation measured earlier in the session are the dominant, and possibly close to the
ONLY, source of perceived artifacts. See [Planned but not built](#planned-but-not-built) for the
fix (decoder-only fine-tune) — not yet built.

### 2. Reference-track conditioning is too strong: it produces a near-exact cover, not inspiration

Measured (2026-07-19): **texture is followed at 97-105% of its own ceiling**, essentially as
tightly as chroma (86% of ceiling). Since chroma pins harmony and texture pins the rhythmic/onset
and energy contour, together they specify the reference at 21 frames/second on every axis except
timbre. A user (correctly) described the output as "just copies whatever the reference track
is." Compounding this: `dit.sample()` has one `cfg_scale` applied uniformly to tags + chroma +
texture together — there's no way to weaken adherence to the reference while keeping tag/text
influence strong. Two redesign plans were scoped (not built) — see below.

### 3. The unresolved divergence

Every attempt to continue training past the healthy `dit_step0030000.pt` checkpoint has
diverged, on TWO different entry paths:

- `--finetune` from `dit_step0030000.pt` (fresh Adam) — diverged, root-caused to the
  fresh-Adam-kick mechanism above. Guarded against now (warning printed).
- `--resume` from `dit_step0030000.pt` (Adam moments restored, this should have been safe) —
  **also diverged**, loss climbing 0.70 -> 1.21 over a few hundred steps, landing at exactly the
  SAME plateau loss (~1.21) as the fresh-Adam failure. This is the current dead end.

Ruled out by direct measurement before giving up on this line of investigation: cached-latent
integrity (byte-perfect vs fresh VAE encoding), data pathology (zero outliers across all 53,609
training latents, max 12.5σ), feature-conditioning distribution shift (0.006σ, negligible),
model generalization (loss on the 43,609 never-before-seen tracks is 0.689 — BETTER than the
0.700 on the originally-memorized 10,000, so the model isn't confused by the new data), and a
"does it recover if the guard is disabled" test (ran 1,200 steps past the guard's abort point
with no guard at all — loss stayed flat at ~1.21-1.26, did not recover).

**Working hypothesis, not confirmed**: the original 10k-track fine-tune memorized hard enough
(train GAP was 102% of the VAE's own ceiling — literally better than ground-truth reconstruction,
only possible by memorization) that the loss landscape around that checkpoint is a narrow,
maybe-not-very-good local minimum, and continuing training on 5.4x more data requires climbing
out of it before any real improvement is visible — and the guards (correctly, by design) never
let a run climb long enough to find out whether it comes back down. This has NOT been tested
(would require disabling or loosening the divergence guard for a multi-hour run) and should not
be treated as established.

**Bottom line: `dit_step0030000.pt` is the best available model. Do not try to improve it via
further training without a new diagnostic idea — five distinct hypotheses have already been
tried and killed by measurement.**

---

## Natural-language generation: what exists, what doesn't

Built 2026-07-19 (`src/text_tags.py`, wired into `generate.py` via `--prompt`).

**What it is**: your text is sentence-embedded (`all-MiniLM-L6-v2`) and matched against all 195
tag names (which are NOT English — `electricguitar`, `drumnbass`, `rocknroll` — expanded via a
phrase table before embedding). Per-category top-k (2 genre / 3 instrument / 2 mood by default,
threshold 0.42) are fed to the model as tags. This is retrieval, not a text encoder: **no
information is created beyond what 195 fixed tags can express.** `--explain` shows the match
without generating.

**What tags-only generation looks like, measured**: NOT mode-collapsed (pairwise cosine
similarity between same-prompt samples is 0.11-0.24, close to real tracks' own 0.14 — genuine
diversity), but individually smoothed: latent STD 0.64 (target 1.0), flatness 0.44x (target
1.0x). Varied but dull.

**What reference-conditioned generation looks like**: much better numbers (STD 1.19, flatness
1.06x) but — see Open Problem #2 — it's a near-exact cover of the reference's harmony AND rhythm,
which the user has explicitly said is not what they want out of "generate through natural
language."

**Conclusion reached in the last design discussion, not yet acted on**: the honest fix for
"generate through natural language properly" requires moving the reference/text conditioning
from a per-frame signal (chroma/texture, 645 values, dictates exact structure) to a GLOBAL
pooled signal (one vector per clip, dictates style/character, not bar-by-bar content). Two plans
were scoped:

- **Plan 1 (cheap, hours)**: block-average chroma/texture down to ~8 segments across the clip
  before feeding them in, so the reference supplies a loose vibe instead of a frame-by-frame map.
  No new dependency; short fine-tune from `dit_step0030000.pt`. Ceiling: still capped by
  tags-only quality (STD 0.64) if no reference given at all — this is a knob, not a fix for the
  "text has no real influence" complaint.
- **Plan 2 (real fix, multi-day)**: CLAP (joint audio/text embedding, e.g.
  `laion/larger_clap_music` via `transformers.ClapModel`). Train on CLAP AUDIO embeddings of each
  track (one 512-dim vector, computed once per track, ~55,609 forward passes); at inference, use
  CLAP TEXT embeddings of the prompt, which live in the same space by construction. Add as a new
  GLOBAL adaLN conditioning path (like `global_cond`, NOT like chroma/texture's per-token
  injection) — architecturally incapable of dictating frame-by-frame structure, which is exactly
  the point. Chroma/texture stay in the code as an explicit opt-in `--cover` mode for users who
  DO want near-exact reference following (today's behavior, unchanged). This is the only path
  where free text carries information beyond the 195-tag vocabulary.

**Neither plan has been implemented.** This is the immediate fork facing whoever continues this
work: which plan (or both, in order) to build.

---

## Planned but not built

In priority order, per the last few design discussions:

1. **Decoder-only fine-tune** for the VAE's −52%/−37% band-energy loss (Open Problem #1).
   Encoder frozen -> all cached latents and the DiT stay valid, no re-caching, no DiT retrain.
   Two components discussed:
   - A band-targeted loss term. Plain mel loss was checked and ruled out as the sole cause: the
     2-6kHz band already gets 32.6% of the mel bins (the largest share of any band) and still
     fails, so more mel won't fix it — needs direct energy-matching or a linear-frequency term.
   - Possibly a Vocos-style magnitude+phase iSTFT head to replace the raw-frame
     `Linear(384->512)` + overlap-add, which is the likely source of the phase-cancellation
     signature (HF loss increases monotonically with frequency band, +1%/-2%/-13%/-34% from OLA
     alone — the fingerprint of destructive interference between overlapping frames, worse at
     higher frequencies because their periods are shorter than a 1-2 sample phase disagreement).
   - Recommendation from the last discussion: try the band-loss alone first, measure, before
     committing to the head rewrite.
2. **CLAP-based conditioning redesign** (Plan 2 above) for natural-language control.
3. **Scale the DiT** (39M -> ~130M, ~DiT-B) — deferred behind both of the above since it would
   still chase the same VAE ceiling and the same per-frame-reference over-specification.
4. Two known-but-inert VAE bugs, not yet fixed (low priority, currently invisible in practice):
   - Encoder double-pads: `F.pad(nfft//2)` before `torch.stft(center=True)` (the default), which
     pads another `nfft//2` — 2582 STFT frames actually produced, not the 2580 assumed elsewhere;
     encoder/decoder frame centers sit 256 samples apart. The trained model has absorbed this.
   - OLA tail: `_overlap_add`'s window-sum normalizer decays to ~0 over the last 162 samples of
     every chunk (up to 1e8x amplification if the numerator weren't already ~0 there). Currently
     masked because the trained decoder emits near-zero in that region.

---

## How to run everything

```bash
cd /home/bhuvan/6DGS/music-gen && source venv/bin/activate

# --- one-time / when data changes ---
python extract_features.py --start 12000 --workers 32                       # training features
python extract_features.py --start 10000 --limit 2000 --out-dir data/features_val   # held-out
python src/cache_latents.py --checkpoint checkpoints/vae/vae_step0200000.pt --deterministic \
    --batch-size 64 --workers 32
python compute_feature_stats.py
python compute_latent_stats.py

# --- training ---
# Resume the SAME run (same architecture already present in the checkpoint):
python src/train_dit.py --resume checkpoints/dit_ft/dit_step0030000.pt
# Only use --finetune when the checkpoint LACKS a module you're adding (e.g. moving from
# tags-only checkpoints/dit/dit_step0115000.pt to a chroma+texture-conditioned model):
python src/train_dit.py --finetune checkpoints/dit/dit_step0115000.pt

# --- generation ---
python generate.py --list-tags
python generate.py --search piano
python generate.py --prompt "upbeat jazzy piano" --explain          # see matched tags, no gen
python generate.py --prompt "upbeat jazzy piano" --reference mysong.mp3 --n 4
python generate.py --list-references

# --- evaluation / listening ---
python sample_dit.py --heldout --checkpoint checkpoints/dit_ft/dit_step0030000.pt
python eval_dit.py
```

---

## Traps for whoever touches this next

- **Never `torch.save` a slice of a batch tensor without `.clone()`** — this bug cost 33 GB and a
  very confusing "why is caching still slow after I raised the batch size" debugging detour.
- **Never `--finetune` from a checkpoint whose `chroma_gate` is already non-zero.** Use
  `--resume`. The code now warns about this, but the warning didn't exist when the damage
  happened — don't assume future-you will see it if the checkpoint path changes.
- **Never put held-out features in `data/features/`.** Training is defined as
  `latents ∩ features`; this is the single easiest way to silently recreate the exact
  contamination bug that cost the most debugging time in the whole project.
- **Never change the chroma/texture extraction recipe in only one of `extract_features.py` /
  `src/features.py` / wherever the metric lives.** They MUST stay one definition. This has
  already gone wrong once.
- **Recomputing `latent_stats.json` or `feature_stats.json` over a different track count moves
  the input space the model was trained in.** Small for features (0.006σ, ignorable), NOT small
  for latents (13.9% median shift, ~+0.077 loss cost) — expect a brief loss bump on the next
  resume after a stats recompute, and don't mistake it for a real divergence.
- **`sf.write` hard-clips.** Always peak-normalize before writing audio, and remember the VAE
  overshoots ±1.0 even from perfect ground-truth latents (peak ~2.0) — this is normal, not a
  bug signal.
- **Before trusting any quality number, check whether it's measured on held-out data.** The
  single largest error in this entire project was an eval set that was, unbeknownst to anyone,
  100% memorized training data for weeks.
