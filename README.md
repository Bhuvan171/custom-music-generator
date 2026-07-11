# Music Generation: Tag-Conditioned Latent Flow Matching

A from-scratch, two-stage music generation pipeline: a spectral waveform VAE
compresses 30-second audio clips into a compact latent sequence, and a
Diffusion Transformer (DiT) trained with flow matching generates latents
conditioned on multi-label tags (genre / instrument / mood) from the
MTG-Jamendo dataset. Every component (VAE, discriminator, DiT, conditioning
modules) is implemented and trained from random initialization on a single
shared A100 80GB GPU — no pretrained weights anywhere in the pipeline.

This document describes what has been built, the results obtained so far,
the reasoning behind the architectural decisions, and what remains open.
For exact hyperparameters and file layout, see `PROJECT.md`.

---

## Status at a glance

- **VAE**: trained for the full 200,000-step schedule. Frozen and in use as
  the latent encoder/decoder for DiT training.
- **DiT**: in progress. A first full training run (raw, unnormalized latents,
  constant learning rate) reached step ~76,000/400,000 before being stopped
  after diagnosis showed it had plateaued. Two root causes were identified
  and fixed (latent normalization, cosine LR decay); training restarted from
  step 0 on the corrected objective and is ongoing.
- **Evaluation tooling**: built this session (`eval_dit.py`, `sample_dit.py`,
  `compute_latent_stats.py`) to measure generation quality beyond the raw
  loss curve, since loss alone was shown to be an unreliable and sometimes
  misleading signal on its own (see Results and Known Issues below).

---

## Pipeline overview

```
Raw waveform (22050 Hz, mono, 30s, 660480 samples)
        |
        v
  Waveform VAE Encoder  ---- frozen after VAE training
        |
   latent z (32, 645)
        |
        v
  Flow Matching DiT  <---- tag conditioning (195-way multi-label)
  (38.8M params)     <---- timestep conditioning
        |
   generated latent
        |
        v
  Waveform VAE Decoder  ---- frozen
        |
        v
  Generated waveform (.wav)
```

No vocoder stage exists in this pipeline. The VAE decodes latents directly to
waveform, collapsing what would otherwise be two trained stages (an audio VAE
plus a separate vocoder) into one — removing a source of artifacts and a
training stage under a fixed compute/time budget. This mirrors the design
used by latent diffusion image models, where a single autoencoder handles
both compression and reconstruction and the generative model never touches
pixel/sample space (Rombach et al., 2022).

---

## Stage 1: Spectral Waveform VAE

### Architecture

Both the encoder and decoder run at STFT frame rate; only the initial FFT and
the final overlap-add synthesis touch sample rate.

- **Encoder**: STFT (n_fft=512, hop=256) -> sign-log1p compression of
  real/imaginary parts -> Conv1d projection to 384 channels -> 8 ConvNeXt1d
  blocks -> strided downsample (4x) -> Conv1d -> `mu`, `logvar`, each
  `(32, 645)`.
- **Decoder**: `z (32, 645)` -> Conv1d projection -> 4x linear upsample ->
  Conv1d refinement -> 8 ConvNeXt1d blocks -> per-frame `Linear(384 -> 512)`
  head -> Hann-windowed overlap-add -> waveform.
- **ConvNeXt1d block**: depthwise `Conv1d(D, D, kernel=7)` -> LayerNorm ->
  pointwise expansion x4 -> GELU -> pointwise projection -> residual.
- 20.4M parameters (encoder + decoder). 645 latent frames x 4x upsample x 256
  hop = 660,480 samples exactly, so there are no boundary padding artifacts.

**Discriminator**: an EnCodec-style multi-scale STFT discriminator — three
independent 2D-convolutional discriminators over `n_fft in {512, 1024,
2048}`, each operating on stacked real/imaginary STFT channels (Défossez et
al., 2022). 5.3M parameters.

**Losses**:

| Loss | Purpose | Weight |
|---|---|---|
| Multi-scale log-mel L1 | reconstruction, equal weight per frequency band so high frequencies cannot be silently smoothed away by a linear-magnitude loss | 1.0 |
| Waveform L1 (time domain) | anchors phase, which spectral-magnitude losses alone do not constrain | 5.0 |
| KL divergence | regularizes the latent toward a unit Gaussian prior; weight warmed up over 10,000 steps to avoid early posterior collapse | 1e-4 |
| Adversarial (hinge, generator) | recovers high-frequency detail that L1/L2 reconstruction losses tend to average away | 2.0 x adaptive lambda |
| Feature matching | L1 between discriminator intermediate activations, stabilizes adversarial training | 2.0 |

The multi-scale mel loss, hinge adversarial loss, and feature-matching loss
follow the recipe established for neural vocoders (Kong et al., 2020,
HiFi-GAN). The adaptive adversarial weight — scaling the generator's
adversarial loss so its gradient at the last decoder layer never exceeds the
reconstruction loss's gradient at that layer — is the mechanism introduced by
Esser, Rombach & Ommer, 2021 (VQGAN) to prevent the discriminator from
dominating training early on, implemented directly in `src/losses.py`.

### Why this architecture

- **ConvNeXt blocks instead of plain ResNet/dilated-conv stacks**: depthwise
  convolution plus an inverted-bottleneck MLP gives most of a transformer
  block's capacity at convolutional cost (Liu et al., 2022, "A ConvNet for
  the 2020s").
- **Frame-rate processing with a lightweight synthesis head instead of
  transposed-convolution upsampling**: transposed convolutions are a known
  source of checkerboard/ringing artifacts in audio and image generation
  alike. Running the whole network at STFT frame rate and only expanding to
  sample rate via a linear per-frame projection plus overlap-add follows the
  approach introduced by Vocos (Siuzdak, 2023) and refined by WaveNeXt
  (Kaneko et al., 2024), which replaces even the inverse-STFT synthesis step
  with a directly-learned linear head. This was also a deliberate speed
  choice: keeping the heavy convolutional stack at 2580 frames instead of
  660,480 samples gives a 3-5x training speedup over the strided-Conv /
  ConvTranspose architecture originally planned (see `project_context_transfer_v2.md`).
- **Waveform-native VAE instead of mel-spectrogram + vocoder**: an earlier
  design considered computing a mel spectrogram and using a separate vocoder
  (Griffin-Lim or HiFi-GAN) to get back to audio. This was rejected: a
  waveform-native VAE collapses two trained stages into one, removing an
  entire additional source of artifacts and a training stage from a fixed
  one-week budget.

### Training

- AdamW, betas=(0.5, 0.9), eps=1e-6, lr=2e-4 (the low-momentum beta1=0.5 is
  standard practice for GAN-style adversarial training, reducing oscillation).
- bfloat16 autocast; KL and waveform-L1 losses computed in fp32.
- Batch 16, 3-second random crops during training (full 30-second clips are
  only ever encoded once, at latent-caching time).
- Adversarial loss activates at step 5,000, after the reconstruction losses
  have had time to establish a reasonable baseline.
- Trained for the full 200,000-step schedule. ~18.2GB VRAM.

### Results

Before committing to the full 200,000-step run, a single-batch overfit
sanity test (`overfit_test.py`, B=4, 8000 steps, deterministic decode)
verified that the 32x645 latent bottleneck was not the limiting factor:
**final SNR 18.58 dB**, well above the 6 dB pass gate.

On general (non-overfit) data, measured via `eval_dit.py` by decoding
ground-truth latents from held-out real clips:

- **Reconstruction SNR: 6.59 dB**
- **High-frequency energy ratio: 0.0142** (vs. 0.0198 for real audio) —
  real, measured high-frequency loss from the VAE's compression, though
  modest.

Direct listening to these reconstructions was judged qualitatively
"decent." This is not a contradiction of the 6.59 dB figure: time-domain
SNR is a well-known harsh and phase-sensitive metric for audio — a
reconstruction that is a few samples out of phase with the reference can
score very poorly on SNR while sounding nearly identical. The HF-energy
ratio (a spectral, phase-insensitive metric) is the more trustworthy signal
here, and it shows a real but modest loss of high-frequency detail, which is
expected for any compressive audio autoencoder.

**This reconstruction ceiling is a hard limit on final output quality.**
No amount of downstream DiT training can produce audio better than what the
frozen VAE decoder can reconstruct from a perfect latent.

---

## Stage 2: Music DiT (flow matching)

### Architecture

A Diffusion Transformer (Peebles & Xie, 2023) operating directly on VAE
latents, using adaLN-Zero conditioning throughout:

```
z_noisy (32, 645)
  -> Linear(32 -> 512) + fixed sinusoidal position embedding
  -> 8x DiTBlock(D=512, heads=8)
  -> adaLN-modulated LayerNorm -> Linear(512 -> 32), zero-initialized
  -> predicted velocity (32, 645)
```

Each `DiTBlock` computes six modulation vectors (shift/scale/gate for both
the attention and feed-forward sub-layers) from the conditioning vector, and
its output projection is zero-initialized so every block is the identity
function at initialization — this is what lets a DiT of this depth train
stably without additional warmup tricks (Peebles & Xie, 2023). Attention
uses `F.scaled_dot_product_attention` (the fused flash-attention kernel),
which never materializes the 645x645 score matrix explicitly.

**Conditioning**: `cond = timestep_embedding(t) + tag_embedding(tags)`.
Timesteps use a sinusoidal embedding through a small MLP. Tags use a
195-way multi-label embedding (`genre---*`, `instrument---*`,
`mood/theme---*` from MTG-Jamendo), mean-pooled over the active tags per
clip and projected to the model dimension; a single learnable null vector
stands in for empty tag lists and for classifier-free-guidance dropout.
AdaLN conditioning was chosen over cross-attention because the conditioning
signal here is a single global vector, not a sequence — cross-attention is
the right tool when conditioning becomes sequence-structured (e.g.
phoneme-level lyrics, which are explicitly out of scope for this project).

**Size**: 38.8M parameters.

### Training objective: flow matching

```
t        ~ U[0, 1]
z0       ~ N(0, I)              (noise)
z1       = clean latent          (data, per-channel normalized — see below)
z_t      = (1 - t) * z0 + t * z1 (linear interpolation)
v_target = z1 - z0                (constant velocity along the linear path)
loss     = MSE(model(z_t, t, tags), v_target)
```

This is the conditional flow matching / rectified flow objective (Lipman et
al., 2023; Liu et al., 2022, "Flow Straight and Fast"). It was chosen over a
standard DDPM-style noise-prediction objective (Ho et al., 2020) because
straight-line probability paths admit accurate sampling in far fewer steps
(50 Euler steps here, vs. hundreds-to-thousands for a typical DDPM noise
schedule) and empirically train more stably from random initialization at
small parameter/data scale — both directly relevant under a fixed one-week,
single-GPU training budget.

Twenty percent of training batches have their tag conditioning replaced with
the null embedding (CFG dropout), which is what allows classifier-free
guidance at inference (Ho & Salimans, 2022):

```
z ~ N(0, I)
for i in range(50):                                    # EULER_STEPS
    t = i / 50
    v_cond, v_uncond = model([z, z], t, [tags, null])   # batched in one pass
    v = v_uncond + 4.0 * (v_cond - v_uncond)             # CFG_SCALE
    z = z + (1/50) * v
# z is now a generated latent -> frozen VAE decoder -> audio
```

### Why a transformer (DiT) instead of a convolutional U-Net

A convolutional U-Net's local receptive field struggles to model long-range
structure across all 645 latent frames representing 30 seconds of audio.
Self-attention models any-to-any dependency across the whole sequence
directly, which matters for musical coherence over that time span.

### Why flow matching instead of an autoregressive token model

An autoregressive transformer over discrete audio codec tokens
(MusicGen-style; Copet et al., 2023) was considered and rejected: at the
small parameter scale available here, AR generation over quantized audio
tokens is more prone to exposure bias and quantization-bottleneck artifacts,
and generally requires more careful training than a continuous flow-matching
objective trained from scratch in a limited time budget.

### Overall design lineage

The two-stage split (train a VAE, freeze it, train a latent generative model
on the frozen latents) follows Latent Diffusion Models (Rombach et al.,
2022). The specific choice of a non-autoregressive, flow-matching DiT
operating on full-length music latents was inspired by DiffRhythm (2025),
identified as the simplest viable full-length-song architecture available;
every component here was nonetheless implemented and trained from random
initialization rather than fine-tuned from any released checkpoint.

### Training configuration

- AdamW, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2.
- Batch size 384 (see "GPU utilization" below for why this differs from the
  original planned batch of 64).
- Peak LR 6e-4, linear warmup over 1,000 steps, then cosine decay to a 5%
  floor over the full 400,000-step schedule.
- CFG dropout 20%. EMA of model weights, decay 0.999 — the EMA weights, not
  the raw training weights, are what `sample_dit.py` / `eval_dit.py` load for
  generation, since EMA weights are standard practice for stabilizing
  generative-model sampling quality (as in DDPM, Ho et al., 2020, and
  subsequent diffusion work).
- Gradient clipping at 1.0.
- Trains on pre-cached latents (`data/latents/*.pt`), not raw audio — no VAE
  encoding cost at DiT training time.
- Checkpoints every 5,000 steps, last 3 kept.

**GPU utilization**: the shared A100 80GB was significantly underutilized at
the originally planned batch size of 64 (~9.4GB VRAM). Batch size was
increased to 384 (~52GB VRAM, empirically verified stable) and
`DIT_NUM_WORKERS` tuned to keep the data pipeline from starving the GPU
between steps, since profiling showed utilization repeatedly dropping to 0%
between batches at the original worker count.

---

## What went wrong, and what was fixed

### Latent caching bugs (data pipeline)

1. **Disk-full crash during initial caching.** The very first full-dataset
   latent-caching pass ran out of disk space and crashed at 3,200/55,609
   files. Recovered by freeing space and resuming with `--skip-existing`.
2. **A single corrupted cached file survived the resume undetected.**
   `--skip-existing` only checks whether a filename exists, not whether the
   file is valid — the specific `.pt` file that was mid-write when the disk
   filled up (a 62KB truncated file, vs. ~330KB for a healthy one) was
   silently skipped on resume and only surfaced as a crash when
   `train_dit.py` tried to load it. Found via a full 55,609-file integrity
   scan (`torch.load` on every cached file), deleted, and regenerated.
3. **Filename convention mismatch.** `cache_latents.py` names its output
   `track_XXXXXXX.pt`, but `dataset.py`'s track-ID parser assumed the
   filename stem's first underscore-delimited segment was always the numeric
   ID — true for raw FLAC stems (e.g. `1002000.flac`) but not for the
   `track_`-prefixed cached latents. Fixed to handle both conventions.

### A host-RAM OOM-kill during training

At step 14,000 of the first full DiT training run, the process was silently
killed (SIGKILL, no Python traceback — the signature of the Linux
OOM-killer, not a CUDA/VRAM error, since GPU memory was stable at the time).
Root cause: `DIT_NUM_WORKERS=16` combined with `pin_memory=True` gives each
worker its own page-locked, non-swappable prefetch buffer, and a concurrent
memory-heavy job from another user on the shared server pushed total system
RAM over the edge. Recovered from the last checkpoint (step 10,000);
`DIT_NUM_WORKERS` reduced to 8.

### The DiT training plateau, and its two root causes

The first full DiT training run reached step ~76,000/400,000. Loss dropped
quickly from ~2.8 to ~1.3 by step 4,000, then **stayed flat within noise for
the remaining ~72,000 steps.** Checkpoint-level evaluation (`eval_dit.py`) at
steps 60k/65k/70k showed generated-latent variance at only 86-88% of real
latent variance (a signature of an undertrained flow-matching model
regressing toward the mean), and elevated high-frequency energy in generated
audio relative to both real audio and the VAE's own reconstruction ceiling.

Two concrete, fixable causes were identified:

1. **No learning-rate decay.** The scheduler only implemented linear warmup
   over 1,000 steps, then held the learning rate constant at its peak value
   forever. A constant high LR after the loss reaches a basin is a classic
   cause of bouncing around a minimum instead of settling into it.
2. **Unnormalized latent channels.** The raw VAE `mu` values cached to disk
   have a 2.2x spread in per-channel standard deviation (1.21 to 2.71 across
   the 32 channels), with no normalization applied anywhere before the flow
   matching MSE loss. Since MSE loss scales with variance, the highest-
   variance channels dominated the gradient signal, at the expense of
   quieter channels that plausibly carry finer detail. This also meant the
   `z1` (data) distribution did not match the `z0 ~ N(0, I)` noise prior on a
   per-channel basis, which the flow-matching path implicitly assumes are
   comparable in scale. Rescaling latents before training a latent
   generative model is standard practice (e.g. Stable Diffusion's fixed
   0.18215 rescaling constant, from Rombach et al., 2022's released
   implementation).

**Both were fixed**, and since normalization changes the actual target
distribution the model fits (not just the training schedule), training was
restarted from step 0 on the corrected objective rather than resumed from an
old checkpoint. The old checkpoints and stats were archived, not deleted, at
`checkpoints/dit_prenorm/` and `dit_stats_prenorm.csv`.

- `compute_latent_stats.py` computes per-channel mean/std over the full
  cached latent set, saved to `data/latent_stats.json`.
- `dataset.py` applies per-channel z-scoring to cached latents when
  `normalize_latents=True` (used only by the DiT training path — the VAE
  training path, which consumes raw audio, is unaffected).
- `train_dit.py`'s scheduler now does linear warmup over 1,000 steps, then
  cosine decay to a 5% floor over the full 400,000-step schedule.
- `sample_dit.py` and `eval_dit.py` denormalize generated latents
  (`z * std + mean`) before decoding through the VAE, since the VAE was
  trained on the raw (unnormalized) latent scale.

---

## Results (current, normalized run)

As of step ~48,500/400,000 (roughly 12% through the intended schedule):

- Loss floor dropped from ~1.3 (raw/unnormalized run) to **~0.9**, a direct
  benefit of removing the channel-scale-driven inflation of the MSE loss.
- The loss is **not** flat, despite visually looking that way at a glance: a
  bucketed-mean regression shows a real ~24% decline since the run began,
  and the across-bucket drift (0.057) is more than double the within-bucket
  minibatch noise (0.024).
- Checkpoint-level generation quality (`eval_dit.py`, steps 35k/40k/45k)
  shows the generated/real latent variance ratio improved to **0.90-0.92**
  (up from 0.858-0.875 in the pre-fix run) — measurably less variance
  collapse, i.e. less of the "regression to the mean" signature of an
  undertrained model.
- **Not yet resolved**: generated audio still shows elevated high-frequency
  energy (0.027-0.030) relative to real audio (0.0198) and the VAE's own
  ceiling (0.0142), essentially unchanged from the pre-fix run. Direct
  listening (`sample_dit.py`) confirms recognizable structure with a
  "muddy" quality — likely explained by still-imperfect generated latents
  landing slightly off the manifold the VAE decoder was trained on, with the
  decoder introducing artifact noise for those. This is distinct from the
  variance-collapse issue that normalization measurably improved, and
  remains open.
- The learning-rate decay's contribution has **not yet been meaningfully
  tested**: at 12% through a cosine schedule over 400,000 steps, LR is still
  at ~97% of its peak value. The decay is deliberately backloaded (cosine
  curves are flattest near both ends), so its effect on any residual noisy-
  basin behavior will not show until much later in training.

---

## Known issues and limitations

- **No train/validation split exists anywhere in the pipeline.** All loss
  and evaluation numbers reported above are computed on data the model was
  trained on; there is currently no way to distinguish memorization from
  generalization.
- **The `latent_cos` and SNR-vs-one-real-clip metrics in `eval_dit.py` are
  weaker signals than they first appear.** They were adapted from
  `overfit_dit.py`'s single-batch memorization test, where high cosine
  similarity to one specific fixed target is the correct thing to check.
  At full scale, DiT is asked to generate *a* plausible sample for a given
  tag set, not reconstruct *that specific* real clip, so low values on these
  two metrics in isolation are not by themselves evidence of a problem — the
  variance-ratio and HF-ratio metrics are more trustworthy.
- **`DIT_KEEP_LAST=3` has twice limited trend analysis** to whatever the
  last few checkpoints happen to be, since older checkpoints are deleted as
  training progresses. Worth increasing before the next long analysis gap.
- **CFG dropout mixes two different objectives into one reported loss
  number.** Conditional and unconditional flow matching are not equally
  hard, so the 20% of every batch with dropped conditioning adds noise to
  the aggregate loss regardless of true model progress; conditional-only
  loss is not currently logged separately.
- **The VAE reconstruction ceiling (6.59 dB SNR, modest HF loss) is a hard
  limit on achievable final quality**, independent of how well DiT training
  goes.

---

## Future work

In rough priority order:

1. **Continue DiT training substantially further.** Only ~12% through the
   intended 400,000-step schedule; the cosine LR decay has not had a chance
   to act yet.
2. **Diagnose the elevated high-frequency-noise / "muddy" artifact**
   specifically, independent of the variance-collapse issue already
   improved. Candidate next steps: log conditional and unconditional loss
   separately to unmask true conditional-branch progress under CFG-dropout
   noise; try more Euler sampling steps or a different CFG scale; check
   whether generated latents lie measurably further from the VAE decoder's
   training manifold than real latents do.
3. **Add a held-out validation split**, so loss and evaluation metrics are
   no longer entirely computed on data the model has seen.
4. **Increase checkpoint retention** (or save periodic evaluation
   snapshots independent of the rolling checkpoint window) so future trend
   analysis is not limited to a narrow recent window.
5. **Re-evaluate the VAE reconstruction ceiling** once DiT training
   saturates — if the ~6.59 dB / reduced-HF ceiling turns out to be the
   binding constraint on final audio quality rather than DiT undertraining,
   revisit VAE capacity or training length.
6. **Lower-confidence, longer-horizon questions**, only worth investigating
   if the loss trend genuinely flattens again despite the fixes above:
   whether 38.8M DiT parameters is enough capacity for the diversity of
   55,609 tracks, and whether MTG-Jamendo's sparse/noisy multi-label tags
   cap how specialized the conditioning signal can get.
7. **Resolve the still-open full-dataset vs. filtered-instrumental-subset
   question** noted in the original project plan (`project_context_transfer_v2.md`)
   — training currently uses the full 55,609-track set, including
   vocal-tagged tracks.

---

## References

- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. (2022).
  High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
- Peebles, W., Xie, S. (2023). Scalable Diffusion Models with Transformers
  (DiT). ICCV.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. (2023). Flow
  Matching for Generative Modeling. ICLR.
- Liu, X., Gong, C., Liu, Q. (2022). Flow Straight and Fast: Learning to
  Generate and Transfer Data with Rectified Flow. arXiv preprint.
- Ho, J., Salimans, T. (2022). Classifier-Free Diffusion Guidance. NeurIPS
  Workshop on Deep Generative Models.
- Ho, J., Jain, A., Abbeel, P. (2020). Denoising Diffusion Probabilistic
  Models (DDPM). NeurIPS.
- Esser, P., Rombach, R., Ommer, B. (2021). Taming Transformers for
  High-Resolution Image Synthesis (VQGAN). CVPR.
- Défossez, A., Copet, J., Synnaeve, G., Adi, Y. (2022). High Fidelity
  Neural Audio Compression (EnCodec). arXiv preprint.
- Kong, J., Kim, J., Bae, J. (2020). HiFi-GAN: Generative Adversarial
  Networks for Efficient and High Fidelity Speech Synthesis. NeurIPS.
- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S.
  (2022). A ConvNet for the 2020s (ConvNeXt). CVPR.
- Siuzdak, H. (2023). Vocos: Closing the Gap Between Time-Domain and
  Fourier-Based Neural Vocoders for High-Quality Audio Synthesis. arXiv
  preprint.
- Kaneko, T., Tanaka, K., Kameoka, H., et al. (2024). WaveNeXt: ConvNeXt-
  Based Fast Neural Vocoder Without ISTFT. ICASSP.
- Bogdanov, D., Won, M., Tovstogan, P., Porter, A., Serra, X. (2019). The
  MTG-Jamendo Dataset for Automatic Music Tagging. Machine Learning for
  Music Discovery Workshop, ICML.
- Copet, J., Kreuk, F., Gat, I., Remez, T., Kant, D., Synnaeve, G., Adi, Y.,
  Défossez, A. (2023). Simple and Controllable Music Generation (MusicGen).
  NeurIPS.
- DiffRhythm (2025). Blazingly Fast and Embarrassingly Simple End-to-End
  Full-Length Song Generation with Latent Diffusion. arXiv preprint. Cited
  here as the stated design inspiration for the overall non-autoregressive,
  flow-matching, full-length-latent architecture; no code or weights from
  this work were used — every component in this repository was implemented
  and trained from random initialization.

---

## Repository layout and setup

See `PROJECT.md` for the full file structure, complete hyperparameter table,
and the three-step training sequence (train VAE, cache latents, train DiT).
