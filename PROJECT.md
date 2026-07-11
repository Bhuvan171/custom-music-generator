# Music Generation Project

Tag-conditioned music generation using a two-stage latent diffusion pipeline:
VAE (audio ↔ latents) + DiT (tags → latents via flow matching).

---

## Architecture Overview

```
[Audio 660480 samples]
        │
        ▼
  ┌──────────┐
  │  VAE     │  encode: audio → (B, 32, 645) latent mu
  │ Encoder  │  decode: latent → audio via overlap-add
  └──────────┘
        │
   z (B,32,645)
        │
        ▼
  ┌──────────┐
  │  DiT     │  flow matching: noise → latent, conditioned on music tags
  │ (38.3M)  │  CFG inference: cfg_scale=4.0
  └──────────┘
        │
    generated z
        │
        ▼
  VAE Decoder → generated audio
```

---

## Stage 1: Spectral VAE

### Architecture (`src/vae.py`)

A symmetric spectral VAE where both encoder and decoder run at STFT frame rate —
only the FFT and the final overlap-add touch sample rate.

**Encoder**
```
audio (B, 1, 660480)
→ STFT (nfft=512, hop=256) → sign·log1p compression → (B, 514, 2580)
→ Conv1d projection → D=384 channels
→ 8× ConvNeXt1d blocks at 2580 frames
→ strided Conv1d ×4 downsample → 645 frames
→ Conv1d → mu, logvar  (B, 32, 645)
```

**Decoder**
```
z (B, 32, 645)
→ Conv1d projection → D=384 channels
→ 4× interpolate upsample → 2580 frames
→ Conv1d refinement
→ 8× ConvNeXt1d blocks
→ Linear(384 → 512) per frame  [WaveNeXt-style head]
→ Hann-windowed overlap-add → audio (B, 1, 660480)
```

**ConvNeXt1d block**: depthwise `Conv1d(D, D, 7)` → LayerNorm → pointwise ×4 → GELU → pointwise → residual.

**Key property**: 645 latent frames × 4 upsample × 256 hop = 660480 samples exactly. No padding artifacts.

### Discriminator (`src/vae.py`)

MS-STFT Discriminator (Encodec-style): three `STFTDiscriminator` instances over n_fft = [512, 1024, 2048].
Each computes `torch.stft` → stacks `[real, imag]` as 2 channels → 5× weight-normed `Conv2d` tower → logit + feature list.

### Losses (`src/losses.py`)

| Loss | Formula | Weight |
|---|---|---|
| Multi-scale mel | log-mel L1 at scales (512/64, 1024/96, 2048/128 mel) | 1.0 |
| Waveform L1 | time-domain L1, anchors phase | 5.0 |
| KL divergence | 0.5·(μ² + σ² − log σ² − 1) | 1e-4 (warmed up over 10k steps) |
| Adversarial (G) | hinge loss vs MS-STFT disc | 2.0 × adaptive λ |
| Feature matching | L1 between disc activations | 2.0 |

Adaptive λ: scales adversarial loss to match reconstruction gradient norm (prevents disc from dominating early).

### Training (`src/train_vae.py`)

- Optimizer: AdamW, betas=(0.5, 0.9), eps=1e-6, lr=2e-4
- Precision: bfloat16 autocast, fp32 KL and wave loss
- Batch: 16 clips × 3-second random crops (65536 samples) — full 30s clips encoded at cache time
- Adversarial training starts at step 5000
- Gradient clip: 1.0 for both G and D
- Checkpoints every 2000 steps, 3 kept

**Sizes**: VAE 20.4M params | Disc 5.3M params | VRAM ~18.2 GB at B=16

### Overfit validation

Before committing to full training, a single-batch overfit test (`overfit_test.py`) verified reconstruction capacity:
- B=4, 8000 steps, LR=1e-3, deterministic decode (mu, no sampling)
- **Final SNR: 18.58 dB** — well above the 6 dB gate
- Confirmed the latent bottleneck (32×645) is not the limiting factor

---

## Stage 2: Music DiT

### Architecture (`src/dit.py`)

Diffusion Transformer (DiT) with adaLN-Zero conditioning. Operates on VAE latents.

```
z_noisy (B, 32, 645)
→ Linear(32 → 512) + sinusoidal pos embed
→ 8× DiTBlock(D=512, heads=8)
→ LayerNorm → Linear(512 → 32)
→ predicted velocity (B, 32, 645)
```

**DiT block (adaLN-Zero)**:
```
cond (B, 512) → SiLU → Linear(512 → 6×512) → [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f]

x = x + gate_a · MultiheadAttn(norm1(x) * (1+scale_a) + shift_a)
x = x + gate_f · FFN(norm2(x) * (1+scale_f) + shift_f)
```
Output projection zero-initialized → identity at init for stable training.

**Conditioning**: `cond = timestep_emb(t) + tag_emb(tags)`

**Timestep embedding**: sinusoidal (dim=256) → MLP → D=512

**Tag embedding**: 195-class multi-label tags (genre, instrument, mood/theme from MTG-Jamendo)
- Each active tag: `Embedding(195, 64)` → mean-pool over active tags → `Linear(64, 512)`
- Null/uncond: learnable `nn.Parameter(zeros(512))`
- Empty tag lists use the null embedding automatically

**DiT size**: 38.3M params

### Training (`src/train_dit.py`)

**Algorithm: linear flow matching**
```
t ~ U[0, 1]
z0 ~ N(0, I)          (noise)
z1 = clean latent      (data)
z_t = (1-t)·z0 + t·z1 (linear interpolation)
v_target = z1 - z0     (constant velocity along linear path)
loss = MSE(model(z_t, t, tags), v_target)
```

- CFG dropout: 20% of batches drop tag conditioning (replaced with null)
- Optimizer: AdamW, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, lr=1e-4
- LR warmup: linear over 1000 steps
- EMA: exponential moving average of weights, decay=0.999
- Gradient clip: 1.0
- Batch: 64 latents (pre-cached, no audio decoding at train time)
- Checkpoints every 5000 steps, 3 kept

### Inference: Euler ODE with CFG (`MusicDiT.sample`)

```python
z ~ N(0, I)         # start from noise
for i in range(EULER_STEPS=50):
    t = i / 50
    # cond + uncond batched in one forward pass for efficiency
    [v_cond, v_uncond] = model([z, z], t, [tags, null_tags])
    v = v_uncond + CFG_SCALE * (v_cond - v_uncond)   # CFG_SCALE=4.0
    z = z + (1/50) * v
# z is now a generated latent → VAE decoder → audio
```

---

## Dataset

**MTG-Jamendo** — ~55,000 royalty-free music tracks with multi-label tags.

Tag vocabulary: **195 tags** across 3 categories:
- `genre---*` (60s, ambient, rock, jazz, …)
- `instrument---*` (guitar, piano, drums, …)
- `mood/theme---*` (happy, relaxing, epic, …)

**Dataset pipeline** (`src/dataset.py`):
- Loads FLAC files or pre-cached latent `.pt` files
- Maps track IDs to tag indices from `data/tag_vocab.json`
- FLAC loading: random 30s crop, zero-padded if shorter
- `.pt` loading: returns cached latent directly (fast, no audio decode)

---

## Config (`config.py`)

All hyperparameters in one place — no hardcoded values anywhere else.

| Group | Key params |
|---|---|
| Audio | `SAMPLE_RATE=22050`, `CLIP_DURATION=30`, `CHUNK_SAMPLES=660480` |
| VAE latent | `VAE_LATENT_DIM=32`, `VAE_LATENT_LEN=645` |
| VAE arch | `VAE_DEC_DIM=384`, `VAE_DEC_BLOCKS=8`, `VAE_DEC_NFFT=512`, `VAE_DEC_HOP=256` |
| VAE train | `VAE_LR=2e-4`, `VAE_BATCH_SIZE=16`, `VAE_TOTAL_STEPS=200000`, `VAE_ADV_START=5000` |
| DiT arch | `DIT_D_MODEL=512`, `DIT_HEADS=8`, `DIT_LAYERS=8`, `DIT_VOCAB_SIZE=195` |
| DiT train | `DIT_LR=1e-4`, `DIT_BATCH_SIZE=64`, `DIT_TOTAL_STEPS=400000`, `DIT_EMA_DECAY=0.999` |
| Inference | `EULER_STEPS=50`, `CFG_SCALE=4.0` |

---

## File Structure

```
music-gen/
├── config.py                  # all hyperparameters
├── src/
│   ├── vae.py                 # WaveformVAE + MS-STFT discriminator
│   ├── losses.py              # MultiScaleMelLoss, GAN losses, adaptive weight
│   ├── dataset.py             # JamendoDataset (FLAC + latent .pt)
│   ├── train_vae.py           # VAE-GAN training loop
│   ├── dit.py                 # MusicDiT (flow matching transformer)
│   ├── train_dit.py           # DiT training loop with EMA
│   └── cache_latents.py       # encode all FLACs → latent .pt files
├── overfit_test.py            # single-batch sanity test for VAE
├── plot_vae_stats.py          # plot vae_stats.csv training curves
├── data/
│   ├── stems/                 # FLAC audio files (55k tracks)
│   ├── latents/               # cached VAE latents (created by cache_latents.py)
│   └── tag_vocab.json         # 195-tag vocabulary
├── checkpoints/
│   ├── vae/                   # VAE checkpoints
│   └── dit/                   # DiT checkpoints
└── samples/
    ├── vae/                   # reconstruction samples during VAE training
    └── dit/                   # generated samples during DiT training
```

---

## Three-Step Training Sequence

### Step 1 — Train VAE (in progress)
```bash
nohup python src/train_vae.py 2>&1 | tee training.log &
```
- **Status**: running, step ~105k / 200k, ~18.2 GB VRAM, ETA ~Jul 3
- Mel loss trending down, discriminator stable (adv_d ~0.6–1.0)

### Step 2 — Cache latents (run once after VAE finishes)
```bash
python src/cache_latents.py --checkpoint checkpoints/vae/vae_step0200000.pt
```
- Encodes all 55k FLAC files with frozen VAE, saves `data/latents/track_XXXXXXX.pt`
- Each file: (32, 645) float32 = ~83 KB. Total ~4.5 GB.
- Estimated time: 1–2 hours on A100

### Step 3 — Train DiT
```bash
nohup python src/train_dit.py 2>&1 | tee dit_training.log &
```
- 400k steps, batch=64 latents, flow matching MSE
- Checkpoints every 5k steps, EMA model saved alongside

---

## Hardware

- GPU: shared 80 GB A100 (multiple users; personal budget capped at 40 GB)
- VRAM used by VAE training: **18.2 GB** (well within budget)
- Estimated VRAM for DiT training: < 10 GB (latents are small, no audio decode)

---

## Design Principles

- **Simple and hackable**: no abstraction beyond what the task requires; all values in `config.py`
- **No VRAM surprises**: per-process `torch.cuda.max_memory_allocated()` used for sizing, not system `nvidia-smi`
- **Spectral throughout**: VAE encoder and decoder both run at STFT frame rate; only FFT and OLA touch sample rate
- **Modern stack**: WaveNeXt-style OLA decoder (no transposed convs → no ringing), MS-STFT discriminator, log-mel loss, flow matching DiT
- **Pre-caching**: DiT trains on stored latents, not raw audio — 64× faster data loading, no GPU bottleneck on decode
