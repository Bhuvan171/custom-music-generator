# Project Context Transfer: From-Scratch Waveform Latent Diffusion Music Generator

## Objective
Build and train, **entirely from scratch (no pretrained weights anywhere)**, a text/tag-conditioned instrumental music generator producing 30s clips.

- **Hardware:** Single NVIDIA A100 80GB, local server, no sudo access.
- **Timeline constraint:** Training must complete in ≤ 1 week, single GPU.
- **Scope:** Purely instrumental generation. No lyrics/vocal-content modeling.
- **Philosophy:** Architecture is *inspired by* DiffRhythm (simplest viable design we found), but every component — VAE, discriminator, DiT, conditioning modules — is implemented and trained from random init by the user, not loaded from any checkpoint.

---

## Finalized Architecture

### Pipeline (high level)
```
Raw waveform (22050Hz mono, 30s)
  → Waveform VAE Encoder (frozen after VAE training)
  → Latent (16 × 645 @ 21.5Hz)
  → Flow Matching DiT  ←  Tag conditioning (embed → mean-pool → 1-layer LSTM → AdaLN)
                       ←  Timestep conditioning (sinusoidal → MLP)
  → Waveform VAE Decoder (frozen)
  → Generated waveform (.wav, ready to play)
```
No vocoder stage. The VAE decodes latent directly to waveform — mel spectrogram + vocoder (Griffin-Lim/HiFi-GAN) was an earlier design that was **replaced**, see Alternatives.

### 1. Audio Preprocessing
- Resample all source audio to 22050Hz mono.
- Chunk into 30s windows (random offset during training for augmentation).
- No mel spectrogram computed anywhere in this pipeline — VAE operates on raw waveform.

### 2. Waveform VAE (~20M params, train first, then freeze)
- **Encoder:** `Conv1d(1→32, k=7)` → 4 EncoderBlocks (each: 3× ResBlock1d + strided Conv1d) with strides `[4,4,8,8]` = **1024× total compression** → split into `mu`, `logvar`, each shape `(16, 645)`.
- **Decoder:** mirrors encoder with `ConvTranspose1d` upsampling blocks, outputs raw waveform directly.
- **Loss:** `L_stft (multi-resolution, FFT sizes 256/512/1024/2048/4096) + L_adv (multi-scale waveform discriminator, 3 scales) + 1e-4 × KL`.
- Reparameterize: `z = mu + eps * exp(0.5*logvar)`.
- After training: encode entire dataset once, cache latents to disk (`.pt`), VAE frozen for all subsequent stages.

### 3. Flow Matching DiT (~40M params, the trainable generator)
- **Backbone:** 8 LLaMA-style decoder layers (RMSNorm + RoPE self-attention + SwiGLU FFN), `d_model=512`, 8 heads, **bidirectional attention** (no causal mask — this is diffusion, not autoregressive).
- **Input/output:** `Linear(16→512)` in, `Linear(512→16)` out, operating on latent `(B, 16, 645)`.
- **Conditioning:** `nn.Embedding(195, 512)` (one row per Jamendo tag) → mean-pool active tags per track → 1-layer LSTM (final hidden state) = `style_hidden`. Timestep → sinusoidal Fourier features → MLP = `timestep_emb`. `cond = style_hidden + timestep_emb`, injected into every layer via **AdaLN** (not cross-attention — cheaper, sufficient for global conditioning). 20% conditioning dropout during training for classifier-free guidance at inference.
- **Training objective (flow matching / rectified flow):**
  ```
  z1 = cached_latent; z0 ~ N(0,I); t ~ logit_normal(0,1)
  zt = (1-t)*z0 + t*z1
  loss = MSE(model(zt, t, cond), z1 - z0)
  ```
- **Optimizer:** AdamW (β=0.9, 0.95), lr=1e-4, 1000-step warmup + cosine decay, EMA decay=0.999, FlashAttention2 + gradient checkpointing.

### 4. Inference
- Sample `z0 ~ N(0,I)`, Euler ODE solve over 50 steps with classifier-free guidance (scale 3–5).
- Decode through frozen VAE decoder → waveform directly. No further processing needed.

---

## Dataset

**Chosen: MTG-Jamendo, `raw_30s` subset, `audio-low` quality.**
- ~55,000 tracks, ~3,500 hours total, 195 tags (genre/instrument/mood-theme) per `autotagging.tsv`.
- Download via **official script only**: `MTG/mtg-jamendo-dataset` repo, `scripts/download/download.py --dataset raw_30s --type audio-low --from mtg-fast --unpack --remove`.
- Tag structure directly drives the DiT's `nn.Embedding(195, ...)` conditioning — this is *why* Jamendo was chosen over alternatives (see below).

**Open decision, not yet finalized:** train on full 55K tracks vs. filter down to ~20-25K instrumental-leaning tracks (drop vocal-tagged tracks, optionally run HTDemucs vocal-energy filtering). Tradeoffs:
- Full set: more data (better generalization, less overfitting), but splits fixed model capacity across vocal+instrumental modes, adds ~12-24h preprocessing/encoding time, and only partially matches the original instrumental-only project scope.
- Filtered subset: full model capacity dedicated to instrumental fidelity, faster preprocessing, matches original scope exactly, but less total data.
- **If keeping vocals:** the `voice` instrument tag (already in the 195-tag vocabulary) can act as a soft on/off conditioning switch at inference. Expect vocal-*flavored texture* (formants, syllable-rate rhythm), **not intelligible lyrics** — there is no phoneme/G2P pathway in this architecture (see Alternatives: Lyrics subsystem).

---

## Expected Output Quality (calibrated expectations)
- Reference point: roughly early-Riffusion-era quality — recognizable genre/instrument texture, audible artifacts, no real compositional structure (no intentional verse/chorus).
- Short clips (5-10s) hold together better than full 30s clips; coherence drifts past ~10-15s.
- Estimated FAD ≈ 10-20 (vs. pretrained SOTA ≈ 1-3, MusicGen-small ≈ 4-5). This is a real, audible gap, not "nearly as good."
- **Ranked bottlenecks:** (1) DiT undertraining — dominant factor, ~40M params / 55K tracks / 4-5 days is far below SOTA data+compute scale. (2) VAE reconstruction ceiling — hard floor, whatever fidelity is lost here can never be recovered downstream. (3) Adversarial VAE training instability — a *risk multiplier*, not guaranteed, but the most likely thing to blow the timeline.
- **Fallback if VAE GAN training destabilizes by day 2:** drop the adversarial loss entirely, train with STFT + KL only, accept blurrier output, protect DiT training time (higher-leverage than VAE polish given fixed budget).

---

## Alternatives Considered and Rejected
*(Listed with rationale. Claude Code may resuggest any of these if it has a concrete, well-reasoned argument for why the tradeoff calculus has changed — these were not absolute rules, they were judgment calls under the stated constraints.)*

| Alternative | Why rejected |
|---|---|
| **Fine-tune pretrained DiffRhythm weights (LoRA or full FT)** | User explicitly wants to implement and train every component from scratch as a learning exercise, not fine-tune existing checkpoints. |
| **Ollama/Qwen as the generative backbone** | Wrong token space — Qwen's vocab is text BPE tokens, our DiT operates on continuous audio latents. Architectural *style* (RoPE, SwiGLU, RMSNorm) was borrowed; weights were not. |
| **Autoregressive transformer over discrete audio codec tokens (MusicGen/YuE-style)** | Rejected in favor of latent diffusion: AR models showed exposure bias, quantization bottlenecks, and training instability at small parameter scales — flow matching trains more stably from random init in a 1-week budget. |
| **Mel spectrogram + vocoder (Griffin-Lim or HiFi-GAN)** | Early design choice, since replaced. A waveform-native VAE collapses two trained stages (audio VAE + vocoder) into one, removing a whole source of artifacts and a training stage from the timeline. |
| **DDPM-style diffusion (ε-prediction, standard noise schedule)** | Flow matching needs far fewer inference steps (~50 Euler steps vs. hundreds-to-thousands), has a simpler loss surface, and tends to train more stably at small scale — directly relevant under the 1-week constraint. |
| **CNN U-Net generative backbone** | Local receptive fields struggle to capture long-range temporal/structural coherence across 645 latent frames (30s). Self-attention (DiT) models any-to-any temporal dependency directly. |
| **Cross-attention for tag conditioning** | AdaLN is far cheaper compute-wise for global (non-sequential) conditioning signals; appropriate given the small fixed parameter/compute budget. Cross-attention is the right tool *only* if conditioning becomes sequence-structured (e.g., phoneme-level lyrics). |
| **Lyrics/phoneme conditioning subsystem (G2P, forced alignment, phoneme embeddings, cross-attention injection)** | Cut entirely — purely instrumental scope makes it irrelevant, and it was DiffRhythm's own most training-unstable component per their ablations. Reintroducing this is scoped as a separate follow-on project, not a feature of this build. |
| **FMA, MagnaTagATune, Slakh2100, MusicCaps, AudioSet as primary training data** | FMA: sparse single-genre labels, no mood/instrument tags, breaks the 195-tag conditioning design. MagnaTagATune: small (~170h), older/lower-quality source audio. Slakh2100: synthetic MIDI renders, no real tag metadata. MusicCaps: rich captions but only ~15h, eval-only scale. AudioSet: weakly labeled, scraping/legal complexity, wrong fit for a 1-week scoped project. |
| **HuggingFace community mirror (`rkstgr/mtg-jamendo`)** | Investigated as a faster-download alternative to MTG's academic mirrors. Rejected after inspection revealed it repackages audio as Opus (not verified to match official `audio-low` spec, uncertain provenance/freshness — last updated ~3 years ago). Reverted to the official `MTG/mtg-jamendo-dataset` `download.py` script with the `mtg-fast` mirror. |
| **Precomputed mel-spectrogram download (official Jamendo `melspecs` type)** | N/A to this pipeline — no mel spectrogram stage exists anywhere in the current waveform-VAE architecture. Would only matter if the project reverts to a mel-based design. |
| **Full 55K vs. filtered ~20-25K instrumental-only training set** | Still genuinely open — see Dataset section above. Not a closed decision. |

---

## Current Pipeline Status (as of last session)
- Dataset download in progress via official `download.py` script (`raw_30s`, `audio-low`, `mtg-fast` mirror), run inside a persistent background session (`tmux` or `nohup`) to survive intermittent server-side wifi drops.
- ~97 of 100 shards (`raw_30s_audio-low-00.tar` through `-96.tar`) successfully downloaded (~152GB). Remaining: shards 97-99.
- Script confirmed to skip already-downloaded shards on rerun (safe to re-invoke without redundant redownloads).
- Python env: dedicated `venv` at `~/6DGS/music-gen/venv`, must be reactivated each new SSH session (`source ~/6DGS/music-gen/venv/bin/activate`) — does not persist automatically.
- Disk usage should be monitored (`du -sh`, `df -h`) before the `--unpack` phase proceeds at scale, since unpacking temporarily increases peak disk usage before `--remove` cleans up source tars.

## Immediate Next Steps
1. Finish dataset download (3 remaining shards), verify full `autotagging.tsv` parse and tag distribution.
2. Decide full-dataset vs. filtered-instrumental-subset (see open decision above) before writing the DataLoader.
3. Implement audio preprocessing (resample/chunk, no mel step).
4. Implement Waveform VAE (encoder/decoder/discriminator), begin training, monitor for GAN instability (fallback plan above).
5. Implement Flow Matching DiT + conditioning modules, begin training once VAE latents are cached.
