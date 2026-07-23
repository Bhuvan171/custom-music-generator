# Text-to-Music Generation

A from-scratch, locally-run music generator: a spectral waveform VAE compresses
30-second clips into a compact latent, and a Diffusion Transformer (DiT)
trained with flow matching generates latents from a **natural-language
prompt**, with an optional reference track for secondary harmonic control.
Trained on MTG-Jamendo. Everything except the text encoder (frozen T5, kept
pretrained deliberately — see `project_transfer.md`) is trained from random
initialization on a single A100 80GB.

For full architectural reasoning, the measurement-audit history, config
reference, and known open problems, see **`project_transfer.md`** — this file
is deliberately a short front door, not the full technical record.

---

## Pipeline

```
prompt ──► Qwen2-Audio captions (offline) ──► frozen T5 ──┐
                                                            ├─► DiT (flow matching, cross-attn) ──► VAE decoder (iSTFT head) ──► audio
reference track (optional) ──► chroma/texture features ───┘
audio ──► VAE encoder (frozen) ──► cached latents ─────────┘  (training target)
```

Text is the primary conditioning signal, via cross-attention to a frozen T5
token sequence. A reference track is optional secondary control (per-frame
chroma + texture), same role a melody reference plays in MusicGen.

## Status (2026-07-22)

- **VAE**: encoder frozen throughout. Decoder head replaced (raw-frame
  overlap-add → Vocos-style magnitude+phase iSTFT) and fine-tuned 5,000 steps
  to fix a measured phase-cancellation defect. Listened to and confirmed good.
- **DiT**: rebuilt with T5 cross-attention and scaled 39M → 158.6M params.
  Training from scratch, in progress (~47,000 / 400,000 steps). Gradient
  checkpointing keeps it under ~60GB VRAM on the shared GPU.
- **Held-out results so far** (16 fixed clips, never trained on):
  - Chroma-following (reference-track path): **100% of ceiling**, matches
    train closely — fully learned, generalizes cleanly.
  - CLAP score (does generated audio match its *prompt*, vs a shuffled one):
    newly positive and real (train/held-out both clearing their ceiling as of
    step 47,000) after finding and replacing a broken CLAP checkpoint —
    `laion/larger_clap_music` is an HTSAT-base HuggingFace port with a
    documented, unresolved conversion bug ([LAION-AI/CLAP#126](https://github.com/LAION-AI/CLAP/issues/126));
    swapped to `laion/clap-htsat-unfused`, verified via direct diagnostic
    before trusting it.
  - Text-conditioning gate is small but has grown every eval checkpoint since
    step 1,000 (0.000 → 0.030) — real but still early; reference-track gates
    are far more developed (~0.98) at this point in training.
- Still very early in the training budget (~12%) — expect all of the above to
  keep moving.

## Running it

Order matters — each stage's output is the next stage's input.

```bash
# 1. Caption every track with an audio-language model (one-time, offline)
python caption_audio.py --backend vllm

# 2. Frozen T5 encodes captions to cached token embeddings
python compute_text_embeddings.py

# 3. Decoder fine-tune (iSTFT head, encoder frozen — see project_transfer.md)
python finetune_decoder.py

# 4. Train the DiT from scratch
python src/train_dit.py

# 5. Generate — text is primary, --reference is optional secondary control
python generate.py --prompt "a slow, moody piano ballad with brushed drums"
python generate.py --prompt "heavy metal guitar" --reference song.mp3
```

## Why these choices, briefly

- **Captions from an audio-language model, not tag paraphrasing.** An LLM
  rewording a track's tags adds zero new information (the LP-MusicCaps trap).
  Qwen2-Audio actually *listening* to the track adds real content beyond the
  195-tag vocabulary this project started with.
- **Frozen pretrained T5, not a from-scratch text encoder.** Language
  generalization needs far more than 55k captions can provide; a from-scratch
  encoder would be brittle to any phrasing outside its training set — the
  opposite of the natural-language goal. Deliberate exception to training
  everything else from scratch.
- **iSTFT decoder head over raw-frame + overlap-add.** The old head had no
  phase representation at all — a loss patch (`BandEnergyLoss`) recovered some
  of a measured 52%/37% energy loss at 2-6kHz/0.5-2kHz but plateaued and made
  output peakier, direct evidence the architecture, not the loss, was the
  wall.
- **Flow matching + DiT over autoregressive tokens**: no vector-quantization
  stage, straight-line probability path, sampling in tens of steps.

## Known limitations

- Bounded by a from-scratch decoder/DiT on ~55k tracks — will not match
  Stable Audio Open's polish (7,300h of data, years of tuning).
- MTG-Jamendo is CC BY-NC-SA; any model trained on it inherits that
  non-commercial license regardless of code license (non-blocking for local/
  research use).
- A persistent, three-times-measured plateau in 2-6kHz decoder reconstruction
  (~-40% vs real) that phase-aware synthesis improved but did not fully close.
- See `project_transfer.md` for the full measurement-audit history, config
  reference, and everything that's been tried and ruled out.
