# --- Paths ---
STEMS_DIR         = "data/stems"
LATENTS_DIR       = "data/latents"
TSV_PATH          = "mtg-jamendo-dataset/data/autotagging.tsv"
VOCAB_PATH        = "data/tag_vocab.json"
LATENT_STATS_PATH = "data/latent_stats.json"   # per-channel mean/std, see compute_latent_stats.py
FEATURES_DIR      = "data/features"            # per-track musical features, see extract_features.py
FEATURE_STATS_PATH = "data/feature_stats.json" # per-channel mean/std, see compute_feature_stats.py

# HELD-OUT eval features. MUST be a separate directory from FEATURES_DIR: train_dit.py defines
# its training set as (cached latents) INTERSECT (tracks with features), so dropping held-out
# features into FEATURES_DIR would silently pull those tracks INTO training — and their cached
# latents are stale random-crop ones, so they would also be misaligned. Eval never reads cached
# latents (SNREvaluator encodes the FLAC fresh), so held-out clips need features and nothing else.
FEATURES_VAL_DIR  = "data/features_val"

# --- Audio ---
SAMPLE_RATE   = 22050
CLIP_DURATION = 30                                                # seconds
CHUNK_SAMPLES = (CLIP_DURATION * SAMPLE_RATE // 1024) * 1024      # 660480

# --- VAE ---
VAE_LATENT_DIM = 32
VAE_LATENT_LEN = CHUNK_SAMPLES // 1024                           # 645
VAE_BATCH_SIZE = 16
VAE_TRAIN_SECONDS = 3                                            # train on short random crops (VAE is conv;
VAE_TRAIN_SAMPLES = (VAE_TRAIN_SECONDS * SAMPLE_RATE // 1024) * 1024   # full clips encoded at cache time). 65536
VAE_LR         = 2e-4
VAE_KL_WEIGHT  = 1e-4
VAE_ADV_START  = 5_000

# --- Spectral VAE architecture ---
VAE_DEC_DIM      = 384       # ConvNeXt channel width (enc + dec share same D)
VAE_DEC_BLOCKS   = 8         # number of ConvNeXt1d blocks in each of enc/dec body
VAE_DEC_NFFT     = 512       # STFT window size (enc input, dec OLA output)
VAE_DEC_HOP      = 256       # STFT hop = nfft/2 (50% overlap, exact OLA reconstruction)
VAE_DEC_FRAME_UP = 4         # latent→frame upsampling factor (645→2580, 64→256)

# Original decoder head: Linear(D, nfft) -> raw time-domain frame -> Hann-window overlap-add.
# NO PHASE REPRESENTATION. Measured (finetune_decoder.py, this session): the decoder loses 52%
# of real energy at 2-6kHz / 37% at 0.5-2kHz even from GROUND-TRUTH latents, and the loss is
# split between the decoder under-generating that content in the first place (frames already
# -39% before overlap-add) AND overlap-add destructively cancelling MORE of it on top, with the
# damage increasing monotonically with frequency (+1%/-2%/-13%/-34% by band) -- the textbook
# signature of phase incoherence between overlapping windowed frames (a 4kHz tone's ~5.5-sample
# period means a 1-2 sample phase disagreement between adjacent frames destroys it; a 200Hz
# tone's ~110-sample period does not care). A band-normalized ENERGY loss recovered 3 of 4 bands
# substantially but plateaued on the hardest one and made output measurably PEAKIER/noisier
# (crest factor +1.7 to +3.6dB across 5 of 6 tested clips) -- a loss cannot fully fix an
# architecture with no phase representation to loss against.
# VAE_DEC_USE_ISTFT_HEAD replaces the raw-frame head with a Vocos/WaveNeXt-style head: predict
# magnitude + phase (as a unit (cos,sin) vector) per STFT bin per frame, then torch.istft. This
# is a genuine architecture change (dec_head's shape and meaning both change), not a loss patch.
VAE_DEC_USE_ISTFT_HEAD = True

# --- MS-STFT discriminator ---
STFTD_FFTS       = [512, 1024, 2048]

# --- Multi-scale mel reconstruction loss ---
MEL_FFT_SIZES    = [512, 1024, 2048]
MEL_N_MELS       = [64, 96, 128]
MEL_EPS          = 1e-5

# --- VAE training / GAN stability ---
VAE_TOTAL_STEPS  = 200_000
VAE_KL_WARMUP    = 10_000    # steps over which KL weight ramps 0 → VAE_KL_WEIGHT
VAE_ADV_WEIGHT   = 2.0
VAE_WAVE_WEIGHT  = 5.0       # time-domain L1 on the waveform — anchors PHASE
VAE_FEAT_WEIGHT  = 2.0
VAE_GRAD_CLIP    = 1.0
VAE_LAMBDA_MAX   = 1_000.0
STFT_FFT_SIZES   = [256, 512, 1024, 2048, 4096]   # kept for MultiResSTFTLoss (unused in training)
STFT_EPS         = 1e-7

# --- Decoder-only fine-tune (finetune_decoder.py) ---
# Targets the measured VAE defect: even decoding GROUND-TRUTH latents (best case, no DiT
# involved), the reconstruction loses 52% of real energy at 2-6kHz and 37% at 0.5-2kHz vs real
# audio, confirmed via a frames-before-overlap-add probe to originate in the decoder, not the
# encoder (a linear probe recovers real 2-6kHz energy from the frozen latent at R^2=0.518 -- the
# information survives encoding; the decoder just never learned to reproduce it, because nothing
# in its original training loss pushed it to).
#
# MultiResSTFTLoss (defined above, NEVER previously wired into training) is the first thing to
# try: it compares LINEAR-frequency STFT bins directly (both raw magnitude and log-magnitude L1)
# across 5 resolutions, unlike the log-MEL loss, which first collapses many linear bins into
# fewer, wider mel filterbank outputs before comparing — individual narrow-band errors can
# average out inside one wide mel bin in a way they cannot when compared bin-for-bin on a linear
# scale. The mel loss already gives 2-6kHz the largest share of bins of any band (32.6%) and
# still fails to reproduce it, which is why this is worth trying before anything more invasive
# (e.g. a Vocos-style phase-aware synthesis head).
#
# Encoder stays FROZEN: training reads pre-cached latents directly (mu, no reparameterization
# noise — exactly what decode() receives in production) rather than re-running the encoder, which
# is both faster and keeps every cached latent (and the current DiT checkpoint) valid regardless
# of outcome. No adversarial loss in this first pass, so any change in results is attributable to
# the one new ingredient (the STFT loss), not a confound with GAN training dynamics.
# MultiResSTFTLoss was tried FIRST (linear-frequency, but summed across all bins into one
# scalar) and measured, over 1000 real steps / 4 consecutive eval checkpoints, to leave the
# 2-6kHz band completely unchanged (-69% -> -68%, noise-level) -- because it sums magnitude
# error across all frequencies together, and real audio is bass-heavy (86% of total energy
# below 500Hz), so the loss is dominated by bass accuracy almost regardless of what happens in
# a band carrying under 7% of total energy. BandEnergyLoss (src/losses.py) fixes this by
# weighting each band EQUALLY regardless of natural magnitude. This is now the primary new
# ingredient; MultiResSTFTLoss is kept at a small weight since it wasn't harmful, just
# insufficient on its own.
FT_DEC_LR           = 1e-4
FT_DEC_BAND_WEIGHT  = 2.0    # BandEnergyLoss weight -- the fix for the band-domination problem
FT_DEC_STFT_WEIGHT  = 0.25   # MultiResSTFTLoss -- demoted, not the primary fix (see above)
FT_DEC_STEPS       = 5_000
FT_DEC_BATCH       = 16
FT_DEC_LOG_EVERY    = 50
FT_DEC_EVAL_EVERY   = 250
FT_DEC_CKPT_EVERY   = 500
FT_DEC_CKPT_DIR     = "checkpoints/vae_decoder_ft"   # separate from checkpoints/vae/ — the
                                                      # original encoder+decoder stays untouched
                                                      # as the fallback / known-good baseline.

# --- VAE logging / checkpoints ---
VAE_LOG_EVERY    = 100
VAE_CKPT_EVERY   = 2_000
VAE_SAMPLE_EVERY = 2_000
VAE_KEEP_LAST    = 3
VAE_NUM_WORKERS  = 4
CKPT_DIR         = "checkpoints/vae"
SAMPLE_DIR       = "samples/vae"
STATS_CSV        = "vae_stats.csv"
OVERFIT_DIR      = "results/overfit"       # overfit_test.py writes timestamped run dirs here
DIT_OVERFIT_DIR  = "results/overfit_dit"   # overfit_dit.py writes timestamped run dirs here

# --- DiT architecture ---
# D_MODEL/LAYERS/HEADS were 512/8/8 (39.1M, ~DiT-S) for the tags+chroma-conditioned model. That
# checkpoint does NOT transfer to this shape (DiT-B: 768/12/12, ~113M trainable + the new
# cross-attention layers) — text-primary generation is a from-scratch run, by design (see
# project_transfer.md's text-primary redesign). D_MODEL matches TEXT_DIM (T5-base hidden size)
# deliberately: cross-attention's K/V projection is then a plain 768->768 Linear, no separate
# text-side projection needed.
DIT_D_MODEL      = 768
DIT_HEADS        = 12
DIT_LAYERS       = 12
DIT_TAG_DIM      = 64     # per-tag embedding size (before projection to D). Tags are now
                          # AUXILIARY (into adaLN cond), not primary — kept for the cheap global
                          # signal (genre/instrument/mood) text captions might occasionally miss.
DIT_T_EMBED_DIM  = 256    # sinusoidal timestep embedding hidden size
DIT_VOCAB_SIZE   = 195    # MTG-Jamendo tag vocabulary size

# --- Text conditioning (audio-captions -> frozen T5 -> DiT cross-attention) ---
# THE central redesign decision: captions are generated by an AUDIO-LANGUAGE MODEL LISTENING TO
# THE WAVEFORM (Qwen2-Audio), not an LLM paraphrasing the existing 195 tags. Paraphrasing tags
# adds ZERO information (LP-MusicCaps confirmed this empirically) -- the whole point of this
# redesign is to break the ~30-40 bit/track ceiling that tags-only conditioning hit, and that
# requires a genuinely richer training SIGNAL, not a better-dressed version of the same one.
#
# T5 is FROZEN and PRETRAINED, not trained from scratch, which breaks this project's original
# "no pretrained weights" rule -- deliberately. Language generalization (responding sensibly to
# phrasing never seen in training) needs orders of magnitude more text than the ~55k captions
# this project can produce; a from-scratch encoder would only generalize near its exact training
# captions. T5-base's pretraining (external, on huge text corpora) is exactly the thing that
# can't be reproduced locally and is the reason to adopt it rather than train it.
CAPTIONS_DIR       = "data/captions"       # {track_id}.txt or .json, one audio-LM caption each
CAPTIONS_VAL_DIR   = "data/captions_val"   # held-out captions, SEPARATE dir (same reason
                                           # features_val is separate: train = latents ∩ captions)
CAPTION_VERSION    = 1       # bump on any change to the captioning prompt/model (mirrors
                              # FEATURE_VERSION's re-extract-on-mismatch pattern)
QWEN_AUDIO_MODEL   = "Qwen/Qwen2-Audio-7B-Instruct"
CAPTION_MAX_NEW_TOKENS = 120

T5_MODEL_NAME    = "t5-base"     # frozen, ~220M params, hidden size 768 == DIT_D_MODEL
TEXT_DIM         = 768
TEXT_MAX_TOKENS  = 64            # captions are cropped/padded to this many T5 tokens for a fixed
                                  # cache shape (T5 tokenizes a ~30-40 word caption to ~40-55
                                  # tokens; 64 gives headroom without much wasted attention).
TEXT_EMB_DIR     = "data/text_emb"       # cached (TEXT_MAX_TOKENS, 768) fp16 + mask per track
TEXT_EMB_VAL_DIR = "data/text_emb_val"
TEXT_EMB_VERSION = 1   # bump if T5_MODEL_NAME/TEXT_MAX_TOKENS changes independently of captions;
                       # regeneration is also triggered by a CAPTION_VERSION bump (new caption text).

# Text CFG dropout, INDEPENDENT of the chroma/texture dropout -- exactly the DIT_CHROMA_DROPOUT
# pattern (see below), so the model learns p(z|tags), p(z|tags,text), p(z|tags,text,chroma) etc.
# rather than only ever seeing all conditioning at once.
DIT_TEXT_DROPOUT = 0.1

CLAP_MODEL_NAME  = "laion/clap-htsat-unfused"  # held-out eval metric: does GENERATED audio
                                                 # actually match the PROMPT? (cosine sim, matched
                                                 # vs mismatched-prompt GAP, same logic as chroma_gap)
                                                 # NOT laion/larger_clap_music: that's an HTSAT-BASE
                                                 # checkpoint, and LAION-AI/CLAP#126 documents a
                                                 # confirmed, unresolved ~45-point R@1 regression
                                                 # from HTSAT-base -> HuggingFace conversion (verified
                                                 # independently here: matched vs mismatched caption
                                                 # similarity was statistically identical, traced to
                                                 # exactly-zero biases in the checkpoint's projection
                                                 # heads). htsat-unfused is HTSAT-TINY, ~3% conversion
                                                 # loss per the same issue -- verify with the same
                                                 # diagnostic before trusting it in a real run.

# --- DiT training ---
DIT_BATCH_SIZE   = 512   # gradient checkpointing (src/dit.py) cut memory enough that 256 only
                         # used ~20GB of the 80GB GPU -- pushing up to use the available budget
                         # (target ~60GB, leaving margin for eval-step spikes: sampling + VAE
                         # decode + CLAP scoring run above steady-state training memory). Estimated
                         # from one data point (256 -> ~20GB); watch nvidia-smi on the next run and
                         # adjust if it lands far from ~60GB or OOMs.
DIT_CKPT_GROUP_SIZE = 3   # gradient-checkpoint 3 DiT blocks per segment (4 recompute segments
                         # instead of 12, DIT_LAYERS=12 divides evenly): GPU sat at 100%
                         # utilization (compute-bound, not a data-loading stall) with per-block
                         # checkpointing at 53GB/80GB used -- real memory headroom to trade back
                         # for less recompute overhead. Bumped 2->3 for more speed; watch
                         # nvidia-smi memory.used doesn't get too close to 80GB.
DIT_LR           = 3e-4   # was 6e-4, tuned for the old 39.1M/512-dim/8-layer/no-cross-attn model.
                         # The 158.6M/768-dim/12-layer + cross-attention model destabilized ~80
                         # steps after DIT_WARMUP=1000 completed and LR hit peak -- a classic
                         # "peak LR too high for this depth/width" signature (more layers + an
                         # extra attention op per block generally need a LOWER peak LR, not the
                         # same one, without a formal scaling recipe like muP). Halved as a
                         # conservative first correction; may still need tuning.
DIT_LR_MIN_RATIO = 0.05    # cosine decay floor, as a fraction of DIT_LR
DIT_WARMUP       = 1_000

# --- Fine-tuning (--finetune) ---
# The pretrained backbone must not be hammered at the from-scratch peak LR, but the
# new zero-init feature modules have to GROW from nothing — so they get their own,
# higher LR via a second param group.
# These were 1e-4 / 2e-4, which were STABLE for the chroma-only fine-tune but destroyed the run
# once texture was added. The evidence is unambiguous: with the resume warmup ramping the LR, the
# model was healthy (loss 0.679, chroma GAP 98% of ceiling — the best ever measured); it diverged
# within 200 steps of the warmup ending and the LR reaching full value. Texture growing fast from
# zero-init, plus chroma dropout 0.3 -> 0.0 (conditioning now in 80% of batches, not 56%), pushes
# more signal into the token stream than the pretrained backbone can absorb at that rate.
DIT_FINETUNE_LR      = 3e-5      # pretrained backbone — nudge it, do not retrain it
DIT_FINETUNE_FEAT_LR = 5e-5      # new zero-init feature modules (was 6e-4 -> diverged,
                                 # then 2e-4 -> stable for chroma alone but not with texture)
# Cosine decays over this, not DIT_TOTAL_STEPS. Raised 30k -> 60k so that RESUMING the finished
# 30k run lands mid-schedule (53.8% of peak: backbone 1.6e-5, feats 2.7e-5) instead of on the 5%
# floor, where training would be effectively frozen. Resume, not --finetune: see the guard in
# train_dit.py — --finetune from an already-converged checkpoint builds a FRESH Adam, and that
# blew up the 53.6k-track run (loss 0.70 -> 1.21, tracking the warmup ramp exactly).
DIT_FINETUNE_STEPS   = 60_000
DIT_FT_CKPT_DIR      = "checkpoints/dit_ft"   # SEPARATE dir: fine-tune steps restart at 0, so
                                 # dit_step0005000.pt sorts BEFORE dit_step0115000.pt and the
                                 # "keep last N" prune silently deleted every fine-tune checkpoint.
DIT_FT_CKPT_EVERY    = 1_000     # save often so a divergence (or an accidental kill) can't erase
                                 # much progress. Aligned with DIT_EVAL_EVERY so every checkpoint
                                 # has a matching held-out eval row in the CSV.
DIT_FT_KEEP_LAST     = 10        # 628 MB each -> ~6.3 GB. The default 3 was NOT passed on the
                                 # fine-tune save path, so it silently kept only 3; at 1k-step
                                 # saves that would be a 3k-step window to recover from.

# Skip an optimizer step whose loss is a wild outlier vs the running average. A single bad
# batch was enough to blow the last run up; this bounds the damage instead of letting one
# spike destroy 6k steps of progress.
DIT_LOSS_SPIKE_FACTOR = 2.0   # was 3.0 — too loose: batches at ~1.4x the average slipped
                              # through and poisoned the model over ~100 steps without ever
                              # tripping a 3x guard.

# Re-warm the LR whenever the optimizer state is DISCARDED (e.g. resuming into a model with a
# new conditioning module: the param count changes, so Adam's moments cannot be restored).
# A fresh Adam has exp_avg_sq = 0, which makes its first updates effectively lr * sign(grad) —
# a full-LR step on EVERY parameter. Dropping that onto an already-trained model at 93% of peak
# LR destroyed a run: loss slid 0.68 -> 1.96 (the predict-zero baseline) over ~600 steps and the
# model abandoned chroma entirely (GAP 0.493 -> 0.010).
DIT_RESUME_WARMUP = 1_000

# The spike guard only catches a single outlier BATCH. It cannot see a slow slide, which is how
# that collapse actually happened (0.68 -> 0.71 -> 1.38, never a 3x jump). Abort if the running
# loss stays this far above its best for a sustained stretch, instead of burning hours on a
# model that is already dead.
DIT_DIVERGE_FACTOR = 1.35
DIT_EMA_DECAY    = 0.999
DIT_CFG_DROPOUT  = 0.2
DIT_TOTAL_STEPS  = 400_000
DIT_LOG_EVERY    = 100
DIT_EVAL_EVERY   = 1_000   # sample -> decode -> SNR vs the real clip (and the VAE ceiling)
DIT_EVAL_CLIPS   = 16      # 4 was far too few: the measured VAE ceiling swung 7.93 -> 4.26 dB
                           # between runs purely from which clips got drawn.
DIT_EVAL_SEED    = 1234    # fixed clip selection, so the ceiling and SNR are comparable ACROSS runs

# Evaluate on HELD-OUT tracks as well as training ones.
# Every metric reported up to now was train-set-only, and not by accident: SNREvaluator skips any
# clip with `valid < 0.5` (i.e. no feature file), and ONLY the 10,000 training tracks had feature
# files — so the eval set was a strict SUBSET of the training set, 16/16, verified. At ~384*2600
# samples over 10k tracks (~100 epochs) "GAP 98% of ceiling" is a memorization score, not a
# generalization one. The train-vs-val gap below is the number that tells us if this actually works.
DIT_EVAL_HELDOUT = True
DIT_CKPT_EVERY   = 5_000
DIT_KEEP_LAST    = 3
DIT_NUM_WORKERS  = 8
DIT_CKPT_DIR     = "checkpoints/dit"
DIT_STATS_CSV    = "dit_stats.csv"
DIT_SAMPLE_DIR   = "samples/dit"

# --- DiT timestep sampling ---
# Uniform t spends most of its budget near t=0/1, where predicting the velocity is
# nearly trivial. Logit-normal concentrates on the hard middle of the trajectory.
DIT_LOGIT_NORMAL_T = True
DIT_T_SIGMA        = 1.0     # std of the pre-sigmoid normal; 1.0 -> moderate concentration

# --- Text conditioning: THE PRIMARY signal in the text-primary redesign ---
# Audio-derived captions (caption_audio.py) -> frozen T5 (compute_text_embeddings.py) -> DiT
# cross-attention (src/dit.py CrossAttention, one per block, zero-init gated). Chroma/texture
# below are now the SECONDARY, opt-in reference-track path, not the primary conditioning —
# they stay because reference tracks remain a supported input, just no longer the main one.
DIT_USE_TEXT = True

# --- Conditioning features (extract_features.py) ---
# The 195-way tags carry only ~30-40 bits, while a 30s latent needs orders of
# magnitude more to be determined. These add musical content bits back.
#   global : tempo (BPM) + key + mode           -> summed into the adaLN cond vector
#   chroma : per-frame 12-bin chroma + energy   -> added to the input tokens
# Chroma is the high-bit signal but changes the inference interface (needs a
# reference/melody at sample time), so it is toggleable and ablatable.
DIT_USE_GLOBAL_FEATS = True
DIT_USE_CHROMA       = True

# Per-frame TEXTURE conditioning: onset strength, percussive ratio (HPSS), spectral
# centroid, spectral-flatness envelope. Chroma says WHICH NOTES to play but nothing about
# how noisy/percussive a frame is, so generated audio came out measurably TOO TONAL
# (median spectral flatness 0.78x the VAE's own output = missing transients/cymbals/air).
# These four channels tell the model how much noise-like content each frame needs.
#
# STATUS: NOT yet supported by evidence. The gate is live (-0.056, ~5% of the token norm) but
# FLAT moved 0.77x -> 0.70x, i.e. AWAY from the 1.00x target, and FLAT sits at 0.70-0.79 across
# every CFG scale from 1.0 to 3.0. The plan's own pre-registered falsification criterion was
# "if FLAT stays ~0.78x once texture is actually being used, the limiter is capacity, not a
# conditioning blind spot". The gate is still small, so this is "unsupported", not "refuted" —
# the held-out eval is what will settle it.
DIT_USE_TEXTURE      = True
DIT_TEXTURE_IN       = 4

# Chroma+texture dropout, INDEPENDENT of the CFG mask.
#
# 0.3 was believed to be "the only value ever observed to be stable", on the grounds that it was
# the one difference from a run that survived 6,000 steps. That was wrong: reverting to 0.3 and
# re-running diverged at step 8600 anyway. Chroma dropout is EXONERATED — as are, by direct
# measurement at the last healthy checkpoint (step 8000), the Adam state (max |m|/sqrt(v) = 2.4,
# perfectly healthy), gradient clipping (global grad norm 0.12, the 1.0 clip never engages),
# texture (bounded, 5% of the token norm), the LR (decreasing at the failure point) and the data
# (all 10k training latents verified deterministic-crop and feature-aligned).
# The cause is still UNKNOWN. Every run so far reset Adam mid-schedule via --resume into a
# checkpoint with a changed param count; --finetune never does. That is the remaining untested
# difference, and the restart below takes that path.
DIT_CHROMA_DROPOUT   = 0.3
DIT_FEAT_DIM         = 64     # embedding width for tempo/key/mode before projection to D
DIT_CHROMA_BINS      = 12
DIT_CHROMA_IN        = 13     # 12 chroma bins + 1 RMS energy channel
DIT_TEMPO_MIN        = 40.0
DIT_TEMPO_MAX        = 220.0

# Standardize chroma/texture channels with dataset-level mean/std (compute_feature_stats.py),
# exactly as latents already are. This is the fix for the DC-bias defect: the raw channels have
# large non-zero means (chroma 0.399, percussive 0.503, flatness 0.243), so `proj(c)` decomposes
# into W*mean + W*(c - mean) and the first term is a CONSTANT vector added identically to all 645
# tokens. Measured at step 8000: of the 5.35 per-token chroma injection, 4.48 was that constant
# and only 2.93 was actual per-frame harmony — i.e. ~70% of the conditioning bandwidth was
# spent on a bias the model could have learned for free. Zero-mean inputs make W*mean vanish.
DIT_STANDARDIZE_FEATS = True

# Feature schema version. extract_features.py re-extracts any file whose stored version differs,
# so a change to the chroma/texture definition can never be silently skipped by an "already
# exists" check. Bump this whenever the meaning of a channel changes.
#   1 = chroma_stft(norm=inf)  <- every frame rescaled to max=1: SILENCE LOOKED LIKE A CHORD
#   2 = chroma_stft(norm=None), log-compressed against a per-clip 99th-percentile scale
FEATURE_VERSION      = 2

# --- Inference ---
EULER_STEPS = 50
# "euler" (1st order) or "heun" (2nd order: predict, then correct with the velocity at the
# predicted endpoint). Heun costs 2x forward passes per step but halves integration error,
# so it is strictly better at the same total cost when you halve the step count.
ODE_SOLVER  = "heun"
# 4.0 was tuned when conditioning was weak tags. Chroma carries far more signal, so 4x
# guidance now OVER-drives: latent std 1.62 (vs real 1.00) and chroma gap >100% of ceiling,
# i.e. the model exaggerates the requested harmony. Measured sweep on dit_ft step 6000:
#   cfg 1.0 -> gap 87%, std 0.84     cfg 2.0 -> gap 100%, std 1.17
#   cfg 1.5 -> gap 97%, std 1.01     cfg 4.0 -> gap 103%, std 1.62
CFG_SCALE   = 1.5
