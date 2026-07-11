# --- Paths ---
STEMS_DIR         = "data/stems"
LATENTS_DIR       = "data/latents"
TSV_PATH          = "mtg-jamendo-dataset/data/autotagging.tsv"
VOCAB_PATH        = "data/tag_vocab.json"
LATENT_STATS_PATH = "data/latent_stats.json"   # per-channel mean/std, see compute_latent_stats.py

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
DIT_D_MODEL      = 512
DIT_HEADS        = 8
DIT_LAYERS       = 8
DIT_TAG_DIM      = 64     # per-tag embedding size (before projection to D)
DIT_T_EMBED_DIM  = 256    # sinusoidal timestep embedding hidden size
DIT_VOCAB_SIZE   = 195    # MTG-Jamendo tag vocabulary size

# --- DiT training ---
DIT_BATCH_SIZE   = 384
DIT_LR           = 6e-4
DIT_LR_MIN_RATIO = 0.05    # cosine decay floor, as a fraction of DIT_LR
DIT_WARMUP       = 1_000
DIT_EMA_DECAY    = 0.999
DIT_CFG_DROPOUT  = 0.2
DIT_TOTAL_STEPS  = 400_000
DIT_LOG_EVERY    = 100
DIT_CKPT_EVERY   = 5_000
DIT_KEEP_LAST    = 3
DIT_NUM_WORKERS  = 8
DIT_CKPT_DIR     = "checkpoints/dit"
DIT_STATS_CSV    = "dit_stats.csv"
DIT_SAMPLE_DIR   = "samples/dit"

# --- Inference ---
EULER_STEPS = 50
CFG_SCALE   = 4.0
