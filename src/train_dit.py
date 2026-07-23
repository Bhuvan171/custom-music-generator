"""
DiT training loop — flow matching on cached, per-channel-normalized VAE latents.

Prerequisites:
  1. src/cache_latents.py   — populates config.LATENTS_DIR with raw *.pt latents
  2. compute_latent_stats.py — writes config.LATENT_STATS_PATH (per-channel mean/std)

Targets are normalized to ~unit variance per channel before flow matching so the
z1 (data) distribution matches the z0 ~ N(0,1) noise prior scale-for-scale, and so
MSE loss doesn't over-weight the highest-variance latent channels. Generated
latents are in normalized space — denormalize (z * std + mean) before vae.decode().

Usage:
  python src/train_dit.py
  python src/train_dit.py --resume checkpoints/dit/dit_step0010000.pt
"""

import argparse
import copy
import csv
import glob
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.dataset import JamendoDataset, collate_fn, _track_id_from_path
from src.dit import MusicDiT
from src.vae import WaveformVAE


# ── Helpers ──────────────────────────────────────────────────────────────────

def ema_update(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.lerp_(p, 1.0 - decay)


def snr_db(ref, x):
    return (10.0 * torch.log10(ref.pow(2).mean() / ((ref - x).pow(2).mean() + 1e-12))).item()


def spectral_flatness(audio_batch):
    """Per-clip mean spectral flatness (Wiener entropy), 0..1. Returns one value per clip."""
    import librosa
    import numpy as np
    return np.array([
        float(librosa.feature.spectral_flatness(
            y=audio_batch[i, 0].detach().cpu().float().numpy(), hop_length=1024).mean())
        for i in range(audio_batch.shape[0])
    ])


def flatness_stats(gen_audio, ref_flatness):
    """
    Compare generated flatness against the VAE reconstruction's, PER CLIP.

    Reported as the MEDIAN of per-clip ratios, not a ratio of means. Per-clip flatness spans
    a ~31x range across clips, so a ratio of means is dominated by whichever clip is loudest
    in this metric — two outlier clips alone were producing a fake "1.95x" signal when the
    median was 0.78x.

    Deviation from 1.0 is bad in BOTH directions, and they are different defects:
      > 1  broadband noise / hiss (e.g. overlap-add phase incoherence)
      < 1  TOO tonal — missing the noise-like texture, transients and air of real music,
           i.e. dull and smeared
    `n_noisy` counts clips that blew up into genuine noise (>3x), which is a distinct,
    occasional failure the median deliberately hides.
    """
    import numpy as np
    ratios = spectral_flatness(gen_audio) / np.maximum(ref_flatness, 1e-9)
    return float(np.median(ratios)), int((ratios > 3.0).sum())


def chroma_adherence(audio_batch, target_chroma):
    """
    THE metric for chroma conditioning: does the generated audio actually play the notes
    we asked for? Extract chroma from the generated waveform and cosine-compare it, frame
    by frame, against the chroma we conditioned on.

    Waveform SNR CANNOT measure this. Chroma is deliberately octave-, timbre- and
    phase-invariant (that is exactly why it leaves the model free to choose instrumentation),
    while SNR is dominated by octave, timbre and phase. A model can follow the harmony
    perfectly and still score terrible SNR. This measures the thing chroma actually controls.

    1.0 = plays exactly the requested harmony. ~0.5-0.6 = what you get by chance between
    two unrelated music clips, so only a clear rise above that baseline means anything.

    `target_chroma` must be RAW (un-standardized) chroma, in the same space chroma_from_audio
    returns. Passing the standardized tensor the dataset hands the model compares a non-negative
    vector against a zero-mean one: the measured ceiling collapsed +0.512 -> +0.313 from exactly
    that mismatch. SNREvaluator keeps `chroma_raw` for this reason.
    """
    import numpy as np
    from src.features import chroma_from_audio
    sims = []
    for i in range(audio_batch.shape[0]):
        y = audio_batch[i, 0].detach().cpu().float().numpy()
        # Same recipe as the conditioning signal — see src/features.py.
        gen = torch.from_numpy(chroma_from_audio(y, config.SAMPLE_RATE, config.VAE_LATENT_LEN)).float()
        tgt = target_chroma[i, :12].detach().cpu().float()     # (12, L) — drop the rms row
        sims.append(F.cosine_similarity(gen, tgt, dim=0).mean().item())
    return float(np.mean(sims))


class ClapScorer:
    """
    THE metric for text-primary conditioning: does the generated audio actually match the
    PROMPT? This is the metric that was missing for the entire tags/chroma era of this project —
    every prior metric (chroma GAP, flatness, SNR) measured internal consistency, never "does
    this sound like what was asked for" in the sense a listener would judge.

    Uses laion/larger_clap_music (frozen, NOT the same model as the T5 text encoder used for
    DiT conditioning — CLAP's audio tower embeds the generated WAVEFORM, its text tower embeds
    the CAPTION STRING, and cosine similarity between the two is the score). Reported as a GAP
    (matched caption vs a shuffled/mismatched one), exactly the chroma_adherence pattern: raw
    similarity is confounded (generic-sounding audio can score deceptively high against any
    caption), the gap cancels that — a model ignoring text should score ~0.
    """
    def __init__(self, device):
        from transformers import ClapModel, ClapProcessor
        self.device = device
        self.model = ClapModel.from_pretrained(config.CLAP_MODEL_NAME).to(device)
        self.model.eval(); self.model.requires_grad_(False)
        self.processor = ClapProcessor.from_pretrained(config.CLAP_MODEL_NAME)

    @torch.no_grad()
    def text_embed(self, captions):
        inputs = self.processor(text=captions, return_tensors="pt", padding=True).to(self.device)
        return F.normalize(self.model.get_text_features(**inputs).pooler_output, dim=-1)   # (B, D)

    @torch.no_grad()
    def audio_embed(self, audio_batch, sr):
        # CLAP's own feature extractor expects 48kHz mono; resample from this project's 22050Hz.
        import torchaudio
        wav = audio_batch[:, 0].float().cpu()                             # (B, T) @ sr
        wav48 = torchaudio.functional.resample(wav, sr, 48_000)
        inputs = self.processor(audio=list(wav48.numpy()), sampling_rate=48_000,
                                return_tensors="pt").to(self.device)
        return F.normalize(self.model.get_audio_features(**inputs).pooler_output, dim=-1)  # (B, D)

    def gap(self, audio_batch, sr, captions, shuf_idx):
        """matched cosine sim - mismatched (shuffled captions) cosine sim, mean over the batch."""
        ae = self.audio_embed(audio_batch, sr)
        te = self.text_embed(captions)
        matched    = (ae * te).sum(-1)
        mismatched = (ae * te[shuf_idx]).sum(-1)
        return (matched - mismatched).mean().item()


class SNREvaluator:
    """
    Periodic end-to-end progress check: sample latents from noise (EMA weights, CFG),
    decode through the frozen VAE, and measure SNR against the REAL clip.

    Note this metric only became meaningful once chroma conditioning existed. With
    tag-only conditioning the model is not trying to reproduce any particular clip,
    so SNR-vs-a-real-clip was noise. With per-frame chroma the content IS pinned, so
    SNR is a real signal — and the VAE reconstruction SNR is the ceiling it chases.

    `features_dir` selects WHICH tracks get evaluated, and it is the whole ballgame.
    Clips are skipped unless `valid >= 0.5`, i.e. unless they have a feature file in this
    directory — so pointing at config.FEATURES_VAL_DIR yields an eval set of tracks that
    have no cached latent and therefore CANNOT be in training. Note this evaluator encodes
    the FLAC fresh through the VAE and never reads a cached latent, which is why a held-out
    eval set needs feature files only.

    Every metric reported before this existed was train-set-only: the sole feature directory
    was the training one, so the eval set was a strict subset of the training set, 16/16
    verified, after ~100 epochs over 10k tracks.
    """
    def __init__(self, vae_ckpt, device, features_dir=None, text_dir=None, captions_dir=None,
                label="train", vae=None, clap=None):
        self.device = device
        self.label  = label
        # Both evaluators want the same frozen VAE (and, if used, the same CLAP scorer); pass
        # them in rather than loading twice.
        if vae is not None:
            self.vae = vae
        else:
            self.vae = WaveformVAE().to(device)
            self.vae.load_state_dict(
                torch.load(vae_ckpt, map_location=device, weights_only=False)["vae"])
            self.vae.eval(); self.vae.requires_grad_(False)
        self.clap = clap

        with open(config.LATENT_STATS_PATH) as f:
            st = json.load(f)
        self.lmean = torch.tensor(st["mean"], device=device).view(1, -1, 1)
        self.lstd  = torch.tensor(st["std"],  device=device).view(1, -1, 1)

        self.use_feats = config.DIT_USE_GLOBAL_FEATS or config.DIT_USE_CHROMA or config.DIT_USE_TEXTURE
        self.use_text  = config.DIT_USE_TEXT
        captions_dir   = captions_dir or config.CAPTIONS_DIR
        # Fixed eval clips, deterministic crop so the cached features describe this audio.
        # The generator is SEEDED: without it, shuffle=True drew a different set of clips every
        # run and the reported "VAE ceiling" moved with them (7.93 dB one run, 4.26 dB the next),
        # making runs incomparable. Same seed -> same clips -> same ceiling, every time.
        eds = JamendoDataset(config.STEMS_DIR, config.TSV_PATH, config.VOCAB_PATH,
                             deterministic_crop=True, load_features=self.use_feats,
                             features_dir=features_dir, load_text=self.use_text, text_dir=text_dir)
        # Index the dataset directly over a seeded permutation rather than going through a
        # DataLoader. Same clip selection (RandomSampler draws exactly this randperm), but it
        # also tells us WHICH tracks were picked — without that, train/val contamination is
        # invisible, which is how the eval set silently became 100% training data.
        g = torch.Generator()
        g.manual_seed(config.DIT_EVAL_SEED)
        clips, self.tags, fl, self.track_ids, self.captions = [], [], [], [], []
        for idx in torch.randperm(len(eds), generator=g).tolist():
            # Pre-filter on metadata BEFORE touching the audio. eds[idx] decodes a 30s FLAC, and
            # the val set is ~2k of 55.6k tracks, so a load-then-check loop would decode ~445
            # clips to find 16. Deciding from the path alone costs a stat() instead.
            tid = _track_id_from_path(eds.files[idx])
            if not eds.track_tags.get(tid):
                continue
            if self.use_feats and not os.path.exists(os.path.join(eds.features_dir, f"{tid}.pt")):
                continue
            caption_path = os.path.join(captions_dir, f"{tid}.json")
            if self.use_text and not os.path.exists(caption_path):
                continue
            item = eds[idx]
            if self.use_feats or self.use_text:
                fl.append({k: v.unsqueeze(0) for k, v in item[2].items()})
            if self.use_text:
                with open(caption_path) as cf:
                    self.captions.append(json.load(cf)["caption"])
            clips.append(item[0].unsqueeze(0))
            self.tags.append(item[1])
            self.track_ids.append(tid)
            if len(clips) == config.DIT_EVAL_CLIPS:
                break
        if len(clips) < config.DIT_EVAL_CLIPS:
            raise SystemExit(
                f"[{label}] only found {len(clips)} of {config.DIT_EVAL_CLIPS} eval clips with "
                f"features in {features_dir or config.FEATURES_DIR} and captions in "
                f"{captions_dir}.\n"
                f"For the held-out set run:\n"
                f"  python extract_features.py --start 10000 --limit 2000 "
                f"--out-dir {config.FEATURES_VAL_DIR}\n"
                f"  python caption_audio.py --start 10000 --limit 2000 "
                f"--out-dir {config.CAPTIONS_VAL_DIR}\n"
                f"  python compute_text_embeddings.py --captions-dir {config.CAPTIONS_VAL_DIR} "
                f"--out-dir {config.TEXT_EMB_VAL_DIR}"
            )
        self.audio = torch.cat(clips, dim=0).to(device)
        self.feats = None
        self.chroma_raw = None
        if self.use_feats or self.use_text:
            self.feats = {k: torch.cat([f[k] for f in fl], dim=0).to(device) for k in fl[0]}
        if self.use_feats:
            # The model gets STANDARDIZED chroma; chroma_adherence must score against the RAW
            # signal, in the same space chroma_from_audio returns. Undo the dataset's transform
            # rather than re-reading the files, so the two can never disagree.
            self.chroma_raw = self.feats["chroma"]
            if eds.feat_stats is not None:
                cm, cs = eds.feat_stats["chroma"]
                self.chroma_raw = self.feats["chroma"] * cs.to(device) + cm.to(device)

        # The ceiling: decode the GROUND-TRUTH latent. The DiT can never beat this.
        with torch.no_grad():
            mu, _ = self.vae.encode(self.audio)
            recon = self.vae.decode(mu.float()).float()
            self.ceiling = snr_db(self.audio.float(), recon)

        # References the generated audio is scored against.
        # real_std : if generated latents have LESS variance than real, the model is
        #            regressing to the mean of the conditional distribution — the direct
        #            cause of smearing. std_ratio -> 1.0 is the target.
        # flatness : the VAE recon is the fair reference (the DiT must decode through it),
        #            so flatness ABOVE this is mud the DiT itself is adding.
        self.real_std      = mu.float().std().item()
        self.real_flatness = spectral_flatness(recon)

        self.chroma_ceiling = float("nan")
        if self.use_feats and config.DIT_USE_CHROMA:
            # Fixed derangement (every clip gets SOMEONE ELSE'S chroma) used as the mismatched
            # control on every eval, so the reported gap is comparable step to step.
            n = self.chroma_raw.shape[0]
            self.shuf_idx = torch.roll(torch.arange(n), 1)
            # Ceiling GAP: what the VAE reconstruction of the real audio scores. No model that
            # decodes through this VAE can exceed it.
            self.chroma_ceiling = (chroma_adherence(recon, self.chroma_raw)
                                   - chroma_adherence(recon, self.chroma_raw[self.shuf_idx]))

        self.clap_ceiling = float("nan")
        if self.use_text and self.clap is not None:
            n = len(self.captions)
            self.clap_shuf_idx = torch.roll(torch.arange(n), 1)
            # Ceiling: what the VAE reconstruction of the real audio scores against its OWN
            # (matched) caption vs a shuffled one. No generated model can exceed this either —
            # it bounds both the VAE's fidelity AND the fact that captions themselves are
            # imperfect descriptions (an audio-LM caption is not a lossless description).
            self.clap_ceiling = self.clap.gap(recon, config.SAMPLE_RATE, self.captions,
                                              self.clap_shuf_idx)

    @torch.no_grad()
    def __call__(self, ema_model):
        was_training = ema_model.training
        ema_model.eval()
        # Same starting noise every eval, so a change in SNR reflects the MODEL improving
        # rather than a luckier noise draw. Restore RNG afterwards so training is unaffected.
        rng = torch.cuda.get_rng_state(self.device)
        torch.cuda.manual_seed(config.DIT_EVAL_SEED)
        z = ema_model.sample(self.tags, steps=config.EULER_STEPS,
                             cfg_scale=config.CFG_SCALE, device=self.device, feats=self.feats)
        torch.cuda.set_rng_state(rng, self.device)
        z = z.float() * self.lstd + self.lmean
        gen = self.vae.decode(z).float()
        if was_training:
            ema_model.train()

        m = {}

        # 0. CLAP GAP — THE metric for the text-primary redesign: does the generated audio
        # actually match the PROMPT it was given? Same matched-minus-mismatched logic as chroma
        # GAP below, but this is the only metric in this whole file that measures "does this
        # sound like what was asked for" the way a listener would judge, rather than internal
        # consistency (chroma-following, flatness-vs-VAE, etc.).
        m["clap_gap"] = float("nan")
        if self.use_text and self.clap is not None:
            m["clap_gap"] = self.clap.gap(gen, config.SAMPLE_RATE, self.captions, self.clap_shuf_idx)

        # 1. CHROMA GAP — is it following the harmony it was given?
        # The GAP (matched - mismatched), not raw adherence. Raw adherence is confounded: a
        # flat/smeared signal scores ~0.80 against ANY target, because chroma is non-negative
        # and a flat vector has high cosine similarity with everything. The gap cancels that
        # bias — a model that ignores chroma scores exactly 0.000 (verified empirically).
        m["chroma_gap"] = float("nan")
        if self.use_feats and config.DIT_USE_CHROMA:
            m["chroma_gap"] = (chroma_adherence(gen, self.chroma_raw)
                               - chroma_adherence(gen, self.chroma_raw[self.shuf_idx]))

        # 2. STD RATIO — is it regressing to the mean? <1 means shrunken/averaged latents,
        #    which is exactly what produces smeared audio. Target 1.0.
        m["std_ratio"] = z.std().item() / self.real_std

        # 3. FLATNESS RATIO — the direct "muddiness" number. >1 means the generated spectrum
        #    is smearier/noisier than what the VAE itself produces. Target 1.0.
        m["flatness_ratio"], m["n_noisy"] = flatness_stats(gen, self.real_flatness)

        # 4. SNR — retained only for the CSV. It is NOT a quality signal here: the same song
        #    shifted 10ms scores -0.79 dB and silence scores 0.00 dB, so generation (a
        #    different waveform by design) is punished no matter how good it sounds.
        m["snr_db"] = snr_db(self.audio.float(), gen)
        return m


def save_checkpoint(step, model, ema_model, opt, scheduler, ckpt_dir=None, keep_last=None):
    # ckpt_dir is separate for fine-tuning. A fine-tune restarts the step counter at 0, so its
    # checkpoints (dit_step0005000.pt) sort BEFORE the pretrained ones (dit_step0115000.pt) and
    # the sorted()[:-keep_last] prune deleted the NEW checkpoints instead of the old ones.
    ckpt_dir  = ckpt_dir  or config.DIT_CKPT_DIR
    keep_last = keep_last or config.DIT_KEEP_LAST
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"dit_step{step:07d}.pt")
    torch.save({
        "step":      step,
        "dit":       model.state_dict(),
        "ema":       ema_model.state_dict(),
        "opt":       opt.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "dit_step*.pt")))
    for old in ckpts[:-keep_last]:
        os.remove(old)
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Path to DiT checkpoint (full resume: weights+opt+step)")
    parser.add_argument("--finetune", default=None,
                        help="Load weights from a checkpoint but restart the step/optimizer/schedule. "
                             "Use when the conditioning changed (new feature modules) — those modules "
                             "are missing from the old checkpoint and start from their init.")
    parser.add_argument("--vae-ckpt", default=None,
                        help="VAE checkpoint for eval-time decoding. Defaults to the newest "
                             f"{config.FT_DEC_CKPT_DIR}/dec_step*.pt (the retrained iSTFT-head "
                             "decoder); falls back to checkpoints/vae/vae_step0200000.pt only if "
                             "no decoder-ft checkpoint exists yet, which will fail to load under "
                             "the current architecture unless VAE_DEC_USE_ISTFT_HEAD is False.")
    args = parser.parse_args()

    device = torch.device("cuda")

    use_feats = config.DIT_USE_GLOBAL_FEATS or config.DIT_USE_CHROMA or config.DIT_USE_TEXTURE
    use_text  = config.DIT_USE_TEXT   # THE primary conditioning signal in the text-redesign

    # Dataset: pre-cached latent .pt files from config.LATENTS_DIR
    ds = JamendoDataset(config.LATENTS_DIR, config.TSV_PATH, config.VOCAB_PATH,
                        normalize_latents=True, load_features=use_feats,
                        load_text=use_text, text_dir=config.TEXT_EMB_DIR)
    if not any(f.endswith(".pt") for f in ds.files):
        print("ERROR: No .pt latent files found in", config.LATENTS_DIR)
        print("Run: python src/cache_latents.py --checkpoint <vae_checkpoint>")
        sys.exit(1)
    # Keep only .pt files (latents, not raw audio)
    ds.files = [f for f in ds.files if f.endswith(".pt")]

    if use_feats:
        # ALIGNMENT GUARANTEE. Latents cached before the --deterministic switch came from
        # RANDOM 30s crops, while features describe the FIRST 30s. Training on such a pair
        # feeds the model chroma from a different slice of audio than the latent it
        # conditions — an actively wrong signal, not just a weak one. Every track that has
        # a feature file has been re-cached deterministically, so restricting to those
        # makes the mismatch impossible (and drops the stale random-crop latents).
        from src.dataset import _track_id_from_path
        have = {f[:-3] for f in os.listdir(config.FEATURES_DIR) if f.endswith(".pt")}
        before = len(ds.files)
        ds.files = [f for f in ds.files if _track_id_from_path(f) in have]
        print(f"DiT dataset: {len(ds.files)} latent files "
              f"(filtered from {before} to those with aligned features)")
        if not ds.files:
            print("ERROR: no latents have matching features. Run extract_features.py and "
                  "cache_latents.py --deterministic --limit N with the SAME N.")
            sys.exit(1)
    else:
        print(f"DiT dataset: {len(ds.files)} latent files")

    dl = DataLoader(
        ds,
        batch_size=config.DIT_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.DIT_NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model     = MusicDiT().to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    # Resuming a checkpoint that lives in the fine-tune dir must keep the fine-tune setup:
    # the same two param groups (or the saved optimizer state will not load), the fine-tune
    # LRs, the 30k schedule, and the fine-tune checkpoint dir. Without this, --resume would
    # silently revert to backbone-at-6e-4 on a 400k schedule and save into checkpoints/dit/.
    is_ft = bool(args.finetune) or (args.resume and config.DIT_FT_CKPT_DIR in args.resume)

    # AdamW with default betas — flow matching is not adversarial, no GAN instability
    if is_ft:
        # Two groups: the pretrained backbone is nudged gently (hammering it at the
        # from-scratch peak LR would undo 115k steps of training), while the zero-init
        # feature modules need the full LR to grow from nothing.
        feat_params = [p for n, p in model.named_parameters() if n.startswith("feat_embed.")]
        base_params = [p for n, p in model.named_parameters() if not n.startswith("feat_embed.")]
        opt = torch.optim.AdamW(
            [{"params": base_params, "lr": config.DIT_FINETUNE_LR},
             {"params": feat_params, "lr": config.DIT_FINETUNE_FEAT_LR}],
            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
        )
        total_steps = config.DIT_FINETUNE_STEPS
        ckpt_dir, ckpt_every = config.DIT_FT_CKPT_DIR, config.DIT_FT_CKPT_EVERY
        keep_last = config.DIT_FT_KEEP_LAST
        print(f"Fine-tune LRs: backbone {config.DIT_FINETUNE_LR:.1e} "
              f"({len(base_params)} tensors) | new feature modules {config.DIT_FINETUNE_FEAT_LR:.1e} "
              f"({len(feat_params)} tensors) | {total_steps} steps")
        print(f"Checkpoints -> {ckpt_dir}/ every {ckpt_every} steps")
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=config.DIT_LR,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
        )
        total_steps = config.DIT_TOTAL_STEPS
        ckpt_dir, ckpt_every = config.DIT_CKPT_DIR, config.DIT_CKPT_EVERY
        keep_last = config.DIT_KEEP_LAST

    # Set when the optimizer state has to be thrown away (param count changed). Everything
    # after this step gets an extra warmup ramp, because a fresh Adam's first updates are
    # effectively lr * sign(grad) on every parameter at once — lethal to a trained model.
    reset_at = {"step": None}

    def lr_lambda(s):
        if s < config.DIT_WARMUP:
            base = s / max(1, config.DIT_WARMUP)
        else:
            progress = (s - config.DIT_WARMUP) / max(1, total_steps - config.DIT_WARMUP)
            cosine   = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            base = config.DIT_LR_MIN_RATIO + (1 - config.DIT_LR_MIN_RATIO) * cosine
        if reset_at["step"] is not None:
            rewarm = (s - reset_at["step"]) / max(1, config.DIT_RESUME_WARMUP)
            base *= min(1.0, max(0.0, rewarm))
        return base

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # strict=False: a checkpoint predating a conditioning module (e.g. texture) simply has
        # no weights for it. Those modules are zero-init, so they stay a no-op and the model
        # resumes behaving EXACTLY as the checkpoint did, then learns to use the new signal.
        missing, unexpected = model.load_state_dict(ckpt["dit"], strict=False)
        ema_model.load_state_dict(ckpt["ema"], strict=False)
        step = ckpt["step"]
        if missing:
            print(f"  newly initialized (absent from checkpoint): {len(missing)} tensors")
            for m in missing:
                print(f"    + {m}")

        # The optimizer state is keyed by PARAMETER COUNT per group. Adding a module changes
        # that count, so the saved state no longer matches and must be discarded — otherwise
        # load_state_dict raises. Weights are preserved; only Adam's moments restart.
        try:
            opt.load_state_dict(ckpt["opt"])
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"Resumed from step {step} (optimizer state restored)")
        except (ValueError, KeyError) as e:
            reset_at["step"] = step        # -> lr_lambda re-warms from here (see above)
            for _ in range(step):          # fast-forward the LR schedule to the right point
                scheduler.step()
            print(f"Resumed from step {step} — optimizer state DISCARDED "
                  f"({e.__class__.__name__}: param count changed). Weights kept; Adam moments "
                  f"restart, so the LR RE-WARMS over {config.DIT_RESUME_WARMUP} steps "
                  f"(skipping this destroyed the previous run).")
    elif args.finetune:
        # strict=False: the feature-conditioning modules are new and absent from the
        # old checkpoint. chroma_proj is zero-init, so the model starts out behaving
        # EXACTLY like the loaded checkpoint and then learns to exploit the new signal.
        ckpt = torch.load(args.finetune, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["dit"], strict=False)
        ema_model.load_state_dict(ckpt["ema"], strict=False)
        print(f"Fine-tuning from {args.finetune} (was step {ckpt['step']}), step counter reset to 0")
        print(f"  newly initialized (not in checkpoint): {len(missing)} tensors")
        for m in missing:
            print(f"    + {m}")
        if unexpected:
            print(f"  unexpected in checkpoint (ignored): {unexpected}")

        # GUARD: --finetune always builds a FRESH optimizer. At init Adam has exp_avg_sq = 0, so
        # its update is m/sqrt(v) = sign(grad) — EVERY parameter moves by exactly +/-lr regardless
        # of whether its gradient is 1e-8 or 1e2. On a model that is still learning this is
        # survivable; on a CONVERGED one it kicks it straight out of its minimum. beta2=0.999 means
        # the second moment needs ~1000 steps of history to damp this, which is why the damage
        # accrues over exactly the warmup window and then plateaus.
        # Observed: --finetune from the converged dit_ft/dit_step0030000.pt ran loss 0.70 -> 1.21
        # in lockstep with the LR ramp, while chroma GAP stayed at 75% — the model had not
        # forgotten anything, it was being shaken apart. --resume restores the moments and does not
        # have this failure mode.
        # A non-zero chroma_gate means this checkpoint has ALREADY been trained with conditioning,
        # i.e. it is converged for this task and --finetune is the wrong entry point.
        if use_feats and config.DIT_USE_CHROMA:
            g = model.feat_embed.chroma_gate.abs().item()
            if g > 1e-3:
                print(f"\n  !! WARNING: this checkpoint is ALREADY conditioning-trained "
                      f"(chroma_gate = {model.feat_embed.chroma_gate.item():+.3f}, not ~0).")
                print(f"  !! --finetune gives it a FRESH Adam, whose first updates are "
                      f"lr*sign(grad) on every parameter.")
                print(f"  !! That blew up the previous attempt (loss 0.70 -> 1.21). Prefer:")
                print(f"  !!     python src/train_dit.py --resume {args.finetune}")
                print(f"  !! (--resume restores Adam's moments; raise DIT_FINETUNE_STEPS so the "
                      f"cosine schedule is not already exhausted at that step.)\n")

    # Append only if the existing header MATCHES. Appending new columns under a stale header
    # silently shifts every value (SNR was being read as chroma_proj_norm), which made the CSV
    # actively misleading during a debug session.
    _fields = ["step", "loss", "loss_avg", "lr", "vram",
               "clap_gap", "clap_gap_ceiling",
               "chroma_gap", "chroma_gap_ceiling", "std_ratio", "flatness_ratio", "n_noisy",
               "snr_db",
               # held-out counterparts — the train-vs-val gap is the point of this whole run
               "val_clap_gap", "val_clap_gap_ceiling",
               "val_chroma_gap", "val_chroma_gap_ceiling", "val_std_ratio",
               "val_flatness_ratio", "val_n_noisy", "val_snr_db",
               "text_gate", "chroma_gate", "chroma_proj_norm", "texture_gate", "texture_proj_norm",
               "skipped"]
    csv_exists = False
    if args.resume and os.path.exists(config.DIT_STATS_CSV):
        with open(config.DIT_STATS_CSV) as _f:
            csv_exists = (_f.readline().strip() == ",".join(_fields))
        if not csv_exists:
            os.rename(config.DIT_STATS_CSV, config.DIT_STATS_CSV + ".old")
            print(f"stats CSV header changed -> archived old one to {config.DIT_STATS_CSV}.old")
    stats_file  = open(config.DIT_STATS_CSV, "a" if csv_exists else "w", newline="")
    csv_writer  = csv.DictWriter(stats_file, fieldnames=_fields)
    if not csv_exists:
        csv_writer.writeheader()

    # THE eval metric for text-primary conditioning: shared once, both evaluators use it.
    clap_scorer = ClapScorer(device) if use_text else None

    vae_ckpt_path = args.vae_ckpt
    if vae_ckpt_path is None:
        ft_ckpts = sorted(glob.glob(os.path.join(config.FT_DEC_CKPT_DIR, "dec_step*.pt")))
        vae_ckpt_path = ft_ckpts[-1] if ft_ckpts else "checkpoints/vae/vae_step0200000.pt"
    print(f"Eval-time VAE checkpoint: {vae_ckpt_path}"
          + ("  (retrained iSTFT-head decoder)" if "vae_decoder_ft" in vae_ckpt_path
             else "  (WARNING: pre-iSTFT-head checkpoint — run finetune_decoder.py first if "
                  "VAE_DEC_USE_ISTFT_HEAD is True, or this will crash on load)"))

    evaluator = SNREvaluator(vae_ckpt_path, device, label="train",
                             text_dir=config.TEXT_EMB_DIR, captions_dir=config.CAPTIONS_DIR,
                             clap=clap_scorer)
    # HELD-OUT evaluator: tracks with features/captions in the VAL dirs only. Those tracks have
    # no deterministic latent cached, so they cannot be in the training set — this is the number
    # that says whether any of this generalizes. Shares the frozen VAE + CLAP with the train one.
    val_evaluator = None
    if config.DIT_EVAL_HELDOUT:
        val_evaluator = SNREvaluator(None, device, features_dir=config.FEATURES_VAL_DIR,
                                     text_dir=config.TEXT_EMB_VAL_DIR,
                                     captions_dir=config.CAPTIONS_VAL_DIR,
                                     label="val", vae=evaluator.vae, clap=clap_scorer)
        train_ids = {_track_id_from_path(f) for f in ds.files}
        overlap = sum(t in train_ids for t in val_evaluator.track_ids)
        print(f"\nHeld-out eval: {len(val_evaluator.track_ids)} clips from "
              f"{config.FEATURES_VAL_DIR}, {overlap} of which are in the training set "
              f"({'CONTAMINATED — fix before trusting val' if overlap else 'clean'}).")

    print(f"\nEval every {config.DIT_EVAL_EVERY} steps on {config.DIT_EVAL_CLIPS} fixed clips "
          f"(seed {config.DIT_EVAL_SEED}). What each number means:")
    print(f"  CLAP  THE metric for text-primary conditioning: does the generated audio actually "
          f"match the PROMPT?  0.000 = ignoring text, ceiling {evaluator.clap_ceiling:+.3f}")
    print(f"  GAP   is it following the harmony it was given (SECONDARY, reference-track path)? "
          f"0.000 = ignoring chroma, ceiling {evaluator.chroma_ceiling:+.3f}")
    print( "  STD   is it regressing to the mean? (<1 = shrunken, averaged latents = smeared)"
           "   target 1.00")
    print( "  FLAT  MEDIAN per-clip spectral flatness vs the VAE's own output.  target 1.00x")
    print( "        >1 = hissy/noisy   <1 = TOO tonal (missing texture/transients = dull).")
    print( "        NOISY counts clips that blew up into real noise (>3x) — a separate,")
    print( "        occasional failure that the median deliberately hides.")
    print( "  SNR   IGNORE. Same song shifted 10ms scores -0.79 dB; silence scores 0.00 dB.")
    print( "        It measures waveform phase alignment, not quality. Kept in the CSV only.")
    if val_evaluator is not None:
        print(f"  val_* the same metrics on {config.DIT_EVAL_CLIPS} HELD-OUT clips (ceiling "
              f"{val_evaluator.chroma_ceiling:+.3f}). A large train-val GAP gap = memorization.")
    print()

    # Baseline BEFORE any training. Feature/text modules are zero-init, so this is exactly the
    # loaded checkpoint's behaviour — the number every later eval is measured against.
    ev  = evaluator(ema_model)
    vev = val_evaluator(ema_model) if val_evaluator is not None else None
    # The eval sampling loop (50 Euler steps x CFG x 16 clips, no_grad) leaves PyTorch's caching
    # allocator holding a large, fragmented reserved pool. The next op is a differently-shaped
    # training forward/backward pass, which can fail to find a contiguous free block and OOM
    # even though total allocated memory is well under capacity. Release it back so the
    # allocator can serve the training shapes cleanly.
    torch.cuda.empty_cache()
    print(f"BASELINE (step {step}): CLAP {ev['clap_gap']:+.3f} | GAP {ev['chroma_gap']:+.3f} "
          f"| STD {ev['std_ratio']:.2f} | FLAT {ev['flatness_ratio']:.2f}x | NOISY {ev['n_noisy']}")
    if vev:
        print(f"BASELINE (step {step}) HELD-OUT: CLAP {vev['clap_gap']:+.3f} "
              f"| GAP {vev['chroma_gap']:+.3f} | STD {vev['std_ratio']:.2f} "
              f"| FLAT {vev['flatness_ratio']:.2f}x | NOISY {vev['n_noisy']}")
    print()

    print(f"DiT params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Latent: ({config.VAE_LATENT_DIM}, {config.VAE_LATENT_LEN})   "
          f"Tags: {config.DIT_VOCAB_SIZE}   D={config.DIT_D_MODEL}   L={config.DIT_LAYERS}")
    print()

    loss_avg  = None    # running mean for the divergence guard
    skipped   = 0
    consec_skips = 0    # consecutive (not lifetime) spike-guard skips -- see abort check below
    best_loss = float("inf")
    diverged  = 0
    abort     = False
    # Don't police the loss during warmup — it is legitimately unsettled there.
    warmup_end = (reset_at["step"] or 0) + config.DIT_RESUME_WARMUP + 200
    # Refreshed on each eval (every DIT_EVAL_EVERY) but read by the CSV row written every
    # DIT_LOG_EVERY, so these must exist from the first logged step onward.
    def _gates():
        fe = model.feat_embed if use_feats else None
        cg = fe.chroma_gate.item()             if fe and config.DIT_USE_CHROMA  else 0.0
        cn = fe.chroma_proj.weight.norm().item()  if fe and config.DIT_USE_CHROMA  else 0.0
        tg = fe.texture_gate.item()            if fe and config.DIT_USE_TEXTURE else 0.0
        tn = fe.texture_proj.weight.norm().item() if fe and config.DIT_USE_TEXTURE else 0.0
        # Text has ONE cross_gate PER BLOCK (12), not a single scalar like chroma/texture — mean
        # absolute value across blocks is the single-number summary: near 0 means text is being
        # ignored everywhere, which is the falsification check for the whole redesign.
        txg = (sum(b.cross_gate.abs().item() for b in model.blocks) / len(model.blocks)
              if use_text else 0.0)
        return cg, cn, tg, tn, txg

    cg, cn, tg_, tn, txg = _gates()

    model.train()
    while step < total_steps and not abort:
        for batch in dl:
            if step >= total_steps:
                break

            if use_feats or use_text:
                latents, tag_lists, feats = batch
            else:
                latents, tag_lists = batch
                feats = None

            # z1: clean latents (B, 32, 645)
            z1 = latents.to(device)
            B  = z1.shape[0]

            # ── Flow matching ─────────────────────────────────────────────────
            # Logit-normal t: concentrates samples near t=0.5, where the velocity
            # field is hardest to predict (near t=0/1 the answer is nearly linear).
            # Uniform t spends most of its budget on the easy ends. See SD3
            # (Esser et al., 2024), which found this measurably improves samples.
            if config.DIT_LOGIT_NORMAL_T:
                t = torch.sigmoid(torch.randn(B, device=device) * config.DIT_T_SIGMA)
            else:
                t = torch.rand(B, device=device)                                # (B,) ∈ [0,1]
            z0       = torch.randn_like(z1)                                    # noise
            z_t      = (1 - t[:, None, None]) * z0 + t[:, None, None] * z1   # interpolated
            v_target = z1 - z0                                                 # velocity

            # CFG: randomly drop conditioning
            drop_mask = torch.rand(B) < config.DIT_CFG_DROPOUT               # (B,) bool

            # Chroma is dropped independently ON TOP of the CFG mask, so the model sees
            # all three regimes: p(z|tags,chroma), p(z|tags) and p(z|nothing). Without the
            # extra drop it would never learn to generate without a reference melody.
            # CFG-dropped samples must lose chroma too, or the "unconditional" branch
            # would still be seeing the content and CFG would be meaningless.
            chroma_drop = drop_mask | (torch.rand(B) < config.DIT_CHROMA_DROPOUT)
            # Text dropped independently too (config.DIT_TEXT_DROPOUT), same reasoning: the
            # model must also see p(z|tags,chroma) WITHOUT text, or CFG guidance on text alone
            # (turning it up/down at inference) has nothing to push against.
            text_drop = drop_mask | (torch.rand(B) < config.DIT_TEXT_DROPOUT)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_pred = model(z_t, t, tag_lists, drop_mask=drop_mask, feats=feats,
                               chroma_drop=chroma_drop, text_drop=text_drop)
                loss   = F.mse_loss(v_pred.float(), v_target)

            # Outlier-BATCH guard: drop a single wild batch rather than let it poison the model.
            lv = loss.item()
            # Anchored to the BEST loss ever seen, not the CURRENT running average. Anchoring to
            # loss_avg made the guard self-defeating: as the model degraded the average rose, so
            # the bar rose with it.
            # HONEST LIMIT: this would NOT have caught the step-8600 collapse. Reconstructing the
            # per-batch losses from loss_avg there gives ~1.07 rising to ~1.44 against a
            # 2.0 x 0.695 = 1.39 bar — the collapse is a slow slide, not a spike, and `skipped`
            # was 0 for that entire run. The slow-slide detector below is what catches that case;
            # this only bounds the damage from a genuine one-off outlier.
            ref   = best_loss if math.isfinite(best_loss) else loss_avg
            spike = (not math.isfinite(lv)) or (
                ref is not None and lv > config.DIT_LOSS_SPIKE_FACTOR * ref)
            if spike:
                skipped += 1
                consec_skips += 1
                opt.zero_grad(set_to_none=True)
                if skipped % 50 == 1:
                    print(f"  [spike guard] step {step}: loss {lv:.3f} vs avg {loss_avg:.3f} "
                          f"— update skipped ({skipped} total)", flush=True)
                # DEADLOCK FIX: best_loss/loss_avg are only updated below, which a spike never
                # reaches -- so once loss jumps and STAYS elevated, the bar this compares against
                # is frozen forever and every future step spikes too, forever, silently (the
                # slow-slide abort below is also never reached, since spike always `continue`s
                # first). Cap consecutive skips and abort loudly instead of spinning.
                if consec_skips >= 300:
                    print(f"\nSTALLED: {consec_skips} consecutive spike-guard skips at step "
                          f"{step} (loss ~{lv:.3f} vs frozen bar {config.DIT_LOSS_SPIKE_FACTOR * ref:.3f}). "
                          f"Aborting instead of spinning forever with zero progress.\n"
                          f"Last good checkpoint: {ckpt_dir}/ — likely cause: peak LR too high for "
                          f"this model, lower DIT_LR and resume.", flush=True)
                    abort = True
                    break
                step += 1
                continue
            consec_skips = 0

            loss_avg = lv if loss_avg is None else 0.99 * loss_avg + 0.01 * lv

            # Slow-slide divergence detector. The spike guard above only catches a single bad
            # BATCH; it cannot see a gradual collapse, which is exactly how the last run died
            # (0.68 -> 0.71 -> 1.38 -> 1.96, never a 3x jump). Bail out loudly rather than
            # spend hours training a model that has already given up.
            # Track the best ALWAYS (the healthy pre-divergence loss is the only meaningful
            # reference), but only ABORT after warmup, where the loss is legitimately unsettled.
            # Previously best_loss was only tracked after warmup_end, so it anchored to an
            # already-diverging 0.97 instead of the true best 0.678 — making the guard far
            # less sensitive than intended and letting the collapse run further.
            if step > 300:
                best_loss = min(best_loss, loss_avg)
            if step > warmup_end:
                if loss_avg > config.DIT_DIVERGE_FACTOR * best_loss:
                    diverged += 1
                    if diverged >= 300:
                        print(f"\nDIVERGED: loss(avg) {loss_avg:.4f} has stayed above "
                              f"{config.DIT_DIVERGE_FACTOR}x its best ({best_loss:.4f}) for 300 "
                              f"steps. Aborting instead of destroying the model.\n"
                              f"Last good checkpoint: {ckpt_dir}/  — lower "
                              f"DIT_FINETUNE_FEAT_LR or raise DIT_RESUME_WARMUP and resume.",
                              flush=True)
                        abort = True
                        break
                else:
                    diverged = 0

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            ema_update(ema_model, model, config.DIT_EMA_DECAY)

            step += 1

            if step % config.DIT_EVAL_EVERY == 0:
                ev = evaluator(ema_model)
                if val_evaluator is not None:
                    vev = val_evaluator(ema_model)
                torch.cuda.empty_cache()   # see baseline-eval comment above: avoid post-eval
                                           # allocator fragmentation OOM-ing the next train step
                # All gates are zero-init. If a gate stays ~0 the model is IGNORING that signal
                # and no downstream metric can move. txg (mean |cross_gate| across all 12 blocks)
                # is THE falsification check for the whole text-primary redesign: if it never
                # grows, text conditioning is inert and clap_gap cannot move regardless of
                # anything else in this file.
                cg, cn, tg_, tn, txg = _gates()
                cpct = (100 * ev["clap_gap"] / evaluator.clap_ceiling
                       if evaluator.clap_ceiling else 0)
                pct = (100 * ev["chroma_gap"] / evaluator.chroma_ceiling
                       if evaluator.chroma_ceiling else 0)
                print(f"step {step:7d} | CLAP {ev['clap_gap']:+.3f} ({cpct:3.0f}% of ceiling) "
                      f"| GAP {ev['chroma_gap']:+.3f} ({pct:3.0f}%) "
                      f"| STD {ev['std_ratio']:.2f} | FLAT {ev['flatness_ratio']:.2f}x "
                      f"| NOISY {ev['n_noisy']}/{config.DIT_EVAL_CLIPS} "
                      f"| loss(avg) {(loss_avg or 0):.4f} "
                      f"| text_gate {txg:.3f} c_gate {cg:+.3f} tex_gate {tg_:+.3f}", flush=True)
                if vev:
                    vcpct = (100 * vev["clap_gap"] / val_evaluator.clap_ceiling
                            if val_evaluator.clap_ceiling else 0)
                    vpct = (100 * vev["chroma_gap"] / val_evaluator.chroma_ceiling
                            if val_evaluator.chroma_ceiling else 0)
                    print(f"         HELD-OUT | CLAP {vev['clap_gap']:+.3f} ({vcpct:3.0f}%) "
                          f"| GAP {vev['chroma_gap']:+.3f} ({vpct:3.0f}%) "
                          f"| STD {vev['std_ratio']:.2f} | FLAT {vev['flatness_ratio']:.2f}x "
                          f"| NOISY {vev['n_noisy']}/{config.DIT_EVAL_CLIPS} "
                          f"| train-val CLAP gap {ev['clap_gap'] - vev['clap_gap']:+.3f}",
                          flush=True)

            if step % config.DIT_LOG_EVERY == 0:
                lr   = opt.param_groups[0]["lr"]
                vram = torch.cuda.max_memory_allocated() / 1e9
                # loss_avg (running mean), not the single-batch loss — per-batch loss is far too
                # noisy to read a trend from (within-batch noise swamped the real drift before).
                la = loss_avg if loss_avg is not None else loss.item()
                print(f"step {step:7d} | loss {la:.4f} | lr {lr:.2e} | vram {vram:.1f}GB",
                      flush=True)
                row = {
                    "step": step, "loss": loss.item(), "loss_avg": la, "lr": lr, "vram": vram,
                    "clap_gap": ev["clap_gap"], "clap_gap_ceiling": evaluator.clap_ceiling,
                    "chroma_gap": ev["chroma_gap"], "chroma_gap_ceiling": evaluator.chroma_ceiling,
                    "std_ratio": ev["std_ratio"], "flatness_ratio": ev["flatness_ratio"],
                    "n_noisy": ev["n_noisy"], "snr_db": ev["snr_db"],
                    "text_gate": txg, "chroma_gate": cg, "chroma_proj_norm": cn,
                    "texture_gate": tg_, "texture_proj_norm": tn, "skipped": skipped,
                }
                if vev:
                    row.update({
                        "val_clap_gap": vev["clap_gap"],
                        "val_clap_gap_ceiling": val_evaluator.clap_ceiling,
                        "val_chroma_gap": vev["chroma_gap"],
                        "val_chroma_gap_ceiling": val_evaluator.chroma_ceiling,
                        "val_std_ratio": vev["std_ratio"],
                        "val_flatness_ratio": vev["flatness_ratio"],
                        "val_n_noisy": vev["n_noisy"], "val_snr_db": vev["snr_db"],
                    })
                csv_writer.writerow(row)
                stats_file.flush()

            if step % ckpt_every == 0:
                path = save_checkpoint(step, model, ema_model, opt, scheduler, ckpt_dir, keep_last)
                print(f"  → saved {path}")

    stats_file.close()
    if abort:
        print("Aborted on divergence — NOT saving a final checkpoint "
              "(it would overwrite/prune a healthy one).")
    else:
        save_checkpoint(step, model, ema_model, opt, scheduler, ckpt_dir, keep_last)
        print("Training complete.")


if __name__ == "__main__":
    main()
