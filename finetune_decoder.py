"""
Decoder-only fine-tune: targets the measured VAE defect (52% energy loss at 2-6kHz, 37% at
0.5-2kHz vs real audio, even decoding GROUND-TRUTH latents) without touching the encoder.

Why decoder-only: a linear probe recovers real 2-6kHz energy from the frozen latent at R^2=0.518
-- the information survives encoding, so the encoder is not the problem. Retraining it would
invalidate every cached latent AND the current DiT checkpoint for no measured benefit. See
config.py's FT_DEC_* block and project_transfer.md for the full reasoning.

What's different from src/train_vae.py:
  - Encoder FROZEN. Training reads pre-cached latents directly (config.LATENTS_DIR) instead of
    re-running the encoder -- faster, and exactly matches what decode() receives in production
    (mu, no reparameterization noise).
  - Adds MultiResSTFTLoss (defined in src/losses.py, never previously wired into training): a
    LINEAR-frequency multi-resolution STFT loss, as opposed to the existing log-MEL loss which
    first collapses many linear bins into fewer/wider mel filterbank outputs before comparing.
  - No adversarial loss in this first pass, so any change in results is attributable to the one
    new ingredient, not a confound with GAN training dynamics.

Usage:
  python finetune_decoder.py                       # fresh from checkpoints/vae/vae_step0200000.pt
  python finetune_decoder.py --resume checkpoints/vae_decoder_ft/dec_step0002000.pt
  python finetune_decoder.py --steps 2000 --eval-only checkpoints/vae/vae_step0200000.pt
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.dataset import _track_id_from_path
from src.losses import MultiScaleMelLoss, MultiResSTFTLoss, BandEnergyLoss
from src.vae import WaveformVAE


# ── Paired (cached latent, real audio) dataset ──────────────────────────────────────────────

class PairedVAEDataset(Dataset):
    """
    Yields (frozen latent mu, real audio) for the SAME track. Latents come straight from
    data/latents/ (the encoder's output on the deterministic first-30s crop) rather than being
    recomputed, since the encoder is frozen for this whole script -- this is both cheaper (no
    encoder forward/backward, no discriminator) and exactly matches production: decode() always
    receives mu directly, never a reparameterized sample.
    """
    def __init__(self, latents_dir, stems_dir):
        flacs = glob.glob(os.path.join(stems_dir, "**", "*.flac"), recursive=True)
        flac_by_id = {_track_id_from_path(p): p for p in flacs}
        lat_files = glob.glob(os.path.join(latents_dir, "*.pt"))
        self.pairs = [(f, flac_by_id[_track_id_from_path(f)])
                      for f in lat_files if _track_id_from_path(f) in flac_by_id]
        missing = len(lat_files) - len(self.pairs)
        print(f"Paired dataset: {len(self.pairs)} (latent, audio) pairs"
              + (f"  ({missing} latents had no matching FLAC)" if missing else ""))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        lat_path, flac_path = self.pairs[i]
        z = torch.load(lat_path, weights_only=True).float()          # (32, 645)
        audio, _ = sf.read(flac_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        n = config.CHUNK_SAMPLES
        audio = audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))
        return z, torch.from_numpy(audio).float().unsqueeze(0)        # (1, 660480)


# ── Band-energy diagnostic (the metric this whole fine-tune targets) ───────────────────────

BANDS = [(0, 500, "0-0.5k"), (500, 2000, "0.5-2k"), (2000, 6000, "2-6k"), (6000, 11025, "6-11k")]


@torch.no_grad()
def band_energy_report(real, recon, label, n_fft=1024, hop=256):
    win = torch.hann_window(n_fft, device=real.device)
    Sr = torch.stft(real.squeeze(1).float(), n_fft, hop, n_fft, win, return_complex=True).abs().pow(2)
    Sg = torch.stft(recon.squeeze(1).float(), n_fft, hop, n_fft, win, return_complex=True).abs().pow(2)
    freqs = torch.linspace(0, config.SAMPLE_RATE / 2, Sr.shape[1], device=real.device)
    parts = []
    for lo, hi, name in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        er = (Sr[:, mask].sum() / Sr.sum()).item()
        eg = (Sg[:, mask].sum() / Sg.sum()).item()
        parts.append(f"{name} {eg:.4f}(real {er:.4f}, {100*(eg/er-1):+.0f}%)")
    peak = recon.abs().max().item()
    print(f"  {label:20s} " + "  ".join(parts) + f"   peak {peak:.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="checkpoints/vae/vae_step0200000.pt",
                   help="VAE checkpoint to load the frozen encoder + starting decoder from")
    p.add_argument("--resume", default=None, help="resume a decoder fine-tune checkpoint")
    p.add_argument("--steps", type=int, default=config.FT_DEC_STEPS)
    p.add_argument("--eval-only", metavar="CKPT", default=None,
                   help="just run the band-energy report on this checkpoint and exit")
    p.add_argument("--dump-audio", metavar="DIR", default=None,
                   help="with --eval-only: write the 8 fixed real/recon clips as WAVs to this dir")
    args = p.parse_args()

    device = torch.device("cuda")
    vae = WaveformVAE().to(device)
    base_ckpt = args.eval_only or args.resume or args.base
    sd = torch.load(base_ckpt, map_location=device, weights_only=False)["vae"]
    # strict=False: base_ckpt may predate VAE_DEC_USE_ISTFT_HEAD, in which case dec_head_mag/
    # dec_head_phase start fresh (missing) and the old dec_head.weight is discarded (unexpected).
    # Encoder + dec_proj/dec_up/dec_body still transfer -- that's the intended warm start.
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  Partial load from {base_ckpt} (head architecture differs -- expected when "
              f"swapping decoder heads):")
        if missing:    print(f"    missing (fresh-init):   {missing}")
        if unexpected: print(f"    unexpected (discarded): {unexpected}")

    ds = PairedVAEDataset(config.LATENTS_DIR, config.STEMS_DIR)
    g = torch.Generator().manual_seed(config.DIT_EVAL_SEED)
    fixed_idx = torch.randperm(len(ds), generator=g)[:8].tolist()
    fixed_z    = torch.stack([ds[i][0] for i in fixed_idx]).to(device)
    fixed_real = torch.stack([ds[i][1] for i in fixed_idx]).to(device)

    if args.eval_only:
        vae.eval()
        with torch.no_grad():
            recon = vae.decode(fixed_z)
        print(f"\n=== {args.eval_only} ===")
        band_energy_report(fixed_real, recon, "this checkpoint")
        if args.dump_audio:
            os.makedirs(args.dump_audio, exist_ok=True)
            recon_np = recon.squeeze(1).float().cpu().numpy()
            real_np  = fixed_real.squeeze(1).float().cpu().numpy()
            for i in range(recon_np.shape[0]):
                sf.write(os.path.join(args.dump_audio, f"clip{i}_real.wav"),
                         real_np[i], config.SAMPLE_RATE)
                sf.write(os.path.join(args.dump_audio, f"clip{i}_recon.wav"),
                         recon_np[i], config.SAMPLE_RATE)
            print(f"Wrote {2 * recon_np.shape[0]} WAVs to {args.dump_audio}/")
        return

    # Freeze the encoder entirely; only decoder submodules get gradients.
    for m in (vae.enc_proj, vae.enc_body, vae.enc_down, vae.enc_out):
        m.requires_grad_(False)
    dec_params = (list(vae.dec_proj.parameters()) + list(vae.dec_up.parameters())
                 + list(vae.dec_body.parameters()))
    # Head is architecture-dependent: config.VAE_DEC_USE_ISTFT_HEAD swaps dec_head (legacy raw-
    # frame) for dec_head_mag + dec_head_phase (Vocos-style). Hardcoding dec_head here crashed
    # with AttributeError the moment the iSTFT head became the default.
    if config.VAE_DEC_USE_ISTFT_HEAD:
        dec_params += list(vae.dec_head_mag.parameters()) + list(vae.dec_head_phase.parameters())
    else:
        dec_params += list(vae.dec_head.parameters())
    print(f"Decoder params (trainable): {sum(p.numel() for p in dec_params)/1e6:.1f}M")

    mel_loss  = MultiScaleMelLoss().to(device)
    stft_loss = MultiResSTFTLoss().to(device)
    band_loss = BandEnergyLoss().to(device)
    opt = torch.optim.AdamW(dec_params, lr=config.FT_DEC_LR, betas=(0.5, 0.9), eps=1e-6)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print(f"Resumed from step {step}")

    dl = DataLoader(ds, batch_size=config.FT_DEC_BATCH, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True)

    os.makedirs(config.FT_DEC_CKPT_DIR, exist_ok=True)
    csv_path = os.path.join(config.FT_DEC_CKPT_DIR, "stats.csv")
    csv_exists = os.path.exists(csv_path) and args.resume
    stats_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.DictWriter(stats_file, fieldnames=["step", "mel", "wave", "stft", "band", "gN"])
    if not csv_exists:
        csv_writer.writeheader()

    print(f"\nBASELINE ({base_ckpt}):")
    vae.eval()
    with torch.no_grad():
        recon0 = vae.decode(fixed_z)
    band_energy_report(fixed_real, recon0, "before fine-tune")

    vae.train()
    target_steps = step + args.steps
    print(f"\nTraining decoder for {args.steps} steps (target step {target_steps})...")
    first_batch = True
    while step < target_steps:
        for z, real in dl:
            if step >= target_steps:
                break
            z, real = z.to(device), real.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                recon = vae.decode(z)

            l_mel  = mel_loss(real, recon)
            l_wave = F.l1_loss(recon.float(), real)
            l_stft = stft_loss(real, recon)
            l_band = band_loss(real, recon)

            if first_batch:
                print(f"  [first-batch magnitudes] mel {l_mel.item():.4f}  "
                      f"wave*{config.VAE_WAVE_WEIGHT} {config.VAE_WAVE_WEIGHT*l_wave.item():.4f}  "
                      f"stft*{config.FT_DEC_STFT_WEIGHT} {config.FT_DEC_STFT_WEIGHT*l_stft.item():.4f}  "
                      f"band*{config.FT_DEC_BAND_WEIGHT} {config.FT_DEC_BAND_WEIGHT*l_band.item():.4f}"
                      f"  (sanity-check these are comparable scale before trusting a long run)")
                first_batch = False

            loss = (l_mel + config.VAE_WAVE_WEIGHT * l_wave
                   + config.FT_DEC_STFT_WEIGHT * l_stft + config.FT_DEC_BAND_WEIGHT * l_band)

            opt.zero_grad()
            loss.backward()
            gN = clip_grad_norm_(dec_params, config.VAE_GRAD_CLIP).item()
            opt.step()
            step += 1

            if step % config.FT_DEC_LOG_EVERY == 0:
                print(f"step {step:6d} | mel {l_mel.item():.4f} | wave {l_wave.item():.4f} | "
                      f"stft {l_stft.item():.4f} | band {l_band.item():.4f} | gN {gN:.2f}", flush=True)
                csv_writer.writerow({"step": step, "mel": l_mel.item(), "wave": l_wave.item(),
                                     "stft": l_stft.item(), "band": l_band.item(), "gN": gN})
                stats_file.flush()

            if step % config.FT_DEC_EVAL_EVERY == 0:
                vae.eval()
                with torch.no_grad():
                    recon_eval = vae.decode(fixed_z)
                band_energy_report(fixed_real, recon_eval, f"step {step}")
                vae.train()

            if step % config.FT_DEC_CKPT_EVERY == 0:
                path = os.path.join(config.FT_DEC_CKPT_DIR, f"dec_step{step:07d}.pt")
                torch.save({"step": step, "vae": vae.state_dict(), "opt": opt.state_dict()}, path)
                ckpts = sorted(glob.glob(os.path.join(config.FT_DEC_CKPT_DIR, "dec_step*.pt")))
                for old in ckpts[:-5]:
                    os.remove(old)
                print(f"  -> saved {path}")

    stats_file.close()
    print(f"\nFINAL (step {step}):")
    vae.eval()
    with torch.no_grad():
        recon_final = vae.decode(fixed_z)
    band_energy_report(fixed_real, recon0, "BEFORE")
    band_energy_report(fixed_real, recon_final, "AFTER")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
