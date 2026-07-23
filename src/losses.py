import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T_audio

import config


class MultiScaleMelLoss(nn.Module):
    """Log-mel L1 at multiple scales. Weights every frequency band equally so
    HF cannot be silently suppressed (unlike linear-STFT magnitude loss)."""
    def __init__(self):
        super().__init__()
        self.mels = nn.ModuleList([
            T_audio.MelSpectrogram(
                sample_rate=config.SAMPLE_RATE,
                n_fft=n_fft,
                hop_length=n_fft // 4,
                n_mels=n_mels,
                power=1,
            )
            for n_fft, n_mels in zip(config.MEL_FFT_SIZES, config.MEL_N_MELS)
        ])

    def forward(self, real, pred):
        x = real.squeeze(1).float()
        y = pred.squeeze(1).float()
        loss = 0.0
        for mel in self.mels:
            Mx = mel(x).clamp(min=config.MEL_EPS).log()
            My = mel(y).clamp(min=config.MEL_EPS).log()
            loss += F.l1_loss(Mx, My)
        return loss / len(self.mels)


class MultiResSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft_sizes = config.STFT_FFT_SIZES

    def forward(self, real, pred):
        # stft requires fp32 — cast explicitly regardless of autocast context
        x = real.squeeze(1).float()
        y = pred.squeeze(1).float()
        loss = 0.0
        for n in self.fft_sizes:
            hop = n // 4
            win = torch.hann_window(n, device=x.device)
            Sx = torch.stft(x, n, hop, n, win, return_complex=True).abs().clamp(config.STFT_EPS)
            Sy = torch.stft(y, n, hop, n, win, return_complex=True).abs().clamp(config.STFT_EPS)
            loss += F.l1_loss(Sx, Sy) + F.l1_loss(Sx.log(), Sy.log())
        return loss / len(self.fft_sizes)


class BandEnergyLoss(nn.Module):
    """
    Per-band log-energy L1, EACH BAND WEIGHTED EQUALLY regardless of its natural magnitude.

    MultiResSTFTLoss (linear-frequency, but summed across ALL bins into one scalar) was tried
    first for the same purpose (recovering the VAE's measured -52%/-37% energy loss at
    2-6kHz/0.5-2kHz) and measured, over 1000 real fine-tuning steps, to leave the 2-6kHz band
    completely unchanged (-69% -> -68%, noise-level, four consecutive eval checkpoints). The
    mechanism: real audio is bass-heavy (86% of total energy sits below 500Hz), so a loss that
    sums magnitude error across all frequencies together is dominated by getting the loud bass
    right, almost regardless of what happens in a band carrying under 7% of total energy. The
    log-magnitude term in that loss is scale-invariant PER BIN, but the SUM across bins is not
    scale-invariant PER BAND — a band with more bins, or louder bins, contributes more to the
    total regardless of the log compression.

    This loss instead computes total energy for each band SEPARATELY, in log domain, and
    averages the per-band errors with equal weight — so a quiet band being 50% wrong costs
    exactly as much as a loud band being 50% wrong, which a raw energy-weighted loss cannot
    express.
    """
    def __init__(self, n_fft=2048, hop=512,
                bands=((0, 500), (500, 2000), (2000, 6000), (6000, 11025))):
        super().__init__()
        self.n_fft, self.hop, self.bands = n_fft, hop, bands
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, real, pred):
        x = real.squeeze(1).float()
        y = pred.squeeze(1).float()
        Sx = torch.stft(x, self.n_fft, self.hop, self.n_fft, self.window,
                        return_complex=True).abs().pow(2)
        Sy = torch.stft(y, self.n_fft, self.hop, self.n_fft, self.window,
                        return_complex=True).abs().pow(2)
        freqs = torch.linspace(0, config.SAMPLE_RATE / 2, Sx.shape[1], device=x.device)
        loss = 0.0
        for lo, hi in self.bands:
            mask = (freqs >= lo) & (freqs < hi)
            ex = torch.log1p(Sx[:, mask, :].sum(dim=1))   # (B, T) per-frame band energy
            ey = torch.log1p(Sy[:, mask, :].sum(dim=1))
            loss = loss + F.l1_loss(ey, ex)
        return loss / len(self.bands)


def adv_g_loss(d_fake):
    # d_fake: list of (logit, feats) from MultiScaleDiscriminator
    return sum(-logit.mean() for logit, _ in d_fake) / len(d_fake)


def adv_d_loss(d_real, d_fake):
    # Hinge loss — bounded, stable
    loss = 0.0
    for (r, _), (f, _) in zip(d_real, d_fake):
        loss += F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
    return loss / len(d_real)


def feat_match_loss(d_real, d_fake):
    # L1 between discriminator intermediate feature maps (real detached)
    loss, n = 0.0, 0
    for (_, real_feats), (_, fake_feats) in zip(d_real, d_fake):
        for rf, ff in zip(real_feats, fake_feats):
            loss += F.l1_loss(ff, rf.detach())
            n += 1
    return loss / n


def adaptive_weight(recon_loss, adv_loss, last_layer, max_weight):
    # VQGAN trick: scale adv loss so its gradient at the last decoder layer
    # never exceeds the reconstruction gradient. Recon always dominates.
    g_recon = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    g_adv   = torch.autograd.grad(adv_loss,   last_layer, retain_graph=True)[0]
    lam = g_recon.norm() / (g_adv.norm() + 1e-4)
    return lam.clamp(0.0, max_weight).detach()
