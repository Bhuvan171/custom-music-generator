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
