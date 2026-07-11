import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvNeXt1d(nn.Module):
    """ConvNeXt block at frame rate (channels-first). No dilations, no upsampling."""
    def __init__(self, dim):
        super().__init__()
        self.dw   = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1  = nn.Conv1d(dim, dim * 4, 1)
        self.pw2  = nn.Conv1d(dim * 4, dim, 1)

    def forward(self, x):
        h = self.dw(x)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(self.pw1(h))
        h = self.pw2(h)
        return x + h


class WaveformVAE(nn.Module):
    """
    Symmetric spectral VAE.

    Encoder: waveform → STFT → frame-rate ConvNeXt body → strided down → mu/logvar
    Decoder: z → interpolate up → frame-rate ConvNeXt body → Linear head → overlap-add

    Both halves run at STFT frame rate (2580 frames for 30s at 22050 Hz).
    Only the FFT and the final OLA touch sample rate → 3-5× faster than the old
    strided-Conv/ConvTranspose architecture.

    Arithmetic:
      CHUNK_SAMPLES=660480, hop=256, pad=256 → 2580 STFT frames
      2580 // VAE_DEC_FRAME_UP(4) = 645 latent frames  (= old CHUNK_SAMPLES//1024)
      VAE_TRAIN_SAMPLES=65536, hop=256, pad=256 → 256 STFT frames → 64 latent frames
    """
    def __init__(self):
        super().__init__()
        D    = config.VAE_DEC_DIM
        N    = config.VAE_DEC_BLOCKS
        lat  = config.VAE_LATENT_DIM
        nfft = config.VAE_DEC_NFFT
        n_bins = nfft // 2 + 1   # 257 for nfft=512

        # Encoder
        self.enc_proj = nn.Conv1d(n_bins * 2, D, 1)
        self.enc_body = nn.Sequential(*[ConvNeXt1d(D) for _ in range(N)])
        self.enc_down = nn.Conv1d(D, D, config.VAE_DEC_FRAME_UP, stride=config.VAE_DEC_FRAME_UP)
        self.enc_out  = nn.Conv1d(D, lat * 2, 1)

        # Decoder
        self.dec_proj = nn.Conv1d(lat, D, 1)
        self.dec_up   = nn.Conv1d(D, D, 3, padding=1)
        self.dec_body = nn.Sequential(*[ConvNeXt1d(D) for _ in range(N)])
        self.dec_head = nn.Linear(D, nfft)

    @staticmethod
    def _stft_feat(audio, nfft, hop):
        # audio: (B, T) float32
        # Returns (B, (nfft//2+1)*2, n_frames) — sign·log1p compressed real+imag
        win   = torch.hann_window(nfft, device=audio.device)
        x_pad = F.pad(audio, (nfft // 2, 0))   # left-pad so n_frames = T // hop
        X     = torch.stft(x_pad, nfft, hop, nfft, win, return_complex=True)
        Xr, Xi = X.real, X.imag
        return torch.cat([
            Xr.sign() * torch.log1p(Xr.abs()),
            Xi.sign() * torch.log1p(Xi.abs()),
        ], dim=1)

    def encode(self, x):
        nfft = config.VAE_DEC_NFFT
        hop  = config.VAE_DEC_HOP
        feat = self._stft_feat(x.squeeze(1).float(), nfft, hop)
        h    = self.enc_proj(feat.to(x.dtype))
        h    = self.enc_body(h)
        h    = self.enc_down(h)
        out  = self.enc_out(h)
        mu, logvar = out.chunk(2, dim=1)
        return mu, logvar.clamp(-30.0, 20.0)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def _overlap_add(self, frames, T_out):
        # frames: (B, T_frames, nfft) — linear synthesis frames
        # T_out:  target waveform length (samples)
        nfft = config.VAE_DEC_NFFT
        hop  = config.VAE_DEC_HOP
        frames = frames.float()
        B, T, N = frames.shape

        win    = torch.hann_window(N, device=frames.device)
        frames = frames * win

        # F.fold: input (B, N, T) → output (B, 1, 1, out_len)
        out_len  = (T - 1) * hop + N
        frames_t = frames.transpose(1, 2)   # (B, N, T)
        signal   = F.fold(frames_t, (1, out_len), (1, N), stride=(1, hop)).squeeze(2)  # (B, 1, out_len)

        # Normalize by window-squared sum (constant for 50% Hann, but computed for correctness)
        win_sq = win.pow(2).view(1, N, 1).expand(1, N, T)
        norm   = F.fold(win_sq, (1, out_len), (1, N), stride=(1, hop)).squeeze(2)      # (1, 1, out_len)

        signal = signal / norm.clamp(min=1e-8)
        # Remove the nfft//2 left-padding introduced in _stft_feat
        return signal[..., nfft // 2: nfft // 2 + T_out]

    def decode(self, z):
        T_out = z.shape[-1] * config.VAE_DEC_FRAME_UP * config.VAE_DEC_HOP
        h      = self.dec_proj(z)
        h      = F.interpolate(h, scale_factor=config.VAE_DEC_FRAME_UP,
                               mode='linear', align_corners=False)
        h      = self.dec_up(h)
        h      = self.dec_body(h)
        frames = self.dec_head(h.transpose(1, 2))   # (B, T_frames, nfft)
        return self._overlap_add(frames, T_out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z     = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @property
    def last_layer(self):
        return self.dec_head.weight


# ---------------------------------------------------------------------------
# MS-STFT Discriminator  (Encodec-style — 2D Conv on stacked real/imag STFT)
# ---------------------------------------------------------------------------

class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft):
        super().__init__()
        self._nfft = n_fft
        norm = nn.utils.weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(2,   32,  (3, 9), padding=(1, 4))),
            norm(nn.Conv2d(32,  64,  (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(64,  128, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(128, 256, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(256, 256, (3, 3), padding=(1, 1))),
        ])
        self.out = norm(nn.Conv2d(256, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        nfft = self._nfft
        hop  = nfft // 4
        win  = torch.hann_window(nfft, device=x.device)
        X    = torch.stft(x.squeeze(1).float(), nfft, hop, nfft, win, return_complex=True)
        h    = torch.stack([X.real, X.imag], dim=1)   # (B, 2, F, T)
        feats = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.2)
            feats.append(h)
        return self.out(h), feats


class MultiSTFTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([STFTDiscriminator(n) for n in config.STFTD_FFTS])

    def forward(self, x):
        return [disc(x) for disc in self.discs]
