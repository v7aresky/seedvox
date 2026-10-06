"""JDC-style differentiable F0 + voicing estimator.

Pretrained on mimi-decoded audio against pyin targets, then FROZEN and reused as
the differentiable pitch-loss for the token operator:

    operator soft logits -> soft-mixture mimi decode -> est(wav) -> L1 vs target

Input:  mono waveform at 24 kHz [B, 1, N] with N a multiple of hop (1920 = 12.5 Hz).
Output: log-F0 in Hz [B, T] (natural log) and voicing logit [B, T].

Frontend: STFT (n_fft 2048, hop 1920, no centering -> exactly one frame per
token frame) -> log mel magnitude (128 bins, fixed linear basis).
Body: JDC-style 2D conv stack with time dilation 1,2,4,8,16, frequency reduced
to a single row.
Heads: log-F0 regression (L1) + voicing BCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from librosa.filters import mel as librosa_mel


class F0Estimator(nn.Module):
    def __init__(self, sr=24000, n_fft=2048, hop=1920, n_mels=128,
                 fmin=50.0, fmax=8000.0, channels=48, blocks=4):
        super().__init__()
        self.sr, self.n_fft, self.hop = sr, n_fft, hop
        basis = librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.register_buffer('mel_basis', torch.from_numpy(basis).float())  # [n_mels, 1+n_fft//2]
        self.register_buffer('win', torch.hann_window(n_fft, periodic=True))

        dil = [1, 2, 4, 8, 16]
        c = 1
        self.body = nn.ModuleList()
        for i in range(blocks):
            d = dil[i % len(dil)]
            self.body.append(nn.Sequential(
                nn.Conv2d(c, channels, (3, 5), padding=(1, 2 * d), dilation=(1, d)),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ))
            c = channels
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.f0_head = nn.Conv1d(channels, 1, 1)
        self.v_head = nn.Conv1d(channels, 1, 1)

    def _feats(self, wav):
        """wav [B, 1, N] -> log-mel magnitude [B, 1, n_mels, T] with T = N // hop.

        torch.stft(center=False) yields 1 + (N - n_fft)//hop frames; pad to
        n_fft + (T-1)*hop so the frame grid lines up exactly with the 12.5 Hz
        token/target grid (one frame per hop)."""
        x = wav.squeeze(1)  # [B, N]
        N = x.shape[-1]
        T = N // self.hop
        need = self.n_fft + (T - 1) * self.hop
        if N < need:
            x = F.pad(x, (0, need - N))
        elif N > need:
            x = x[..., :need]
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.n_fft, window=self.win,
                          center=False, return_complex=True)  # [B, F, T]
        mag = spec.abs()                                        # [B, F, T]
        mel = self.mel_basis @ mag                              # [B, n_mels, T]
        return torch.log(mel + 1e-5).unsqueeze(1)               # [B, 1, n_mels, T]

    def _forward_frame(self, x):
        # x [B, 1, n_mels, T]
        for blk in self.body:
            x = blk(x)
        x = F.adaptive_avg_pool2d(x, (1, x.shape[-1]))          # [B, C, 1, T]
        x = x.squeeze(2)                                        # [B, C, T]
        x = self.proj(x)
        return x

    def forward(self, wav):
        """wav [B, 1, N] -> (logf0 [B, T] natural log Hz, vlogit [B, T])."""
        x = self._feats(wav)
        h = self._forward_frame(x)
        return self.f0_head(h).squeeze(1), self.v_head(h).squeeze(1)

    def estimate_f0_hz(self, wav):
        """Convenience for eval: returns log-F0 [B, T] (natural log, Hz)."""
        logf0, _ = self.forward(wav)
        return logf0
