import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def mel_spectrogram(x, n_fft, hop_len, win_len, n_mels, sample_rate, fmin=0.0, fmax=None,
                    mel_fb=None):
    """Compute log-mel spectrogram from waveform. Matches BigVGAN's implementation."""
    if fmax is None:
        fmax = sample_rate / 2.0

    if mel_fb is None:
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=fmin,
            f_max=fmax,
            norm="slaney",
        ).to(device=x.device, dtype=x.dtype)

    # center=False with explicit padding (matches BigVGAN)
    pad_len = (n_fft - hop_len) // 2
    x_padded = F.pad(x, (pad_len, pad_len), mode="reflect")
    spec = torch.stft(x_padded, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                      return_complex=True, center=False,
                      window=torch.hann_window(win_len, device=x.device, dtype=x.dtype))
    spec_mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    mel_spec = torch.matmul(mel_fb.t(), spec_mag)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return mel_spec


class MultiResolutionMelLoss(nn.Module):
    """BigVGAN-v2 multi-scale mel spectrogram loss.
    Uses different n_mels per scale to capture coarse-to-fine spectral detail."""

    def __init__(self, sample_rate=24000,
                 n_mels=(5, 10, 20, 40, 80, 160, 320),
                 window_lengths=(32, 64, 128, 256, 512, 1024, 2048),
                 fmin=0.0, fmax=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2.0
        self.stft_params = [(w, w // 4) for w in window_lengths]
        self._mel_fb_cache = {}

    def _get_mel_fb(self, n_fft, n_mels, device, dtype):
        key = (n_fft, n_mels, device, dtype)
        if key not in self._mel_fb_cache:
            self._mel_fb_cache[key] = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                n_mels=n_mels,
                sample_rate=self.sample_rate,
                f_min=self.fmin,
                f_max=self.fmax,
                norm="slaney",
            ).to(device=device, dtype=dtype)
        return self._mel_fb_cache[key]

    def forward(self, y_pred, y_true):
        total_loss = 0.0
        for n_mels, (win_len, hop_len) in zip(self.n_mels, self.stft_params):
            n_fft = win_len
            mel_fb = self._get_mel_fb(n_fft, n_mels, y_pred.device, y_pred.dtype)
            mel_pred = mel_spectrogram(y_pred.squeeze(1), n_fft, hop_len, win_len,
                                       n_mels, self.sample_rate, self.fmin, self.fmax,
                                       mel_fb=mel_fb)
            mel_true = mel_spectrogram(y_true.squeeze(1), n_fft, hop_len, win_len,
                                       n_mels, self.sample_rate, self.fmin, self.fmax,
                                       mel_fb=mel_fb)
            min_t = min(mel_pred.shape[-1], mel_true.shape[-1])
            total_loss += F.l1_loss(mel_pred[..., :min_t], mel_true[..., :min_t])
        return total_loss / len(self.n_mels)


class PitchIndependentSpectralLoss(nn.Module):
    """Pitch-independent reconstruction loss using MFCCs.

    MFCCs separate the spectral envelope (formants, timbre — low quefrency)
    from pitch harmonics (high quefrency).  Using only the first n_mfcc
    coefficients gives a representation that is invariant to F0.
    """

    def __init__(self, sample_rate=24000, n_mfcc=13,
                 window_lengths=(1024, 4096)):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.window_lengths = window_lengths
        self._mel_fb_cache = {}
        self._dct_cache = {}
        self._window_cache = {}

    def _get_dct_basis(self, n_mfcc, n_mels, device, dtype):
        key = (n_mfcc, n_mels, device, dtype)
        if key not in self._dct_cache:
            import numpy as np
            dct = np.zeros((n_mfcc, n_mels), dtype=np.float32)
            for i in range(n_mfcc):
                for j in range(n_mels):
                    dct[i, j] = np.cos(np.pi * i * (j + 0.5) / n_mels)
            dct[0] /= np.sqrt(n_mels)
            dct[1:] *= np.sqrt(2.0 / n_mels)
            self._dct_cache[key] = torch.from_numpy(dct).to(device=device, dtype=dtype)
        return self._dct_cache[key]

    def _compute_mfcc(self, wav, n_fft, hop_len, device, dtype):
        """wav [B, N] → MFCC [B, n_mfcc, T]"""
        n_mels = min(80, n_fft // 2)
        key = (n_fft, n_mels, device, dtype)
        if key not in self._mel_fb_cache:
            self._mel_fb_cache[key] = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1, n_mels=n_mels,
                sample_rate=self.sample_rate,
                f_min=0.0, f_max=self.sample_rate / 2.0,
                norm="slaney",
            ).to(device=device, dtype=dtype)
        mel_fb = self._mel_fb_cache[key]

        pad_len = (n_fft - hop_len) // 2
        x_padded = F.pad(wav, (pad_len, pad_len), mode="reflect")
        win_key = (n_fft, device, dtype)
        if win_key not in self._window_cache:
            self._window_cache[win_key] = torch.hann_window(n_fft, device=device, dtype=dtype)
        spec = torch.stft(x_padded, n_fft=n_fft, hop_length=hop_len,
                          win_length=n_fft, return_complex=True, center=False,
                          window=self._window_cache[win_key])
        spec_mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        log_mel = torch.log(torch.clamp(torch.matmul(mel_fb.t(), spec_mag), min=1e-5))

        n_mfcc = min(self.n_mfcc, n_mels)
        dct_basis = self._get_dct_basis(n_mfcc, n_mels, device, dtype)
        mfcc = torch.einsum("ij,bjt->bit", dct_basis, log_mel)
        return mfcc

    def forward(self, y_pred, y_true):
        """Returns pitch-independent spectral loss (scalar)."""
        wav_pred = y_pred.squeeze(1)
        wav_true = y_true.squeeze(1)

        total_loss = 0.0
        for win_len in self.window_lengths:
            hop_len = win_len // 4
            mfcc_pred = self._compute_mfcc(wav_pred, win_len, hop_len,
                                           wav_pred.device, wav_pred.dtype)
            mfcc_true = self._compute_mfcc(wav_true, win_len, hop_len,
                                           wav_true.device, wav_true.dtype)
            min_t = min(mfcc_pred.shape[-1], mfcc_true.shape[-1])
            total_loss += F.l1_loss(mfcc_pred[..., :min_t], mfcc_true[..., :min_t])
        return total_loss / len(self.window_lengths)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (spectral convergence + log magnitude).
    Use for validation only — BigVGAN does not include this in training."""

    def __init__(self, fft_sizes=(512, 1024, 2048),
                 hop_sizes=(120, 240, 480), win_sizes=(480, 960, 1920)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def stft_loss(self, x, y, n_fft, hop_len, win_len):
        pad_len = (n_fft - hop_len) // 2
        x_padded = F.pad(x, (pad_len, pad_len), mode="reflect")
        y_padded = F.pad(y, (pad_len, pad_len), mode="reflect")
        x_stft = torch.stft(x_padded, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                            return_complex=True, center=False,
                            window=torch.hann_window(win_len, device=x.device, dtype=x.dtype))
        y_stft = torch.stft(y_padded, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                            return_complex=True, center=False,
                            window=torch.hann_window(win_len, device=y.device, dtype=y.dtype))
        x_mag = torch.sqrt(x_stft.real.pow(2) + x_stft.imag.pow(2) + 1e-9)
        y_mag = torch.sqrt(y_stft.real.pow(2) + y_stft.imag.pow(2) + 1e-9)
        diff = (y_mag - x_mag).clamp(-10, 10)
        sc_loss = diff.pow(2).mean() / (y_mag.pow(2).mean() + 1e-6)
        mag_loss = F.l1_loss(x_mag.clamp(max=10), y_mag.clamp(max=10))
        return sc_loss + mag_loss

    def forward(self, y_pred, y_true):
        total_loss = 0.0
        for n_fft, hop_len, win_len in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            total_loss += self.stft_loss(y_pred.squeeze(1), y_true.squeeze(1), n_fft, hop_len, win_len)
        return total_loss / len(self.fft_sizes)


class MimiDecoderReconLoss(nn.Module):
    """BigVGAN-v2 multi-scale reconstruction loss for Mimi decoder LoRA training.
    Uses L1 log-mel loss across 7 scales (n_mels 5→320) as the primary training signal.
    STFT loss is excluded from training by default (BigVGAN only uses it for validation)."""

    def __init__(self, sample_rate=24000, mel_weight=45.0, stft_weight=0.0):
        super().__init__()
        self.mel_loss = MultiResolutionMelLoss(sample_rate=sample_rate)
        self.stft_loss = MultiResolutionSTFTLoss() if stft_weight > 0 else None
        self.mel_weight = mel_weight
        self.stft_weight = stft_weight

    def forward(self, y_pred, y_true):
        min_t = min(y_pred.shape[-1], y_true.shape[-1])
        y_pred = y_pred[..., :min_t]
        y_true = y_true[..., :min_t]
        mel_l = self.mel_loss(y_pred, y_true)
        if self.stft_loss is not None and self.stft_weight > 0:
            stft_l = self.stft_loss(y_pred, y_true)
            return self.mel_weight * mel_l + self.stft_weight * stft_l
        return self.mel_weight * mel_l
