"""Pitch-controllable Mimi decoder via two-stage adversarial disentanglement.

Stage 1 (pitch removal):
    LoRA-adapted Mimi decoder + gradient-reversed pitch discriminator.
    Trains the decoder to reconstruct speech while making its internal
    representations pitch-agnostic — the discriminator cannot predict F0
    from decoder hidden states.

Stage 2 (pitch re-injection):
    Freeze Stage 1 LoRA. Add a learned pitch embedding to the RVQ latent
    before decoding. Train a new LoRA layer so the decoder learns to use
    the explicit pitch signal — now a strong, clean conditioning input
    because Stage 1 suppressed implicit pitch from the codebook path.

Architecture:

    Stage 1                          Stage 2
    ──────                           ──────
    RVQ codes → z                    RVQ codes → z
    z → decoder (LoRA-1)             z + pitch_emb → decoder (LoRA-1 frozen, LoRA-2)
    hidden → pitch_disc → F0_pred    hidden → wav
    ↑ grad reversal on F0_pred
    wav → recon loss
    F0_pred vs F0_target → disc loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .finetune_lora_fusion import (
    LoRALinear, LoRAConv1d, LoRAConvTranspose1d,
    inject_lora, MIMI_DECODER_TARGETS,
)
from .f0_estimator import F0Estimator
from .losses import MimiDecoderReconLoss

# LoRA targets: decoder_transformer (Linear layers) + SEANet decoder (Conv1d / ConvTranspose1d).
# RawStreamingConv1d inherits from nn.Conv1d, so LoRAConv1d can wrap it.
MIMI_DECODER_LORA_TARGETS = ["decoder_transformer", "decoder"]


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------

class _GradReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reversal(x, alpha=1.0):
    return _GradReversal.apply(x, alpha)


# ---------------------------------------------------------------------------
# Pitch discriminator (operates on decoder hidden states)
# ---------------------------------------------------------------------------

class PitchDiscriminator(nn.Module):
    """Predicts F0 from decoder hidden states. Used adversarially in Stage 1:
    gradients are reversed so the decoder learns to suppress pitch info."""

    def __init__(self, hidden_dim=512, n_frames=32, hidden=256):
        super().__init__()
        # Global average pool over time, then classify
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):
        # h: [B, T, C] → pool over T → [B, C] → [B, 1]
        h = h.mean(dim=1)
        return self.net(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Pitch embedding (added to RVQ latent in Stage 2)
# ---------------------------------------------------------------------------

class PitchEmbedding(nn.Module):
    """Maps a scalar F0 value to a 256-dim vector added to the RVQ latent."""

    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, f0):
        """f0: [B, T] (Hz or log-Hz) → [B, T, dim]"""
        return self.net(f0.unsqueeze(-1))


# ---------------------------------------------------------------------------
# Pitch-controllable Mimi wrapper
# ---------------------------------------------------------------------------

class PitchControllableMimi(nn.Module):
    """Wraps a frozen Mimi model with pitch-aware LoRA decoding.

    Args:
        mimi: frozen MimiModel
        stage: 1 (pitch removal) or 2 (pitch re-injection)
        lora_rank: rank for LoRA adapters
        lora_alpha: alpha scaling for LoRA
        grad_rev_alpha: strength of gradient reversal in Stage 1
    """

    def __init__(self, mimi, stage=1, lora_rank=8, lora_alpha=16,
                 grad_rev_alpha=1.0):
        super().__init__()
        self.mimi = mimi
        self.stage = stage
        self.grad_rev_alpha = grad_rev_alpha

        # Freeze entire Mimi
        for p in mimi.parameters():
            p.requires_grad = False

        # ── Stage 1: LoRA on decoder (transformer + SEANet convs) + pitch discriminator ──
        if stage == 1:
            inject_lora(mimi, rank=lora_rank, alpha=lora_alpha,
                        targets=MIMI_DECODER_LORA_TARGETS)
            self.pitch_disc = PitchDiscriminator(hidden_dim=512)

        # ── Stage 2: freeze Stage-1 LoRA, add pitch embedding + new LoRA ──
        elif stage == 2:
            # Freeze existing LoRA parameters
            for name, param in mimi.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = False
            self.pitch_emb = PitchEmbedding(dim=512)
            inject_lora(mimi, rank=lora_rank, alpha=lora_alpha,
                        targets=MIMI_DECODER_LORA_TARGETS)

        # Frozen F0 estimator for extracting pitch targets
        self.f0_estimator = F0Estimator()
        for p in self.f0_estimator.parameters():
            p.requires_grad = False

    def _decode_rvq(self, codes):
        """codes [B, K, T] → z [B, 256, T_rvq] (quantizer latent)."""
        return self.mimi.quantizer.decode(codes)

    def _to_waveform(self, z):
        """z [B, 256, T_rvq] → wav [B, 1, N], hidden [B, T, C] (channels-last)."""
        e = self.mimi._to_encoder_framerate(z)
        (e,) = self.mimi.decoder_transformer(e)
        # hidden for discriminator: transpose to channels-last [B, T, C]
        hidden = e.transpose(1, 2) if e.dim() == 3 else e
        return self.mimi.decoder(e), hidden

    def _extract_f0(self, wav):
        """wav [B, 1, N] → f0 [B, T] (Hz)."""
        with torch.no_grad():
            log_f0, vlogit = self.f0_estimator(wav)
            f0 = torch.exp(log_f0)  # natural-log Hz → Hz
        return f0

    def forward(self, codes, target_wav=None):
        """
        Args:
            codes: [B, K, T] RVQ codes
            target_wav: [B, 1, N] ground truth waveform (needed for F0 target)

        Returns:
            wav: [B, 1, N] reconstructed waveform
            info: dict with loss components
        """
        z = self._decode_rvq(codes)

        # Stage 2: add pitch embedding
        pitch_emb = None
        if self.stage == 2 and hasattr(self, 'pitch_emb'):
            if target_wav is not None:
                f0 = self._extract_f0(target_wav)  # [B, T]
            else:
                # Inference: use zeros (or provide pitch externally)
                T_rvq = z.shape[-1]
                f0 = torch.zeros(z.shape[0], T_rvq, device=z.device)
            pitch_emb = self.pitch_emb(f0)  # [B, T, 256]
            pitch_emb = pitch_emb.transpose(1, 2)  # [B, 256, T]
            # Align lengths
            T_min = min(z.shape[-1], pitch_emb.shape[-1])
            z = z[:, :, :T_min] + pitch_emb[:, :, :T_min]

        wav, hidden = self._to_waveform(z)

        info = {'wav': wav}

        # Stage 1: pitch discriminator with gradient reversal
        if self.stage == 1 and target_wav is not None:
            f0_target = self._extract_f0(target_wav)  # [B, T] in Hz
            f0_pred_raw = self.pitch_disc(hidden)       # [B]
            # Normalize: log-F0 centered at 100 Hz, scaled to ~unit variance
            f0_target_mean = f0_target.mean(dim=1)     # [B] Hz
            f0_target_norm = (torch.log(f0_target_mean.clamp(min=1.0)) - 4.6) / 0.5  # ~N(0,1)
            f0_pred_norm = f0_pred_raw / 500.0  # normalize discriminator output
            # Apply gradient reversal
            f0_pred_rev = grad_reversal(f0_pred_norm, self.grad_rev_alpha)
            info['f0_pred'] = f0_pred_raw
            info['f0_pred_rev'] = f0_pred_rev
            info['f0_target_norm'] = f0_target_norm

        return wav, info


# ---------------------------------------------------------------------------
# Combined loss for Stage 1
# ---------------------------------------------------------------------------

class PitchControlLoss(nn.Module):
    """Stage 1 loss: reconstruction + pitch discriminator adversarial.

    Total = mel_weight * mel_loss + disc_weight * disc_loss

    disc_loss = MSE(f0_pred_rev, f0_target) — gradients reversed so
    decoder learns to hide pitch from the discriminator.
    """

    def __init__(self, sample_rate=24000, mel_weight=45.0, disc_weight=1.0):
        super().__init__()
        self.recon = MimiDecoderReconLoss(sample_rate=sample_rate)
        self.mel_weight = mel_weight
        self.disc_weight = disc_weight

    def forward(self, wav_pred, wav_true, info):
        """Returns (total_loss, loss_dict)."""
        mel_loss = self.recon(wav_pred, wav_true)

        disc_loss = torch.tensor(0.0, device=wav_pred.device)
        if 'f0_pred_rev' in info:
            disc_loss = F.mse_loss(info['f0_pred_rev'], info['f0_target_norm'])

        total = self.mel_weight * mel_loss + self.disc_weight * disc_loss
        return total, {'mel': mel_loss.item(), 'disc': disc_loss.item()}
