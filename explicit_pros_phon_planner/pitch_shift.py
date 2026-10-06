"""Gradient-guided pitch shift via z-space F0 gradient direction.

Uses d(predictor(z))/dz to find the correct z-space direction for F0 change.
FilmShift predicts only a scalar magnitude (alpha) conditioned on log_delta.
beta = alpha * grad_direction.

Trainable: pitch_predictor + FilmShift. Mimi frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .f0_estimator import F0Estimator
from .losses import MultiResolutionMelLoss, PitchIndependentSpectralLoss


class PitchPredictor(nn.Module):
    """Predict F0 from decoded z-space (512-dim continuous latent)."""

    def __init__(self, z_dim=512, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(z_dim, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, 1, 1),
            nn.Softplus(),
        )

    def forward(self, z):
        """z [B, 512, T_z] -> f0 [B, 1, T_z] in Hz."""
        return self.net(z)


class FilmShift(nn.Module):
    """Predict scalar shift magnitude from log F0 delta.

    Input: log_delta [B, T_z, 1]
    Output: alpha [B, 1, T_z] (scalar magnitude per timestep)
    """

    def __init__(self, hidden=128, n_layers=3):
        super().__init__()
        layers = []
        in_dim = 1
        for i in range(n_layers):
            out_dim = hidden
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())
            in_dim = out_dim
        self.encoder = nn.Sequential(*layers)
        self.alpha_proj = nn.Linear(hidden, 1)

        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.zeros_(self.alpha_proj.bias)

    def forward(self, log_delta):
        """log_delta [B, T_z, 1] -> alpha [B, 1, T_z]"""
        h = self.encoder(log_delta)          # [B, T_z, hidden]
        alpha = self.alpha_proj(h)           # [B, T_z, 1]
        return alpha.permute(0, 2, 1)        # [B, 1, T_z]


class PitchShiftModel(nn.Module):
    """Gradient-guided pitch control for Mimi decoder.

    Forward:
        codes -> quantizer.decode -> z
        z -> pitch_predictor -> f0_current
        grad_z = d(f0_current)/dz  (F0 gradient direction in z-space)
        direction = grad_z / ||grad_z||
        log_delta = log(target_f0) - log(f0_current)
        alpha = film_shift(log_delta)
        beta = alpha * direction
        z_shifted = z + beta
        z_shifted -> decoder -> wav
    """

    def __init__(self, mimi, predictor_hidden=128, shift_hidden=128,
                 shift_layers=3, f0_checkpoint="checkpoints/f0_estimator_best.pt",
                 beta_max=2.0, predictor_checkpoint=None, freeze_predictor=False):
        super().__init__()
        self.mimi = mimi
        self.beta_max = beta_max
        self._predictor_pretrained = None

        for p in mimi.parameters():
            p.requires_grad = False

        self.f0_estimator = F0Estimator()
        ckpt = torch.load(f0_checkpoint, map_location="cpu", weights_only=False)
        self.f0_estimator.load_state_dict(ckpt["model"])
        del ckpt
        for p in self.f0_estimator.parameters():
            p.requires_grad = False

        self.pitch_predictor = PitchPredictor(z_dim=512, hidden=predictor_hidden)

        if predictor_checkpoint is not None:
            ckpt = torch.load(predictor_checkpoint, map_location="cpu", weights_only=False)
            self.pitch_predictor.load_state_dict(ckpt["model"])
            self._predictor_pretrained = {
                k: v.clone().detach()
                for k, v in self.pitch_predictor.state_dict().items()
            }
            del ckpt
            print(f"  Loaded pretrained predictor from {predictor_checkpoint}")
            if freeze_predictor:
                for p in self.pitch_predictor.parameters():
                    p.requires_grad = False
                print(f"  Predictor FROZEN")
            else:
                print(f"  Predictor TRAINABLE")

        self.film_shift = FilmShift(hidden=shift_hidden, n_layers=shift_layers)

    def _decode_to_wav(self, z):
        e = self.mimi._to_encoder_framerate(z)
        (e,) = self.mimi.decoder_transformer(e)
        wav = self.mimi.decoder(e)
        return wav

    def _compute_grad_direction(self, z, f0_sum):
        """Compute normalized F0 gradient direction in z-space.

        grad_z = d(sum(predictor(z)))/dz [B, 512, T_z]
        direction = grad_z / ||grad_z||_dim1
        """
        grad_z = torch.autograd.grad(
            f0_sum, z, create_graph=False, retain_graph=False
        )[0]  # [B, 512, T_z]

        # Normalize per-sample across z_dim (dim=1)
        norm = grad_z.norm(dim=1, keepdim=True) + 1e-8  # [B, 1, T_z]
        direction = grad_z / norm  # [B, 512, T_z]
        return direction, norm

    def forward(self, codes, target_f0=None):
        z = self.mimi.quantizer.decode(codes)  # [B, 512, T_z]
        z.requires_grad_(True)

        f0_current = self.pitch_predictor(z)  # [B, 1, T_z]
        f0_sum = f0_current.sum()
        f0_current_sq = f0_current.squeeze(1)  # [B, T_z]

        info = {'f0_current': f0_current_sq, 'z': z}

        if target_f0 is not None:
            direction, grad_norm = self._compute_grad_direction(z, f0_sum)
            info['grad_norm'] = grad_norm.squeeze(1).detach()

            B = min(target_f0.shape[0], f0_current_sq.shape[0])
            T = min(target_f0.shape[1], f0_current_sq.shape[2])
            target_f0_trim = target_f0[:B, :T]
            f0_curr_trim = f0_current_sq[:B, :T]

            log_delta = torch.log(target_f0_trim + 1.0) - torch.log(f0_curr_trim + 1.0)
            log_delta_t = log_delta.unsqueeze(2)  # [B, T, 1]

            alpha = self.film_shift(log_delta_t)  # [B, 1, T]

            # Broadcast alpha to full z dims: [B, 1, T] * [B, 512, T] -> [B, 512, T]
            direction_trim = direction[:B, :, :T]
            beta = alpha * direction_trim  # [B, 512, T]
            beta = beta.clamp(-self.beta_max, self.beta_max)

            z_trim = z[:B, :, :T]
            z_shifted = z_trim + beta

            info['alpha'] = alpha.detach()
            info['beta'] = beta.detach()
            info['log_delta'] = log_delta
            info['direction_scale'] = direction_trim.abs().mean().item()
            info['alpha_scale'] = alpha.abs().mean().item()
            info['z_scale'] = z_trim.abs().mean().item()
            info['beta_scale'] = beta.abs().mean().item()
        else:
            z_shifted = z

        info['z_shifted'] = z_shifted
        wav = self._decode_to_wav(z_shifted)
        return wav, info


class PitchShiftLoss(nn.Module):
    """Separate losses for baseline (reconstruction) and shifted (pitch control)."""

    def __init__(self, sample_rate=24000, mel_weight=5.0, f0_weight=5.0,
                 mfc_weight=2.0, f0_short_weight=3.0, f0_estimator=None):
        super().__init__()
        self.mel_weight = mel_weight
        self.f0_weight = f0_weight
        self.mfc_weight = mfc_weight
        self.f0_short_weight = f0_short_weight
        self.mel_loss = MultiResolutionMelLoss(sample_rate=sample_rate)
        self.mfc_loss = PitchIndependentSpectralLoss(sample_rate=sample_rate)
        self.f0_estimator = f0_estimator

    def baseline_loss(self, wav_pred, wav_target):
        mel = self.mel_loss(wav_pred, wav_target)
        total = mel * self.mel_weight
        return total, {'mel': mel.item(), 'total': total.item()}

    def shifted_loss(self, wav_pred, wav_target, target_f0, info=None):
        loss_dict = {}

        if self.f0_estimator is not None:
            log_f0_pred = self.f0_estimator(wav_pred)[0]
            f0_pred = torch.exp(log_f0_pred)

            T = min(f0_pred.shape[1], target_f0.shape[1])
            f0_pred = f0_pred[:, :T]
            target_f0_trimmed = target_f0[:, :T]

            f0_pred_log = torch.log(f0_pred + 1.0)
            target_log = torch.log(target_f0_trimmed + 1.0)

            voiced_mask = (f0_pred > 20.0).float() * (target_f0_trimmed > 20.0).float()

            if voiced_mask.sum() > 10:
                f0_loss = ((f0_pred_log - target_log) * voiced_mask).pow(2).sum() / (voiced_mask.sum() + 1)
            else:
                f0_loss = torch.tensor(0.0, device=wav_pred.device)
        else:
            f0_loss = torch.tensor(0.0, device=wav_pred.device)

        f0_short_loss = torch.tensor(0.0, device=wav_pred.device)
        if info is not None and 'f0_from_shifted' in info:
            f0_from_shifted = info['f0_from_shifted']
            T_s = min(f0_from_shifted.shape[1], target_f0.shape[1])
            f0_shifted_t = f0_from_shifted[:, :T_s]
            target_f0_t = target_f0[:, :T_s]
            f0_short_log = torch.log(f0_shifted_t + 1.0)
            target_short_log = torch.log(target_f0_t + 1.0)
            f0_short_loss = (f0_short_log - target_short_log).pow(2).mean()

        mfc = self.mfc_loss(wav_pred, wav_target)

        total = (f0_loss * self.f0_weight +
                 f0_short_loss * self.f0_short_weight +
                 mfc * self.mfc_weight)
        loss_dict['f0'] = f0_loss.item()
        loss_dict['f0_short'] = f0_short_loss.item()
        loss_dict['mfc'] = mfc.item()
        loss_dict['total'] = total.item()
        return total, loss_dict

    def forward(self, wav_pred, wav_target, info, target_f0=None):
        if target_f0 is None:
            return self.baseline_loss(wav_pred, wav_target)
        else:
            return self.shifted_loss(wav_pred, wav_target, target_f0, info=info)
