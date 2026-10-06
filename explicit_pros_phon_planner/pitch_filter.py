"""Code-space pitch filter for Mimi decoder.

Stage 1 (pitch filter):
    z [B, 512, T] → filter → z' [B, 512, T]
    z' → decoder → wav
    Trained with: L_mfcc + λ_delta · ||z'-z||² + λ_pitch · L_periodicity(wav')
    Pitch loss uses autocorrelation pitch strength (fixed DSP, not a neural net —
    cannot be fooled by adversarial perturbations).

Stage 2 (pitch injection):
    z' + pitch_emb(F0) → decoder → wav with desired pitch
    Filter frozen, new LoRA decoder trained to use explicit pitch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .f0_estimator import F0Estimator
from .finetune_lora_fusion import inject_lora
from .losses import PitchIndependentSpectralLoss, MultiResolutionMelLoss


# ---------------------------------------------------------------------------
# Autocorrelation pitch strength (fixed DSP — cannot be gamed by a neural filter)
# ---------------------------------------------------------------------------

class AutocorrelationPitchStrength(nn.Module):
    """Differentiable pitch-strength measure via autocorrelation.

    Measures how periodic a waveform is by computing the normalized
    autocorrelation peak in the pitch period range (50–500 Hz).
    This is a fixed signal-processing operation — no learned parameters —
    so a neural filter cannot find adversarial perturbations to minimize it
    without actually reducing periodicity.
    """

    def __init__(self, sr=24000, fmin=50.0, fmax=500.0):
        super().__init__()
        self.sr = sr
        self.lag_min = int(sr / fmax)  # min lag in samples
        self.lag_max = int(sr / fmin)  # max lag in samples

    def forward(self, wav):
        """wav [B, 1, N] → scalar pitch strength in [0, 1]."""
        x = wav.squeeze(1)  # [B, N]
        B, N = x.shape
        # Normalize per-utterance
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-8)

        # Full autocorrelation via FFT (differentiable)
        n_fft = 1
        while n_fft < N:
            n_fft *= 2
        X = torch.fft.rfft(x, n=n_fft)          # [B, n_fft//2+1]
        acf = torch.fft.irfft(X * X.conj())      # [B, n_fft]
        # Normalize by zero-lag value
        acf = acf / (acf[:, :1].abs() + 1e-8)

        # Extract peak in pitch period range
        lags = acf[:, self.lag_min:self.lag_max]  # [B, lag_range]
        pitch_strength = lags.max(dim=1).values    # [B]
        return pitch_strength.mean()               # scalar


# ---------------------------------------------------------------------------
# Pitch filter: lightweight residual MLP on z
# ---------------------------------------------------------------------------

class PitchFilter(nn.Module):
    """Per-frame residual MLP that suppresses pitch in the quantizer latent z.

    Operates on z [B, C, T] where C=512 (Mimi quantizer dimension).
    Uses residual connections to ensure the output is a small modification
    of the input — only suppressing what's necessary for pitch removal.
    """

    def __init__(self, dim=512, hidden=1024, n_layers=4, dropout=0.1):
        super().__init__()
        self.dim = dim

        layers = []
        for i in range(n_layers):
            in_d = dim if i == 0 else hidden
            out_d = dim if i == n_layers - 1 else hidden
            layers.append(nn.Conv1d(in_d, out_d, kernel_size=1))
            if i < n_layers - 1:
                layers.append(nn.GroupNorm(1, out_d))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

        # Initialize last layer to zero for identity residual at init
        nn.init.zeros_(self.net[-4].weight)  # last Conv1d
        nn.init.zeros_(self.net[-4].bias)

    def forward(self, z):
        """z [B, C, T] → z' [B, C, T]"""
        delta = self.net(z)
        return z + delta


# ---------------------------------------------------------------------------
# Pitch-controllable model wrapping the filter
# ---------------------------------------------------------------------------

class PitchFilterModel(nn.Module):
    """Wraps frozen Mimi + pitch filter + optional pitch embedding.

    Stage 1: filter + decoder LoRA
    Stage 2: frozen filter + pitch embedding + new decoder LoRA
    """

    def __init__(self, mimi, stage=1, lora_rank=32, lora_alpha=64,
                 filter_hidden=1024, filter_layers=4,
                 f0_checkpoint="checkpoints/f0_estimator_best.pt"):
        super().__init__()
        self.mimi = mimi
        self.stage = stage

        # Freeze everything in Mimi
        for p in mimi.parameters():
            p.requires_grad = False

        # Frozen F0 estimator — load pretrained weights
        self.f0_estimator = F0Estimator()
        ckpt = torch.load(f0_checkpoint, map_location="cpu", weights_only=False)
        self.f0_estimator.load_state_dict(ckpt["model"])
        del ckpt
        for p in self.f0_estimator.parameters():
            p.requires_grad = False

        # ── Stage 1: pitch filter only, decoder FROZEN (no LoRA) ──
        if stage == 1:
            self.pitch_filter = PitchFilter(
                dim=512, hidden=filter_hidden, n_layers=filter_layers
            )

        # ── Stage 2: frozen filter + pitch concat adapter + pitch predictor ──
        elif stage == 2:
            self.pitch_filter = PitchFilter(
                dim=512, hidden=filter_hidden, n_layers=filter_layers
            )
            for p in self.pitch_filter.parameters():
                p.requires_grad = False

            # Concat F0 with z, project back to 512
            self.pitch_adapter = nn.Sequential(
                nn.Conv1d(513, 512, 1),
                nn.SiLU(),
                nn.Conv1d(512, 512, 1),
            )

            # Predict F0 from codes (for internal prediction / inference without target)
            self.pitch_predictor = nn.Sequential(
                nn.Conv1d(16, 128, 3, padding=1),
                nn.SiLU(),
                nn.Conv1d(128, 128, 3, padding=1),
                nn.SiLU(),
                nn.Conv1d(128, 1, 1),
            )

            inject_lora(mimi, rank=lora_rank, alpha=lora_alpha,
                        targets=["decoder_transformer", "decoder"])

    def train(self, mode=True):
        super().train(mode)
        self.f0_estimator.eval()
        return self

    def eval(self):
        super().eval()
        self.f0_estimator.eval()
        return self

    def _extract_f0(self, wav):
        """wav [B, 1, N] → f0 [B, T_f0] in Hz.
        No torch.no_grad() — gradients flow through F0 estimator for f0_match loss.
        F0 estimator params are frozen (not in optimizer) so they won't update."""
        log_f0, _ = self.f0_estimator(wav)
        f0 = torch.exp(log_f0)
        return f0

    def _decode_to_wav(self, z):
        """z [B, 512, T_rvq] → wav [B, 1, N]."""
        e = self.mimi._to_encoder_framerate(z)
        (e,) = self.mimi.decoder_transformer(e)
        return self.mimi.decoder(e)

    def forward(self, codes, target_wav=None, f0_override=None, predict_pitch=False):
        """
        Args:
            codes: [B, K, T] RVQ codes
            target_wav: [B, 1, N] ground truth (for F0 extraction)
            f0_override: [B, T_f0] F0 in Hz to inject directly
            predict_pitch: if True, use pitch_predictor instead of extracting from target_wav

        Returns:
            wav: [B, 1, N] reconstructed waveform
            info: dict with loss components
        """
        # Quantizer decode
        z = self.mimi.quantizer.decode(codes)  # [B, 512, T_rvq]

        # Apply pitch filter
        z_filtered = self.pitch_filter(z)  # [B, 512, T_rvq]

        # Stage 2: concatenate F0 with z and predict pitch
        if self.stage == 2:
            # Get F0: from override, target_wav, or predictor
            if f0_override is not None:
                f0 = f0_override
            elif predict_pitch:
                f0 = self.pitch_predictor(codes.float())[:, 0]  # [B, T_pred]
                f0 = torch.clamp(f0, min=1.0)  # ensure positive Hz
            elif target_wav is not None:
                f0 = self._extract_f0(target_wav)
            else:
                f0 = None

            if f0 is not None:
                # Upsample F0 to match z_filtered temporal dim
                f0_2d = f0.unsqueeze(1)  # [B, 1, T_f0]
                f0_up = F.interpolate(f0_2d, size=z_filtered.shape[2], mode='linear', align_corners=False)
                f0_up = f0_up.squeeze(1)  # [B, T_rvq]

                # Concatenate F0 with z_filtered and project
                z_in = torch.cat([z_filtered, f0_up.unsqueeze(1)], dim=1)  # [B, 513, T_rvq]
                z_filtered = self.pitch_adapter(z_in)  # [B, 512, T_rvq]

        # Decode to waveform
        wav = self._decode_to_wav(z_filtered)

        info = {'wav': wav, 'z': z, 'z_filtered': z_filtered}

        if target_wav is not None:
            info['f0_target'] = self._extract_f0(target_wav)

        # Store injected F0 and output F0 for matching loss
        if self.stage == 2:
            if f0 is not None:
                info['f0_injected'] = f0
                info['f0_output'] = self._extract_f0(wav)

        if self.stage == 2 and hasattr(self, 'pitch_predictor'):
            f0_pred = self.pitch_predictor(codes.float())[:, 0]  # [B, T_pred]
            info['f0_predicted'] = torch.clamp(f0_pred, min=1.0)

        return wav, info


# ---------------------------------------------------------------------------
# Loss for pitch filter training
# ---------------------------------------------------------------------------

class PitchFilterLoss(nn.Module):
    """Pitch filter training loss.

    Stage 1: MFCC reconstruction + delta regularization + autocorrelation pitch strength
    Stage 2: full mel reconstruction (including pitch)
    """

    def __init__(self, sample_rate=24000, stage=1, spectral_weight=1.0,
                 mel_weight=45.0, delta_weight=1.0, pitch_weight=10.0):
        super().__init__()
        self.stage = stage
        self.spectral_loss = PitchIndependentSpectralLoss(sample_rate=sample_rate)
        self.mel_loss = MultiResolutionMelLoss(sample_rate=sample_rate)
        self.pitch_strength = AutocorrelationPitchStrength(sr=sample_rate)
        self.spectral_weight = spectral_weight
        self.mel_weight = mel_weight
        self.delta_weight = delta_weight
        self.pitch_weight = pitch_weight

    def forward(self, wav_pred, wav_true, info):
        """Returns (total_loss, loss_dict)."""
        min_t = min(wav_pred.shape[-1], wav_true.shape[-1])
        wav_pred_t = wav_pred[..., :min_t]
        wav_true_t = wav_true[..., :min_t]

        loss_dict = {}
        total = torch.tensor(0.0, device=wav_pred.device)

        if self.stage == 1:
            spectral_l = self.spectral_loss(wav_pred_t, wav_true_t)
            total = total + self.spectral_weight * spectral_l
            loss_dict['spectral'] = spectral_l.item()

            delta_l = (info['z_filtered'] - info['z']).pow(2).mean()
            total = total + self.delta_weight * delta_l
            loss_dict['delta'] = delta_l.item()

            # Pitch strength via autocorrelation (fixed DSP, cannot be gamed)
            ps_pred = self.pitch_strength(wav_pred_t)
            ps_true = self.pitch_strength(wav_true_t)
            # Minimize predicted pitch strength (flatten to zero pitch)
            pitch_l = ps_pred
            total = total + self.pitch_weight * pitch_l
            loss_dict['pitch'] = pitch_l.item()
            loss_dict['ps_pred'] = ps_pred.item()
            loss_dict['ps_true'] = ps_true.item()

        elif self.stage == 2:
            mel_l = self.mel_loss(wav_pred_t, wav_true_t)
            total = total + self.mel_weight * mel_l
            loss_dict['mel'] = mel_l.item()

            # F0 matching loss: force output pitch to follow injected F0
            if 'f0_target' in info and 'f0_injected' in info:
                f0_out = info['f0_output']     # [B, T_f0] extracted from predicted wav
                f0_inj = info['f0_injected']   # [B, T_f0] the F0 we injected
                T_f0 = min(f0_out.shape[1], f0_inj.shape[1])
                log_out = torch.log(f0_out[:, :T_f0].clamp(min=1.0))
                log_inj = torch.log(f0_inj[:, :T_f0].clamp(min=1.0))
                f0_match = (log_out - log_inj).pow(2).mean()
                total = total + 2.0 * f0_match
                loss_dict['f0_match'] = f0_match.item()

            # F0 prediction loss (trains pitch_predictor for inference without target)
            if 'f0_predicted' in info and 'f0_target' in info:
                f0_pred = info['f0_predicted']
                f0_tgt = info['f0_target']
                T_f0 = min(f0_pred.shape[1], f0_tgt.shape[1])
                log_pred = torch.log(f0_pred[:, :T_f0].clamp(min=1.0))
                log_tgt = torch.log(f0_tgt[:, :T_f0].clamp(min=1.0))
                f0_mse = (log_pred - log_tgt).pow(2).mean()
                total = total + 0.5 * f0_mse
                loss_dict['f0_pred'] = f0_mse.item()

        return total, loss_dict
