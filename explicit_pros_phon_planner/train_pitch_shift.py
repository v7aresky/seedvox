"""Train gradient-guided pitch shift model.

Usage:
    python -m explicit_pros_phon_planner.train_pitch_shift \
        --globe_pt /path/to/globe.pt \
        --lj_pt /path/to/lj.pt \
        --epochs 30 --batch_size 8
"""

import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from seedvox.modules.mimi import get_mimi_model
from .pitch_shift import PitchShiftModel, PitchShiftLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProsodyCodecDataset(Dataset):
    """Loads pre-computed tokens + F0 from .pt files."""

    def __init__(self, pt_path, min_duration=0.5, max_duration=30.0,
                 sample_rate=24000, name=None):
        self.sr = sample_rate
        self.name = name or Path(pt_path).stem
        self.min_samples = int(min_duration * sample_rate)
        self.max_samples = int(max_duration * sample_rate)

        print(f"Loading {self.name}: {pt_path}")
        ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
        self.data = ckpt['data']

        self.data = [d for d in self.data
                     if self.min_samples <= int(d['dur_sec'] * self.sr) <= self.max_samples]
        print(f"  {self.name}: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        wav_path = d['wav_path']
        if not Path(wav_path).exists():
            wav_path = wav_path.replace('.vad', '.vad.wav')
        wav, sr = torchaudio.load(wav_path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)

        codes = d['audio_tokens'].squeeze(0)  # [16, T]

        log_f0 = d['log_f0_center'] + d['mu_logF0']
        f0_hz = torch.exp(log_f0)

        return wav, codes, f0_hz


def collate_fn(batch):
    wavs, codes_list, f0s = zip(*batch)

    max_wav = max(w.shape[0] for w in wavs)
    wav_lens = torch.tensor([w.shape[0] for w in wavs])
    wavs_padded = torch.zeros(len(wavs), max_wav)
    for i, w in enumerate(wavs):
        wavs_padded[i, :w.shape[0]] = w

    max_code_t = max(c.shape[1] for c in codes_list)
    code_lens = torch.tensor([c.shape[1] for c in codes_list])
    codes_padded = torch.zeros(len(codes_list), codes_list[0].shape[0], max_code_t,
                               dtype=codes_list[0].dtype)
    for i, c in enumerate(codes_list):
        codes_padded[i, :, :c.shape[1]] = c

    max_f0 = max(f.shape[0] for f in f0s)
    f0_padded = torch.zeros(len(f0s), max_f0)
    for i, f in enumerate(f0s):
        f0_padded[i, :f.shape[0]] = f

    return wavs_padded, codes_padded, wav_lens, code_lens, f0_padded


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PitchShiftTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        print("Loading Mimi model...")
        self.mimi = get_mimi_model().to(self.device)
        self.mimi.eval()
        self.mimi.set_num_codebooks(16)

        self.model = PitchShiftModel(
            self.mimi,
            predictor_hidden=args.predictor_hidden,
            shift_hidden=args.shift_hidden,
            shift_layers=args.shift_layers,
            beta_max=args.beta_max,
            predictor_checkpoint=args.predictor_checkpoint,
            freeze_predictor=args.freeze_predictor,
        ).to(self.device)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        total = sum(p.numel() for p in trainable)
        print(f"Trainable parameters: {total:,}")

        self.criterion = PitchShiftLoss(
            sample_rate=24000,
            mel_weight=args.mel_weight,
            f0_weight=args.f0_weight,
            mfc_weight=args.mfc_weight,
            f0_short_weight=args.f0_short_weight,
            f0_estimator=self.model.f0_estimator,
        )

        self.optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )

        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.global_step = 0

        # Validation
        self.val_codes = None
        self.val_wav_gt = None
        self.val_f0 = None
        val_wav_path = getattr(args, 'val_wav', None)
        if val_wav_path and Path(val_wav_path).exists():
            wav, sr = torchaudio.load(val_wav_path)
            if sr != 24000:
                wav = torchaudio.transforms.Resample(sr, 24000)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            self.val_wav_gt = wav.to(self.device)
            with torch.no_grad():
                self.val_codes = self.mimi.encode(wav.unsqueeze(0).to(self.device))[0]
                self.val_f0 = torch.exp(
                    self.model.f0_estimator(
                        wav.unsqueeze(0).to(self.device)
                    )[0].squeeze(0).detach()
                )
            val_out = Path(args.output_dir) / "val"
            val_out.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(val_out / "ground_truth.wav"), wav.cpu(), 24000)
            print(f"  Validation wav: {val_wav_path} ({wav.shape[1]/24000:.2f}s)")
            print(f"  Validation F0: mean={self.val_f0.mean().item():.1f} Hz")
        self.val_every = getattr(args, 'val_every', 500)

    def _random_f0_shift(self, f0):
        """Random F0 shift for training: scale by 0.5x-2.0x."""
        scale = torch.empty(f0.shape[0]).uniform_(0.5, 2.0).to(f0.device)
        return f0 * scale.unsqueeze(1)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        self.model.mimi.eval()
        self.model.f0_estimator.eval()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            wavs, codes, wav_lens, code_lens, f0_hz = batch
            wavs = wavs.to(self.device)
            codes = codes.to(self.device)
            f0_hz = f0_hz.to(self.device)

            wav_target = wavs.unsqueeze(1)
            self.optimizer.zero_grad()

            # Pass 1: Baseline (no shift) — mel reconstruction only
            wav_pred_base, info_base = self.model(codes, target_f0=None)
            min_len_b = min(wav_pred_base.shape[2], wav_target.shape[2])
            loss_base, dict_base = self.criterion.baseline_loss(
                wav_pred_base[:, :, :min_len_b],
                wav_target[:, :, :min_len_b],
            )

            # Pass 2: Shifted — alpha prediction loss + MFCC timbre
            target_f0 = self._random_f0_shift(f0_hz)
            wav_pred_shift, info_shift = self.model(codes, target_f0=target_f0)

            # --- Alpha target via inner optimization on predictor ---
            z_det = info_shift['z'].detach()
            direction_det = info_shift.get('beta', torch.zeros_like(z_det)).detach()
            # Recompute direction from z (detached) for inner opt
            z_inner = z_det.clone().requires_grad_(True)
            f0_inner_init = self.model.pitch_predictor(z_inner)
            grad_z = torch.autograd.grad(f0_inner_init.sum(), z_inner, create_graph=False)[0]
            direction_det = grad_z / (grad_z.norm(dim=1, keepdim=True) + 1e-8)

            B, _, T_z = z_det.shape
            T = min(target_f0.shape[1], T_z)
            target_f0_trim = target_f0[:, :T]
            z_inner = z_det[:, :, :T]
            direction_trim = direction_det[:, :, :T]

            # Inner optimization: find alpha that makes predictor(z + alpha*direction) = target_f0
            alpha_inner = torch.zeros(B, 1, T, device=self.device, requires_grad=True)
            opt_inner = torch.optim.SGD([alpha_inner], lr=self.args.beta_pretrain_lr)
            for _ in range(self.args.beta_pretrain_steps):
                z_shift_inner = z_inner + alpha_inner * direction_trim
                f0_inner = self.model.pitch_predictor(z_shift_inner).squeeze(1)
                loss_inner = F.mse_loss(f0_inner, target_f0_trim)
                opt_inner.zero_grad()
                loss_inner.backward()
                opt_inner.step()
            alpha_target = alpha_inner.detach()  # [B, 1, T]

            # --- FilmShift prediction + alpha prediction loss ---
            log_delta = info_shift['log_delta'][:, :T]
            log_delta_t = log_delta.unsqueeze(2)  # [B, T, 1]
            alpha_pred = self.model.film_shift(log_delta_t)  # [B, 1, T]
            alpha_pred_loss = F.mse_loss(alpha_pred, alpha_target) * self.args.beta_weight

            # --- MFCC timbre preservation (small weight) ---
            min_len_s = min(wav_pred_shift.shape[2], wav_target.shape[2])
            mfc = self.criterion.mfc_loss(
                wav_pred_shift[:, :, :min_len_s],
                wav_target[:, :, :min_len_s],
            )
            mfc_loss = mfc * self.args.mfc_weight

            loss = loss_base + alpha_pred_loss + mfc_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            for k, v in dict_base.items():
                self.writer.add_scalar(f'train/base_{k}', v, self.global_step)
            self.writer.add_scalar('train/alpha_pred', alpha_pred_loss.item(), self.global_step)
            self.writer.add_scalar('train/mfc', mfc.item(), self.global_step)
            self.writer.add_scalar('train/total', loss.item(), self.global_step)

            self.writer.add_scalar('diag/alpha_scale', info_shift.get('alpha_scale', 0), self.global_step)
            self.writer.add_scalar('diag/beta_scale', info_shift.get('beta_scale', 0), self.global_step)
            self.writer.add_scalar('diag/alpha_target_scale', alpha_target.abs().mean().item(), self.global_step)
            self.global_step += 1

            if (self.val_codes is not None and
                    self.global_step % self.val_every == 0):
                self._validate()
                self.model.mimi.eval()
                self.model.f0_estimator.eval()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mel': f"{dict_base.get('mel', 0):.4f}",
                'apred': f"{alpha_pred_loss.item():.4f}",
                'mfc': f"{mfc.item():.4f}",
                'atgt': f"{alpha_target.abs().mean().item():.4f}",
            })

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}")
        self.scheduler.step()
        return avg_loss

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        f0_est = self.model.f0_estimator

        def _measure_f0(wav):
            log_f0 = f0_est(wav.unsqueeze(0))[0].squeeze(0)
            f0_hz = torch.exp(log_f0)
            voiced = f0_hz[f0_hz > 20.0]
            if voiced.numel() > 0:
                return voiced.mean().item()
            return f0_hz.mean().item()

        # Baseline: no shift
        val_pred, val_info = self.model(self.val_codes.unsqueeze(0))
        val_path = Path(self.args.output_dir) / "val" / f"step_{self.global_step}_baseline.wav"
        torchaudio.save(str(val_path), val_pred.squeeze(0).cpu(), 24000)
        f0_base_actual = _measure_f0(val_pred.squeeze(0))

        # Shift up 1.5x
        target_f0_up = self.val_f0 * 1.5
        val_pred_up, info_up = self.model(
            self.val_codes.unsqueeze(0), target_f0=target_f0_up.unsqueeze(0)
        )
        val_path_up = Path(self.args.output_dir) / "val" / f"step_{self.global_step}_up.wav"
        torchaudio.save(str(val_path_up), val_pred_up.squeeze(0).cpu(), 24000)
        f0_up_actual = _measure_f0(val_pred_up.squeeze(0))
        alpha_up = info_up.get('alpha', torch.zeros(1)).abs().mean().item()
        beta_up = info_up.get('beta', torch.zeros(1)).abs().mean().item()
        grad_up = info_up.get('grad_norm', torch.zeros(1)).mean().item()

        # Shift down 0.5x
        target_f0_down = self.val_f0 * 0.5
        val_pred_down, info_down = self.model(
            self.val_codes.unsqueeze(0), target_f0=target_f0_down.unsqueeze(0)
        )
        val_path_down = Path(self.args.output_dir) / "val" / f"step_{self.global_step}_down.wav"
        torchaudio.save(str(val_path_down), val_pred_down.squeeze(0).cpu(), 24000)
        f0_down_actual = _measure_f0(val_pred_down.squeeze(0))
        alpha_down = info_down.get('alpha', torch.zeros(1)).abs().mean().item()
        beta_down = info_down.get('beta', torch.zeros(1)).abs().mean().item()
        grad_down = info_down.get('grad_norm', torch.zeros(1)).mean().item()

        target_up = target_f0_up.mean().item()
        target_down = target_f0_down.mean().item()
        print(f"  -> val step {self.global_step}: "
              f"baseline={f0_base_actual:.1f}Hz, "
              f"up={f0_up_actual:.1f}Hz (tgt {target_up:.1f}, alpha={alpha_up:.4f}, beta={beta_up:.4f}, grad={grad_up:.2f}), "
              f"down={f0_down_actual:.1f}Hz (tgt {target_down:.1f}, alpha={alpha_down:.4f}, beta={beta_down:.4f}, grad={grad_down:.2f})")

        self.model.train()

    def save_checkpoint(self, epoch, loss):
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'args': vars(self.args),
        }
        path = Path(self.args.output_dir) / f"pitch_shift_epoch{epoch}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)

    def train(self, dataloader):
        print(f"\n{'='*60}")
        print(f"Gradient-guided pitch shift training")
        print(f"{'='*60}\n")

        best_loss = float('inf')
        for epoch in range(1, self.args.epochs + 1):
            loss = self.train_epoch(dataloader, epoch)
            self.save_checkpoint(epoch, loss)
            if loss < best_loss:
                best_loss = loss
                best_path = Path(self.args.output_dir) / "pitch_shift_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'args': vars(self.args),
                }, best_path)

        self.writer.close()
        print(f"\nTraining complete. Best loss: {best_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gradient-guided pitch shift for Mimi decoder"
    )
    parser.add_argument('--globe_pt', type=str, default=None)
    parser.add_argument('--lj_pt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints/pitch_shift')
    parser.add_argument('--log_dir', type=str, default='runs/pitch_shift')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_duration', type=float, default=30.0)

    # Model capacity
    parser.add_argument('--predictor_hidden', type=int, default=128)
    parser.add_argument('--shift_hidden', type=int, default=128)
    parser.add_argument('--shift_layers', type=int, default=3)
    parser.add_argument('--beta_max', type=float, default=2.0)
    parser.add_argument('--predictor_checkpoint', type=str, default=None)
    parser.add_argument('--freeze_predictor', action='store_true')

    # Inner optimization
    parser.add_argument('--beta_pretrain_steps', type=int, default=5,
                        help='Inner optimization steps to compute alpha targets')
    parser.add_argument('--beta_pretrain_lr', type=float, default=0.5,
                        help='Learning rate for inner optimization')
    parser.add_argument('--beta_weight', type=float, default=20.0,
                        help='Weight for alpha prediction loss')

    # Loss
    parser.add_argument('--f0_weight', type=float, default=5.0)
    parser.add_argument('--f0_short_weight', type=float, default=3.0)
    parser.add_argument('--mel_weight', type=float, default=5.0)
    parser.add_argument('--mfc_weight', type=float, default=0.1)

    # Validation
    parser.add_argument('--val_wav', type=str, default=None)
    parser.add_argument('--val_every', type=int, default=500)

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    assert args.globe_pt or args.lj_pt, "Provide at least --globe_pt or --lj_pt"

    datasets = []
    if args.globe_pt:
        datasets.append(ProsodyCodecDataset(args.globe_pt, name="Globe",
                                            max_duration=args.max_duration))
    if args.lj_pt:
        datasets.append(ProsodyCodecDataset(args.lj_pt, name="LJ",
                                            max_duration=args.max_duration))

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)
        print(f"  Combined: {len(dataset)} total samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    trainer = PitchShiftTrainer(args)
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
