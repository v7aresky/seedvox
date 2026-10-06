"""Train pitch filter (Stage 1) and pitch injection (Stage 2).

Stage 1: Train PitchFilter to suppress pitch in z, decoder LoRA to reconstruct.
Stage 2: Freeze filter, add PitchEmbedding, train new decoder LoRA.

Usage:
    # Stage 1 — train filter + decoder
    python -m explicit_pros_phon_planner.train_pitch_filter \
        --stage 1 --data_dir ../autovoc/dataset/wavs \
        --epochs 50 --batch_size 4 --lr 1e-4 \
        --output_dir checkpoints/pitch_filter \
        --val_wav ../autovoc/dataset/wavs/LJ001-0001.wav --val_every 500

    # Stage 2 — inject pitch
    python -m explicit_pros_phon_planner.train_pitch_filter \
        --stage 2 --data_dir ../autovoc/dataset/wavs \
        --stage1_checkpoint checkpoints/pitch_filter/pitch_filter_stage1_best.pt \
        --epochs 50 --batch_size 4 --lr 1e-4 \
        --output_dir checkpoints/pitch_filter \
        --val_wav ../autovoc/dataset/wavs/LJ001-0001.wav --val_every 500
"""

import argparse
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from seedvox.modules.mimi import get_mimi_model
from .pitch_filter import PitchFilterModel, PitchFilterLoss
from .train_pitch_control import WaveformDataset, collate_fn


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PitchFilterTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.stage = args.stage

        # Load Mimi
        print("Loading Mimi model...")
        self.mimi = get_mimi_model().to(self.device)
        self.mimi.eval()
        self.mimi.set_num_codebooks(16)

        # Build model
        self.model = PitchFilterModel(
            self.mimi,
            stage=self.stage,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            filter_hidden=args.filter_hidden,
            filter_layers=args.filter_layers,
        ).to(self.device)

        # Load checkpoint
        if args.stage == 2 and args.stage1_checkpoint:
            print(f"Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
            ckpt = torch.load(args.stage1_checkpoint, map_location='cpu')
            self.model.load_state_dict(ckpt['model'], strict=False)

        # Count trainable
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        total = sum(p.numel() for p in trainable)
        print(f"Trainable parameters: {total:,}")

        # Loss
        self.criterion = PitchFilterLoss(
            sample_rate=24000,
            stage=self.stage,
            spectral_weight=args.spectral_weight,
            mel_weight=args.mel_weight,
            delta_weight=args.delta_weight,
            pitch_weight=args.pitch_weight,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )

        # Logging
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.global_step = 0

        # Validation vocoding
        self.val_codes = None
        self.val_wav_gt = None
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
            val_out = Path(args.output_dir) / "val"
            val_out.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(val_out / "ground_truth.wav"), wav.cpu(), 24000)
            print(f"  Validation wav: {val_wav_path} ({wav.shape[1]/24000:.2f}s)")
        self.val_every = getattr(args, 'val_every', 500)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        n_batches = 0
        epoch_totals = {}

        pbar = tqdm(dataloader, desc=f"Stage {self.stage} Epoch {epoch}")
        for wav_true, codes, wav_lens, code_lens in pbar:
            wav_true = wav_true.to(self.device)
            codes = codes.to(self.device)

            wav_target = wav_true.unsqueeze(1)

            wav_pred, info = self.model(codes, target_wav=wav_target)

            loss, loss_dict = self.criterion(
                wav_pred, wav_target, info
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            for k, v in loss_dict.items():
                self.writer.add_scalar(f'train/{k}', v, self.global_step)
            self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            self.global_step += 1

            # Validation vocoding
            if self.val_codes is not None and self.global_step % self.val_every == 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred, val_info = self.model(
                        self.val_codes.unsqueeze(0).to(self.device),
                        target_wav=self.val_wav_gt,
                    )
                    f0_out_wav = self.model._extract_f0(val_pred)
                val_path = Path(self.args.output_dir) / "val" / f"step_{self.global_step}.wav"
                torchaudio.save(str(val_path), val_pred.squeeze(0).cpu(), 24000)
                self.model.train()
                f0_in = val_info['f0_target'][0].mean().item()
                f0_out = f0_out_wav[0].mean().item()
                print(f"  → val step {self.global_step}: F0 in={f0_in:.1f}Hz out={f0_out:.1f}Hz")

            # Accumulate for epoch summary
            for k, v in loss_dict.items():
                if k not in epoch_totals:
                    epoch_totals[k] = 0.0
                epoch_totals[k] += v

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                **{k: f"{v:.4f}" for k, v in loss_dict.items()},
            })

        avg_loss = total_loss / max(n_batches, 1)
        parts = " ".join(f"{k}={v/max(n_batches,1):.4f}" for k, v in epoch_totals.items())
        print(f"  Epoch {epoch}: loss={avg_loss:.4f} {parts}")
        self.scheduler.step()
        return avg_loss

    def save_checkpoint(self, epoch, loss):
        ckpt = {
            'epoch': epoch,
            'stage': self.stage,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'args': vars(self.args),
        }
        path = Path(self.args.output_dir) / f"pitch_filter_stage{self.stage}_epoch{epoch}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        print(f"  Saved checkpoint: {path}")

    def train(self, dataloader):
        print(f"\n{'='*60}")
        print(f"Stage {self.stage} — {'Pitch filter' if self.stage == 1 else 'Pitch injection'}")
        print(f"{'='*60}\n")

        best_loss = float('inf')
        for epoch in range(1, self.args.epochs + 1):
            loss = self.train_epoch(dataloader, epoch)
            self.save_checkpoint(epoch, loss)
            if loss < best_loss:
                best_loss = loss
                best_path = Path(self.args.output_dir) / f"pitch_filter_stage{self.stage}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'stage': self.stage,
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
        description="Train pitch filter for Mimi decoder"
    )
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2],
                        help='1=pitch filter, 2=pitch injection')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--stage1_checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints/pitch_filter')
    parser.add_argument('--log_dir', type=str, default='runs/pitch_filter')
    parser.add_argument('--cache_dir', type=str, default=None)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_duration', type=float, default=30.0)

    # Model
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--lora_alpha', type=float, default=64.0)
    parser.add_argument('--filter_hidden', type=int, default=1024)
    parser.add_argument('--filter_layers', type=int, default=4)

    # Loss
    parser.add_argument('--spectral_weight', type=float, default=1.0,
                        help='MFCC spectral loss weight (Stage 1 only)')
    parser.add_argument('--mel_weight', type=float, default=45.0,
                        help='Full mel loss weight (Stage 2 only)')
    parser.add_argument('--delta_weight', type=float, default=1.0,
                        help='L2 penalty on filter magnitude')
    parser.add_argument('--pitch_weight', type=float, default=50.0,
                        help='Per-frame F0 L1 weight (Stage 1 only)')

    # Validation
    parser.add_argument('--val_wav', type=str, default=None)
    parser.add_argument('--val_every', type=int, default=500)

    # Device
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Build dataset
    print("Loading Mimi for encoding...")
    mimi = get_mimi_model()
    mimi.eval()
    mimi.set_num_codebooks(16)
    for p in mimi.parameters():
        p.requires_grad = False

    dataset = WaveformDataset(
        args.data_dir, mimi,
        sample_rate=24000,
        max_duration=args.max_duration,
        cache_dir=args.cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Train
    trainer = PitchFilterTrainer(args)
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
