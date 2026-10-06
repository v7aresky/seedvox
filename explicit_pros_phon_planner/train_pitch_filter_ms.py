"""Multi-speaker Stage 2 training with pre-computed tokens + F0.

Supports Globe prosody codec (.pt) and/or LJ Speech prosody codec (.pt).

Usage:
    python -m explicit_pros_phon_planner.train_pitch_filter_ms \
        --globe_pt /path/to/globe.pt \
        --lj_pt /path/to/lj.pt \
        --stage1_checkpoint checkpoints/pitch_filter/pitch_filter_stage1_best.pt
"""

import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from seedvox.modules.mimi import get_mimi_model
from .pitch_filter import PitchFilterModel, PitchFilterLoss


# ---------------------------------------------------------------------------
# Shared collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    wavs, codes_list, f0s = zip(*batch)

    max_wav = max(w.shape[0] for w in wavs)
    wav_lens = torch.tensor([w.shape[0] for w in wavs])
    wavs_padded = torch.zeros(len(wavs), max_wav)
    for i, w in enumerate(wavs):
        wavs_padded[i, :w.shape[0]] = w

    max_code_t = max(c.shape[1] for c in codes_list)
    code_lens = torch.tensor([c.shape[1] for c in codes_list])
    codes_padded = torch.zeros(len(codes_list), codes_list[0].shape[0], max_code_t, dtype=codes_list[0].dtype)
    for i, c in enumerate(codes_list):
        codes_padded[i, :, :c.shape[1]] = c

    max_f0 = max(f.shape[0] for f in f0s)
    f0_padded = torch.zeros(len(f0s), max_f0)
    for i, f in enumerate(f0s):
        f0_padded[i, :f.shape[0]] = f

    return wavs_padded, codes_padded, wav_lens, code_lens, f0_padded


# ---------------------------------------------------------------------------
# Prosody codec dataset (works for both Globe and LJ Speech .pt files)
# ---------------------------------------------------------------------------

class ProsodyCodecDataset(Dataset):
    def __init__(self, pt_path, min_duration=0.5, max_duration=30.0, sample_rate=24000, name=None):
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PitchFilterTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.stage = args.stage

        print("Loading Mimi model...")
        self.mimi = get_mimi_model().to(self.device)
        self.mimi.eval()
        self.mimi.set_num_codebooks(16)

        self.model = PitchFilterModel(
            self.mimi,
            stage=self.stage,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            filter_hidden=args.filter_hidden,
            filter_layers=args.filter_layers,
        ).to(self.device)

        if args.stage == 2 and args.stage1_checkpoint:
            print(f"Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
            ckpt = torch.load(args.stage1_checkpoint, map_location='cpu', weights_only=False)
            self.model.load_state_dict(ckpt['model'], strict=False)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        total = sum(p.numel() for p in trainable)
        print(f"Trainable parameters: {total:,}")

        self.criterion = PitchFilterLoss(
            sample_rate=24000,
            stage=self.stage,
            spectral_weight=args.spectral_weight,
            mel_weight=args.mel_weight,
            delta_weight=args.delta_weight,
            pitch_weight=args.pitch_weight,
        )

        self.optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

        if args.resume_checkpoint:
            print(f"Resuming from: {args.resume_checkpoint}")
            ckpt = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
            self.model.load_state_dict(ckpt['model'], strict=False)
            if 'optimizer' in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                    print("  Optimizer state restored")
                except Exception as e:
                    print(f"  Could not restore optimizer: {e}")
            self.start_epoch = ckpt.get('epoch', 0)
            print(f"  Resuming from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs - self.start_epoch
        )
        for _ in range(self.start_epoch):
            self.scheduler.step()

        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.global_step = 0

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

        pbar = tqdm(dataloader, desc=f"Stage {self.stage} Epoch {epoch}")
        for batch in pbar:
            wavs, codes, wav_lens, code_lens, f0_hz = batch
            wavs = wavs.to(self.device)
            codes = codes.to(self.device)
            f0_hz = f0_hz.to(self.device)

            wav_target = wavs.unsqueeze(1)

            wav_pred, info = self.model(codes, target_wav=wav_target,
                                         f0_override=f0_hz if self.stage == 2 else None)

            loss, loss_dict = self.criterion(wav_pred, wav_target, info)

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
                print(f"  -> val step {self.global_step}: F0 in={f0_in:.1f}Hz out={f0_out:.1f}Hz")

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                **{k: f"{v:.4f}" for k, v in loss_dict.items()},
            })

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}")
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

    def train(self, dataloader):
        print(f"\n{'='*60}")
        print(f"Stage {self.stage} -- Pitch injection (multi-speaker)")
        print(f"{'='*60}\n")

        best_loss = float('inf')
        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
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
    parser = argparse.ArgumentParser(description="Stage 2 pitch injection (multi-speaker)")
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2])
    parser.add_argument('--globe_pt', type=str, default=None,
                        help='Path to Globe prosody codec .pt file')
    parser.add_argument('--lj_pt', type=str, default=None,
                        help='Path to LJ Speech prosody codec .pt file')
    parser.add_argument('--stage1_checkpoint', type=str, default=None)
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Resume from a Stage 2 checkpoint (loads model weights + optimizer)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/pitch_filter_ms')
    parser.add_argument('--log_dir', type=str, default='runs/pitch_filter_ms')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_duration', type=float, default=30.0)

    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--lora_alpha', type=float, default=64.0)
    parser.add_argument('--filter_hidden', type=int, default=1024)
    parser.add_argument('--filter_layers', type=int, default=4)

    parser.add_argument('--spectral_weight', type=float, default=1.0)
    parser.add_argument('--mel_weight', type=float, default=45.0)
    parser.add_argument('--delta_weight', type=float, default=1.0)
    parser.add_argument('--pitch_weight', type=float, default=50.0)
    parser.add_argument('--val_wav', type=str, default=None)
    parser.add_argument('--val_every', type=int, default=500)

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    assert args.globe_pt or args.lj_pt, "Provide at least --globe_pt or --lj_pt"
    if args.stage == 2:
        assert args.stage1_checkpoint, "Stage 2 requires --stage1_checkpoint"

    datasets = []
    if args.globe_pt:
        datasets.append(ProsodyCodecDataset(args.globe_pt, name="Globe", max_duration=args.max_duration))
    if args.lj_pt:
        datasets.append(ProsodyCodecDataset(args.lj_pt, name="LJ", max_duration=args.max_duration))

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

    trainer = PitchFilterTrainer(args)
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
