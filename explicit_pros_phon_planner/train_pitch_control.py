"""Two-stage training for pitch-controllable Mimi decoder.

Stage 1 — Pitch removal:
    Train LoRA-adapted decoder to reconstruct speech while a gradient-reversed
    pitch discriminator cannot predict F0 from hidden states.

Stage 2 — Pitch re-injection:
    Freeze Stage 1 LoRA. Add pitch embedding to RVQ latent. Train new LoRA
    so the decoder learns to use the explicit pitch signal.

Usage:
    # Stage 1: remove implicit pitch from decoder
    python -m explicit_pros_phon_planner.train_pitch_control \
        --stage 1 \
        --data_dir /path/to/wavs \
        --checkpoint checkpoints/mimi_pretrained.pt \
        --epochs 50 --lr 1e-4

    # Stage 2: add explicit pitch control
    python -m explicit_pros_phon_planner.train_pitch_control \
        --stage 2 \
        --data_dir /path/to/wavs \
        --stage1_checkpoint checkpoints/pitch_control_stage1.pt \
        --epochs 50 --lr 5e-5
"""

import os
import json
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from seedvox.modules.mimi import get_mimi_model
from .mimi_pitch_control import PitchControllableMimi, PitchControlLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WaveformDataset(Dataset):
    """Loads wav files and returns (wav, codes) pairs.

    Codes are pre-computed once and cached to disk, then loaded from cache
    on subsequent runs.  First run encodes everything through the frozen Mimi
    encoder (slow); all later runs just load the cached tensors (fast).
    """

    def __init__(self, data_dir, mimi, sample_rate=24000, min_duration=0.5,
                 max_duration=30.0, cache_dir=None):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.min_samples = int(min_duration * sample_rate)
        self.max_samples = int(max_duration * sample_rate)
        self.mimi = mimi

        # Cache path
        if cache_dir is None:
            cache_dir = Path(data_dir) / ".cache_codes"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Find all wav files
        self.wav_files = sorted(
            list(self.data_dir.glob("**/*.wav")) +
            list(self.data_dir.glob("**/*.flac"))
        )
        print(f"Found {len(self.wav_files)} audio files")

        # Filter out files that are too short based on duration hint
        self.wav_files = [f for f in self.wav_files
                          if self._estimate_duration(f) >= min_duration]
        print(f"After filtering (<{min_duration}s): {len(self.wav_files)} files")

        # Check if cache is valid
        manifest_path = self.cache_dir / "manifest.pt"
        if manifest_path.exists():
            manifest = torch.load(manifest_path, weights_only=False)
            cached_names = set(manifest.get("names", []))
            current_names = [f.name for f in self.wav_files]
            if cached_names == set(current_names):
                print(f"Using cached codes from {self.cache_dir}")
                self.codes_cache = manifest["codes"]      # [N, K, max_T] tensor
                self.wav_lens_cache = manifest["wav_lens"] # [N] tensor
                return

        # Pre-compute codes
        print(f"Pre-computing RVQ codes (this may take a while)...")
        self._precompute_codes()

    def _estimate_duration(self, wav_path):
        """Quick duration estimate without loading full audio."""
        try:
            import soundfile as sf
            info = sf.info(str(wav_path))
            return info.duration
        except Exception:
            return 0.0

    def _precompute_codes(self):
        """Encode all wav files and cache codes + metadata."""
        self.mimi.eval()
        device = next(self.mimi.parameters()).device

        # Step 1: Parallel load all wavs (I/O-bound → threads help)
        print("  Loading wavs (parallel)...")
        resamplers = {}

        def _load_one(wav_path):
            try:
                wav, sr = torchaudio.load(str(wav_path))
            except Exception:
                return None
            if sr != self.sample_rate:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
                wav = resamplers[sr](wav)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if wav.shape[1] < self.min_samples:
                return None
            if wav.shape[1] > self.max_samples:
                start = torch.randint(0, wav.shape[1] - self.max_samples, (1,)).item()
                wav = wav[:, start:start + self.max_samples]
            return wav.squeeze(0)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        all_wavs = [None] * len(self.wav_files)
        valid_mask = [False] * len(self.wav_files)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(_load_one, p): i
                       for i, p in enumerate(self.wav_files)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="  Loading"):
                idx = futures[future]
                result = future.result()
                if result is not None:
                    all_wavs[idx] = result
                    valid_mask[idx] = True

        # Filter valid
        valid_files = [f for f, v in zip(self.wav_files, valid_mask) if v]
        all_wavs = [w for w, v in zip(all_wavs, valid_mask) if v]
        self.wav_files = valid_files
        print(f"  Loaded {len(self.wav_files)} wavs")

        # Step 2: Batch encode on GPU
        print("  Encoding on GPU...")
        codes_list = []
        batch_size = 64
        for batch_start in tqdm(range(0, len(all_wavs), batch_size), desc="  Encoding"):
            batch = all_wavs[batch_start:batch_start + batch_size]
            max_len = max(w.shape[0] for w in batch)
            max_len = ((max_len + 255) // 256) * 256
            padded = torch.zeros(len(batch), 1, max_len)
            for i, w in enumerate(batch):
                padded[i, 0, :w.shape[0]] = w

            with torch.no_grad():
                batch_codes = self.mimi.encode(padded.to(device))

            for i in range(len(batch)):
                codes_list.append((batch_codes[i], batch[i].shape[0]))

        self.codes_cache = codes_list

        # Convert to padded tensor format for fast save/load
        wav_lens = torch.tensor([l for _, l in codes_list])
        max_T = max(c.shape[1] for c, _ in codes_list)
        K = codes_list[0][0].shape[0]
        padded_codes = torch.zeros(len(codes_list), K, max_T, dtype=torch.long)
        for i, (c, _) in enumerate(codes_list):
            padded_codes[i, :, :c.shape[1]] = c

        self.codes_cache = padded_codes
        self.wav_lens_cache = wav_lens

        # Save cache
        manifest = {
            "names": [f.name for f in self.wav_files],
            "codes": padded_codes,
            "wav_lens": wav_lens,
        }
        torch.save(manifest, self.cache_dir / "manifest.pt")
        print(f"  Cached {len(self.wav_files)} codes to {self.cache_dir}")

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]

        # Load audio
        wav, sr = torchaudio.load(str(wav_path))
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        # Trim to max duration
        if wav.shape[1] > self.max_samples:
            start = torch.randint(0, wav.shape[1] - self.max_samples, (1,)).item()
            wav = wav[:, start:start + self.max_samples]

        # Get cached codes (trimmed to actual length)
        wav_len = self.wav_lens_cache[idx].item()
        codes = self.codes_cache[idx]  # [K, max_T]
        # Trim to actual code length (wav_len // 1920, with rounding)
        code_T = (wav_len + 1919) // 1920
        codes = codes[:, :code_T]

        return wav.squeeze(0), codes


def collate_fn(batch):
    """Pad wavs and codes to batch max length."""
    wavs, codes = zip(*batch)

    wav_lens = torch.tensor([w.shape[0] for w in wavs])
    wav_max = wav_lens.max().item()
    wav_max = ((wav_max + 255) // 256) * 256  # align to 256

    padded_wav = torch.zeros(len(wavs), wav_max)
    for i, w in enumerate(wavs):
        padded_wav[i, :w.shape[0]] = w

    code_lens = torch.tensor([c.shape[1] for c in codes])
    code_max = code_lens.max().item()

    padded_codes = torch.zeros(len(codes), codes[0].shape[0], code_max, dtype=torch.long)
    for i, c in enumerate(codes):
        padded_codes[i, :, :c.shape[1]] = c

    return padded_wav, padded_codes, wav_lens, code_lens


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PitchControlTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.stage = args.stage

        # Load Mimi
        print("Loading Mimi model...")
        self.mimi = get_mimi_model().to(self.device)
        self.mimi.eval()
        self.mimi.set_num_codebooks(16)  # SeedVox uses 16 codebooks

        # Build pitch-controllable wrapper
        self.model = PitchControllableMimi(
            self.mimi,
            stage=self.stage,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            grad_rev_alpha=args.grad_rev_alpha,
        ).to(self.device)

        # Load checkpoint
        if args.stage == 2 and args.stage1_checkpoint:
            print(f"Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
            ckpt = torch.load(args.stage1_checkpoint, map_location='cpu')
            self.model.load_state_dict(ckpt['model'], strict=False)

        # Loss
        self.criterion = PitchControlLoss(
            sample_rate=24000,
            mel_weight=args.mel_weight,
            disc_weight=args.disc_weight if args.stage == 1 else 0.0,
        )

        # Optimizer (only LoRA params + stage-specific modules)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
        self.optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.epochs * args.steps_per_epoch,
        )

        # Logging
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.global_step = 0

        # Validation vocoding: pre-encode a fixed reference wav
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
                self.val_codes = self.mimi.encode(wav.unsqueeze(0).to(self.device))[0]  # [K, T]
            val_out = Path(args.output_dir) / "val"
            val_out.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(val_out / "ground_truth.wav"), wav.cpu(), 24000)
            print(f"  Validation wav loaded: {val_wav_path} ({wav.shape[1]/24000:.2f}s)")
        self.val_every = getattr(args, 'val_every', 500)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        total_mel = 0
        total_disc = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Stage {self.stage} Epoch {epoch}")
        for wav_true, codes, wav_lens, code_lens in pbar:
            wav_true = wav_true.to(self.device)
            codes = codes.to(self.device)

            # Forward
            wav_pred, info = self.model(codes, target_wav=wav_true.unsqueeze(1))

            # Compute loss
            loss, loss_dict = self.criterion(
                wav_pred, wav_true.unsqueeze(1), info
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.optimizer.step()
            self.scheduler.step()

            # Log
            total_loss += loss.item()
            total_mel += loss_dict['mel']
            total_disc += loss_dict['disc']
            n_batches += 1

            self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            self.writer.add_scalar('train/mel', loss_dict['mel'], self.global_step)
            self.writer.add_scalar('train/disc', loss_dict['disc'], self.global_step)
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0],
                                   self.global_step)
            self.global_step += 1

            # Validation vocoding
            if self.val_codes is not None and self.global_step % self.val_every == 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred, _ = self.model(
                        self.val_codes.unsqueeze(0).to(self.device),
                        target_wav=self.val_wav_gt,
                    )
                val_path = Path(self.args.output_dir) / "val" / f"step_{self.global_step}.wav"
                torchaudio.save(str(val_path), val_pred.squeeze(0).cpu(), 24000)
                self.model.train()
                print(f"  → Saved val wav: {val_path}")

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mel': f"{loss_dict['mel']:.4f}",
                'disc': f"{loss_dict['disc']:.4f}",
            })

        avg_loss = total_loss / max(n_batches, 1)
        avg_mel = total_mel / max(n_batches, 1)
        avg_disc = total_disc / max(n_batches, 1)
        print(f"  Epoch {epoch}: loss={avg_loss:.4f} mel={avg_mel:.4f} disc={avg_disc:.4f}")
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
        path = Path(self.args.output_dir) / f"pitch_control_stage{self.stage}_epoch{epoch}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        print(f"  Saved checkpoint: {path}")

    def train(self, dataloader):
        print(f"\n{'='*60}")
        print(f"Stage {self.stage} — {'Pitch removal' if self.stage == 1 else 'Pitch re-injection'}")
        print(f"{'='*60}\n")

        best_loss = float('inf')
        for epoch in range(1, self.args.epochs + 1):
            loss = self.train_epoch(dataloader, epoch)
            self.save_checkpoint(epoch, loss)
            if loss < best_loss:
                best_loss = loss
                # Save best
                best_path = Path(self.args.output_dir) / f"pitch_control_stage{self.stage}_best.pt"
                torch.save(ckpt := {
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
        description="Train pitch-controllable Mimi decoder"
    )
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2],
                        help='Training stage (1=pitch removal, 2=pitch re-injection)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with wav files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pretrained Mimi checkpoint (Stage 1)')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                        help='Stage 1 checkpoint (Stage 2 only)')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs/pitch_control',
                        help='TensorBoard log directory')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--max_duration', type=float, default=30.0,
                        help='Max audio duration in seconds')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to cache pre-computed RVQ codes')

    # Validation
    parser.add_argument('--val_wav', type=str, default=None,
                        help='Path to a reference wav for periodic vocoding validation')
    parser.add_argument('--val_every', type=int, default=500,
                        help='Steps between validation vocoding')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--lora_alpha', type=float, default=64.0)

    # Loss
    parser.add_argument('--mel_weight', type=float, default=45.0)
    parser.add_argument('--disc_weight', type=float, default=10.0,
                        help='Pitch discriminator weight (Stage 1 only)')
    parser.add_argument('--grad_rev_alpha', type=float, default=2.0,
                        help='Gradient reversal strength (Stage 1 only)')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Build dataset
    print("Loading Mimi for encoding...")
    mimi = get_mimi_model()
    mimi.eval()
    mimi.set_num_codebooks(16)  # SeedVox uses 16 codebooks
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
    trainer = PitchControlTrainer(args)
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
