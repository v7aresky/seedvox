"""Pretrain pitch predictor: predict actual decoded F0 from z-space.

Trains predictor(z) → f0_estimator(decode(z)) so the predictor faithfully
estimates the F0 that the Mimi decoder would produce from a given z.

This frozen predictor is then used in pitch_shift training to provide
reliable F0 estimates from z-space, preventing the FilmShift from cheating.
"""

import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from seedvox.modules.mimi import get_mimi_model
from .pitch_shift import PitchPredictor


class ProsodyCodecDataset(Dataset):
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
        codes = d['audio_tokens'].squeeze(0)
        return wav.squeeze(0), codes


def collate_fn(batch):
    wavs, codes_list = zip(*batch)
    max_wav = max(w.shape[0] for w in wavs)
    wavs_padded = torch.zeros(len(wavs), max_wav)
    for i, w in enumerate(wavs):
        wavs_padded[i, :w.shape[0]] = w

    max_code_t = max(c.shape[1] for c in codes_list)
    codes_padded = torch.zeros(len(codes_list), codes_list[0].shape[0], max_code_t,
                               dtype=codes_list[0].dtype)
    for i, c in enumerate(codes_list):
        codes_padded[i, :, :c.shape[1]] = c

    wav_lens = torch.tensor([w.shape[0] for w in wavs])
    code_lens = torch.tensor([c.shape[1] for c in codes_list])
    return wavs_padded, codes_padded, wav_lens, code_lens


def main():
    parser = argparse.ArgumentParser(description="Pretrain pitch predictor")
    parser.add_argument('--lj_pt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs/pretrain_predictor')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading Mimi model...")
    mimi = get_mimi_model().to(device)
    mimi.eval()
    mimi.set_num_codebooks(16)

    # Freeze Mimi
    for p in mimi.parameters():
        p.requires_grad = False

    # Load F0 estimator
    from .f0_estimator import F0Estimator
    f0_estimator = F0Estimator().to(device)
    ckpt = torch.load("checkpoints/f0_estimator_best.pt", map_location="cpu", weights_only=False)
    f0_estimator.load_state_dict(ckpt["model"])
    del ckpt
    f0_estimator.eval()
    for p in f0_estimator.parameters():
        p.requires_grad = False

    predictor = PitchPredictor(z_dim=512, hidden=128).to(device)
    trainable = [p for p in predictor.parameters() if p.requires_grad]
    total = sum(p.numel() for p in trainable)
    print(f"Predictor parameters: {total:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    writer = SummaryWriter(log_dir=args.log_dir)

    dataset = ProsodyCodecDataset(args.lj_pt, name="LJ")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True,
    )

    print(f"\n{'='*60}")
    print(f"Pretraining pitch predictor: z → f0_estimator(decode(z))")
    print(f"{'='*60}\n")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        predictor.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for wavs, codes, wav_lens, code_lens in pbar:
            wavs = wavs.to(device)
            codes = codes.to(device)

            with torch.no_grad():
                z = mimi.quantizer.decode(codes)  # [B, 512, T_z]
                # Batch decode through full Mimi path
                e = mimi._to_encoder_framerate(z)
                (e,) = mimi.decoder_transformer(e)
                wav_decoded = mimi.decoder(e)  # [B, 1, N]

                # Get ground truth F0 from decoded audio
                log_f0_true = f0_estimator(wav_decoded)[0]  # [B, T]
                f0_true = torch.exp(log_f0_true)  # [B, T] Hz

            # Predict F0 from z
            f0_pred = predictor(z).squeeze(1)  # [B, T_z]

            # Align time dimensions
            T = min(f0_pred.shape[1], f0_true.shape[1])
            f0_pred = f0_pred[:, :T]
            f0_true = f0_true[:, :T]

            # Loss in log space
            loss = (torch.log(f0_pred + 1.0) - torch.log(f0_true + 1.0)).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

            writer.add_scalar('train/loss', loss.item(), global_step)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}")

        # Save checkpoint
        path = Path(args.output_dir) / f"predictor_pretrained.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model': predictor.state_dict(),
            'loss': avg_loss,
        }, path)

    writer.close()
    print(f"\nDone. Predictor saved to {path}")


if __name__ == '__main__':
    main()
