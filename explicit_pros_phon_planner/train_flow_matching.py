"""Training script for Pitch-Conditioned Flow Matching TTS.
Uses pre-extracted Mimi continuous latents + F0.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import soundfile as sf

from dataset_continuous import ContinuousLatentDataset, collate_fn
from flow_matching_tts import FlowMatchingTTS


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default="data/manifest.jsonl")
    p.add_argument("--cache_dir", type=str, default="cache/continuous_latents")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--sample_freq", type=int, default=5000)
    p.add_argument("--ckpt_freq", type=int, default=5000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--run_dir", type=str, default="runs/flow_matching")
    return p.parse_args()


def simple_tokenizer(text):
    return [ord(c) % 256 for c in text]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = ContinuousLatentDataset(
        manifest_path=args.manifest,
        tokenizer=simple_tokenizer,
        cache_dir=args.cache_dir,
    )
    print(f"Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    model = FlowMatchingTTS(
        dim=args.dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dim_head=args.dim_head,
        latent_dim=512,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count / 1e6:.1f}M params")

    ema_model = FlowMatchingTTS(
        dim=args.dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dim_head=args.dim_head,
        latent_dim=512,
        dropout=0.0,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda")

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        scheduler.last_epoch = start_step

    print(f"Training from step {start_step} to {args.max_steps}...")
    model.train()

    step = start_step
    epoch = 0
    while step < args.max_steps:
        epoch += 1
        print(f"\nEpoch {epoch}")

        for batch in dataloader:
            if step >= args.max_steps:
                break

            latents = batch["latents"].to(device)
            f0_hz = batch["f0_hz"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_mask = batch["text_mask"].to(device)
            f0_mask = batch["f0_mask"].to(device)

            with torch.amp.autocast("cuda"):
                loss, metrics = model.compute_loss(
                    latents, text_tokens, text_mask, f0_hz, f0_mask
                )
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                with torch.no_grad():
                    for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                        ema_p.data.mul_(0.999).add_(p.data, alpha=0.001)

            step += 1

            if step % 100 == 0:
                print(f"  Step {step}: loss={loss.item()*args.grad_accum_steps:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

            if step % args.sample_freq == 0:
                print(f"\n  Generating sample at step {step}...")
                model.eval()
                ema_model.eval()
                with torch.no_grad():
                    sample_text = text_tokens[:1]
                    sample_text_mask = text_mask[:1]
                    sample_f0 = f0_hz[:1]
                    sample_f0_mask = f0_mask[:1]
                    num_frames = latents.shape[1]

                    generated = ema_model.generate(
                        sample_text, sample_text_mask,
                        sample_f0, sample_f0_mask,
                        num_frames, n_steps=args.n_steps
                    )
                    # Decode with Mimi (continuous latent -> encoder framerate -> decoder)
                    from seedvox.modules.mimi import get_mimi_model
                    mimi = get_mimi_model(device=device)
                    mimi.eval()
                    generated_t = generated.transpose(1, 2)  # [B, D, T] at 12.5Hz
                    emb = mimi._to_encoder_framerate(generated_t)
                    audio = mimi.decoder(emb)  # [B, 1, N]
                    audio_path = run_dir / f"sample_step_{step:06d}.wav"
                    sf.write(str(audio_path), audio[0, 0].detach().cpu().numpy(), 24000)
                    print(f"  Saved {audio_path}")
                    del mimi
                    torch.cuda.empty_cache()

                model.train()

            if step % args.ckpt_freq == 0:
                ckpt_path = run_dir / f"checkpoint_{step:06d}.pt"
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"  Saved {ckpt_path}")

    print("\nTraining complete!")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }, run_dir / "checkpoint_final.pt")


if __name__ == "__main__":
    train(get_args())
