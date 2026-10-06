#!/usr/bin/env python3
"""
Mimi Q2 Pitch Analysis & Lookup Builder.

Two-phase approach:
  1. Train F0 prediction head on frozen Q2 codebook embeddings
  2. Build Q2→F0 lookup table for code replacement

If Q2 already captures enough F0 info (low RMSE), code replacement works.
If not, this provides the diagnostic to decide next steps.

Architecture:
  Q1 = content (semantic, WavLM distillation)
  Q2 = first acoustic level
  Q3+ = acoustic residual

Pitch shifting target: swap Q2 codes for target F0's code.
"""

import os
import sys
import time
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seedvox.modules.mimi import get_mimi_model
from explicit_pros_phon_planner.f0_estimator import F0Estimator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", type=str, nargs="+",
                    default=["/home/vpollet/proj/autovoc/dataset/wavs",
                             "/home/vpollet/proj/autovoc/dataset/globe_v3_r2/wavs",
                             "/home/vpollet/proj/autovoc/dataset/hifitts/wavs"])
    p.add_argument("--f0_estimator_ckpt", type=str,
                    default="checkpoints/f0_estimator_best.pt")
    p.add_argument("--mimi_ckpt", type=str,
                    default="pretrained_models/best_mimi.pt")
    p.add_argument("--out_dir", type=str,
                    default="checkpoints/mimi_q2_pitch")
    p.add_argument("--n_codebooks", type=int, default=16)
    p.add_argument("--n_train_files", type=int, default=5000)
    p.add_argument("--n_val_files", type=int, default=500)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_frames", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class F0PredictionHead(nn.Module):
    """Predict F0 (log Hz) from codebook embeddings."""
    def __init__(self, embed_dim=256, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, embed):
        return self.mlp(embed).squeeze(-1)  # [...]


@torch.no_grad()
def get_q2_codebook(mimi):
    """Get Q2's EuclideanCodebook. rvq_rest.vq.layers[0] is Q2 (first acoustic level)."""
    return mimi.quantizer.rvq_rest.vq.layers[0]._codebook


def encode_dataset(mimi, wav_files_with_dirs, device, max_frames=500, n_files=3000):
    """Encode wav files through Mimi, return Q2 codes and Q2 embeddings."""
    import torchaudio

    all_q2_codes = []
    all_q2_embeds = []
    all_f0s = []
    n = min(n_files, len(wav_files_with_dirs))

    max_samples = max_frames * 1920
    f0_est = None  # lazy init

    for i in range(n):
        wav_path = wav_files_with_dirs[i]
        wav, sr = torchaudio.load(wav_path)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        wav = wav.mean(dim=0, keepdim=True)
        if wav.shape[1] > max_samples:
            start = random.randint(0, wav.shape[1] - max_samples)
            wav = wav[:, start:start + max_samples]
        wav = wav.unsqueeze(0).to(device)  # [1, 1, T]

        # Encode through Mimi
        codes = mimi.encode(wav)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        q2_codes = codes[:, 1, :]  # [1, T_z]

        # Q2 embedding lookup (from frozen codebook)
        cb = mimi.quantizer.rvq_rest.vq.layers[0]._codebook
        q2_embed = F.embedding(q2_codes, cb.embedding)  # [1, T_z, 256]

        # Extract F0
        if f0_est is None:
            f0_est = F0Estimator().to(device).eval()
            f0_ckpt = torch.load("checkpoints/f0_estimator_best.pt", map_location=device)
            f0_est.load_state_dict(f0_ckpt["model"])

        with torch.no_grad():
            f0_log, _ = f0_est(wav.squeeze(1))  # [1, T_f0]

        # Align
        T = min(q2_codes.shape[1], f0_log.shape[1])

        all_q2_codes.append(q2_codes[0, :T].cpu())
        all_q2_embeds.append(q2_embed[0, :T].cpu())
        all_f0s.append(f0_log[0, :T].cpu())

        if (i + 1) % 500 == 0:
            print(f"    Encoded {i+1}/{n}")

    return (torch.cat(all_q2_codes),   # [N_total]
            torch.cat(all_q2_embeds),  # [N_total, 256]
            torch.cat(all_f0s))         # [N_total]


def train_f0_head(q2_embeds, f0s, device, args):
    """Train F0 prediction head on frozen Q2 embeddings."""
    print(f"\n=== Training F0 head ({q2_embeds.shape[0]} frames) ===")

    # Split train/val
    n = q2_embeds.shape[0]
    perm = torch.randperm(n)
    n_val = min(args.n_val_files * 200, n // 5)  # ~200 frames per val file
    n_train = n - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_dataset = torch.utils.data.TensorDataset(q2_embeds[train_idx], f0s[train_idx])
    val_dataset = torch.utils.data.TensorDataset(q2_embeds[val_idx], f0s[val_idx])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"  Train: {n_train}, Val: {n_val}")

    # Model
    f0_head = F0PredictionHead(256, 128).to(device)
    optimizer = torch.optim.AdamW(f0_head.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        f0_head.train()
        total_loss = 0
        n_batches = 0

        for embed_batch, f0_batch in train_loader:
            embed_batch = embed_batch.to(device)
            f0_batch = f0_batch.to(device)

            pred = f0_head(embed_batch)
            loss = F.mse_loss(pred, f0_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validate
        f0_head.eval()
        all_val_pred = []
        all_val_gt = []
        with torch.no_grad():
            for embed_batch, f0_batch in val_loader:
                embed_batch = embed_batch.to(device)
                pred = f0_head(embed_batch)
                all_val_pred.append(pred.cpu())
                all_val_gt.append(f0_batch)

        val_pred = torch.cat(all_val_pred)
        val_gt = torch.cat(all_val_gt)
        val_rmse_log = torch.sqrt(F.mse_loss(val_pred, val_gt)).item()
        val_rmse_hz = torch.sqrt(F.mse_loss(torch.exp(val_pred), torch.exp(val_gt))).item()

        scheduler.step()

        marker = ""
        if val_rmse_hz < best_val_rmse:
            best_val_rmse = val_rmse_hz
            best_state = {k: v.clone() for k, v in f0_head.state_dict().items()}
            marker = " ★"

        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={total_loss/n_batches:.6f} | "
              f"val_RMSE={val_rmse_hz:.1f} Hz{marker}")

    f0_head.load_state_dict(best_state)
    print(f"\n  Best val RMSE: {best_val_rmse:.1f} Hz")
    return f0_head, best_val_rmse


def analyze_q2_codes(q2_codes, f0s, out_dir):
    """Analyze Q2 code distribution vs F0."""
    print("\n=== Q2 Code Analysis ===")

    f0_hz = torch.exp(f0s)
    voiced = (f0_hz > 50) & (f0_hz < 500)
    q2_voiced = q2_codes[voiced]
    f0_voiced = f0_hz[voiced]

    print(f"  Total frames: {len(q2_codes)}")
    print(f"  Voiced frames: {voiced.sum().item()}")
    print(f"  Unique Q2 codes used: {len(q2_voiced.unique())}/2048")

    # Per-code F0 statistics
    print(f"\n  Per-code F0 stats (codes with >50 voiced frames):")
    code_f0_map = {}
    for code in range(2048):
        mask = q2_voiced == code
        if mask.sum() > 50:
            code_f0_map[code] = {
                "mean_f0": f0_voiced[mask].mean().item(),
                "std_f0": f0_voiced[mask].std().item(),
                "count": mask.sum().item(),
            }

    sorted_codes = sorted(code_f0_map.items(), key=lambda x: x[1]["mean_f0"])
    for code, stats in sorted_codes[:20]:
        print(f"    Q2={code:4d} → mean_F0={stats['mean_f0']:.0f}±{stats['std_f0']:.0f} Hz (n={stats['count']})")
    if len(sorted_codes) > 20:
        print(f"    ... ({len(sorted_codes)} codes with >50 frames)")

    # Correlation: how well does Q2 code predict F0?
    # Assign each code the mean F0 of its frames
    code_mean_f0 = torch.full((2048,), float("nan"))
    for code, stats in code_f0_map.items():
        code_mean_f0[code] = stats["mean_f0"]

    # For voiced frames, predict F0 from code
    pred_f0 = code_mean_f0[q2_voiced]
    valid_pred = ~torch.isnan(pred_f0)
    if valid_pred.sum() > 0:
        rmse = torch.sqrt(F.mse_loss(pred_f0[valid_pred], f0_voiced[valid_pred])).item()
        print(f"\n  Code→F0 lookup RMSE: {rmse:.1f} Hz (from {valid_pred.sum().item()} frames)")

    return code_f0_map


def build_lookup(q2_codes, f0s, f0_head, q2_embeds, device, out_dir):
    """Build Q2→F0 lookup table using the trained F0 head."""
    print("\n=== Building Q2→F0 Lookup Table ===")

    f0_head.eval()
    with torch.no_grad():
        # Predict F0 for each of the 2048 codebook entries
        all_embeds = torch.zeros(2048, 256)
        # We need the actual codebook embeddings
        # Use the ones we have
        unique_codes = q2_codes.unique()
        for code in unique_codes:
            mask = q2_codes == code
            if mask.sum() > 0:
                mean_embed = q2_embeds[mask].mean(dim=0)
                all_embeds[code] = mean_embed

        # Predict F0 for each code
        pred_f0 = f0_head(all_embeds.to(device)).cpu()  # [2048] in log Hz

    # Build lookup: bin by predicted F0
    n_bins = 50
    f0_lo, f0_hi = 50.0, 500.0
    f0_hz_pred = torch.exp(pred_f0)

    # Find codes for each F0 bin
    lookup = []
    f0_edges = torch.linspace(f0_lo, f0_hi, n_bins + 1)

    for i in range(n_bins):
        lo, hi = f0_edges[i], f0_edges[i + 1]
        # Find codes whose predicted F0 falls in this bin
        mask = (f0_hz_pred >= lo) & (f0_hz_pred < hi)
        codes_in_bin = mask.nonzero(as_tuple=True)[0]

        if len(codes_in_bin) == 0:
            lookup.append({
                "f0_lo": round(lo.item(), 1),
                "f0_hi": round(hi.item(), 1),
                "f0_center": round(((lo + hi) / 2).item(), 1),
                "q2_code": -1,
                "n_candidates": 0,
            })
        else:
            # Pick the code closest to the bin center
            target_f0 = (lo + hi) / 2
            code_f0s = f0_hz_pred[codes_in_bin]
            dists = torch.abs(code_f0s - target_f0)
            best_idx = dists.argmin()
            best_code = codes_in_bin[best_idx].item()

            lookup.append({
                "f0_lo": round(lo.item(), 1),
                "f0_hi": round(hi.item(), 1),
                "f0_center": round(target_f0.item(), 1),
                "q2_code": best_code,
                "n_candidates": len(codes_in_bin),
                "pred_f0_hz": round(f0_hz_pred[best_code].item(), 1),
            })

    # Save
    os.makedirs(out_dir, exist_ok=True)
    lookup_path = os.path.join(out_dir, "q2_f0_lookup.json")
    with open(lookup_path, "w") as f:
        json.dump(lookup, f, indent=2)

    valid = [e for e in lookup if e["q2_code"] >= 0]
    print(f"  Lookup: {len(valid)}/{n_bins} bins filled")
    print(f"  Saved to {lookup_path}")

    # Print summary
    print(f"\n  F0 → Q2 code mapping:")
    for e in lookup:
        if e["q2_code"] >= 0:
            print(f"    {e['f0_center']:6.0f} Hz → Q2={e['q2_code']:4d} "
                  f"(pred={e.get('pred_f0_hz', '?')} Hz, {e['n_candidates']} candidates)")

    return lookup


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load Mimi ---
    print("Loading Mimi...")
    mimi = get_mimi_model(checkpoint_path=args.mimi_ckpt, device=device, num_codebooks=args.n_codebooks)
    mimi.set_num_codebooks(args.n_codebooks)
    mimi.eval()
    print(f"  Mimi loaded ({sum(p.numel() for p in mimi.parameters())/1e6:.1f}M params)")

    # --- List wav files from all directories ---
    all_wav_paths = []
    for wav_dir in args.wav_dir:
        if os.path.isdir(wav_dir):
            files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")]
            print(f"  {wav_dir}: {len(files)} files")
            all_wav_paths.extend(files)
    random.shuffle(all_wav_paths)
    print(f"  Total wav files: {len(all_wav_paths)}")

    # --- Encode dataset ---
    print("\n=== Encoding dataset through Mimi ===")
    q2_codes, q2_embeds, f0s = encode_dataset(
        mimi, all_wav_paths, device,
        max_frames=args.max_frames,
        n_files=args.n_train_files + args.n_val_files
    )
    print(f"  Q2 codes: {q2_codes.shape}, Q2 embeds: {q2_embeds.shape}, F0: {f0s.shape}")

    # --- Analyze Q2 codes ---
    code_f0_map = analyze_q2_codes(q2_codes, f0s, args.out_dir)

    # --- Train F0 head ---
    f0_head, baseline_rmse = train_f0_head(q2_embeds, f0s, device, args)

    # --- Save F0 head ---
    f0_head_path = os.path.join(args.out_dir, "f0_head.pt")
    torch.save(f0_head.state_dict(), f0_head_path)
    print(f"  F0 head saved to {f0_head_path}")

    # --- Build lookup ---
    lookup = build_lookup(q2_codes, f0s, f0_head, q2_embeds, device, args.out_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  F0 head RMSE (frozen Q2): {baseline_rmse:.1f} Hz")
    print(f"  Lookup table: {args.out_dir}/q2_f0_lookup.json")
    print(f"  F0 head: {args.out_dir}/f0_head.pt")
    print()
    if baseline_rmse < 30:
        print("  → Q2 has GOOD F0 information. Code replacement should work.")
    elif baseline_rmse < 60:
        print("  → Q2 has MODERATE F0 information. Code replacement may work roughly.")
    else:
        print("  → Q2 has WEAK F0 information. Fine-tuning Mimi may be needed.")
    print()
    print("  Next step: test code replacement with the lookup table.")


if __name__ == "__main__":
    main()
