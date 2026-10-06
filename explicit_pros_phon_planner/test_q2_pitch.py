#!/usr/bin/env python3
"""
Test token-level pitch shifting by Q2 code replacement.

Pipeline:
  1. Load Mimi (frozen) + F0 head + lookup table
  2. Encode test wav → Q2 codes
  3. For each frame: predict F0 from Q2, compute target F0, find target Q2 code
  4. Replace Q2 codes, decode → shifted audio
  5. Measure F0 of shifted audio, compare with target
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seedvox.modules.mimi import get_mimi_model
from explicit_pros_phon_planner.f0_estimator import F0Estimator
from explicit_pros_phon_planner.mimi_pitch_finetune import F0PredictionHead, get_q2_codebook


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", type=str, required=True, help="Input wav file")
    p.add_argument("--out_dir", type=str, default="checkpoints/mimi_q2_pitch/test")
    p.add_argument("--mimi_ckpt", type=str, default="pretrained_models/best_mimi.pt")
    p.add_argument("--f0_head_ckpt", type=str, default="checkpoints/mimi_q2_pitch/f0_head.pt")
    p.add_argument("--lookup_json", type=str, default="checkpoints/mimi_q2_pitch/q2_f0_lookup.json")
    p.add_argument("--f0_estimator_ckpt", type=str, default="checkpoints/f0_estimator_best.pt")
    p.add_argument("--n_codebooks", type=int, default=16)
    p.add_argument("--shift_ratios", type=float, nargs="+",
                    default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                    help="F0 shift ratios (e.g., 1.5 = +50%)")
    return p.parse_args()


def load_lookup(path):
    with open(path) as f:
        lookup = json.load(f)
    # Build: f0_center -> q2_code
    f0_to_code = {}
    for entry in lookup:
        if entry["q2_code"] >= 0:
            f0_to_code[entry["f0_center"]] = entry["q2_code"]
    return f0_to_code


def find_nearest_code(target_f0, f0_to_code):
    """Find Q2 code closest to target F0."""
    best_f0 = min(f0_to_code.keys(), key=lambda f: abs(f - target_f0))
    return f0_to_code[best_f0], best_f0


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load models ---
    print("Loading models...")
    mimi = get_mimi_model(checkpoint_path=args.mimi_ckpt, device=device, num_codebooks=args.n_codebooks)
    mimi.set_num_codebooks(args.n_codebooks)
    mimi.eval()

    f0_head = F0PredictionHead(256, 128).to(device)
    f0_head.load_state_dict(torch.load(args.f0_head_ckpt, map_location=device))
    f0_head.eval()

    f0_est = F0Estimator().to(device).eval()
    f0_est.load_state_dict(torch.load(args.f0_estimator_ckpt, map_location=device)["model"])

    f0_to_code = load_lookup(args.lookup_json)
    print(f"  Lookup: {len(f0_to_code)} F0 bins")

    # --- Load audio ---
    wav, sr = torchaudio.load(args.wav)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    wav = wav.mean(dim=0, keepdim=True).to(device)  # [1, T]
    print(f"  Audio: {wav.shape[1]/24000:.2f}s")

    # --- Encode ---
    with torch.no_grad():
        codes = mimi.encode(wav.unsqueeze(0))  # [1, K, T_z]
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

    q2_codes_orig = codes[0, 1, :].clone()  # [T_z]

    # --- Get Q2 embeddings → predict F0 ---
    cb = get_q2_codebook(mimi)
    with torch.no_grad():
        q2_embeds = F.embedding(q2_codes_orig.unsqueeze(0), cb.embedding)  # [1, T_z, 256]
        f0_pred_log = f0_head(q2_embeds).squeeze(0)  # [T_z]
        f0_pred_hz = torch.exp(f0_pred_log)

    print(f"  Predicted F0: {f0_pred_hz.mean():.0f} ± {f0_pred_hz.std():.0f} Hz")
    print(f"  F0 range: {f0_pred_hz.min():.0f} - {f0_pred_hz.max():.0f} Hz")

    # --- Save original ---
    with torch.no_grad():
        wav_orig = mimi.decode(codes).squeeze()  # [T]
    orig_path = os.path.join(args.out_dir, "original.wav")
    torchaudio.save(orig_path, wav_orig.unsqueeze(0).cpu(), 24000)
    print(f"  Saved: {orig_path}")

    # --- Apply shifts ---
    for ratio in args.shift_ratios:
        print(f"\n--- Shift ratio: {ratio}x (target F0 = original × {ratio}) ---")

        # For each frame, find target Q2 code
        q2_codes_shifted = q2_codes_orig.clone()
        n_matched = 0
        n_total = len(q2_codes_orig)

        for i in range(n_total):
            current_f0 = f0_pred_hz[i].item()
            if current_f0 < 50 or current_f0 > 500:
                continue  # skip unvoiced

            target_f0 = current_f0 * ratio
            target_f0 = max(50, min(500, target_f0))  # clamp to lookup range

            target_code, _ = find_nearest_code(target_f0, f0_to_code)
            q2_codes_shifted[i] = target_code
            n_matched += 1

        print(f"  Replaced {n_matched}/{n_total} frames")

        # --- Decode shifted ---
        shifted_codes = codes.clone()
        shifted_codes[0, 1, :] = q2_codes_shifted

        with torch.no_grad():
            wav_shifted = mimi.decode(shifted_codes).squeeze()  # [T]

        # --- Measure F0 of shifted audio ---
        with torch.no_grad():
            f0_shifted_log, _ = f0_est(wav_shifted.unsqueeze(0))
            f0_shifted_hz = torch.exp(f0_shifted_log)

        print(f"  Original F0:  {f0_pred_hz.mean():.0f} ± {f0_pred_hz.std():.0f} Hz")
        print(f"  Shifted F0:   {f0_shifted_hz.mean():.0f} ± {f0_shifted_hz.std():.0f} Hz")
        print(f"  Target F0:    {f0_pred_hz.mean() * ratio:.0f} Hz (ratio={ratio})")

        actual_ratio = f0_shifted_hz.mean() / f0_pred_hz.mean()
        print(f"  Actual ratio: {actual_ratio:.3f}")

        # Save
        out_path = os.path.join(args.out_dir, f"shifted_{ratio}x.wav")
        torchaudio.save(out_path, wav_shifted.unsqueeze(0).cpu(), 24000)
        print(f"  Saved: {out_path}")

    # --- Also test: uniform code replacement (all frames same Q2 code) ---
    print(f"\n--- Uniform replacement test ---")
    for target_f0 in [100, 150, 200, 250]:
        target_code, matched_f0 = find_nearest_code(target_f0, f0_to_code)
        print(f"  Target {target_f0} Hz → Q2={target_code} (lookup matched {matched_f0:.0f} Hz)")

        shifted_codes = codes.clone()
        shifted_codes[0, 1, :] = target_code  # all frames same code

        with torch.no_grad():
            wav_shifted = mimi.decode(shifted_codes).squeeze()  # [T]

        with torch.no_grad():
            f0_shifted_log, _ = f0_est(wav_shifted.unsqueeze(0))
            f0_shifted_hz = torch.exp(f0_shifted_log)

        print(f"    Actual F0: {f0_shifted_hz.mean():.0f} ± {f0_shifted_hz.std():.0f} Hz")

        out_path = os.path.join(args.out_dir, f"uniform_{target_f0}hz.wav")
        torchaudio.save(out_path, wav_shifted.unsqueeze(0).cpu(), 24000)
        print(f"    Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
