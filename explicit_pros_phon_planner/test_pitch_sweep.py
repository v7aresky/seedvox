"""Test Stage 2 pitch control: sweep F0 multipliers and internal prediction."""

import torch
import torchaudio
import argparse
from pathlib import Path

from seedvox.modules.mimi import get_mimi_model
from explicit_pros_phon_planner.pitch_filter import PitchFilterModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pitch_filter/pitch_filter_stage2_best.pt")
    parser.add_argument("--ref_wav", default="../autovoc/dataset/wavs/LJ001-0001.wav")
    parser.add_argument("--output_dir", default="checkpoints/pitch_filter/s2_test")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Mimi
    mimi = get_mimi_model().to(device)
    mimi.eval()
    mimi.set_num_codebooks(16)

    # Build Stage 2 model
    model = PitchFilterModel(mimi, stage=2, lora_rank=32, lora_alpha=64)

    # Load checkpoint
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().to(device)

    # Load reference audio
    wav_ref, sr = torchaudio.load(args.ref_wav)
    if sr != 24000:
        wav_ref = torchaudio.functional.resample(wav_ref, sr, 24000)
    if wav_ref.shape[0] > 1:
        wav_ref = wav_ref.mean(0, keepdim=True)
    wav_ref = wav_ref.unsqueeze(0).to(device)  # [1, 1, N]

    # Extract codes (content)
    with torch.no_grad():
        codes = mimi.encode(wav_ref)[0]  # [1, 16, T]
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

    # Extract reference F0
    with torch.no_grad():
        f0_ref = model._extract_f0(wav_ref)  # [1, T_f0]
    mean_f0 = f0_ref.mean().item()
    print(f"Reference F0: {mean_f0:.1f} Hz")

    # Output dir
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out / "00_reference.wav"), wav_ref.squeeze(0).cpu(), 24000)

    # Test 1: Internal prediction (model predicts F0 from codes)
    with torch.no_grad():
        wav_pred_f0, info_pred = model(codes, predict_pitch=True)
    f0_pred = info_pred['f0_predicted']
    f0_out = model._extract_f0(wav_pred_f0)
    print(f"\n[Predicted F0] model={f0_pred.mean().item():.1f}Hz, output={f0_out.mean().item():.1f}Hz")
    torchaudio.save(str(out / "10_predicted_f0.wav"), wav_pred_f0.squeeze(0).cpu(), 24000)

    # Test 2: External F0 override at different scales
    for scale in args.scales:
        f0_scaled = (f0_ref * scale).clamp(min=1.0)
        with torch.no_grad():
            wav_out, info = model(codes, f0_override=f0_scaled)

        f0_out = model._extract_f0(wav_out)
        mean_out = f0_out.mean().item()

        fname = f"scale_{scale:.2f}_f0_{mean_out:.0f}hz.wav"
        torchaudio.save(str(out / fname), wav_out.squeeze(0).cpu(), 24000)
        print(f"  scale={scale:.2f}x -> target={mean_f0*scale:.0f}Hz, actual={mean_out:.1f}Hz -> {fname}")

    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
