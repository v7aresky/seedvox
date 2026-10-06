"""Test single-stage pitch shift with F0 sweep.

Usage:
    python -m explicit_pros_phon_planner.test_pitch_shift \
        --checkpoint checkpoints/pitch_shift/pitch_shift_best.pt \
        --output_dir checkpoints/pitch_shift/test
"""

import argparse
from pathlib import Path
import torch
import torchaudio

from seedvox.modules.mimi import get_mimi_model
from .pitch_shift import PitchShiftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_wav', type=str,
                        default='../autovoc/dataset/wavs/LJ001-0001.wav')
    parser.add_argument('--output_dir', type=str, default='checkpoints/pitch_shift/test')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading Mimi...")
    mimi = get_mimi_model().to(device)
    mimi.eval()
    mimi.set_num_codebooks(16)

    model = PitchShiftModel(mimi).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    # Load test wav
    wav, sr = torchaudio.load(args.test_wav)
    if sr != 24000:
        wav = torchaudio.transforms.Resample(sr, 24000)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.to(device)

    # Encode
    with torch.no_grad():
        codes = mimi.encode(wav.unsqueeze(0))[0]
        f0_ref = model.f0_estimator(wav.unsqueeze(0))[0].squeeze(0)

    print(f"Reference F0: {f0_ref.mean().item():.1f} Hz")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save baseline (no shift)
    with torch.no_grad():
        wav_base, info_base = model(codes)
        f0_base = info_base['f0_current'][0].mean().item()
    torchaudio.save(str(out_dir / "baseline.wav"), wav_base.squeeze(0).cpu(), 24000)
    print(f"Baseline F0: {f0_base:.1f} Hz")

    # Sweep
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    print("\n[F0 Sweep]")
    for scale in scales:
        target_f0 = f0_ref * scale
        with torch.no_grad():
            wav_out, info = model(codes, target_f0=target_f0.unsqueeze(0))
            f0_out = info['f0_current'][0].mean().item()
        fname = f"scale_{scale:.2f}.wav"
        torchaudio.save(str(out_dir / fname), wav_out.squeeze(0).cpu(), 24000)
        print(f"  scale={scale:.2f}x -> target={f0_ref.mean().item()*scale:.0f}Hz, "
              f"predicted_f0={f0_out:.1f}Hz -> {fname}")

    print(f"\nSaved to {out_dir}")


if __name__ == '__main__':
    main()
