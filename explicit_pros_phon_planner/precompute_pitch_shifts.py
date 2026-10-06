"""Pre-compute pitch-shifted versions of all wavs for Stage 2 augmentation."""

import torchaudio
import torch
import math
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp


def shift_one(args_tuple):
    src, dst, n_steps, sr = args_tuple
    if dst.exists():
        return
    wav, sr_orig = torchaudio.load(src)
    if sr_orig != sr:
        wav = torchaudio.functional.resample(wav, sr_orig, sr)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    shifted = torchaudio.functional.pitch_shift(wav, sr, n_steps=n_steps)
    torchaudio.save(str(dst), shifted, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../autovoc/dataset/wavs")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.7, 0.85, 1.0, 1.15, 1.3])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.data_dir) / "pitch_shifted"
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(Path(args.data_dir).glob("*.wav"))
    print(f"{len(wavs)} wavs × {len(args.scales)} scales → {len(wavs)*len(args.scales)} files")

    tasks = []
    for wav in wavs:
        for scale in args.scales:
            n_steps = 12.0 * math.log2(scale)
            dst = out_dir / f"{wav.stem}_s{scale:.2f}.wav"
            tasks.append((str(wav), str(dst), n_steps, 24000))

    with mp.Pool(args.workers) as pool:
        list(tqdm(pool.imap(shift_one, tasks), total=len(tasks)))

    print(f"Done. Saved to {out_dir}/")


if __name__ == "__main__":
    main()
