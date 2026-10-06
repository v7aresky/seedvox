"""Pre-extract Mimi continuous latents + F0 for all files in manifest.

Run this BEFORE training to build the cache. Extracts on GPU in main process,
then training loads from CPU cache with num_workers=0.

Usage:
    python extract_latents.py --cache_dir /data/flow_matching_cache
"""
import argparse
import json
import time
from pathlib import Path

import torch
import soundfile as sf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seedvox.modules.mimi import get_mimi_model
from explicit_pros_phon_planner.f0_estimator import F0Estimator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str,
                    default="/home/vpollet/proj/seedvox/explicit_pros_phon_planner/data/manifest.jsonl")
    p.add_argument("--cache_dir", type=str,
                    default="/home/vpollet/proj/seedvox/explicit_pros_phon_planner/cache/continuous_latents")
    p.add_argument("--batch_size", type=int, default=32,
                    help="Files to encode per forward pass (sequential, but batched through Mimi)")
    p.add_argument("--max_duration_sec", type=float, default=30.0)
    p.add_argument("--sample_rate", type=int, default=24000)
    p.add_argument("--skip_existing", action="store_true", default=True)
    return p.parse_args()


def load_audio(path, target_sr):
    """Load audio and resample to target_sr."""
    wav, sr = sf.read(str(path))
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav).float()
    if sr != target_sr:
        target_len = int(wav.shape[-1] * target_sr / sr)
        wav = torch.nn.functional.interpolate(
            wav.unsqueeze(0).unsqueeze(0), size=target_len, mode='linear', align_corners=False
        ).squeeze()
    return wav


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading Mimi...")
    mimi = get_mimi_model(device=device).eval()

    print("Loading F0 estimator...")
    f0_est = F0Estimator(sr=24000, n_fft=2048, hop=1920, n_mels=128,
                         fmin=50.0, fmax=8000.0, channels=48, blocks=4)
    ckpt = torch.load("/home/vpollet/proj/seedvox/checkpoints/f0_estimator_best.pt", map_location=device)
    f0_est.load_state_dict(ckpt.get("model", ckpt))
    f0_est = f0_est.to(device).eval()

    # Create cache dirs
    cache_dir = Path(args.cache_dir)
    latent_dir = cache_dir / "latents"
    f0_dir = cache_dir / "f0"
    latent_dir.mkdir(parents=True, exist_ok=True)
    f0_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    entries = []
    with open(args.manifest) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"Manifest: {len(entries)} entries")

    # Filter already-cached
    todo = []
    skipped = 0
    for entry in entries:
        key = Path(entry["path"]).stem
        if args.skip_existing and (latent_dir / f"{key}.pt").exists():
            skipped += 1
            continue
        todo.append(entry)
    print(f"Already cached: {skipped}, remaining: {len(todo)}")

    if not todo:
        print("Nothing to do!")
        return

    # Extract
    t0 = time.time()
    for i, entry in enumerate(todo):
        wav_path = Path(entry["path"])
        if not wav_path.is_absolute():
            continue  # skip if path is broken

        try:
            wav = load_audio(wav_path, args.sample_rate)
            max_samples = int(args.max_duration_sec * args.sample_rate)
            if wav.shape[0] > max_samples:
                wav = wav[:max_samples]
            wav = wav.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, N]

            with torch.no_grad():
                latents = mimi.encode_to_latent(wav, quantize=False)  # [1, D, T]
                latents = latents.transpose(1, 2).squeeze(0).cpu()  # [T, D]

                log_f0 = f0_est.estimate_f0_hz(wav)  # [1, T]
                f0_hz = torch.exp(log_f0).squeeze(0).cpu()  # [T]

                # Align F0 to latent length
                T_lat = latents.shape[0]
                if f0_hz.shape[0] > T_lat:
                    f0_hz = f0_hz[:T_lat]
                elif f0_hz.shape[0] < T_lat:
                    f0_hz = torch.nn.functional.pad(f0_hz, (0, T_lat - f0_hz.shape[0]))

            key = wav_path.stem
            torch.save(latents, latent_dir / f"{key}.pt")
            torch.save(f0_hz, f0_dir / f"{key}.pt")

        except Exception as e:
            print(f"  ERROR {wav_path.name}: {e}")
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(todo)}] {rate:.1f} files/s, ETA {eta:.1f} min")

    elapsed = time.time() - t0
    print(f"Done! Extracted {len(todo)} files in {elapsed/60:.1f} min")
    print(f"Cache: {cache_dir}")


if __name__ == "__main__":
    main()
