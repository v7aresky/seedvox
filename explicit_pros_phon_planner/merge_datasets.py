"""Merge LJ Speech cached codes into Globe prosody codec .pt format."""

import torch
import torchaudio
import math
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_f0_from_wav(wav_path, sr=24000):
    """Simple F0 extraction via autocorrelation (no neural net needed)."""
    wav, sr = torchaudio.load(wav_path)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze(0), sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--globe_pt", default="/home/vpollet/proj/autovoc/train_tokens_globe_prosody_codec.pt")
    parser.add_argument("--lj_wav_dir", default="/home/vpollet/proj/autovoc/dataset/wavs")
    parser.add_argument("--output_pt", default="/home/vpollet/proj/autovoc/train_tokens_globe_lj.pt")
    args = parser.parse_args()

    # Load Globe data
    print(f"Loading Globe data: {args.globe_pt}")
    globe = torch.load(args.globe_pt, weights_only=False)
    globe_data = globe['data']
    print(f"  {len(globe_data)} Globe samples")

    # Find LJ Speech cached codes
    cache_dir = Path(args.lj_wav_dir) / ".cache_codes"
    lj_caches = sorted(cache_dir.glob("*.pt"))
    print(f"  {len(lj_caches)} LJ Speech cached codes")

    # Load F0 estimator (JDC-style)
    from explicit_pros_phon_planner.f0_estimator import F0Estimator
    f0_est = F0Estimator().eval()
    ckpt = torch.load("checkpoints/f0_estimator_best.pt", map_location="cpu", weights_only=False)
    f0_est.load_state_dict(ckpt["model"])
    del ckpt

    # Build LJ Speech entries
    lj_data = []
    for cache_path in tqdm(lj_caches, desc="Processing LJ Speech"):
        wav_name = cache_path.stem  # e.g., LJ001-0001
        wav_path = Path(args.lj_wav_dir) / f"{wav_name}.wav"
        if not wav_path.exists():
            continue

        # Load cached codes
        codes = torch.load(cache_path, weights_only=True)  # [1, 16, T]

        # Load wav and extract F0
        wav, sr = extract_f0_from_wav(str(wav_path))
        duration = wav.shape[0] / 24000

        # Extract F0 using estimator
        with torch.no_grad():
            log_f0, voiced_prob = f0_est(wav.unsqueeze(0).unsqueeze(0))
            log_f0 = log_f0.squeeze(0).squeeze(0)  # [T_f0]
            voiced_prob = voiced_prob.squeeze(0).squeeze(0)

        # Compute utterance-level mean F0 (only voiced frames)
        voiced = voiced_prob > 0.5
        if voiced.any():
            mu_logF0 = log_f0[voiced].mean().item()
        else:
            mu_logF0 = log_f0.mean().item()

        # Center F0 (remove utterance mean)
        log_f0_center = log_f0 - mu_logF0

        # Energy (simple RMS)
        frame_len = 1920
        n_frames = wav.shape[0] // frame_len
        e_center = torch.zeros(n_frames)
        for i in range(n_frames):
            frame = wav[i*frame_len:(i+1)*frame_len]
            e_center[i] = torch.log(frame.pow(2).mean() + 1e-8)
        e_center = e_center - e_center.mean()

        # Pad/truncate to match codes length
        T_code = codes.shape[2]
        if log_f0_center.shape[0] > T_code:
            log_f0_center = log_f0_center[:T_code]
            voiced_prob = voiced_prob[:T_code]
        elif log_f0_center.shape[0] < T_code:
            pad = T_code - log_f0_center.shape[0]
            log_f0_center = torch.nn.functional.pad(log_f0_center, (0, pad))
            voiced_prob = torch.nn.functional.pad(voiced_prob, (0, pad))

        if e_center.shape[0] > T_code:
            e_center = e_center[:T_code]
        elif e_center.shape[0] < T_code:
            pad = T_code - e_center.shape[0]
            e_center = torch.nn.functional.pad(e_center, (0, pad))

        entry = {
            'text': wav_name,
            'audio_tokens': codes,  # [1, 16, T]
            'ph_ids': torch.zeros(1, dtype=torch.long),  # placeholder
            'log_f0_center': log_f0_center,
            'e_center': e_center,
            'voicing': voiced[:T_code] if voiced.shape[0] >= T_code else torch.nn.functional.pad(voiced, (0, T_code - voiced.shape[0])),
            'voiced_prob': voiced_prob,
            'mu_logF0': torch.tensor(mu_logF0),
            'mu_logE': torch.tensor(0.0),
            'dur_sec': duration,
            'wav_path': str(wav_path),
        }
        lj_data.append(entry)

    print(f"  {len(lj_data)} LJ Speech entries built")

    # Merge
    merged_data = globe_data + lj_data
    merged = {
        'data': merged_data,
        'prosody_stats': globe['prosody_stats'],
        'sources': ['globe', 'ljspeech'],
        'n_globe': len(globe_data),
        'n_ljspeech': len(lj_data),
    }

    print(f"Saving merged dataset: {len(merged_data)} total samples")
    torch.save(merged, args.output_pt)
    print(f"Saved to {args.output_pt}")


if __name__ == "__main__":
    main()
