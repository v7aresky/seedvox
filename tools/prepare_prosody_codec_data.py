#!/usr/bin/env python
"""prepare_prosody_codec_data.py — Extract prosody-codec targets from audio.

One-time offline preprocessing for the FSQ prosody codec (docs/prosody_codec_design.md).

Per utterance (24 kHz audio) at the Mimi token frame rate (12.5 Hz, hop 1920):
  - log_f0_center[T]:  log-F0, centered by per-utterance median over voiced frames;
                       unvoiced frames set to 0.0 (neutral). NaN removed.
  - e_center[T]:       log-RMS energy, centered by per-utterance mean.
  - voicing[T]:        bool voiced flag (pyin HMM-smoothed).
  - voiced_prob[T]:    pyin voicing probability (auxiliary, for loss weighting).
  - mu_logF0:          per-utterance median log-F0 over voiced frames (register scalar).
  - mu_logE:           per-utterance mean log-energy (loudness scalar).
  - dur_sec:           utterance duration in seconds.
  - wav_path:          resolved absolute path to the 24 kHz audio (for codec input).

Center-only normalization per spec: excursion magnitude IS prosody; never standardize.

Usage:
  # Augment an existing token file (adds the fields above to each item):
  python tools/prepare_prosody_codec_data.py --tokens <tokens.pt> --manifest <manifest> \
      --out <augmented.pt> --data_dir <base> --num_workers 10
  # --limit N limits to the first N utterances (smoke test).
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

_IS_TTY = sys.stderr.isatty()

SR = 24000
FRAME_RATE = 12.5
HOP = int(round(SR / FRAME_RATE))  # 1920
FMIN, FMAX = 50.0, 600.0
PYIN_FRAME = 2048
MEDFILT_K = 1  # masked-median window radius (design: median-filtered F0)


def _load_mono_24k(path):
    try:
        import numpy as _np
        import soundfile as sf
        waveform, sr = sf.read(path, dtype="float32")
        if sr != SR:
            import scipy.signal as sig
            waveform = sig.resample_poly(waveform, SR, sr)
        return _np.asarray(waveform, dtype=_np.float32)
    except Exception:
        pass
    waveform, sr = torchaudio.load(path)
    if sr != SR:
        waveform = torchaudio.transforms.Resample(sr, SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy()  # [T_samples]


def _masked_median_logf0(log_f0, voiced, k=MEDFILT_K):
    """Median-filter log-F0 over voiced frames only; unvoiced stay NaN."""
    out = np.full_like(log_f0, np.nan)
    idx = np.where(voiced)[0]
    for i in idx:
        lo = max(i - k, 0)
        hi = min(i + k + 1, len(log_f0))
        window = np.argwhere(voiced[lo:hi]).ravel() + lo
        out[i] = np.median(log_f0[window])
    return out


def extract_prosody(wav_path):
    """Extract prosody targets for one utterance. Returns dict or None on error."""
    try:
        y = _load_mono_24k(wav_path)
        if y.shape[0] < SR // 2:
            return None

        import librosa
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='pkg_resources')
            f0, vflag, vprob = librosa.pyin(
                y, fmin=FMIN, fmax=FMAX, sr=SR,
                frame_length=PYIN_FRAME, hop_length=HOP,
            )
        T = f0.shape[0]

        voiced = vflag.astype(bool)
        log_f0 = np.full(T, np.nan, dtype=np.float32)
        log_f0[voiced] = np.log(np.maximum(f0[voiced], 1.0))
        log_f0 = _masked_median_logf0(log_f0, voiced)

        if voiced.any():
            mu_logF0 = float(np.median(log_f0[voiced]))
        else:
            mu_logF0 = 0.0

        log_f0_center = log_f0 - mu_logF0
        log_f0_center[~voiced] = 0.0  # neutral constant for unvoiced

        rms = np.zeros(T, dtype=np.float32)
        n_samples = y.shape[0]
        for t in range(T):
            seg = y[t * HOP:(t + 1) * HOP]
            rms[t] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
        log_E = np.log(np.maximum(rms, 1e-7))
        mu_logE = float(log_E.mean())
        e_center = log_E - mu_logE

        return {
            'log_f0_center': log_f0_center.astype(np.float32),
            'e_center': e_center.astype(np.float32),
            'voicing': voiced.astype(np.bool_),
            'voiced_prob': vprob.astype(np.float32),
            'mu_logF0': mu_logF0,
            'mu_logE': mu_logE,
            'dur_sec': float(n_samples) / SR,
            'wav_path': os.path.abspath(wav_path),
        }
    except Exception as e:
        return {'_error': f"{type(e).__name__}: {e}"}


def _attach(item, pr, T_mimi):
    for key in ('log_f0_center', 'e_center', 'voicing', 'voiced_prob'):
        arr = pr[key]
        if arr.shape[0] != T_mimi:
            if arr.shape[0] < T_mimi:
                pad = T_mimi - arr.shape[0]
                arr = np.pad(arr, (0, pad), mode='constant', constant_values=(0,))
            else:
                arr = arr[:T_mimi]
        item[key] = torch.from_numpy(arr)
    item['mu_logF0'] = torch.tensor(pr['mu_logF0'], dtype=torch.float32)
    item['mu_logE'] = torch.tensor(pr['mu_logE'], dtype=torch.float32)
    item['dur_sec'] = pr['dur_sec']
    item['wav_path'] = pr['wav_path']
    return item


def _resolve_path(filepath, data_dir=None):
    if filepath and os.path.exists(filepath):
        return filepath
    if filepath and data_dir:
        joined = os.path.join(data_dir, filepath)
        if os.path.exists(joined):
            return joined
        alt = os.path.join(data_dir, os.path.basename(filepath))
        if os.path.exists(alt):
            return alt
    return None


def get_audio_path(manifest_entry, data_dir=None):
    for key in ('audio_filepath', 'audio_path', 'wav_path'):
        if key in manifest_entry:
            return _resolve_path(manifest_entry[key], data_dir)
    return None


def main():
    ap = argparse.ArgumentParser(description="Extract prosody-codec targets and attach to a token file.")
    ap.add_argument("--tokens", required=True, help="Path to existing train_tokens .pt to augment")
    ap.add_argument("--manifest", required=True, help="JSONL manifest with audio paths")
    ap.add_argument("--out", default=None, help="Output path (default: <tokens>_prosody_codec.pt)")
    ap.add_argument("--data_dir", default=None, help="Base dir for relative audio paths")
    ap.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=0, help="Only process first N utterances (smoke test)")
    args = ap.parse_args()

    raw = torch.load(args.tokens, map_location='cpu', weights_only=False)
    data = raw['data'] if isinstance(raw, dict) and 'data' in raw else raw
    n = len(data)
    print(f"Tokens: {n} items from {args.tokens}")

    with open(args.manifest, 'r') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    print(f"Manifest: {len(lines)} lines from {args.manifest}")

    if len(lines) == n:
        audio_paths = [get_audio_path(m, args.data_dir) for m in lines]
        matched_texts = [data[i].get('text', '') for i in range(n)]
    else:
        text_map = {}
        for i, m in enumerate(lines):
            text_map.setdefault(m.get('normalized_text', m.get('text', '')), []).append(i)
        audio_paths = []
        used = set()
        for item in data:
            key = item.get('text', '')
            cands = [i for i in text_map.get(key, []) if i not in used]
            if cands:
                used.add(cands[0])
                audio_paths.append(get_audio_path(lines[cands[0]], args.data_dir))
            else:
                audio_paths.append(None)
        print(f"Text-matched {sum(p is not None for p in audio_paths)}/{n}")

    missing = [i for i, p in enumerate(audio_paths) if p is None]
    if missing:
        print(f"WARNING: {len(missing)} items have no resolvable audio path (will be skipped).")
    n_work = n if args.limit <= 0 else min(n, args.limit)
    if n_work < n:
        print(f"Smoke-test mode: processing first {n_work} of {n} utterances.")

    results = {}
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(extract_prosody, audio_paths[i]): i
                   for i in range(n_work) if audio_paths[i] is not None}
        for fut in tqdm(futures, total=len(futures), desc="pyin", disable=not _IS_TTY):
            i = futures[fut]
            pr = fut.result()
            if pr is None or '_error' in (pr or {}):
                results[i] = None
                if pr and '_error' in pr and n_fail < 20:
                    print(f"  item {i} FAILED: {pr['_error']}")
                n_fail += 1
            else:
                results[i] = pr

    ok = sum(r is not None for r in results.values())
    print(f"Extracted {ok}/{len(results)} utterances.")

    out_path = args.out or (args.tokens.rsplit('.pt', 1)[0] + '_prosody_codec.pt')
    added = 0
    for i, pr in results.items():
        if pr is None:
            continue
        T_mimi = data[i]['audio_tokens'].shape[2]
        _attach(data[i], pr, T_mimi)
        added += 1

    ok_items = [data[i] for i in results if results.get(i) is not None]
    stats = {
        'frame_rate': FRAME_RATE, 'hop': HOP, 'normalization': 'center_only',
        'n_augmented': added, 'n_total': n,
    }
    if ok_items:
        all_f0 = torch.cat([it['log_f0_center'] for it in ok_items])
        all_e = torch.cat([it['e_center'] for it in ok_items])
        all_v = torch.cat([it['voicing'].float() for it in ok_items])
        stats.update({
            'log_f0_center_std': float(all_f0.std()),
            'e_center_std': float(all_e.std()),
            'voiced_frac': float(all_v.mean()),
        })
    print(f"prosody_stats: {stats}")

    output = {'data': data, 'prosody_stats': stats}
    torch.save(output, out_path)
    print(f"Saved {added}/{n} augmented items to {out_path}")


if __name__ == "__main__":
    main()
