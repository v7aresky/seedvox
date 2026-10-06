#!/usr/bin/env python
"""prepare_token_operator_data.py — Offline pairs for the decoupled prosody knob.

For sampled utterances from the token .pt files:
  - pitch-shift the original wav (librosa, duration-preserving) by each semitone
    delta in the configured set;
  - mimi.encode the shifted wav -> tokens_out (first 16 codebooks);
  - extract F0/energy/voicing from the shifted wav -> frozen ProsodyCodec latent
    (the target plan for the operator);
  - optionally add free "identity" pairs (tokens_out == tokens_in, target == the
    item's ORIGINAL latent) so the operator learns a no-op when plan ~= input.

Saved pair item:
  { 'tokens_in':  [16, T] int64, 'tokens_out': [16, T] int64,
    'target_latent': [32, 512] float32, 'semitone': float, 'dur_sec': float,
    'src': str (original wav path) }

GPU usage: only mimi.encode of the shifted wavs (batched, small). Run alongside
training with a modest --encode_batch.
"""

import os
import sys
import json
import argparse
import random
import shutil
import tempfile
import warnings
import numpy as np
import torch
import torchaudio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from seedvox.prosody_codec import ProsodyCodec
from seedvox.modules.mimi import get_mimi_model
from tools.prepare_prosody_codec_data import extract_prosody

SR = 24000
N_Q = 16
NUM_BLOCKS = 32
DIM = 512


def _load_mono_24k(path):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).numpy()


def _write_mono_24k(path, y):
    torchaudio.save(path, torch.from_numpy(y).unsqueeze(0), SR)


def _feats_to_latent(feats, codec, device, dtype):
    """feats [T,3] (log_f0_center, e_center, voicing) -> [1, 32, 512] latent."""
    T = feats.shape[0]
    T_pad = max(NUM_BLOCKS, ((T + NUM_BLOCKS - 1) // NUM_BLOCKS) * NUM_BLOCKS)
    ft = np.zeros((T_pad, 3), dtype=np.float32)
    ft[:T] = feats
    ft = torch.from_numpy(ft).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        return codec.encode(ft).float().cpu()


def _variant_and_extract(args):
    """WORLD f0-excursion/shift variants for ONE utterance (decompose once, synthesize N).
    Returns list of {'tmp': shifted wav path, 'feats': [T,3], 'dur': sec,
    'f0_scale': float, 'semitone': float, 'key': str} or {'_error': ...}."""
    wav_path, variants, scratch = args
    try:
        import pyworld as pw
        y = _load_mono_24k(wav_path)
        if y.shape[0] < SR // 2:
            return None
        y64 = y.astype(np.float64)
        f0, t = pw.dio(y64, SR, frame_period=5.0)
        f0 = pw.stonemask(y64, f0, t, SR)
        sp = pw.cheaptrick(y64, f0, t, SR)
        ap = pw.d4c(y64, f0, t, SR)
        voiced = f0 > 0
        med = float(np.median(f0[voiced])) if voiced.any() else 150.0
        out = []
        for (scale, st, key) in variants:
            f0n = f0.copy()
            if voiced.any():
                f0n[voiced] = med * 2 ** (st / 12) + scale * (f0[voiced] - med)
                f0n[voiced] = np.maximum(np.minimum(f0n[voiced], 800.0), 40.0)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                ys = pw.synthesize(f0n, sp, ap, SR, frame_period=5.0).astype(np.float32)
            ys = np.minimum(np.maximum(ys, -1.0), 1.0)
            tmp = os.path.join(scratch, f"{os.getpid()}_{st}_{key}.wav")
            _write_mono_24k(tmp, ys)
            pr = extract_prosody(tmp)
            if pr is None or '_error' in pr:
                continue
            feats = np.stack(
                [pr['log_f0_center'], pr['e_center'], pr['voicing'].astype(np.float32)], axis=-1)
            out.append({'tmp': tmp, 'feats': feats.astype(np.float32),
                        'dur': float(y.shape[0]) / SR, 'semitone': float(st),
                        'f0_scale': scale, 'key': key})
        return out
    except Exception as e:
        return {'_error': f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", action="append", required=True,
                    help="Token .pt file(s); repeatable. If given as a JSON array path, read it.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_file", type=int, default=2000, help="max sampled utterances per file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--f0_scales", type=float, nargs="+", default=[0.6, 1.0, 1.6],
                    help="f0 excursion scale (around per-utterance median)")
    ap.add_argument("--semitones", type=int, nargs="+", default=[-3, 3],
                    help="global semitone shifts combined with each f0_scale")
    ap.add_argument("--identity_frac", type=float, default=0.2,
                    help="fraction of utterances that also produce a free no-op pair")
    ap.add_argument("--num_workers", type=int, default=max(2, os.cpu_count() - 4))
    ap.add_argument("--encode_batch", type=int, default=8)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--mimi_checkpoint", default="pretrained_models/best_mimi.pt")
    ap.add_argument("--codec_checkpoint", default="checkpoints/prosody_codec.pt")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(args.seed)

    codec = ProsodyCodec(dim=DIM, num_blocks=NUM_BLOCKS)
    ckpt = torch.load(args.codec_checkpoint, map_location='cpu', weights_only=False)
    codec.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt,
                          strict=False)
    codec.eval().to(device)
    dtype = next(codec.parameters()).dtype

    print(f"[prepare_token_operator_data] Loading frozen Mimi...")
    mimi = get_mimi_model(device=device, checkpoint_path=args.mimi_checkpoint).eval()
    for p in mimi.parameters():
        p.requires_grad = False

    pairs = []
    scratch = args.scratch or tempfile.mkdtemp(prefix='tokop_')
    os.makedirs(scratch, exist_ok=True)
    print(f"[prepare_token_operator_data] scratch={scratch}")

    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    from concurrent.futures import as_completed

    for tp in args.tokens:
        parts = tp.rsplit(':', 1)
        if len(parts) == 2 and parts[1].isdigit():
            tp, per_file = parts[0], int(parts[1])
        else:
            per_file = args.per_file
        raw = torch.load(tp, map_location='cpu', weights_only=False)
        items = raw['data'] if isinstance(raw, dict) and 'data' in raw else raw
        items = [it for it in items if isinstance(it, dict) and it.get('wav_path')
                 and isinstance(it.get('audio_tokens'), torch.Tensor)]
        if per_file > 0:
            items = rng.sample(items, min(len(items), per_file))
        print(f"[prepare_token_operator_data] {os.path.basename(tp)}: {len(items)} utterances")

        tasks = []
        meta = []
        for it in items:
            at = it['audio_tokens'].squeeze(0)  # [16, T]
            if at.shape[0] < N_Q or at.shape[1] < 16:
                continue
            wav_path = it['wav_path']
            if not os.path.exists(wav_path):
                continue
            variants = [(s, st, f"{i}") for i, (s, st) in enumerate(
                [(s, st) for s in args.f0_scales for st in args.semitones])]
            tasks.append((wav_path, variants, scratch))
            meta.append(it)

        # Bounded-window parallel dispatch (a per-task hang cannot stall the loop).
        window = max(2 * args.num_workers, 32)
        pending = {}
        it = iter(zip(tasks, meta))
        n_done = 0
        while True:
            while len(pending) < window:
                try:
                    task, mit = next(it)
                    pending[executor.submit(_variant_and_extract, task)] = mit
                except StopIteration:
                    break
            if not pending:
                break
            for f in as_completed(pending, timeout=120.0):
                mit = pending.pop(f)
                res = f.result(timeout=5.0)
                n_done += 1
                if res is not None and '_error' in res:
                    print(f"[prepare_token_operator_data] task error: {res['_error']}")
                    continue
                if res is None:
                    continue
                at = mit['audio_tokens'].squeeze(0)
                for r in res:
                    pairs.append({'r': r, 'tokens_in': at, 'orig_latent': None})
                if args.identity_frac > 0 and rng.random() < args.identity_frac:
                    if all(k in mit for k in ('log_f0_center', 'e_center', 'voicing')):
                        feats = torch.stack(
                            [mit['log_f0_center'], mit['e_center'], mit['voicing'].float()],
                            dim=-1)
                        pairs.append({'r': {'feats': feats.numpy().astype(np.float32),
                                            'dur': mit.get('dur_sec', at.shape[1] / 12.5),
                                            'semitone': 0.0, 'f0_scale': 1.0,
                                            'key': 'id', 'tmp': None},
                                      'tokens_in': at, 'orig_latent': True})
                break  # refill window
        print(f"[prepare_token_operator_data] {n_done} utterances processed")
    executor.shutdown(wait=True)

    print(f"[prepare_token_operator_data] {len(pairs)} raw results -> encoding on {device}")

    out_items = []
    n_fail = 0
    for i in range(0, len(pairs), args.encode_batch):
        chunk = pairs[i:i + args.encode_batch]
        valid = [p for p in chunk if p['r']['tmp'] is not None or p['orig_latent']]
        if not valid:
            continue
        wavs, rmeta, tin = [], [], []
        for p in valid:
            if p['r']['tmp'] is not None:
                wavs.append(_load_mono_24k(p['r']['tmp']))
            rmeta.append(p['r'])
            tin.append(p['tokens_in'])
        if wavs:
            L = max(w.shape[0] for w in wavs)
            ws = np.zeros((len(wavs), 1, L), dtype=np.float32)
            for j, w in enumerate(wavs):
                ws[j, 0, :w.shape[0]] = w
            wt = torch.from_numpy(ws).to(device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                codes = mimi.encode(wt)
            codes = codes[:, :N_Q].long().cpu()
        else:
            codes = None

        ci = 0
        for p in chunk:
            if p['r']['tmp'] is not None and p['orig_latent'] is None:
                tin_use = p['tokens_in']
                Tout = codes[ci]
                ci += 1
                L = min(tin_use.shape[1], Tout.shape[1])
                if L < 16 or abs(tin_use.shape[1] - Tout.shape[1]) > 4:
                    continue
                lat = _feats_to_latent(p['r']['feats'], codec, device, dtype)
                # Content-locked target: keep the ORIGINAL semantic codebook (c0),
                # take the shifted acoustic codebooks 1..15 (prosody lives there).
                Tout = Tout[:, :L].clone()
                Tout[0] = tin_use[0, :L]
            elif p['orig_latent']:
                tin_use = p['tokens_in']
                Tout = tin_use
                L = tin_use.shape[1]
                lat = _feats_to_latent(p['r']['feats'], codec, device, dtype)
            else:
                continue
            out_items.append({
                'tokens_in': tin_use[:, :L].clone(),
                'tokens_out': Tout[:, :L].clone(),
                'target_latent': lat[0],
                'semitone': p['r']['semitone'],
                'f0_scale': p['r'].get('f0_scale', 1.0),
                'dur_sec': p['r']['dur'],
                'src': None,
            })
            if out_items[-1]['target_latent'].isnan().any():
                n_fail += 1
                out_items.pop()

    print(f"[prepare_token_operator_data] {len(out_items)} pairs saved (failed {n_fail})")
    torch.save({'data': out_items, 'semitones': args.semitones,
                'identity_frac': args.identity_frac,
                'frame_rate': 12.5, 'sr': SR,
                'codec': {'dim': DIM, 'num_blocks': NUM_BLOCKS},
                'n_q': N_Q}, args.out)
    print(f"[prepare_token_operator_data] saved -> {args.out}")

    if args.scratch is None:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
