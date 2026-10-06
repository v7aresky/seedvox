"""Precompute contour-scaled exemplar tokens for token-operator training (Run 6).

For each dataset item (T <= max_len): decode orig audio -> est log-F0 trace ->
for each grid scale s synthesize harmonic audio whose absolute log-F0 trace is
mean + s*(lf-mean) (same synthesis as feasibility_contour_scale) -> mimi encode
-> keep the first 16 levels (c0 + 15 acoustic, matching the operator layout) ->
store keyed by wav_path. These are the ground-truth token edits that realize a
contour excursion scale s; the trainer supervises the operator's logits against
them with dense cross-entropy (giving the plan path CE-strength gradients).

Run:
  python explicit_pros_phon_planner/build_exemplars.py \
      --data ../autovoc/dataset/train_tokens_prosody_zp_prosody_codec.pt \
      --out checkpoints/operator_exemplars.pt
"""
import sys
import os
import argparse
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from seedvox.modules.mimi import get_mimi_model
from explicit_pros_phon_planner.f0_estimator import F0Estimator
from explicit_pros_phon_planner.feasibility_contour_scale import synth_contour, HOP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs='+', required=True)
    ap.add_argument("--out", default="checkpoints/operator_exemplars.pt")
    ap.add_argument("--grid", default="0.6,0.8,1.0,1.2,1.5,1.8")
    ap.add_argument("--max_len", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mimi_checkpoint", default="pretrained_models/best_mimi.pt")
    ap.add_argument("--estimator_checkpoint", default="checkpoints/f0_estimator_best.pt")
    args = ap.parse_args()
    grid = [float(x) for x in args.grid.split(",")]
    dev = args.device

    torch.set_grad_enabled(False)
    mimi = get_mimi_model(device=dev, checkpoint_path=args.mimi_checkpoint).eval()
    est = F0Estimator().eval().to(dev)
    est.load_state_dict(torch.load(args.estimator_checkpoint, map_location='cpu',
                                   weights_only=False)['model'])

    items = []
    for p in args.data:
        d = torch.load(p, map_location='cpu', weights_only=False)
        data = d['data'] if isinstance(d, dict) and 'data' in d else d
        items.extend(data)
    items = [it for it in items if 'audio_tokens' in it and 'wav_path' in it
             and it['audio_tokens'].shape[-1] <= args.max_len]
    print(f"[exemplars] {len(items)} items, grid {grid}", flush=True)

    exemplars = {}
    for i, it in enumerate(items):
        tok = it['audio_tokens'].to(dev)
        vo = it['voicing'].float().to(dev)
        T = tok.shape[-1]
        N = T * HOP
        wav0 = mimi.decode(tok)[:, :, :N]
        lf0, _ = est(wav0)
        xs = torch.stack([synth_contour(wav0[0, 0], lf0[0], vo, T, s) for s in grid], 0)
        xs = xs[:, None, :]                                # [n_s, 1, N]
        codes = mimi.encode(xs)                            # [n_s, nq, T]
        if codes.dim() == 4:
            codes = codes[:, 0]
        if codes.shape[-1] != T:
            print(f"[exemplars] len mismatch {codes.shape[-1]} vs {T}, skipping", flush=True)
            continue
        out = {}
        for k, s in enumerate(grid):
            out[s] = codes[k, :16, :T].to('cpu', dtype=torch.int16)
        exemplars[it['wav_path']] = out
        if (i + 1) % 200 == 0:
            print(f"[exemplars] {i + 1}/{len(items)}", flush=True)
    torch.save({'grid': grid, 'max_len': args.max_len, 'exemplars': exemplars}, args.out)
    print(f"[exemplars] saved {len(exemplars)} items -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
