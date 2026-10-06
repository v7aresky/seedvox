#!/usr/bin/env python3
"""
prosody_audio_metrics.py — Audio-level prosody-planning metrics.

The JEPA prosody signal is a GLOBAL latent (mean-centered, num_prosody_tokens
pooled blocks), so frame-wise F0 is the wrong instrument. This evaluates the
generated audio against the SAME global latent space:

1. PLAN-FOLLOW: run the generated wav back through the ProsodyCodec and
   compare its latent to the planner's prediction `pred` vs the no-prosody
   baseline `null`. If the decoder realizes the plan, cos(gen, pred) should
   exceed cos(gen, null), and that gap should grow with exagg. Include a
   `random` condition (injected prosody = null + noise) as a control:
   planned gap ≈ random gap means the signal is a content confound, not
   plan-follow.

2. WITHIN vs ACROSS noise: MFCC distance between different seeds of the SAME
   condition (sampling noise) vs distance between conditions on MATCHED seeds
   (prosody effect). ratio = across/within; >1 means exagg produces a reliable
   acoustic change above sampling noise. A global style needs only a modest
   ratio — it does NOT have to move frame-wise pitch.

Usage:
  python scripts/prosody_audio_metrics.py \\
      --config configs/light_fusion_r3.json \\
      --checkpoint checkpoints/seedvox_light_fusion_epoch_93.pt \\
      --demo_file demo_prosody.txt \\
      --output_dir demos/prosody_test \\
      --conditions exagg00_0 exagg00_5 exagg01_0 random \\
      [--num_seeds 3] [--sr 24000] [--device cuda]
"""

import os, sys, json, argparse, glob, re, torch, torchaudio
import numpy as np
from librosa.feature import mfcc
from librosa.sequence import dtw

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from seedvox.utils.tokenizer import CharTokenizer
from seedvox.utils.text import normalize_text


def encode_prosody_codec(model, wav_path, dev):
    """Latent in the planner's training space (stage-1 ProsodyCodec, NOT the
    old Mimi-latent bottleneck). Same pyin pipeline and mult-32 padding as
    training: tools/prepare_prosody_codec_data.py -> model.prosody_codec.encode.
    """
    import numpy as np
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
    from prepare_prosody_codec_data import extract_prosody
    pr = extract_prosody(wav_path)
    if pr is None or '_error' in pr:
        return None
    feat = torch.from_numpy(np.stack(
        [pr['log_f0_center'], pr['e_center'], pr['voicing'].astype(np.float32)], axis=-1))  # [T,3]
    a_pros = ((feat.shape[0] + 31) // 32) * 32
    if a_pros > feat.shape[0]:
        feat = torch.nn.functional.pad(feat, (0, 0, 0, a_pros - feat.shape[0]))
    with torch.no_grad():
        z = model.prosody_codec.encode(feat.unsqueeze(0).to(dev)).detach().float()  # [1, K, dim]
    return z[0]


def load_model(cfg_path, ckpt_path, dev):
    cfg = json.load(open(cfg_path))
    tok = CharTokenizer()
    from explicit_pros_phon_planner.model_fusion import FusionPlannerModel
    model = FusionPlannerModel(cfg, tok.vocab_size, phoneme_vocab_size=128).to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()
    return model, tok


def compute_pred(model, tok, text, dev):
    tn = normalize_text(text)
    t_ids = torch.tensor([tok.encode(tn, normalize=False)], device=dev)
    t_lens = torch.tensor([t_ids.shape[1]], device=dev)
    text_feat, _ = model.encode_text(t_ids, t_lens, raw_texts=[tn])
    if model.use_bpe_encoder:
        from seedvox.bpe_char_encoder import BPECharCollator
        bc = BPECharCollator(model.bpe_encoder)
        bpe_ids, bpe_lens, c2b = bc.process_batch_texts([tn], t_lens, dev)
        _, T_char = c2b.shape
        padded_c2b = torch.zeros((1, T_char + 2), dtype=c2b.dtype, device=dev)
        padded_c2b[:, 1:1 + T_char] = c2b
        wcl = t_lens + 2
        bpe_ctx = model.bpe_encoder.forward_bpe(bpe_ids, bpe_lens, device=dev)
        bpe_exp = model.bpe_encoder.expand_to_chars(bpe_ctx, padded_c2b, wcl, device=dev)
        if bpe_exp.shape[1] < text_feat.shape[1]:
            bpe_exp = torch.nn.functional.pad(bpe_exp, (0, 0, 0, text_feat.shape[1] - bpe_exp.shape[1]))
        elif bpe_exp.shape[1] > text_feat.shape[1]:
            bpe_exp = bpe_exp[:, :text_feat.shape[1]]
        text_feat = text_feat + torch.sigmoid(model.bpe_gate) * bpe_exp
    t_mask = torch.arange(text_feat.shape[1], device=dev).unsqueeze(0) >= (t_lens + 2).unsqueeze(1)
    with torch.no_grad():
        pred = model.jepa_planner(text_feat, text_mask=t_mask).float()
    return pred


def gen_latent(model, wav_path, dev):
    z = encode_prosody_codec(model, wav_path, dev)
    if z is None:
        return None
    return z  # [K, dim]


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def _exagg_val(name):
    """Parse the exagg value from a condition name like 'exagg00_5' -> 0.0."""
    m = re.match(r"exagg(\d+(?:\.\d+)?)", name.lower())
    return float(m.group(1)) if m else None


def mfcc_dist(a, b):
    D, wp = dtw(a, b, metric="euclidean")
    return float(D[-1, -1] / len(wp))


def parse_demo(demo_file):
    entries = []
    for i, l in enumerate(open(demo_file)):
        l = l.rstrip("\n")
        if not l.strip() or l.startswith("#"):
            continue
        fields = [f.strip() for f in l.split(" || ")]
        text = fields[0]
        name = fields[1] if len(fields) >= 2 and fields[1] else f"demo_{i+1:03d}"
        if name.lower().endswith(".wav"):
            name = name[:-4]
        entries.append((text, name))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--demo_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--num_seeds", type=int, default=3)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ref_wav_prosody", default=None,
                    help="Reference wav for the 'ref' condition: its stage-1 codec latent is the plan target")
    args = ap.parse_args()

    dev = args.device
    model, tok = load_model(args.config, args.checkpoint, dev)
    null = model.null_prosody.detach()
    ref_latent = None
    if args.ref_wav_prosody:
        ref_latent = encode_prosody_codec(model, args.ref_wav_prosody, dev)
        if ref_latent is None:
            print(f"WARNING: prosody extraction failed for {args.ref_wav_prosody} — 'ref' condition will be NaN")
    entries = parse_demo(args.demo_file)

    def plan_target(c, pred):
        """Plan the generated audio is compared against: the injected reference
        latent for the 'ref' condition (planner was bypassed), else the
        planner's prediction for this line."""
        if c == "ref":
            return ref_latent
        return pred

    # 1. Plan-follow: cos(gen_latent, pred) vs cos(gen_latent, null)
    print("=== PLAN-FOLLOW (global latent of generated audio vs plan/null) ===")
    print(f"{'line':<22}" + "".join(f"{c:>14}" for c in args.conditions))
    plan_gap = {c: [] for c in args.conditions}
    for text, name in entries:
        pred = compute_pred(model, tok, text, dev)
        row = f"{name:<22}"
        for c in args.conditions:
            wavs = sorted(glob.glob(os.path.join(args.output_dir, c, f"{name}_*.wav")))
            if len(wavs) < args.num_seeds:
                wavs = sorted(glob.glob(os.path.join(args.output_dir, c, f"{name}.wav")))
            cs_p, cs_n = [], []
            for wp in wavs:
                lat = gen_latent(model, wp, dev)
                if lat is None:
                    continue
                target = plan_target(c, pred)
                if target is None:
                    continue
                cs_p.append(cos(lat, target)); cs_n.append(cos(lat, null))
            if not cs_p:
                gap = float("nan")
            else:
                gap = np.mean(cs_p) - np.mean(cs_n)
            plan_gap[c].append(gap)
            row += f"{gap:>14.3f}"
        print(row)
    print("  (cell = cos(gen,plan) - cos(gen,null); >0 means output carries the plan,")
    print("   growing with exagg means the dial is realized in the audio;")
    print("   'ref' compares against the INJECTED reference latent, not the planner pred)")
    print(f"  mean gap per condition: " + "  ".join(f"{c}={np.nanmean(plan_gap[c]):.3f}" for c in args.conditions))
    base_gap = np.nanmean(plan_gap[args.conditions[0]])
    print("  increment over exagg0 (prosody effect; exagg0 gap is the content confound):")
    print("  " + "  ".join(f"{c}={np.nanmean(plan_gap[c]) - base_gap:+.3f}" for c in args.conditions[1:]))

    # --- Control-gated verdict ---
    ctrl = [c for c in args.conditions if "random" in c.lower()]
    ctrl_gap = np.nanmean([np.nanmean(plan_gap[c]) for c in ctrl]) if ctrl else float("nan")
    planned = [c for c in args.conditions if "random" not in c.lower()]
    exagg_ordered = sorted(planned, key=lambda c: _exagg_val(c) if _exagg_val(c) is not None else -1.0)
    dial_incs = [(c, np.nanmean(plan_gap[c]) - base_gap)
                 for c in exagg_ordered if c != args.conditions[0]]
    print("\n  VERDICT (plan-follow):")
    if not np.isnan(ctrl_gap):
        over_random = base_gap - ctrl_gap
        if over_random > 0.05:
            print(f"  DECODER CARRIES THE PLAN: planned gap exceeds the random control by {over_random:+.3f}.")
        else:
            print(f"  DECODER DOES NOT CARRY THE PLAN: planned gap ≈ random control "
                  f"(diff {over_random:+.3f}). The gap is a CONTENT CONFOUND, not the injected prosody plan.")
    else:
        print("  NOTE: no 'random' control condition found — add one to separate plan-follow from content confound.")
    if dial_incs and all(i > 0 for _, i in dial_incs):
        print(f"  EXAGG DIAL REALIZED: gap grows with exagg "
              f"(increments {' '.join(f'{i:+.3f}' for _, i in dial_incs)}).")
    elif dial_incs:
        print(f"  EXAGG DIAL NOT REALIZED: gap does NOT grow with exagg "
              f"(increments {' '.join(f'{i:+.3f}' for _, i in dial_incs)}; all should be > 0).")

    # 2. Within vs across MFCC distance
    print()
    print("=== MFCC WITHIN vs ACROSS (sampling noise vs prosody effect) ===")
    print(f"{'line':<22}" + "".join(f"{c:>14}" for c in args.conditions))
    base = args.conditions[0]
    ratios = []
    for text, name in entries:
        feats = {}
        for c in args.conditions:
            wavs = sorted(glob.glob(os.path.join(args.output_dir, c, f"{name}_*.wav")))
            if len(wavs) < args.num_seeds:
                wavs = sorted(glob.glob(os.path.join(args.output_dir, c, f"{name}.wav")))
            feats[c] = []
            for wp in wavs:
                w, sr = torchaudio.load(wp)
                y = w.mean(0).numpy().astype(np.float32)
                feats[c].append(mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512))
        # within-condition seed noise (avg over all conditions)
        within = [mfcc_dist(feats[c][i], feats[c][j])
                  for c in args.conditions
                  for i in range(len(feats[c])) for j in range(i + 1, len(feats[c]))]
        wmean = np.mean(within) if within else float("nan")
        # across-condition on matched seeds (vs baseline condition)
        n_seed = min(len(feats[c]) for c in args.conditions)
        across = [mfcc_dist(feats[c][s], feats[base][s])
                  for c in args.conditions if c != base for s in range(n_seed)]
        amean = np.mean(across) if across else float("nan")
        ratio = amean / wmean if wmean > 0 else float("nan")
        ratios.append(ratio)
        row = f"{name:<22}"
        for c in args.conditions:
            if c == base:
                cell = f"within={wmean:.0f}"
            else:
                d = np.mean([mfcc_dist(feats[c][s], feats[base][s]) for s in range(n_seed)])
                cell = f"{d:.0f}"
            row += f"{cell:>14}"
        row += f"  across/within={ratio:.2f}"
        print(row)
    print("  interpretation: across/within > 1 means exagg changes the acoustics above")
    print("  seed noise (a global style needs only a modest ratio — not frame-wise F0).")
    ok = [r for r in ratios if not np.isnan(r)]
    if ok:
        m = np.mean(ok)
        if m > 1.2:
            print(f"  VERDICT: STRONG — mean across/within {m:.2f}; exagg reliably changes the audio (probe reference ~1.2-1.3).")
        elif m > 1.05:
            print(f"  VERDICT: WEAK/MARGINAL — mean across/within {m:.2f}; barely above seed noise, not a reliable dial yet.")
        else:
            print(f"  VERDICT: NONE — mean across/within {m:.2f}; exagg does not change the audio above seed noise.")


if __name__ == "__main__":
    main()
