#!/usr/bin/env python3
"""
validate_prosody_planning.py — Planning-level validation of the JEPA prosody planner.

The JEPA prosody is a GLOBAL style latent (num_prosody_tokens pooled blocks over
the whole utterance, mean-centered), so its impact is evaluated in that same
global, frame-independent space — NOT on frame-wise F0.

The model's own teacher (ProsodyCodec) turns a real audio clip into the
ground-truth prosody latent `gt` (F0/E/voicing -> K block vectors, the SAME
space the planner is trained against; NOT the old Mimi-latent bottleneck). The
planner (JEPAProsodyPlanner) predicts a latent `pred` from text alone. If
planning works:

  cos(pred_i, gt_i)   >> cos(pred_j, gt_i)   (text-specific plan)
  cos(pred_i, gt_i)   >> cos(null , gt_i)    (plan beats the no-prosody baseline)
  pred_i is the best match to gt_i among all texts in the batch (discrimination).

Usage:
  python scripts/validate_prosody_planning.py \\
      --config configs/light_fusion_r3.json \\
      --checkpoint checkpoints/seedvox_light_fusion_epoch_93.pt \\
      [--manifest ../autovoc/dataset/train_manifest_jepa.jsonl] \\
      [--num 16] [--device cuda]
"""

import os, sys, json, argparse, torch
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from seedvox.utils.tokenizer import CharTokenizer
from seedvox.utils.text import normalize_text


def encode_prosody_codec(model, wav_path, dev):
    """GT prosody latent in the planner's training space.

    Replicate the stage-2 training teacher exactly: extract F0/E/voicing at
    12.5 Hz (same pyin pipeline as tools/prepare_prosody_codec_data.py), stack
    to [T, 3], pad T to a multiple of num_blocks (32), then run the frozen
    stage-1 ProsodyCodec.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="../autovoc/dataset/train_manifest_jepa.jsonl")
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    cfg = json.load(open(args.config))
    tok = CharTokenizer()

    from explicit_pros_phon_planner.model_fusion import FusionPlannerModel
    model = FusionPlannerModel(cfg, tok.vocab_size, phoneme_vocab_size=128).to(dev)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()

    import os as _os
    if not _os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}")
        return
    lines = [json.loads(l) for l in open(args.manifest)]
    cands = [d for d in lines if d.get("wav_path", "").endswith(".wav") and 3.0 <= float(d.get("duration", 4)) <= 8.0]
    if len(cands) < args.num:
        cands = [d for d in lines if d.get("wav_path", "").endswith(".wav")]
    step = max(1, len(cands) // args.num)
    sel = cands[::step][:args.num]
    print(f"using {len(sel)} utterances")

    null = model.null_prosody.detach()
    preds, gts = [], []
    for i, d in enumerate(sel):
        wav_path = d["wav_path"]
        tn = normalize_text(d["text"])
        t_ids = torch.tensor([tok.encode(tn, normalize=False)], device=dev)
        t_lens = torch.tensor([t_ids.shape[1]], device=dev)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                gt = encode_prosody_codec(model, wav_path, dev)
                if gt is None:
                    print(f"  [{i}] {os.path.basename(wav_path)}  prosody extraction failed, skipping", flush=True)
                    continue

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
                pred = model.jepa_planner(text_feat, text_mask=t_mask).float()
        preds.append(pred[0]); gts.append(gt)
        print(f"  [{i}] {os.path.basename(d['wav_path'])} {float(d['duration']):.1f}s  cos(pred,gt)={torch.nn.functional.cosine_similarity(pred[0], gt, dim=-1).mean():.3f}", flush=True)

    P = torch.stack(preds); G = torch.stack(gts)
    N = P.shape[0]

    def cosmat(a, b):
        a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return (a.unsqueeze(1) * b.unsqueeze(0)).sum(-1).mean(-1)

    def l2mat(a, b):
        return (a.unsqueeze(1) - b.unsqueeze(0)).norm(dim=-1).mean(-1)

    C = cosmat(P, G)
    L = l2mat(P, G)
    diag_c, diag_l = torch.diag(C), torch.diag(L)
    off_c = (C.sum() - C.trace()) / (N * N - N)
    off_l = (L.sum() - L.trace()) / (N * N - N)
    null_c = cosmat(null.expand(N, -1, -1), G).mean().item()
    null_l = l2mat(null.expand(N, -1, -1), G).mean().item()
    self_best = (C.argmax(dim=0) == torch.arange(N, device=C.device)).float().mean().item()

    print()
    print("=== PLANNING-LEVEL VALIDATION (global JEPA latent, frame-independent) ===")
    print(f"cos(pred_i, gt_i)  matched     = {diag_c.mean():.4f}  +/- {diag_c.std():.4f}")
    print(f"cos(pred_j, gt_i)  mismatched  = {off_c:.4f}")
    print(f"cos(null , gt_i)   no-prosody  = {null_c:.4f}")
    print(f"L2  (pred_i, gt_i) matched     = {diag_l.mean():.4f}  +/- {diag_l.std():.4f}")
    print(f"L2  (pred_j, gt_i) mismatched  = {off_l:.4f}")
    print(f"L2  (null , gt_i)  no-prosody  = {null_l:.4f}")
    print(f"fraction where pred_i best-matches its OWN gt : {self_best * 100:.1f}%")
    print()
    print("interpretation:")
    print("  - matched cos >> mismatched & null  -> planner predicts TEXT-SPECIFIC prosody plan")
    print("  - matched cos ~ null                -> planner output carries no real prosody plan")


if __name__ == "__main__":
    main()
