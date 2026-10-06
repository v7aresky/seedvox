#!/usr/bin/env python
"""train_operator_diff.py — All-differentiable token-operator training (no WORLD).

End-to-end on real audio, no cross-utterance alignment:
  tokens + plan_latent(warped) -> soft logits -> soft-mixture mimi decode (c0
  hard-locked) -> frozen F0 estimator -> L1 vs warped real f0 (centered),
  plus identity CE against the original tokens (c0 excluded, padding masked).

The warp is a latent-space excursion scale s (plan_latent * s); the target is
the same warp applied to the real centered f0 trace (log_f0_center * s). Only
the operator is trained (mimi / estimator / prosody codec all frozen).

Run 6 (--exemplar_path): dense exemplar CE. The audio-grounded f0 loss alone
cannot move the plan path (its gradient through est->soft-decode->logits is
~100x weaker than the token path), so the operator falls back to a constant
excursion compromise (exc*s == const, slope == 0). With precomputed exemplar
tokens (checkpoints/operator_exemplars.pt, built by build_exemplars.py) the
operator is additionally supervised with dense CE against the ground-truth
token edit that realizes scale s, giving the plan path CE-strength gradients.
The scale is sampled per batch from the exemplar grid (homogeneous batches).

Usage:
  python -m explicit_pros_phon_planner.train_operator_diff \
      --data <file1.pt> [<file2.pt> ...] --out checkpoints/token_operator_diff.pt \
      --exemplar_path checkpoints/operator_exemplars.pt --lambda_ex 1.0
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from explicit_pros_phon_planner.token_operator import TokenOperator
from explicit_pros_phon_planner.soft_mimi import SoftMimi

PAD_T = -100
HOP = 1920


class OpDataset(Dataset):
    def __init__(self, items, max_len=0):
        self.items = [it for it in items
                      if 'audio_tokens' in it and 'log_f0_center' in it and 'voicing' in it]
        if max_len:
            self.items = [it for it in self.items if it['audio_tokens'].shape[-1] <= max_len]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        tok = it['audio_tokens']                     # [1, 16, T]
        f0c = it['log_f0_center'].float()
        e = it['e_center'].float()
        vo = it['voicing'].bool()
        wav = it.get('wav_path', f'item_{i}')
        return tok, f0c, e, vo, wav


def make_collate(grid, ex_map):
    def collate(batch):
        toks, f0c, e, vo, wavs = zip(*batch)
        B = len(batch)
        ex_scale = float(np.random.choice(grid)) if ex_map is not None else None
        Tm = max(t.shape[-1] for t in toks)
        Tp = ((Tm + 31) // 32) * 32                     # pad to a codec-block multiple
        tok_p = torch.zeros(B, 16, Tp, dtype=torch.long)
        f0_p = torch.zeros(B, Tp, dtype=torch.float32)
        e_p = torch.zeros(B, Tp, dtype=torch.float32)
        vo_p = torch.zeros(B, Tp, dtype=torch.bool)
        lens = torch.zeros(B, dtype=torch.long)
        ce_target = torch.full((B, 16, Tp), PAD_T, dtype=torch.long)
        ex_target = torch.full((B, 16, Tp), PAD_T, dtype=torch.long)
        for i, (t, f, e_, v) in enumerate(zip(toks, f0c, e, vo)):
            L = t.shape[-1]
            tok_p[i, :, :L] = t[0, :, :L]
            f0_p[i, :L] = f[:L]
            e_p[i, :L] = e_[:L]
            vo_p[i, :L] = v[:L]
            lens[i] = L
            ce_target[i, :, :L] = t[0, :, :L]
        ce_target[:, 0] = PAD_T                          # c0 content-locked, never supervised
        ex_target[:, 0] = PAD_T
        if ex_map is not None:
            for i, w in enumerate(wavs):
                d = ex_map.get(w)
                if d is None or ex_scale not in d:
                    continue
                c = d[ex_scale]                          # [16, T] int16
                L = int(lens[i])
                ex_target[i, :, :L] = torch.from_numpy(c[:, :L].numpy()).long()
        return tok_p, f0_p, e_p, vo_p, ce_target, lens, ex_target, ex_scale
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs='+', required=True)
    ap.add_argument("--out", default="checkpoints/token_operator_diff.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--max_len", type=int, default=250)
    ap.add_argument("--lambda_ce", type=float, default=0.01,
                    help="identity-CE weight (F0 loss is the driver; CE anchors content)")
    ap.add_argument("--anneal_ce", action="store_true",
                    help="anneal the identity-CE weight linearly from lambda_ce down to "
                         "lambda_ce*0.1 over training: anchor content early, then let the "
                         "plan edits grow once the copy mechanism is learned")
    ap.add_argument("--s_min", type=float, default=0.6)
    ap.add_argument("--s_max", type=float, default=1.8)
    ap.add_argument("--f0_weight", type=float, default=1.0,
                    help="raw multiplier on the (target-normalized) F0 loss")
    ap.add_argument("--lambda_aux", type=float, default=0.3,
                    help="weight of the plan-control pitch-readout aux loss (drives "
                         "the plan path + GST bank with a strong direct gradient)")
    ap.add_argument("--no_ste", action="store_true",
                    help="use soft-mixture decode in the loss instead of straight-through hard decode")
    ap.add_argument("--exemplar_path", default=None,
                    help="precomputed scaled-token exemplars (build_exemplars.py); enables "
                         "dense exemplar CE as the plan-path driver")
    ap.add_argument("--grid", default="0.6,0.8,1.0,1.2,1.5,1.8",
                    help="scale grid for exemplar training (sampled per batch)")
    ap.add_argument("--lambda_ex", type=float, default=1.0,
                    help="weight of the dense exemplar CE (the plan-path driver)")
    ap.add_argument("--scale_in_features", action="store_true",
                    help="build the plan as codec.encode([f0c*s, e, vo]) instead of "
                         "latent-scale L*s (so only the f0 dimension is scaled, not "
                         "energy/voicing latent dims)")
    ap.add_argument("--edit_levels", default=None,
                    help="comma list of codebook rows the operator may edit (e.g. "
                         "'1-8'); all other rows are structurally pinned to the input "
                         "tokens. Prevents corrupting high residual levels (9..15) to "
                         "fool the estimator, which rendered run 6g unintelligible.")
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--eval_scales", type=str, default="0.6,1.0,1.4,1.8")
    ap.add_argument("--mimi_checkpoint", default="pretrained_models/best_mimi.pt")
    ap.add_argument("--codec_checkpoint", default="checkpoints/prosody_codec.pt")
    ap.add_argument("--estimator_checkpoint", default="checkpoints/f0_estimator_best.pt")
    ap.add_argument("--plan_dim", type=int, default=512)
    ap.add_argument("--plan_blocks", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    eval_scales = [float(x) for x in args.eval_scales.split(",")]
    edit_levels = None
    if args.edit_levels:
        edit_levels = []
        for part in args.edit_levels.split(","):
            if "-" in part:
                a, b = map(int, part.split("-"))
                edit_levels += list(range(a, b + 1))
            else:
                edit_levels.append(int(part))
        print(f"[opdiff] editable codebook rows: {edit_levels}")

    items = []
    for p in args.data:
        d = torch.load(p, map_location='cpu', weights_only=False)
        data = d['data'] if isinstance(d, dict) and 'data' in d else d
        items.extend(data)
        print(f"[opdiff] {p}: {len(data)} utts")
    print(f"[opdiff] total {len(items)}")

    ds = OpDataset(items, max_len=args.max_len)
    print(f"[opdiff] after max_len={args.max_len}: {len(ds)}")
    n_val = max(8, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    if args.limit:
        train_ds, _ = random_split(train_ds, [min(args.limit, len(train_ds)),
                                              max(0, len(train_ds) - min(args.limit, len(train_ds)))],
                                   generator=gen)
    ex_map = None
    grid = [float(x) for x in args.grid.split(",")]
    if args.exemplar_path and os.path.exists(args.exemplar_path):
        ed = torch.load(args.exemplar_path, map_location='cpu', weights_only=False)
        ex_map = ed['exemplars']
        print(f"[opdiff] exemplars: {len(ex_map)} items, grid {ed['grid']}")
    collate_fn = make_collate(grid, ex_map)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    print(f"[opdiff] train {len(train_ds)} val {len(val_ds)}")

    from seedvox.modules.mimi import get_mimi_model
    from seedvox.prosody_codec import ProsodyCodec
    from explicit_pros_phon_planner.f0_estimator import F0Estimator

    mimi = get_mimi_model(device=device, checkpoint_path=args.mimi_checkpoint).eval()
    soft = SoftMimi(mimi).to(device).eval()
    codec = ProsodyCodec(dim=args.plan_dim, num_blocks=args.plan_blocks).eval().to(device)
    ck = torch.load(args.codec_checkpoint, map_location='cpu', weights_only=False)
    codec.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck, strict=False)
    for p in codec.parameters():
        p.requires_grad = False
    est = F0Estimator().eval().to(device)
    ck = torch.load(args.estimator_checkpoint, map_location='cpu', weights_only=False)
    est.load_state_dict(ck['model'])
    for p in est.parameters():
        p.requires_grad = False

    model = TokenOperator(plan_dim=args.plan_dim, plan_blocks=args.plan_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[opdiff] operator params {n_params/1e6:.2f}M")

    nq, card = 16, 2048
    w = torch.ones(nq)
    w[:nq // 2] = 1.5
    w[nq // 2:] = 0.5
    level_w = (w / w.sum() * nq).to(device)
    ce_crit = nn.CrossEntropyLoss(ignore_index=PAD_T)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_loader))
    sched = get_cosine_schedule_with_warmup(
        opt, args.warmup_steps, steps_per_epoch * args.epochs, num_cycles=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_epoch, step = 0, 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        start_epoch = ck.get('epoch', 0) + 1
        step = ck.get('step', 0)
        print(f"[opdiff] resumed epoch {start_epoch} step {step}")

    def center(x, v):
        m = (x * v).sum(dim=-1, keepdim=True) / v.sum(dim=-1, keepdim=True).clamp(min=1)
        return x - m

    def run_eval():
        model.eval()
        res = {}
        for s in eval_scales:
            ratios, corrs, idents, n = [], [], [], 0
            with torch.no_grad():
                for toks, f0c, e_, vo, ce_t, lens, ex_t, ex_s in val_loader:
                    B = toks.shape[0]
                    toks, f0c, e_, vo = toks.to(device), f0c.to(device), e_.to(device), vo.to(device)
                    feats = torch.stack([f0c, e_, vo.float()], dim=-1)
                    L = codec.encode(feats)
                    if args.scale_in_features:
                        feats_s = torch.stack([f0c * s, e_, vo.float()], dim=-1)
                        Ls = codec.encode(feats_s)
                        edits = model.edit(toks, Ls, edit_levels=edit_levels)
                    else:
                        edits = model.edit(toks, L * s, edit_levels=edit_levels)
                    wav = soft.hard_decode(edits)
                    est_lf = est(wav.float())[0]
                    wav_in = soft.hard_decode(toks)
                    est_in_lf = est(wav_in.float())[0]
                    F = int(lens.max())
                    vf = vo[:, :F].float()
                    est_c = center(est_lf[:, :F], vf)
                    est_in_c = center(est_in_lf[:, :F], vf)
                    tgt_c = est_in_c * s                    # input decode re-baselined
                    for i in range(B):
                        Li = int(lens[i])
                        v = vo[i, :Li]
                        if v.sum() < 3:
                            continue
                        ec, tc = est_c[i, :Li][v], tgt_c[i, :Li][v]
                        ratios.append((ec.std() / tc.std().clamp(min=1e-3)).item())
                        corrs.append(torch.corrcoef(torch.stack([ec, tc]))[0, 1].item())
                        idents.append((edits[i, 1:, :Li] == toks[i, 1:, :Li]).float().mean().item())
                        n += 1
            if n:
                res[f"exc_{s}"] = float(np.mean(ratios))
                res[f"corr_{s}"] = float(np.mean(corrs))
                res[f"ident_{s}"] = float(np.mean(idents))
        # acceptance: exc_ratio -> 1 at every scale (decoded excursion == target),
        # not just corr (shape preserved trivially by identity output).
        res["rdev"] = float(np.mean([abs(np.log(res[f"exc_{s}"])) for s in eval_scales]))
        if len(eval_scales) >= 2:
            ex = [res[f"exc_{s}"] * s for s in eval_scales]     # decoded excursion (×σ)
            ls = [math.log(s) for s in eval_scales]
            le = [math.log(max(e, 1e-3)) for e in ex]
            res["slope"] = float((le[-1] - le[0]) / (ls[-1] - ls[0]))  # ideal = 1.0
        model.train()
        return res

    pbar = tqdm(total=steps_per_epoch * (args.epochs - start_epoch))
    for epoch in range(start_epoch, args.epochs):
        pbar.set_description(f"epoch {epoch}/{args.epochs}")
        model.train()
        for toks, f0c, e_, vo, ce_t, lens, ex_t, ex_s in train_loader:
            toks, f0c, e_, vo, ce_t, ex_t = (toks.to(device), f0c.to(device), e_.to(device),
                                             vo.to(device), ce_t.to(device), ex_t.to(device))
            B = toks.shape[0]
            # One scale for the whole batch: mixed-s batches cancel gradients at the
            # constant-excursion compromise (some samples push up, some down). A
            # homogeneous batch makes every sample push the same direction, forcing
            # the operator to actually read the (per-batch) warp scale from the plan.
            # With exemplars the scale comes from the collate grid pick.
            if ex_s is not None:
                s = torch.full((B,), ex_s, device=device)
            else:
                s = (args.s_max - args.s_min) * torch.rand(1, device=device) + args.s_min
                s = s.expand(B)
            feats = torch.stack([f0c, e_, vo.float()], dim=-1)
            with torch.no_grad():
                L = codec.encode(feats)
                if args.scale_in_features:
                    feats_s = torch.stack([f0c * s[:, None], e_, vo.float()], dim=-1)
                    plan = codec.encode(feats_s)
                else:
                    plan = L * s[:, None, None]
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, ctl = model(toks, plan, return_ctl=True,
                                    lock_levels=([j for j in range(nq)
                                                  if j not in edit_levels]
                                                 if edit_levels else None))
                wav = soft.soft_decode(logits, c0_hard=toks[:, 0], ste=not args.no_ste)
            wav = wav.float()
            wavs = []
            Np = int(lens.max()) * HOP
            for i in range(B):
                Li = int(lens[i])
                w = wav[i, 0, :Li * HOP]
                wavs.append(w)
            wav_p = torch.zeros(B, 1, Np, device=device)
            for i, w in enumerate(wavs):
                wav_p[i, 0, :w.shape[-1]] = w
            est_lf, _ = est(wav_p)                      # fp32, frozen
            F = int(lens.max())
            vf = vo[:, :F].float()
            est_c = center(est_lf, vf)
            # Re-baselined target: the mimi codec re-decode reads ~0.55-0.9x of the
            # dataset's ground-truth f0 excursion (measured), so targets built from
            # dataset f0c are inflated and the operator learned "do nothing at s>1".
            # Score against the *measured* contour of the input's own decode, scaled
            # by s: identity is optimal at s=1 and every scale is internally consistent.
            with torch.no_grad():
                wav_in = soft.hard_decode(toks).float()
                wav_in_p = torch.zeros(B, 1, Np, device=device)
                for i in range(B):
                    Li = int(lens[i])
                    wav_in_p[i, 0, :Li * HOP] = wav_in[i, 0, :Li * HOP]
                est_in_lf, _ = est(wav_in_p)
            est_in_c = center(est_in_lf[:, :F], vf)
            tgt_c = est_in_c * s[:, None]
            # Per-sample normalized L1: divide by the target's own voiced-excursion
            # so every scale (esp. s<1) contributes equal gradient. Scale-free DC
            # ambiguity is removed by the per-sample centering above.
            vsum = vf.sum(-1).clamp(min=1)
            tstd = (vf * tgt_c.square()).sum(-1).div(vsum).sqrt().clamp(min=1e-3)
            l1_i = ((est_c - tgt_c).abs() * vf).sum(-1).div(vsum)
            loss_f0 = args.f0_weight * (l1_i / tstd).mean()
            # Aux: the post-FiLM plan control must itself predict the target pitch
            # trace. Direct strong gradient into plan_proj/plan_temporal/FiLM/GST.
            plan_f0 = model.plan_f0_head(ctl).squeeze(-1).float()   # [B, T]
            aux_l1 = ((plan_f0[:, :F] - tgt_c).abs() * vf).sum(-1).div(vsum)
            loss_aux = args.lambda_aux * (aux_l1 / tstd).mean()
            loss_ce = 0.0
            for j in range(1, nq):
                loss_ce = loss_ce + level_w[j] * ce_crit(
                    logits[:, j].reshape(-1, card), ce_t[:, j].reshape(-1))
            loss_ce = loss_ce / nq
            loss_ex = 0.0
            if ex_t is not None:
                for j in range(1, nq):
                    loss_ex = loss_ex + level_w[j] * ce_crit(
                        logits[:, j].reshape(-1, card), ex_t[:, j].reshape(-1))
                loss_ex = loss_ex / nq
            loss = args.lambda_ex * loss_ex + loss_f0 + loss_aux + args.lambda_ce * loss_ce
            if args.anneal_ce:
                frac = min(1.0, step / max(1, steps_per_epoch * args.epochs))
                ce_w = args.lambda_ce * (1.0 - 0.9 * frac)
                loss = args.lambda_ex * loss_ex + loss_f0 + loss_aux + ce_w * loss_ce
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            step += 1
            pbar.update(1)
            pbar.set_postfix(f0=f"{loss_f0.item():.4f}", aux=f"{loss_aux.item():.4f}",
                             ce=f"{loss_ce.item():.3f}", ex=f"{loss_ex.item():.3f}",
                             lr=f"{sched.get_last_lr()[0]:.1e}")
            if step % args.eval_every == 0:
                res = run_eval()
                line = "  ".join(f"{k}={v:.3f}" for k, v in res.items())
                print(f"\n[opdiff @step {step}] {line}")
                ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                      'scheduler': sched.state_dict(), 'epoch': epoch, 'step': step}
                torch.save(ck, args.out)
        ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': sched.state_dict(), 'epoch': epoch, 'step': step}
        torch.save(ck, args.out)
        print(f"[opdiff] epoch {epoch} done, saved {args.out}")
    pbar.close()
    res = run_eval()
    line = "  ".join(f"{k}={v:.3f}" for k, v in res.items())
    print(f"[opdiff] FINAL {line}")


if __name__ == "__main__":
    main()
