#!/usr/bin/env python
"""train_f0_estimator.py — Pretrain the JDC-style differentiable F0 estimator.

Inputs:  mimi-decoded audio (from stored 16-q tokens) at 24 kHz, 12.5 Hz frames.
Targets: absolute log-F0 (log_f0_center + mu_logF0) and voicing, from pyin.
Loss:    masked L1 on voiced frames + 0.3 * BCE(voicing).

The estimator is later frozen and used as the differentiable pitch-loss for the
token operator (soft-mixture mimi decode -> estimate -> L1 vs warped target).

Usage:
  python -m explicit_pros_phon_planner.train_f0_estimator \
      --data <file1.pt> [<file2.pt> ...] --out checkpoints/f0_estimator.pt
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

from explicit_pros_phon_planner.f0_estimator import F0Estimator

HOP = 1920


class F0Dataset(Dataset):
    def __init__(self, items, max_len=0):
        self.items = [it for it in items if 'audio_tokens' in it and 'voicing' in it]
        if max_len:
            self.items = [it for it in self.items if it['audio_tokens'].shape[-1] <= max_len]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        tok = it['audio_tokens']                      # [1, 16, T]
        mu = float(it['mu_logF0']) if torch.is_tensor(it['mu_logF0']) else it['mu_logF0']
        logf0 = it['log_f0_center'].float() + mu      # absolute log-F0, unvoiced=dummy
        voic = it['voicing'].bool()
        return tok, logf0, voic


def collate(batch):
    toks, lf, voic = zip(*batch)
    B = len(batch)
    T = max(t.shape[-1] for t in toks)
    tok_p = torch.zeros(B, 16, T, dtype=torch.long)
    lf_p = torch.zeros(B, T, dtype=torch.float32)
    vo_p = torch.zeros(B, T, dtype=torch.bool)
    lens = torch.zeros(B, dtype=torch.long)
    for i, (t, l, v) in enumerate(batch):
        L = t.shape[-1]
        tok_p[i, :, :L] = t[0, :, :L]
        lf_p[i, :L] = l[:L]
        vo_p[i, :L] = v[:L]
        lens[i] = L
    return tok_p, lf_p, vo_p, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs='+', required=True)
    ap.add_argument("--out", default="checkpoints/f0_estimator.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=300,
                    help="drop utterances longer than this many frames (24s @12.5Hz)")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0, help="subsample N train items (0=all)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--mimi_checkpoint", default="pretrained_models/best_mimi.pt")
    ap.add_argument("--save_best", type=str, default="checkpoints/f0_estimator_best.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    items = []
    for p in args.data:
        d = torch.load(p, map_location='cpu', weights_only=False)
        data = d['data'] if isinstance(d, dict) and 'data' in d else d
        items.extend(data)
        print(f"[f0_est] {p}: {len(data)} utts")
    print(f"[f0_est] total {len(items)}")

    ds = F0Dataset(items, max_len=args.max_len)
    print(f"[f0_est] after max_len={args.max_len}: {len(ds)}")

    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    if args.limit:
        train_ds, _ = random_split(train_ds, [min(args.limit, len(train_ds)),
                                              max(0, len(train_ds) - min(args.limit, len(train_ds)))],
                                   generator=gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    print(f"[f0_est] train {len(train_ds)} val {len(val_ds)}")

    from seedvox.modules.mimi import get_mimi_model
    mimi = get_mimi_model(device=device, checkpoint_path=args.mimi_checkpoint).eval()
    for p in mimi.parameters():
        p.requires_grad = False

    model = F0Estimator().to(device)
    print(f"[f0_est] params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
        print(f"[f0_est] resumed epoch {start_epoch} step {step}")

    best = float('inf')

    def run_eval():
        model.eval()
        num = den = 0.0
        vn = vd = 0.0
        sems = []
        with torch.no_grad():
            for toks, lf, voic, lens in val_loader:
                toks = toks.to(device)
                wavs = []
                for i in range(toks.shape[0]):
                    L = int(lens[i])
                    w = mimi.decode(toks[i:i + 1, :, :L])[0, 0].float()
                    wavs.append(w)
                Np = max(w.shape[-1] for w in wavs)
                wav_p = torch.zeros(toks.shape[0], 1, Np, device=device)
                for i, w in enumerate(wavs):
                    wav_p[i, 0, :w.shape[-1]] = w
                lf, voic = lf.to(device), voic.to(device)
                pred_lf, vlog = model(wav_p)
                for i in range(toks.shape[0]):
                    L = int(lens[i])
                    v = voic[i, :L]
                    p = pred_lf[i, :L]
                    t = lf[i, :L]
                    if v.any():
                        pv, tv = p[v], t[v]
                        mp, mt = pv.mean(), tv.mean()
                        num += ((pv - mp) * (tv - mt)).sum().item()
                        den += (((pv - mp) ** 2).sum().item() * ((tv - mt) ** 2).sum().item()) ** 0.5
                        sems.append((12.0 * torch.log2(p[v].exp() / t[v].exp()).abs()).mean().item())
                    vn += (vlog[i, :L].sigmoid() > 0.5).eq(voic[i, :L]).sum().item()
                    vd += L
        model.train()
        corr = (num / den) if den > 0 else 0.0
        v_acc = vn / max(1, vd)
        sem = float(np.mean(sems)) if sems else 0.0
        return corr, v_acc, sem

    pbar = tqdm(total=steps_per_epoch * (args.epochs - start_epoch))
    for epoch in range(start_epoch, args.epochs):
        pbar.set_description(f"epoch {epoch}/{args.epochs}")
        model.train()
        for toks, lf, voic, lens in train_loader:
            toks = toks.to(device)
            with torch.no_grad():
                wavs = []
                for i in range(toks.shape[0]):
                    L = int(lens[i])
                    w = mimi.decode(toks[i:i + 1, :, :L])[0, 0].float()
                    wavs.append(w)
                Np = max(w.shape[-1] for w in wavs)
                wav_p = torch.zeros(toks.shape[0], 1, Np, device=device)
                for i, w in enumerate(wavs):
                    wav_p[i, 0, :w.shape[-1]] = w
            lf, voic = lf.to(device), voic.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pred_lf, vlog = model(wav_p)
                v = voic.float()
                num = (pred_lf * v).sum()
                den = v.sum().clamp(min=1)
                loss_f0 = F.l1_loss(pred_lf * v / den, lf * v / den, reduction='sum')
                loss_v = F.binary_cross_entropy_with_logits(vlog, v, reduction='mean')
                loss = loss_f0 + 0.3 * loss_v
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            step += 1
            pbar.update(1)
            pbar.set_postfix(f0=f"{loss_f0.item():.4f}", v=f"{loss_v.item():.4f}",
                             lr=f"{sched.get_last_lr()[0]:.1e}")
            if step % args.eval_every == 0:
                corr, v_acc, sem = run_eval()
                print(f"\n[f0_est @step {step}] val corr={corr:.4f} voicing_acc={v_acc:.4f} "
                      f"median_abs_err_semitone={sem:.3f}")
                ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                      'scheduler': sched.state_dict(), 'epoch': epoch, 'step': step}
                if sem < best:
                    best = sem
                    torch.save(ck, args.save_best)
                    print(f"[f0_est] new best semitone {sem:.3f} -> {args.save_best}")
                model.train()
        ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': sched.state_dict(), 'epoch': epoch, 'step': step}
        torch.save(ck, args.out)
        print(f"[f0_est] epoch {epoch} done, saved {args.out}")
    pbar.close()
    corr, v_acc, sem = run_eval()
    print(f"[f0_est] FINAL val corr={corr:.4f} voicing_acc={v_acc:.4f} median_semitone={sem:.3f}")


if __name__ == "__main__":
    main()
