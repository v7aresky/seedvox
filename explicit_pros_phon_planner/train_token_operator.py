#!/usr/bin/env python
"""train_token_operator.py — Train the decoupled prosody knob (TokenOperator).

Supervised on offline (tokens_in, target_latent) -> tokens_out pairs from real
pitch-shifted audio. Loss = token cross-entropy over the 16 codebooks (level
weighted, earlier codebooks stronger) vs the real shifted tokens. Optional
eval-time check decodes edited tokens and measures prosody-codec cosine vs the
target (the uncheatable audio-space signal).

Usage:
  python -m explicit_pros_phon_planner.train_token_operator \
      --data /tmp/opencode/token_operator_pairs.pt \
      --config configs/token_operator.json --out checkpoints/token_operator.pt
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from explicit_pros_phon_planner.token_operator import TokenOperator

PAD_T = -100


class PairDataset(Dataset):
    def __init__(self, pairs, max_len=0):
        self.items = []
        for p in pairs:
            T = p['tokens_in'].shape[1]
            if max_len and T > max_len:
                continue
            self.items.append(p)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]
        return p['tokens_in'], p['tokens_out'], p['target_latent'], p['semitone']


def collate(batch):
    t_in, t_out, lat, st = zip(*batch)
    B = len(t_in)
    T = max(t.shape[1] for t in t_out)
    tin = torch.zeros(B, 16, T, dtype=torch.long)
    tout = torch.full((B, 16, T), PAD_T, dtype=torch.long)
    tl = torch.zeros(B, dtype=torch.long)
    lat_p = torch.stack([l for l in lat])
    st_p = torch.tensor(st, dtype=torch.float32)
    for i, (a, b) in enumerate(zip(t_in, t_out)):
        L = b.shape[1]
        tin[i, :, :L] = a[:, :L]
        tout[i, :, :L] = b[:, :L]
        tl[i] = L
    return tin, tout, lat_p, tl, st_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/token_operator.json")
    ap.add_argument("--out", default="checkpoints/token_operator.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_audio", type=int, default=8,
                    help="val utterances to decode + pyin for audio-space cos eval (0=off)")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    mcfg, tcfg = cfg['model'], cfg['training']
    device = torch.device(args.device)
    seed = tcfg.get('seed', 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pairs = torch.load(args.data, map_location='cpu', weights_only=False)
    data = pairs['data'] if isinstance(pairs, dict) and 'data' in pairs else pairs
    print(f"[token_operator] {len(data)} pairs from {args.data}")

    max_len = tcfg.get('max_len_tokens', 0)
    ds = PairDataset(data, max_len=max_len)
    val_ratio = tcfg.get('val_ratio', 0.01)
    n_val = max(1, int(len(ds) * val_ratio))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=tcfg.get('batch_size', 64),
                              shuffle=True, collate_fn=collate,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.get('batch_size', 64),
                            shuffle=False, collate_fn=collate, num_workers=0)
    print(f"[token_operator] train {n_train} val {n_val}")

    model = TokenOperator(**mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[token_operator] model params {n_params/1e6:.2f}M")

    nq = mcfg['n_q']
    w = torch.ones(nq)
    hw = tcfg.get('first_half_weight', 1.5)
    w[:nq // 2] = hw
    w[nq // 2:] = 2 - hw
    level_w = (w / w.sum() * nq).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_T)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.get('lr', 3e-4), weight_decay=0.01)
    steps_per_epoch = max(1, len(train_loader))
    sched = get_cosine_schedule_with_warmup(
        opt, tcfg.get('warmup_steps', 500), steps_per_epoch * tcfg.get('epochs', 20), num_cycles=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_epoch = 0
    step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        start_epoch = ck.get('epoch', 0) + 1
        step = ck.get('step', 0)
        print(f"[token_operator] resumed epoch {start_epoch} step {step}")

    mimi = codec = None
    if args.eval_audio > 0:
        from seedvox.modules.mimi import get_mimi_model
        from seedvox.prosody_codec import ProsodyCodec
        mimi = get_mimi_model(device=device, checkpoint_path=tcfg.get(
            'mimi_checkpoint', 'pretrained_models/best_mimi.pt')).eval()
        for p in mimi.parameters():
            p.requires_grad = False
        codec = ProsodyCodec(dim=mcfg['plan_dim'], num_blocks=mcfg['plan_blocks']).eval().to(device)
        ck = torch.load(tcfg.get('codec_checkpoint', 'checkpoints/prosody_codec.pt'),
                        map_location='cpu', weights_only=False)
        codec.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck, strict=False)
        for p in codec.parameters():
            p.requires_grad = False

    writer = SummaryWriter(log_dir=args.log or f"runs/token_operator")
    pbar = tqdm(total=steps_per_epoch * (tcfg.get('epochs', 20) - start_epoch), desc="epoch 0")

    def audio_prosody_cos(batch):
        """Decode edited tokens, re-extract prosody via pyin, encode with the frozen
        codec, cos vs the target latent. The uncheatable audio-space check."""
        tin, tout, lat, tl, st = batch
        tin, lat = tin.to(device), lat.to(device)
        model.eval()
        with torch.no_grad():
            edits = model.edit(tin, lat)
            cos_acc, n = 0.0, 0
            for i in range(tin.shape[0]):
                toks = edits[i, :, :tl[i]].clamp(0, mcfg['card'] - 1).unsqueeze(0)
                y = mimi.decode(toks)[0, 0].float().cpu().numpy()
                import librosa
                f0, vflag, _ = librosa.pyin(y, fmin=50, fmax=600, sr=24000,
                                            frame_length=2048, hop_length=1920)
                if f0.shape[0] < 1:
                    continue
                T_ = f0.shape[0]
                voiced = vflag.astype(bool)
                logf0 = np.full(T_, np.nan, dtype=np.float32)
                logf0[voiced] = np.log(np.maximum(f0[voiced], 1.0))
                mu = float(np.median(logf0[voiced])) if voiced.any() else 0.0
                lfc = logf0 - mu
                lfc[~voiced] = 0.0
                seg = y[:T_ * 1920].reshape(T_, 1920)
                rms = np.sqrt((seg ** 2).mean(1))
                logE = np.log(np.maximum(rms, 1e-7))
                feats = np.stack([lfc, logE - logE.mean(), voiced.astype(np.float32)], axis=-1)
                z = codec.encode(torch.from_numpy(feats).unsqueeze(0).to(device))
                cos_acc += F.cosine_similarity(z, lat[i:i + 1], dim=-1).mean().item()
                n += 1
        model.train()
        return (cos_acc / n) if n else 0.0

    for epoch in range(start_epoch, tcfg.get('epochs', 20)):
        pbar.set_description(f"epoch {epoch}/{tcfg.get('epochs', 20)}")
        model.train()
        tot, cnt, acc_cnt = 0.0, 0, 0
        for tin, tout, lat, tl, st in train_loader:
            tin, tout, lat = tin.to(device), tout.to(device), lat.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(tin, lat)
                loss = 0.0
                for j in range(1, logits.shape[1]):  # c0 locked (content anchor), not supervised
                    loss = loss + level_w[j] * criterion(
                        logits[:, j].reshape(-1, mcfg['card']), tout[:, j].reshape(-1))
                loss = loss / nq
            scaler.scale(loss).backward()
            if tcfg.get('grad_clip', 5.0) > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg['grad_clip'])
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            step += 1
            tot += loss.item()
            cnt += 1
            pbar.update(1)
            pbar.set_postfix(ce=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")
            if step % 50 == 0:
                writer.add_scalar("op/ce", loss.item(), step)
                writer.add_scalar("op/lr", sched.get_last_lr()[0], step)
            if step % args.eval_every == 0:
                model.eval()
                va = 0.0
                with torch.no_grad():
                    vn = 0
                    for vt, vo, vl, _, _ in val_loader:
                        vt, vo, vl = vt.to(device), vo.to(device), vl.to(device)
                        logits = model(vt, vl)
                        pred = logits.argmax(-1)
                        mask = vo != PAD_T
                        mask[:, 0] = False  # c0 is content-locked, never predicted
                        va += (pred[mask] == vo[mask]).float().mean().item()
                        vn += 1
                va /= max(1, vn)
                cos_audio = 0.0
                if args.eval_audio > 0 and mimi is not None:
                    vb = next(iter(val_loader))
                    N = min(args.eval_audio, len(vb[0]))
                    vb = tuple(v[:N] if torch.is_tensor(v) else v for v in vb)
                    cos_audio = audio_prosody_cos(vb)
                writer.add_scalar("op/val_acc", va, step)
                writer.add_scalar("op/val_cos_audio", cos_audio, step)
                print(f"\n[tokop eval @step {step}] val_token_acc={va:.4f} "
                      f"audio_prosody_cos={cos_audio:.4f}")
                model.train()
        ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': sched.state_dict(), 'epoch': epoch, 'step': step, 'config': cfg}
        torch.save(ck, args.out)
        pbar.set_postfix(ce=f"{(tot/max(cnt,1)):.4f}")
        print(f"\n[token_operator] epoch {epoch} done, avg CE {tot/max(cnt,1):.4f}, saved {args.out}")
    pbar.close()


if __name__ == "__main__":
    main()
