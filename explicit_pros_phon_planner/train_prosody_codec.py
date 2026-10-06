"""train_prosody_codec.py — Stage-1 prosody bottleneck codec training.

Trains the ProsodyCodec (F0/E/voicing -> K block vectors, reconstruction loss)
on the extracted prosody targets from prepare_prosody_codec_data.py.

Usage:
  python -m explicit_pros_phon_planner.train_prosody_codec \
      --train_data ../autovoc/dataset/train_tokens_prosody_16q_prosody_codec.pt \
                   <more files> \
      --out checkpoints/prosody_codec.pt --epochs 60 --batch_size 256
"""

import argparse
import glob
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from seedvox.prosody_codec import ProsodyCodec


class ProsodyFeatDataset(Dataset):
    def __init__(self, paths):
        self.items = []  # each: (feat [T,3] float32, lens int)
        for p in paths:
            if not os.path.exists(p):
                print(f"SKIP missing {p}")
                continue
            raw = torch.load(p, map_location='cpu', weights_only=False)
            data = raw['data'] if isinstance(raw, dict) and 'data' in raw else raw
            got = 0
            for it in data:
                if 'log_f0_center' in it:
                    T = it['log_f0_center'].shape[0]
                    feat = torch.stack([it['log_f0_center'], it['e_center'], it['voicing'].float()], dim=-1)
                    self.items.append((feat, T))
                    got += 1
            print(f"{os.path.basename(p)}: {got}/{len(data)} items with prosody")
        if not self.items:
            raise RuntimeError("No prosody features found in any file")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat, T = self.items[idx]
        return feat, T


def collate(batch, num_blocks=32):
    feats, lens = zip(*batch)
    lens = torch.tensor(lens, dtype=torch.long)
    T = max(num_blocks, ((lens.max().item() + num_blocks - 1) // num_blocks) * num_blocks)
    padded = torch.zeros(len(feats), T, 3)
    for i, f in enumerate(feats):
        padded[i, :lens[i]] = f
    return padded, lens


def run_eval(codec, loader, device, n_batches=None):
    codec.eval()
    pf, pe, tf, te, pv, tv = [], [], [], [], [], []
    tot_f0, tot_e, tot_v, tot_n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for bi, (feats, lens) in enumerate(loader):
            if n_batches and bi >= n_batches:
                break
            feats, lens = feats.to(device), lens.to(device)
            _, rec = codec(feats, lens)
            m = torch.arange(feats.shape[1], device=device)[None, :] < lens[:, None]
            v = m & (feats[..., 2] > 0.5)
            pf.append(rec[..., 0][v]); tf.append(feats[..., 0][v])
            pe.append(rec[..., 1][m]); te.append(feats[..., 1][m])
            pv.append((rec[..., 2] > 0)[m]); tv.append(feats[..., 2][m] > 0.5)
            tot_f0 += ((rec[..., 0] - feats[..., 0]).abs() * m).sum().item()
            tot_e += ((rec[..., 1] - feats[..., 1]).abs() * m).sum().item()
            tot_n += m.sum().item()
            tot_v += m.sum().item()
    pf, tf = torch.cat(pf), torch.cat(tf)
    pe, te = torch.cat(pe), torch.cat(te)
    pv, tv = torch.cat(pv).float(), torch.cat(tv).float()
    corr_f = torch.corrcoef(torch.stack([pf, tf]))[0, 1].item() if pf.numel() > 1 else float('nan')
    r2_f = 1 - ((pf - tf) ** 2).mean().item() / tf.var().item() if tf.numel() > 1 else float('nan')
    corr_e = torch.corrcoef(torch.stack([pe, te]))[0, 1].item() if pe.numel() > 1 else float('nan')
    r2_e = 1 - ((pe - te) ** 2).mean().item() / te.var().item() if te.numel() > 1 else float('nan')
    acc_v = (pv == tv).float().mean().item()
    return {'l1_f0': tot_f0 / max(tot_n, 1), 'l1_e': tot_e / max(tot_n, 1),
            'corr_f0': corr_f, 'r2_f0': r2_f, 'corr_e': corr_e, 'r2_e': r2_e, 'voicing_acc': acc_v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", nargs='+', required=True)
    ap.add_argument("--out", default="checkpoints/prosody_codec.pt")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--num_blocks", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ds = ProsodyFeatDataset(args.train_data)
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_tr = len(ds) - n_val
    idx = torch.randperm(len(ds))
    train_ds = Subset(ds, idx[:n_tr].tolist())
    val_ds = Subset(ds, idx[n_tr:].tolist())
    print(f"train {n_tr}  val {n_val}")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, num_blocks=args.num_blocks), num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, num_blocks=args.num_blocks), num_workers=args.num_workers)

    # global per-feature stds for normalization
    feats = torch.cat([f for f, _ in ds.items])  # may be large; chunk
    del feats
    stds = torch.zeros(3)
    ns = 0
    for i in range(0, len(ds.items), 4096):
        chunk = torch.cat([ds.items[j][0] for j in range(i, min(i + 4096, len(ds.items)))])
        stds += (chunk ** 2).sum(0)
        ns += chunk.shape[0]
    stds = (stds / ns).sqrt()
    stds[:2] = stds[:2].clamp(min=1e-3)
    stds[2] = 1.0
    print(f"feat_std (log-f0, log-e, voicing): {[round(v, 3) for v in stds.tolist()]}")

    model = ProsodyCodec(dim=args.dim, num_blocks=args.num_blocks, hidden=args.hidden)
    model.feat_std.copy_(stds)
    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    best_r2 = -1.0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for feats, lens in tqdm(loader, desc=f"ep{ep}", disable=True):
            feats, lens = feats.to(args.device), lens.to(args.device)
            _, rec = model(feats, lens)
            m = torch.arange(feats.shape[1], device=args.device)[None, :] < lens[:, None]
            resid = (rec - feats).abs() * m.unsqueeze(-1)
            resid = resid / model.feat_std.clamp(min=1e-4)
            l_f0 = resid[..., 0].sum() / m.sum().clamp(min=1)
            l_e = resid[..., 1].sum() / m.sum().clamp(min=1)
            l_v = F.binary_cross_entropy_with_logits(rec[..., 2], feats[..., 2].clamp(0, 1), reduction='none')
            l_v = (l_v * m).sum() / m.sum().clamp(min=1)
            loss = l_f0 + l_e + 0.3 * l_v
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        ev = run_eval(model, val_loader, args.device)
        print(f"ep{ep}: loss={tot / len(loader):.4f} | "
              f"L1f0={ev['l1_f0']:.4f} L1e={ev['l1_e']:.4f} | "
              f"val corr_f0={ev['corr_f0']:.3f} R2_f0={ev['r2_f0']:.3f} "
              f"corr_e={ev['corr_e']:.3f} R2_e={ev['r2_e']:.3f} voicing_acc={ev['voicing_acc']:.3f}", flush=True)
        if ev['r2_f0'] > best_r2:
            best_r2 = ev['r2_f0']
            torch.save({'model': model.state_dict(), 'epoch': ep, 'best_r2': best_r2}, args.out)
            print(f"  saved {args.out} (R2_f0={best_r2:.3f})")

    ev = run_eval(model, val_loader, args.device)
    print(f"FINAL R2_f0={ev['r2_f0']:.3f} corr_f0={ev['corr_f0']:.3f} R2_e={ev['r2_e']:.3f} voicing_acc={ev['voicing_acc']:.3f}")


if __name__ == "__main__":
    main()
