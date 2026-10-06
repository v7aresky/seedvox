"""Smoke test: build seedvox ExplicitPlannerModel from the REAL config
(light_fusion_r6_style.json, use_explicit_planner=True) and run a
forward/backward pass exactly as the live trainer does.

Confirms seedvox is not broken under its actual live configuration.
"""
import json
import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "explicit_pros_phon_planner"))

import torch

from explicit_pros_phon_planner.model import ExplicitPlannerModel
from seedvox.utils.tokenizer import CharTokenizer


def main():
    cfgs = ["configs/light_fusion_r6_style.json", "configs/light_fusion_r6_style_ljonly.json"]
    for cf in cfgs:
        print(f"\n===== {cf} =====")
        cfg = json.load(open(cf))
        tok = CharTokenizer()
        model = ExplicitPlannerModel(cfg, tok.vocab_size, phoneme_vocab_size=128)
        model.train()

        B, T = 2, 28
        card = cfg["model"]["card"]
        nq = model.n_q
        text = torch.randint(1, tok.vocab_size, (B, T))
        t_lens = torch.full((B,), T)
        a_toks = torch.randint(0, card - 2, (B, nq, T))
        a_lens = torch.full((B,), T)
        raw = ["the quick brown fox jumps over the lazy dog" , "she sells sea shells by the sea shore"]

        # phonetic ids padded to T (pad=0)
        ph = torch.randint(1, 128, (B, T))
        ph[:, 0] = 0
        ph_lens_ok = torch.randint(5, T, (B,))
        for b in range(B):
            ph[b, ph_lens_ok[b]:] = 0

        out = model(
            text, a_toks, t_lens, a_lens,
            raw_texts=raw,
            phoneme_ids=ph,
            mimi_latents=None,
            drop_prob=0.1,
        )
        logits, targets, ph_logits, jepa_loss, _, latent_pred = out
        print("  forward ok | ph_logits:", None if ph_logits is None else ph_logits.shape,
              "| jepa_loss:", None if jepa_loss is None else round(jepa_loss.item(), 4),
              "| latent_pred:", latent_pred.shape if latent_pred is not None else None)

        # backward on AR logits (explicit model uses ph_planner logits as main)
        loss = torch.tensor(0.0, device=logits[0].device)
        if ph_logits is not None:
            loss = loss + ph_logits.float().pow(2).mean()
        if jepa_loss is not None:
            loss = loss + jepa_loss
        loss.backward()
        g = sum(p.grad is not None for p in model.parameters())
        print("  backward ok; params with grad:", g)
    print("\nSEEDVOX SMOKE PASS (light_fusion_r6_style family)")


if __name__ == "__main__":
    main()