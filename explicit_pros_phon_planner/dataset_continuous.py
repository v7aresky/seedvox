"""Dataset for pitch-conditioned TTS using Mimi continuous latents.

Lazy-loads pre-extracted latents/F0 from disk, tokenizes text on-the-fly.
"""
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ContinuousLatentDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        tokenizer,
        cache_dir,
        max_duration_sec=30.0,
        max_frames=750,
    ):
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.max_frames = max_frames
        self.latent_dir = self.cache_dir / "latents"
        self.f0_dir = self.cache_dir / "f0"

        self.entries = []
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                key = Path(entry["path"]).stem
                latent_path = self.latent_dir / f"{key}.pt"
                f0_path = self.f0_dir / f"{key}.pt"
                if latent_path.exists() and f0_path.exists():
                    self.entries.append({
                        "key": key,
                        "transcript": entry.get("transcript", ""),
                    })

        print(f"Dataset: {len(self.entries)} entries with cached latents")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        key = entry["key"]

        latents = torch.load(self.latent_dir / f"{key}.pt", weights_only=True)
        f0_hz = torch.load(self.f0_dir / f"{key}.pt", weights_only=True)

        # Normalize shapes: some extracts have [1, T, D] or [1, T] batch dims
        if latents.dim() == 3:
            latents = latents.squeeze(0)
        if f0_hz.dim() == 2:
            f0_hz = f0_hz.squeeze(0)
        if latents.dim() != 2 or f0_hz.dim() != 1 or latents.shape[0] != f0_hz.shape[0]:
            latents = latents[:min(latents.shape[0], f0_hz.shape[0])]
            f0_hz = f0_hz[:latents.shape[0]]

        if latents.shape[0] > self.max_frames:
            latents = latents[:self.max_frames]
            f0_hz = f0_hz[:self.max_frames]

        text_tokens = torch.tensor(self.tokenizer(entry["transcript"]), dtype=torch.long)
        if text_tokens.numel() == 0:
            text_tokens = torch.tensor([0], dtype=torch.long)

        return {
            "latents": latents,
            "f0_hz": f0_hz,
            "text_tokens": text_tokens,
            "key": key,
        }


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: -x["latents"].shape[0])

    max_T = max(b["latents"].shape[0] for b in batch)
    max_L = max(b["text_tokens"].shape[0] for b in batch)
    D = batch[0]["latents"].shape[1]

    latents, f0_hz, text_tokens = [], [], []
    latent_mask, f0_mask, text_mask = [], [], []

    for b in batch:
        T = b["latents"].shape[0]
        L = b["text_tokens"].shape[0]

        lat = b["latents"]
        if T < max_T:
            lat = torch.cat([lat, lat.new_zeros(max_T - T, lat.shape[1])], dim=0)
        latents.append(lat)

        f0 = b["f0_hz"]
        if T < max_T:
            f0 = torch.cat([f0, f0.new_zeros(max_T - T)], dim=0)
        f0_hz.append(f0)

        txt = b["text_tokens"]
        if L < max_L:
            txt = torch.cat([txt, txt.new_zeros(max_L - L, dtype=txt.dtype)], dim=0)
        text_tokens.append(txt)

        latent_mask.append(torch.arange(max_T) >= T)
        f0_mask.append(torch.arange(max_T) >= T)
        text_mask.append(torch.arange(max_L) >= L)

    return {
        "latents": torch.stack(latents),
        "f0_hz": torch.stack(f0_hz),
        "text_tokens": torch.stack(text_tokens),
        "latent_mask": torch.stack(latent_mask),
        "f0_mask": torch.stack(f0_mask),
        "text_mask": torch.stack(text_mask),
    }
