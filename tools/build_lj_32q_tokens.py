import os
import sys
import torch
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

from seedvox.modules.mimi import get_mimi_model


def main():
    src_pt = "/home/vpollet/proj/autovoc/dataset/train_tokens_prosody_16q_prosody_codec.pt"
    out_pt = "/home/vpollet/proj/autovoc/dataset/train_tokens_prosody_32q_prosody_codec.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = torch.load(src_pt, map_location="cpu", weights_only=False)
    items = raw["data"]
    print(f"source items: {len(items)}, keys: {sorted(items[0].keys())}")

    mimi = get_mimi_model(device=device, checkpoint_path="pretrained_models/best_mimi.pt").eval()

    n_total = len(items)
    n_mismatch = 0
    bs = 8
    with torch.no_grad():
        for start in range(0, n_total, bs):
            chunk = items[start:start + bs]
            wav_paths = [it["wav_path"] for it in chunk]
            waves = []
            for wp in wav_paths:
                import torchaudio
                w, sr = torchaudio.load(wp)
                if sr != 24000:
                    w = torchaudio.transforms.Resample(sr, 24000)(w)
                waves.append(w)
            max_len = max(w.shape[1] for w in waves)
            x = torch.stack([torch.nn.functional.pad(w, (0, max_len - w.shape[1])) if w.shape[1] < max_len else w for w in waves]).to(device)
            codes = mimi.encode(x)  # [B, 32, T']
            for i, it in enumerate(chunk):
                old_t = it["audio_tokens"].shape[-1]
                new_t = codes[i].shape[-1]
                if new_t < old_t:
                    n_mismatch += 1
                    print(f"T SHORT {it['wav_path']}: old {old_t} new {new_t}")
                it["audio_tokens"] = codes[i, :, :old_t].unsqueeze(0).cpu().to(torch.int64)  # [1, 32, T]
            if (start // bs) % 125 == 0 or start == n_total - 1:
                print(f"  {start + len(chunk)}/{n_total}  T errs so far: {n_mismatch}")

    print(f"T-length mismatches: {n_mismatch}")
    torch.save(raw, out_pt)
    sz = os.path.getsize(out_pt) / 1e9
    print(f"Saved {out_pt} ({len(raw['data'])} items, {sz:.1f} GB)")


if __name__ == "__main__":
    main()