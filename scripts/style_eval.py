#!/usr/bin/env python3
"""style_eval.py — Repeatable style-conditioning evaluation.

Generates a fixed set of sentences with every style id (0..num_style_tokens-1),
anchored on a FIXED reference speaker, then reports prosody statistics
(f0 mean/std/range, duration) per style across multiple seeds so that
across-style effects can be compared against within-style sampling noise.

Protocol rationale: the model's AR sampling wanders (temp 0.1), so always pass
--ref_wav_speaker and >=2 seeds. Judge style by f0/timing tables + listening;
the speaker-encoder cosine metric is too weak (real same-speaker ~0.4) to use.

Usage:
  python scripts/style_eval.py \
      --config configs/light_fusion_r6_style.json \
      --checkpoint checkpoints/seedvox_light_fusion_r6_style_epoch_120.pt \
      --texts "It looks just like a crystal lattice structure growing across the field!" \
      --ref_wav_speaker feynman_demo_01.wav \
      --seeds 2 \
      --output_dir demos/style_eval
"""
import os, sys, json, argparse, torch, torchaudio
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from explicit_pros_phon_planner.model_fusion import FusionPlannerModel
from explicit_pros_phon_planner.utils import filter_state_dict
from seedvox.utils.tokenizer import CharTokenizer
from seedvox.utils.text import normalize_text
from seedvox.modules.mimi import get_mimi_model
from prepare_prosody_codec_data import extract_prosody

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_audio(path, device, sr=24000):
    wav, orig_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav.to(device)


def _sum_embeddings(emb_list, tokens, n_q):
    e = emb_list[0](tokens[:, 0])
    for q in range(1, n_q):
        e = e + emb_list[q](tokens[:, q])
    return e


def main():
    ap = argparse.ArgumentParser(description="Style-conditioning eval")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--texts", action="append", required=True,
                    help="Sentence to synthesize (repeatable).")
    ap.add_argument("--ref_wav_speaker", default=None,
                    help="Anchor speaker identity. Strongly recommended.")
    ap.add_argument("--styles", default="0-15", help="Range of style ids, e.g. 0-15")
    ap.add_argument("--seeds", type=int, default=2, help="Seeds per (style, sentence).")
    ap.add_argument("--output_dir", default="demos/style_eval")
    ap.add_argument("--seed", type=int, default=42, help="Base seed (incremented per replicate).")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.styles.split("-"))
    styles = list(range(lo, hi + 1))
    device = torch.device(args.device)
    dtype = (torch.bfloat16 if args.dtype == "bf16"
             else torch.float16 if args.dtype == "fp16" else torch.float32)
    autocast_enabled = args.dtype != "fp32"

    with open(args.config) as f:
        cfg = json.load(f)
    tokenizer = CharTokenizer()

    model = FusionPlannerModel(cfg, tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
    if args.dtype != "fp32":
        model = model.to(dtype)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt.get("ema_model", ckpt.get("model", ckpt))
    model.load_state_dict(filter_state_dict(model, sd), strict=False)
    model.eval()
    mimi = get_mimi_model(device=device,
                          checkpoint_path=cfg.get("mimi_checkpoint",
                                                  os.path.join(ROOT, "pretrained_models/best_mimi.pt"))).eval()
    print(f"Loaded: {args.checkpoint}")
    print(f"Styles: {styles}, seeds/style: {args.seeds}, ref speaker: {args.ref_wav_speaker}")

    ext_spk = None
    if args.ref_wav_speaker:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled):
            wav = load_audio(args.ref_wav_speaker, device)
            toks = mimi.encode(wav.unsqueeze(0))
            ae = _sum_embeddings(model.audio_embs, toks, model.n_q)
            ae = model.audio_prenet(model.audio_norm(ae))
            mask = torch.arange(toks.shape[-1], device=device).unsqueeze(0) >= torch.full((1,), toks.shape[-1], device=device)
            ext_spk = model.speaker_encoder(ae, key_padding_mask=mask)
        print("Reference speaker extracted.")

    out_root = args.output_dir
    os.makedirs(out_root, exist_ok=True)
    reports = []

    for si, text in enumerate(args.texts):
        text = normalize_text(text)
        t_ids = torch.tensor([tokenizer.encode(text, normalize=False)], device=device)
        t_lens = torch.tensor([t_ids.shape[1]], device=device)
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled):
            text_feat = model.get_enriched_text_feat(t_ids, t_lens, raw_texts=[text])
            ph_ids = model.phonetic_planner.sample(text_feat, temp=1.0, top_p=0.9, greedy=False)

        print(f"\n=== Sentence {si}: {text[:70]}... ===")
        table = []
        for st in styles:
            stats = []
            for rep in range(args.seeds):
                torch.manual_seed(args.seed + rep)
                np.random.seed(args.seed + rep)
                with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled):
                    context, ctx_mask, _, _, _, spk_vec, prosody_emb = model.encode_context(
                        t_ids, t_lens, raw_texts=[text], phoneme_ids=ph_ids,
                        external_speaker=ext_spk, external_style=st,
                    )
                    audio_tokens, _ = model.sample(
                        t_ids, t_lens, phoneme_ids=ph_ids, temp=args.temp, cfg_scale=1.0,
                        external_speaker=ext_spk, precomputed_context=context,
                        precomputed_mask=ctx_mask, spk_vec=spk_vec,
                        prosody_emb=prosody_emb,
                    )
                eoa = (audio_tokens[:, 0, :] == model.EOA_ID).int().argmax(dim=-1)
                if eoa.max() > 0:
                    audio_tokens = audio_tokens[:, :, :eoa.max()]
                if audio_tokens.shape[-1] > 1:
                    audio_tokens = audio_tokens[:, :, :-1]
                wav = mimi.decode(audio_tokens.clamp(0, model.card - 1))
                expected_len = audio_tokens.shape[-1] * 1920
                if wav.shape[-1] > expected_len:
                    wav = wav[..., :expected_len]
                fade_len = min(240, wav.shape[-1] // 4)
                fade = torch.linspace(0.0, 1.0, fade_len, device=wav.device, dtype=wav.dtype)
                wav[..., :fade_len] *= fade

                sdir = os.path.join(out_root, f"s{st:02d}")
                os.makedirs(sdir, exist_ok=True)
                out_path = os.path.join(sdir, f"sent{si}_seed{rep}.wav")
                torchaudio.save(out_path, wav[0].cpu(), 24000)

                pr = extract_prosody(out_path)
                if pr and "_error" not in pr:
                    f0v = np.exp(pr["log_f0_center"])[pr["voicing"].astype(bool)]
                    if len(f0v):
                        stats.append(dict(f0mean=float(f0v.mean()), f0std=float(f0v.std()),
                                          f0range=float(f0v.max() - f0v.min()), dur=wav.shape[-1] / 24000))
            if len(stats):
                m = {k: np.mean([s[k] for s in stats]) for k in stats[0]}
                v = {k: np.std([s[k] for s in stats]) for k in stats[0]}
                table.append(dict(style=st, **m, f0std_se=1.96 * v["f0std"] / max(np.sqrt(len(stats)), 1)))
                print(f"  style {st:02d}: f0mean={m['f0mean']:.2f}±{v['f0mean']:.2f}  "
                      f"f0std={m['f0std']:.2f}±{v['f0std']:.2f}  f0range={m['f0range']:.2f}  "
                      f"dur={m['dur']:.2f}s  ({len(stats)} reps)")
            else:
                print(f"  style {st:02d}: extraction failed")

        if table:
            print(f"  --- summary (across-style spread vs within-style se) ---")
            for k in ("f0mean", "f0std", "f0range"):
                vals = [r[k] for r in table]
                ses = [r.get("f0std_se", 0) for r in table]
                print(f"  {k:>8}: across-style span {max(vals) - min(vals):.3f}, "
                      f"max within-style se {max(ses):.3f}")
            reports.append(dict(sentence=text, table=table))

    out_json = os.path.join(out_root, "summary.json")
    with open(out_json, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nWavs + {out_json} written to {out_root}/")


if __name__ == "__main__":
    main()
