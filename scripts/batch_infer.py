#!/usr/bin/env python3
"""
Batch inference — load model once, compile once, process all lines from demo.txt.

Usage:
  python scripts/batch_infer.py \\
      --config ./configs/light_fusion_r3.json \\
      --checkpoint ./checkpoints/seedvox_light_fusion_latest.pt \\
      [--lora_checkpoint ./checkpoints/fynmann_lora_epoch_29.pt] \\
      [--refiner_checkpoint ./checkpoints/seedvox_light_fusion_refiner_epoch_0.pt] \\
      [--demo_file demo.txt] \\
      [--output_dir demos] \\
      [--compile] \\
      [--dtype fp32|bf16|fp16] \\
      [--seed 42] \\
      [--ref_wav_speaker path.wav] \\
      [--random_speaker] [--random_prosody] \\
      [--overwrite_phonemes "R IY1 D"] \\
      [--cfg 2.0] [--temp 0.8] \\
      [--variant_n 3] [--variant_axis pros] \\
      [--play] [--view_waveform] [--log_metrics]
"""

import torch
import torchaudio
import argparse
import json
import os
import sys
import time
import random
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from explicit_pros_phon_planner.model import ExplicitPlannerModel
from explicit_pros_phon_planner.model_fusion import FusionPlannerModel
from explicit_pros_phon_planner.finetune_lora_fusion import inject_lora, MIMI_DECODER_TARGETS
from seedvox.utils.tokenizer import CharTokenizer, PhonemeTokenizer
from explicit_pros_phon_planner.utils import PhoneticGenerator, collate_phonemes, filter_state_dict, extract_ref_prosody_latent
from seedvox.utils.text import normalize_text


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_audio(path, device):
    wav, sr = torchaudio.load(path)
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000).to(device)
        wav = resampler(wav.to(device))
    else:
        wav = wav.to(device)
    return wav


def de_emphasize(wav, coeff=0.95):
    if coeff <= 0:
        return wav
    wav = wav.squeeze(0)
    y = torch.empty_like(wav)
    y[0, 0] = wav[0, 0]
    for t in range(1, wav.shape[-1]):
        y[0, t] = wav[0, t] + coeff * y[0, t - 1]
    return y.unsqueeze(0)


def temporal_smooth(wav, strength=0.3, kernel_size=7):
    if strength <= 0 or kernel_size < 3:
        return wav
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    sigma = kernel_size / 6.0
    t = torch.arange(kernel_size, device=wav.device, dtype=wav.dtype) - kernel_size // 2
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    blurred = torch.nn.functional.conv1d(wav, kernel, padding=kernel_size // 2)
    return (1 - strength) * wav + strength * blurred


def _sum_embeddings(emb_list, tokens, n_q):
    num_codebooks = min(tokens.shape[1], n_q)
    ae = emb_list[0](tokens[:, 0])
    for k in range(1, num_codebooks):
        ae = ae + emb_list[k](tokens[:, k])
    return ae


def get_braille_char(dots):
    mask = 0
    if dots[0][0]: mask |= 0x01
    if dots[1][0]: mask |= 0x02
    if dots[2][0]: mask |= 0x04
    if dots[0][1]: mask |= 0x08
    if dots[1][1]: mask |= 0x10
    if dots[2][1]: mask |= 0x20
    if dots[3][0]: mask |= 0x40
    if dots[3][1]: mask |= 0x80
    return chr(0x2800 + mask)


def print_waveform(audio_tensor, width=80, height=3, sample_rate=24000):
    audio = audio_tensor.squeeze().cpu().numpy()
    duration = len(audio) / sample_rate
    w_dots = width * 2
    h_dots_half = height * 4
    bin_size = max(1, len(audio) // w_dots)
    bins = np.zeros(w_dots)
    for i in range(w_dots):
        chunk = audio[i * bin_size:min((i + 1) * bin_size, len(audio))]
        if len(chunk) > 0:
            bins[i] = np.abs(chunk).max()
    peak = max(bins.max(), 1e-6)
    bars = (bins / peak * h_dots_half).astype(int)
    print(f"\n  \033[1mUltra-Res Braille Preview\033[0m ({duration:.2f}s)")
    print(f"  \033[90m\u250c{'─' * width}\u2510\033[0m")
    for tr in range(height - 1, -height - 1, -1):
        line = "  \033[90m\u2502\033[0m"
        for tc in range(width):
            d_cols = [bars[tc*2], bars[tc*2 + 1]]
            dots = [[0,0],[0,0],[0,0],[0,0]]
            for dr in range(4):
                dot_y = (tr * 4) + (3 - dr)
                for c in range(2):
                    if dot_y >= 0:
                        if d_cols[c] >= dot_y + 1: dots[dr][c] = 1
                    else:
                        if d_cols[c] >= abs(dot_y): dots[dr][c] = 1
            char = get_braille_char(dots)
            avg_h = (d_cols[0] + d_cols[1]) / 2 / h_dots_half
            color = "\033[38;5;198m" if avg_h > 0.8 else ("\033[38;5;45m" if avg_h > 0.5 else "\033[38;5;39m")
            line += f"{color}{char}\033[0m"
        line += "\033[90m\u2502\033[0m"
        print(line)
    print(f"  \033[90m\u2514{'─' * width}\u2518\033[0m")
    print(f"  \033[90m0s{' ' * (width - 7)}{duration:.1f}s\033[0m\n")


def play_audio_v5(audio_tensor, sample_rate=24000):
    audio_np = audio_tensor.squeeze().cpu().numpy()
    duration = len(audio_np) / sample_rate
    width = 80
    height = 4
    w_dots = width * 2
    h_dots_half = height * 4
    bin_size = max(1, len(audio_np) // w_dots)
    bins = np.zeros(w_dots)
    for i in range(w_dots):
        chunk = audio_np[i * bin_size:min((i + 1) * bin_size, len(audio_np))]
        if len(chunk) > 0:
            bins[i] = np.abs(chunk).max()
    peak = max(bins.max(), 1e-6)
    bars = (bins / peak * h_dots_half).astype(int)

    import tempfile
    import subprocess
    fd, tmp_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    torchaudio.save(tmp_path, audio_tensor.unsqueeze(0).cpu() if audio_tensor.dim() == 1 else audio_tensor.cpu(), sample_rate)
    player_proc = None
    for player_cmd in ['paplay', 'aplay', 'ffplay -nodisp -autoexit', 'afplay']:
        try:
            cmd = player_cmd.split() + [tmp_path]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.1)
            if p.poll() is None:
                player_proc = p
                break
        except:
            continue
    try:
        sys.stdout.write("\033[?25l")
        total_lines = height * 2
        sys.stdout.write('\n' * total_lines)
        start_time = time.time()
        while True:
            if player_proc and player_proc.poll() is not None:
                break
            elapsed = time.time() - start_time
            if elapsed >= duration:
                elapsed = duration
            play_pos_dot = min(int(elapsed / max(duration, 1e-6) * w_dots), w_dots - 1)
            play_pos_tc = play_pos_dot // 2
            sys.stdout.write(f"\033[{total_lines}A")
            for tr in range(height - 1, -height - 1, -1):
                line = "  "
                for tc in range(width):
                    if tc > play_pos_tc:
                        line += " "
                        continue
                    d_cols = [bars[tc*2], bars[tc*2 + 1]]
                    dots = [[0,0],[0,0],[0,0],[0,0]]
                    for dr in range(4):
                        dot_y = (tr * 4) + (3 - dr)
                        for c in range(2):
                            if (tc * 2 + c) <= play_pos_dot:
                                if dot_y >= 0:
                                    if d_cols[c] >= dot_y + 1: dots[dr][c] = 1
                                else:
                                    if d_cols[c] >= abs(dot_y): dots[dr][c] = 1
                    char = get_braille_char(dots)
                    if tc < play_pos_tc:
                        avg_h = (d_cols[0] + d_cols[1]) / 2 / h_dots_half
                        color = "\033[38;2;255;20;147m" if avg_h > 0.5 else "\033[38;5;205m"
                        line += f"{color}{char}\033[0m"
                    else:
                        line += f"\033[97m{char}\033[0m"
                sys.stdout.write(line + "\033[K\n")
            sys.stdout.flush()
            if elapsed >= duration:
                break
            time.sleep(0.04)
    finally:
        sys.stdout.write("\033[?25h")
        if player_proc and player_proc.poll() is None:
            player_proc.terminate()
        try:
            os.remove(tmp_path)
        except:
            pass
        sys.stdout.write('\n')
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch inference for SeedVox")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ph_checkpoint", default=None, help="Optional separate phonetic pre-train weights")
    parser.add_argument("--lora_checkpoint")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank (auto-detected if not set)")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (auto-detected if not set)")
    parser.add_argument("--no_mimi_lora", action="store_true", help="Skip Mimi decoder LoRA")
    parser.add_argument("--refiner_checkpoint", help="Optional NAR refiner weights")
    parser.add_argument("--demo_file", default="demo.txt")
    parser.add_argument("--output_dir", default="demos")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_linguistic_fusion", action="store_true", help="Force FusionPlannerModel")

    # Speaker / prosody conditioning
    parser.add_argument("--ref_wav_speaker")
    parser.add_argument("--ref_wav_speaker2", help="Optional second ref speaker for mixing")
    parser.add_argument("--spk_mix_ratio", type=float, default=0.5, help="Blend ratio for speaker mixing")
    parser.add_argument("--random_speaker", action="store_true")
    parser.add_argument("--random_prosody", action="store_true")
    parser.add_argument("--ref_wav_prosody", default=None,
                        help="Reference wav whose prosody is injected via the stage-1 codec latent (overrides sampled plan)")
    parser.add_argument("--style", type=int, default=None,
                        help="Explicit style token id (0..num_style_tokens-1). Defaults to the text->style head prediction.")

    # Phoneme control
    parser.add_argument("--overwrite_phonemes", type=str, default=None)
    parser.add_argument("--use_external_g2p", action="store_true", help="Use external G2P instead of phonetic planner")
    parser.add_argument("--g2p", default="espeak", help="G2P backend with --use_external_g2p")
    parser.add_argument("--phoneme_temp", type=float, default=1.0)
    parser.add_argument("--phoneme_top_p", type=float, default=0.9)
    parser.add_argument("--phoneme_greedy", action="store_true", help="Deterministic argmax for phonemes")

    # Sampling
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p sampling: keep tokens with prob >= min_p * max prob (0=off)")
    parser.add_argument("--rep_penalty", type=float, default=1.0, help="Repetition penalty on acoustic tokens (1.0=off, ~1.2 typical)")
    parser.add_argument("--exagg", type=float, default=1.0, help="Prosody exaggeration dial (0=flat/neutral, 1=natural, >1=emphatic)")
    parser.add_argument("--prosody_temperature", type=float, default=None, help="Stochastic prosody planner sampling temperature (None=use config prosody_temperature)")
    parser.add_argument("--mono_slack", type=float, default=0.0, help="Monotone cross-attn window (0=off; window radius in text frames, e.g. 2)")

    # Variants
    parser.add_argument("--variant_n", type=int, default=1, help="Number of variants per text")
    parser.add_argument("--variant_axis", choices=['pros', 'speaker'], default=None)

    # Post-processing
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument("--smooth_kernel", type=int, default=7)
    parser.add_argument("--de_emph", type=float, default=0.0)

    # Interaction / logging
    parser.add_argument("--play", action="store_true", help="Play audio after generation")
    parser.add_argument("--view_waveform", action="store_true", help="Print braille waveform")
    parser.add_argument("--log_metrics", action="store_true", help="Log latency / RTF per line")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- Load config & model ----------
    with open(args.config) as f:
        cfg = json.load(f)

    tokenizer = CharTokenizer()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if 'ema_model' in ckpt:
        state_dict = ckpt['ema_model']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    has_fusion = any(k.startswith("linguistic_fusion") for k in state_dict.keys())
    use_fusion = has_fusion or cfg["model"].get("use_linguistic_fusion", False) or args.use_linguistic_fusion

    if args.refiner_checkpoint:
        cfg['model']['use_refinement'] = True

    model_cls = FusionPlannerModel if use_fusion else ExplicitPlannerModel
    print(f"Model: {model_cls.__name__}")
    model = model_cls(cfg, tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
    state_dict = filter_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    print("  Checkpoint loaded.")

    # Optional phonetic pre-train overrides
    if args.ph_checkpoint:
        print(f"  Phonetic weights: {args.ph_checkpoint}")
        ph_ckpt = torch.load(args.ph_checkpoint, map_location=device)
        ph_sd = ph_ckpt['model_state'] if 'model_state' in ph_ckpt else ph_ckpt
        if 'phonetic_planner.phoneme_emb.weight' in ph_sd and hasattr(model, 'ph_decoder_emb'):
            with torch.no_grad():
                model.ph_decoder_emb.weight.copy_(ph_sd['phonetic_planner.phoneme_emb.weight'])
        model.load_state_dict(ph_sd, strict=False)

    # Refiner
    if args.refiner_checkpoint:
        print(f"  Refiner: {args.refiner_checkpoint}")
        ref_ckpt = torch.load(args.refiner_checkpoint, map_location=device)
        ref_sd = ref_ckpt.get('light_refiner', ref_ckpt)
        if model.light_refiner is not None:
            model.light_refiner.load_state_dict(ref_sd, strict=False)
            model.use_refinement = True
        else:
            print("\033[91m[Warning]\033[0m Refiner checkpoint provided but model has no light_refiner module.")

    # LoRA
    if args.lora_checkpoint:
        print(f"  LoRA: {args.lora_checkpoint}")
        raw = torch.load(args.lora_checkpoint, map_location=device)
        lora_sd = raw.get("lora_state_dict", raw)
        first_a = next(v for k, v in lora_sd.items() if "lora_A" in k)
        inferred_rank = first_a.shape[0]
        inferred_alpha = raw.get("lora_alpha", inferred_rank * 2)
        rank = args.lora_rank if args.lora_rank is not None else inferred_rank
        alpha = args.lora_alpha if args.lora_alpha is not None else inferred_alpha
        print(f"  LoRA rank={rank}, alpha={alpha}")
        model = inject_lora(model, rank=rank, alpha=alpha)
        model.load_state_dict(lora_sd, strict=False)

    dtype = (
        torch.bfloat16 if args.dtype == "bf16"
        else torch.float16 if args.dtype == "fp16"
        else torch.float32
    )
    if args.dtype != "fp32":
        model = model.to(dtype)
    autocast_enabled = args.dtype != "fp32"
    model.eval()

    # Leading-silence runway: attach the same cached mimi silence codes the
    # trainer used, so sampling injects the trained cold-start runway.
    from explicit_pros_phon_planner.utils import attach_leading_sil_codes
    if attach_leading_sil_codes(model, cfg, device=device):
        print(f"  [batch_infer] leading-silence runway active "
              f"({model.leading_sil_codes.shape[1]} frames x {model.leading_sil_codes.shape[0]}q)")

    # ---------- Load Mimi ----------
    from seedvox.modules.mimi import get_mimi_model
    mimi = get_mimi_model(device=device, checkpoint_path=cfg.get("mimi_checkpoint", "pretrained_models/best_mimi.pt")).eval()

    if args.lora_checkpoint and not args.no_mimi_lora:
        raw_lora_m = torch.load(args.lora_checkpoint, map_location=device)
        lora_sd_m = raw_lora_m.get("lora_state_dict", raw_lora_m)
        mimi_lora_keys = {k: v for k, v in lora_sd_m.items() if k.startswith("mimi.")}
        if mimi_lora_keys:
            mimi_lora = {k[5:]: v for k, v in mimi_lora_keys.items()}
            first_a_m = next(v for k, v in mimi_lora.items() if "lora_A" in k)
            mimi_rank = raw_lora_m.get("mimi_lora_rank", first_a_m.shape[0])
            mimi_alpha = raw_lora_m.get("mimi_lora_alpha", mimi_rank * 2)
            mimi = inject_lora(mimi, rank=mimi_rank, alpha=mimi_alpha, targets=MIMI_DECODER_TARGETS)
            mimi.load_state_dict(mimi_lora, strict=False)
            print(f"  Mimi LoRA applied (rank={mimi_rank}, alpha={mimi_alpha}).")
    elif args.lora_checkpoint and args.no_mimi_lora:
        print("  Mimi LoRA skipped (--no_mimi_lora).")
    mimi = mimi.to(device).eval()

    # ---------- Compile (optional) ----------
    comp_time = 0.0
    if args.compile:
        import torch._inductor.config as inductor_config
        inductor_config.fx_graph_cache = True
        inductor_config.compile_threads = 8
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        cache_dir = os.path.join(os.getcwd(), ".torch_compile_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

        print("Compiling...")
        t0 = time.time()
        for i in range(len(model.decoder_layers)):
            model.decoder_layers[i] = torch.compile(model.decoder_layers[i], mode="reduce-overhead", dynamic=True)
        model.dep_transformer = torch.compile(model.dep_transformer, mode="reduce-overhead", dynamic=True)
        model.audio_prenet = torch.compile(model.audio_prenet, mode="reduce-overhead", dynamic=True)
        if hasattr(model, "phonetic_planner"):
            model.phonetic_planner.transformer = torch.compile(model.phonetic_planner.transformer, mode="reduce-overhead", dynamic=True)
            model.phonetic_planner = torch.compile(model.phonetic_planner, mode="reduce-overhead", dynamic=True)
        mimi = torch.compile(mimi, dynamic=True)

        with torch.no_grad():
            dummy_text = torch.zeros((1, 5), dtype=torch.long, device=device)
            dummy_lens = torch.tensor([5], device=device)
            if hasattr(model, "phonetic_planner"):
                dummy_feat = torch.randn(1, 5, model.dim, device=device, dtype=dtype)
                with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled):
                    model.phonetic_planner.sample(dummy_feat, max_len=4)
            with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled):
                model.sample(dummy_text, dummy_lens, max_steps=4)
            mimi.decode(torch.zeros((1, model.n_q, 4), dtype=torch.long, device=device))
        comp_time = time.time() - t0
        print(f"  Ready in {comp_time:.2f}s\n")

    # ---------- Speaker extraction ----------
    def _extract_spk(wav_path):
        wav = load_audio(wav_path, device)
        with torch.no_grad():
            with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled):
                toks = mimi.encode(wav.unsqueeze(0))
                ae = _sum_embeddings(model.audio_embs, toks, model.n_q)
                ae = model.audio_prenet(model.audio_norm(ae))
                mimi_t_len = toks.shape[-1]
                audio_mask = torch.arange(mimi_t_len, device=device).unsqueeze(0) >= torch.full((1,), mimi_t_len, device=device)
                return model.speaker_encoder(ae, key_padding_mask=audio_mask)

    cached_ext_spk = None
    if args.ref_wav_speaker:
        print(f"Speaker: {args.ref_wav_speaker}")
        spk1 = _extract_spk(args.ref_wav_speaker)
        if args.ref_wav_speaker2:
            print(f"Speaker: {args.ref_wav_speaker2}")
            spk2 = _extract_spk(args.ref_wav_speaker2)
            r = args.spk_mix_ratio
            print(f"  Mix: {1-r:.2f} × {Path(args.ref_wav_speaker).stem} + {r:.2f} × {Path(args.ref_wav_speaker2).stem}")
            cached_ext_spk = (1 - r) * spk1 + r * spk2
        else:
            cached_ext_spk = spk1

    # ---------- Read demo lines ----------
    # Per-line syntax (same as run_demos.sh):
    #   Text to speak || output_name.wav || --ref_wav_speaker path.wav --exagg 1.5
    # The ' || ' delimiter strips the filename/flags from the spoken text.
    def _parse_extra_flags(s):
        toks = s.split()
        ovr = {}
        i = 0
        while i < len(toks):
            t = toks[i]
            if t in ("--ref_wav_speaker", "--ref_wav_speaker2", "--ref_wav_prosody") and i + 1 < len(toks):
                ovr[t[2:]] = toks[i + 1]
                i += 2
            elif t in ("--random_speaker", "--random_prosody"):
                ovr[t[2:]] = True
                i += 1
            elif t in ("--exagg", "--mono_slack", "--temp", "--cfg", "--seed",
                       "--prosody_temperature", "--phoneme_temp") and i + 1 < len(toks):
                ovr[t[2:]] = float(toks[i + 1])
                i += 2
            elif t == "--style" and i + 1 < len(toks):
                ovr[t[2:]] = int(toks[i + 1])
                i += 2
            else:
                i += 1
        return ovr

    def _parse_demo_line(line):
        fields = [f.strip() for f in line.split(" || ")]
        text = fields[0]
        outname = fields[1] if len(fields) >= 2 and fields[1] else None
        ovr = _parse_extra_flags(" || ".join(fields[2:])) if len(fields) >= 3 and fields[2] else {}
        return text, outname, ovr

    with open(args.demo_file) as f:
        raw_lines = [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]
    lines = [_parse_demo_line(l) for l in raw_lines]

    print(f"\nProcessing {len(lines)} lines, {args.variant_n} variant(s) each...\n")

    spk_cache = {}

    def _cached_spk(wav_path):
        if wav_path not in spk_cache:
            spk_cache[wav_path] = _extract_spk(wav_path)
        return spk_cache[wav_path]

    prs_cache = {}

    def _cached_prs(wav_path):
        if wav_path not in prs_cache:
            print(f"  Extracting prosody from {wav_path} (stage-1 codec)...")
            prs_cache[wav_path] = extract_ref_prosody_latent(wav_path, model, device)
        return prs_cache[wav_path]

    total_ok = 0
    total_fail = 0
    total_gen_time = 0.0

    for idx, (raw_text, outname, ovr) in enumerate(lines, 1):
        num = f"{idx:03d}"
        line_seed = ovr.get("seed", args.seed or 0)
        exagg = ovr.get("exagg", args.exagg)
        mono_slack = ovr.get("mono_slack", args.mono_slack)
        temp = ovr.get("temp", args.temp)
        cfg = ovr.get("cfg", args.cfg)
        prosody_temp = ovr.get("prosody_temperature", args.prosody_temperature)
        random_spk = bool(ovr.get("random_speaker", False) or (args.random_speaker and cached_ext_spk is None))
        random_prs = bool(ovr.get("random_prosody", False) or args.random_prosody)
        print(f"{'='*60}")
        print(f"  [{num}] {raw_text[:100]}")

        for variant_idx in range(args.variant_n):
            if args.variant_n > 1:
                print(f"  --- Variant {variant_idx + 1}/{args.variant_n} ---")
                set_seed(line_seed + variant_idx)

            variant_suffix = f"_{variant_idx}" if args.variant_n > 1 else ""
            if outname:
                base = outname[:-4] if outname.lower().endswith(".wav") else outname
                out_path = os.path.join(args.output_dir, f"{base}{variant_suffix}.wav")
            else:
                out_path = os.path.join(args.output_dir, f"demo_{num}{variant_suffix}.wav")

            try:
                text = normalize_text(raw_text)
                t_ids = torch.tensor([tokenizer.encode(text, normalize=False)], device=device)
                t_lens = torch.tensor([t_ids.shape[1]], device=device)

                # --- Phonetic planning ---
                ph_start = time.time()
                bpe_ids, bpe_lens, char_to_bpe = None, None, None

                if args.overwrite_phonemes:
                    ph_tokenizer = PhonemeTokenizer()
                    ph_ids_list = ph_tokenizer.encode(args.overwrite_phonemes)
                    ph_ids = torch.tensor([[model.SOS_ID] + ph_ids_list + [model.EOS_ID]], device=device)
                elif args.use_external_g2p:
                    generator = PhoneticGenerator(backend=args.g2p, phoneme_vocab_size=model.EOS_ID)
                    ph_ids_list = generator.generate_targets(text, normalize=False)
                    ph_ids = torch.tensor([ph_ids_list], device=device)
                else:
                    with torch.no_grad():
                        with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled):
                            if model.use_bpe_encoder:
                                from seedvox.utils.tokenizer import BPECharCollator
                                bpe_collator = BPECharCollator(model.bpe_encoder)
                                bpe_ids, bpe_lens, char_to_bpe = bpe_collator.process_batch_texts([text], t_lens, device)

                            text_feat = model.get_enriched_text_feat(
                                t_ids, t_lens, raw_texts=[text],
                                bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
                            )
                            ph_ids = model.phonetic_planner.sample(
                                text_feat, temp=args.phoneme_temp, top_p=args.phoneme_top_p,
                                greedy=args.phoneme_greedy,
                            )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                ph_time = (time.time() - ph_start) * 1000

                # --- Speaker / prosody conditioning ---
                if ovr.get("ref_wav_speaker"):
                    print(f"  Speaker: {ovr['ref_wav_speaker']}")
                    ext_spk = _cached_spk(ovr["ref_wav_speaker"])
                elif random_spk:
                    ext_spk = model.null_speaker + torch.randn_like(model.null_speaker) * 0.5
                else:
                    ext_spk = cached_ext_spk

                ext_prs = None
                ref_prs_wav = ovr.get("ref_wav_prosody") or args.ref_wav_prosody
                if ref_prs_wav:
                    print(f"  Prosody: {ref_prs_wav}")
                    ext_prs = _cached_prs(ref_prs_wav)
                elif random_prs:
                    ext_prs = model.null_prosody + torch.randn_like(model.null_prosody) * 0.3

                style_id = ovr.get("style", args.style)
                if style_id is not None:
                    print(f"  Style: {style_id}")

                if args.variant_axis == 'speaker' and ext_spk is None:
                    ext_spk = model.null_speaker + torch.randn_like(model.null_speaker) * 0.5
                if args.variant_axis == 'pros' and ext_prs is None:
                    ext_prs = model.null_prosody + torch.randn_like(model.null_prosody) * 0.3

                # --- Context encoding (JEPA prosody planning) ---
                cond_start = time.time()
                with torch.no_grad():
                    with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled):
                        context, ctx_mask, _, _, _, spk_vec, prosody_emb = model.encode_context(
                            t_ids, t_lens, raw_texts=[text],
                            phoneme_ids=ph_ids,
                            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
                            external_speaker=ext_spk,
                            external_prosody=ext_prs,
                            external_style=style_id,
                            exagg=exagg,
                            prosody_temperature=prosody_temp,
                        )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                cond_time = (time.time() - cond_start) * 1000

                # --- Acoustic generation ---
                ac_start = time.time()
                with torch.no_grad():
                    with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled):
                        audio_tokens, _ = model.sample(
                            t_ids, t_lens,
                            phoneme_ids=ph_ids,
                            temp=temp,
                            cfg_scale=cfg,
                            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
                            external_speaker=ext_spk,
                            external_prosody=ext_prs,
                            precomputed_context=context,
                            precomputed_mask=ctx_mask,
                            spk_vec=spk_vec,
                            prosody_emb=prosody_emb,
                            min_p=args.min_p,
                            rep_penalty=args.rep_penalty,
                            exagg=exagg,
                            mono_slack=mono_slack,
                            prosody_temperature=args.prosody_temperature,
                        )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                ac_time = (time.time() - ac_start) * 1000

                # --- Mimi decode ---
                dec_start = time.time()
                eoa_positions = (audio_tokens[:, 0, :] == model.EOA_ID).int().argmax(dim=-1)
                if eoa_positions.max() > 0:
                    audio_tokens = audio_tokens[:, :, :eoa_positions.max()]
                # Drop the final token frame: the AR model reliably emits a
                # degenerate noise-burst frame immediately before EOA (the
                # preceding frame is already near-silent, so no speech is lost).
                # That burst decodes to a click/buzz at the end of the audio.
                if audio_tokens.shape[-1] > 1:
                    audio_tokens = audio_tokens[:, :, :-1]
                wav = mimi.decode(audio_tokens.clamp(0, model.card - 1))
                expected_len = audio_tokens.shape[-1] * 1920
                if wav.shape[-1] > expected_len:
                    wav = wav[..., :expected_len]
                fade_len = min(240, wav.shape[-1] // 4)
                fade = torch.linspace(0.0, 1.0, fade_len, device=wav.device, dtype=wav.dtype)
                wav[..., :fade_len] *= fade
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dec_time = (time.time() - dec_start) * 1000

                if args.de_emph > 0:
                    wav = de_emphasize(wav, coeff=args.de_emph)
                if args.smooth > 0:
                    wav = temporal_smooth(wav, strength=args.smooth, kernel_size=args.smooth_kernel)

                torchaudio.save(out_path, wav[0].cpu(), 24000)
                dur = wav.shape[-1] / 24000
                total_elapsed = ph_time + cond_time + ac_time + dec_time
                total_gen_time += total_elapsed

                print(f"  \u2192 {out_path} ({dur:.2f}s, ph={ph_time:.0f}ms, cond={cond_time:.0f}ms, ac={ac_time:.0f}ms, dec={dec_time:.0f}ms)")

                if args.view_waveform:
                    print_waveform(wav[0])

                if args.play:
                    play_audio_v5(wav[0])

                total_ok += 1

            except Exception as e:
                print(f"  \u2717 FAILED: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                total_fail += 1

    # ---------- Summary ----------
    print(f"\n{'='*60}")
    print(f"  Done: {total_ok} OK, {total_fail} failed out of {len(lines) * args.variant_n} total")
    print(f"  Outputs in: {args.output_dir}/")

    if args.log_metrics and total_ok > 0:
        avg_time = total_gen_time / total_ok
        print(f"\n  Average generation time: {avg_time:.0f}ms per utterance")
        if comp_time > 0:
            print(f"  Compilation: {comp_time:.2f}s (one-time)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
