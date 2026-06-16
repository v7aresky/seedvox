import torch
import torchaudio
import argparse
import json
import os
import sys
import time
import tempfile
import subprocess
import numpy as np
from pathlib import Path

# Add project root and src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from explicit_pros_phon_planner.model import ExplicitPlannerModel

from seedvox.utils.tokenizer import CharTokenizer, PhonemeTokenizer
from explicit_pros_phon_planner.utils import PhoneticGenerator, collate_phonemes

def get_braille_char(dots):
    """
    dots: 2x4 list of bits [[col0_row0, col1_row0], ...]
    Returns the Unicode braille character.
    """
    # Braille bit mapping:
    # 1 4
    # 2 5
    # 3 6
    # 7 8
    # Bits for dots: 1:0x01, 2:0x02, 3:0x04, 4:0x08, 5:0x10, 6:0x20, 7:0x40, 8:0x80
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
    """
    Static waveform overview using Braille dots for ultra-high resolution.
    """
    audio = audio_tensor.squeeze().cpu().numpy()
    duration = len(audio) / sample_rate
    
    # 2 dots per char width, 4 dots per char height
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
    print(f"  \033[90m┌{'─' * width}┐\033[0m")
    
    # Render top and bottom halves using braille
    # Each terminal row is 4 dots high
    for tr in range(height - 1, -height - 1, -1):
        line = "  \033[90m│\033[0m"
        for tc in range(width):
            # 2 columns of dots
            d_cols = [bars[tc*2], bars[tc*2 + 1]]
            dots = [[0,0],[0,0],[0,0],[0,0]]
            
            # Dot row 0-3 within this terminal row
            for dr in range(4):
                # Vertical dot index from center
                # Top half: tr*4 + (3-dr)
                # Bottom half: symmetric
                dot_y = (tr * 4) + (3 - dr)
                
                # Simple logic for symmetric bars
                for c in range(2):
                    if dot_y >= 0: # Above or on axis
                        if d_cols[c] >= dot_y + 1: dots[dr][c] = 1
                    else: # Below axis
                        if d_cols[c] >= abs(dot_y): dots[dr][c] = 1
            
            char = get_braille_char(dots)
            # Apply color based on avg height
            avg_h = (d_cols[0] + d_cols[1]) / 2 / h_dots_half
            color = "\033[38;5;198m" if avg_h > 0.8 else ("\033[38;5;45m" if avg_h > 0.5 else "\033[38;5;39m")
            line += f"{color}{char}\033[0m"
        line += "\033[90m│\033[0m"
        print(line)
        
    print(f"  \033[90m└{'─' * width}┘\033[0m")
    print(f"  \033[90m0s{' ' * (width - 7)}{duration:.1f}s\033[0m\n")


def play_audio_v5(audio_tensor, sample_rate=24000):
    """
    Play audio with an ultra-high resolution Braille reveal animation.
    """
    print("\n\033[94m[DEBUG] Active Animation: Ultra-Res Braille (v5)\033[0m")
    
    audio_np = audio_tensor.squeeze().cpu().numpy()
    duration = len(audio_np) / sample_rate
    
    width = 80
    height = 2 # 2 rows above, 2 below = 16 dots total vertical
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
        except: continue

    try:
        sys.stdout.write("\033[?25l")
        total_lines = height * 2
        sys.stdout.write('\n' * total_lines)
        
        start_time = time.time()
        while True:
            if player_proc and player_proc.poll() is not None: break
            elapsed = time.time() - start_time
            if elapsed >= duration: elapsed = duration
            
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
                            # Only show dots that have been reached by the audio
                            if (tc * 2 + c) <= play_pos_dot:
                                if dot_y >= 0:
                                    if d_cols[c] >= dot_y + 1: dots[dr][c] = 1
                                else:
                                    if d_cols[c] >= abs(dot_y): dots[dr][c] = 1
                    
                    char = get_braille_char(dots)
                    
                    if tc < play_pos_tc:
                        avg_h = (d_cols[0] + d_cols[1]) / 2 / h_dots_half
                        color = "\033[96m" if avg_h > 0.5 else "\033[36m"
                        line += f"{color}{char}\033[0m"
                    else:
                        # Cursor leading edge
                        line += f"\033[97m{char}\033[0m"
                
                sys.stdout.write(line + "\033[K\n")
            
            sys.stdout.flush()
            if elapsed >= duration: break
            time.sleep(0.04)
            
    finally:
        sys.stdout.write("\033[?25h")
        if player_proc and player_proc.poll() is None:
            player_proc.terminate()
        try: os.remove(tmp_path)
        except: pass
        sys.stdout.write('\n')
    return True

def load_audio(path, device):
    wav, sr = torchaudio.load(path)
    if sr != 24000: wav = torchaudio.transforms.Resample(sr, 24000)(wav)
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    return wav.to(device)

def _sum_embeddings(emb_list, tokens, n_q):
    ae = emb_list[0](tokens[:, 0])
    for k in range(1, n_q):
        ae = ae + emb_list[k](tokens[:, k])
    return ae

def set_seed(seed):
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for some ops if needed
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def run_inference(args):
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Setting global seed: {args.seed}")
    
    device = torch.device(args.device)
    
    # FlashAttention 2 Check
    if device.type == 'cuda':
        from torch.backends.cuda import flash_sdp_enabled, mem_efficient_sdp_enabled
        fa_ok = flash_sdp_enabled()
        me_ok = mem_efficient_sdp_enabled()
        print(f"\033[94m[GPU Diagnostic]\033[0m {torch.cuda.get_device_name(0)}")
        print(f"  FlashAttention-2: {'\033[92mEnabled\033[0m' if fa_ok else '\033[91mDisabled\033[0m'}")
        print(f"  Memory Efficient: {'\033[92mEnabled\033[0m' if me_ok else '\033[91mDisabled\033[0m'}")
        if args.dtype == 'fp32':
            print(f"  \033[93mNOTE: FlashAttention-2 requires FP16/BF16. Falling back to MemEfficient for FP32.\033[0m")
    
    metrics = {}
    
    # 1. Load config and model
    with open(args.config, "r") as f:
        cfg = json.load(f)
        
    tokenizer = CharTokenizer()
    model = ExplicitPlannerModel(cfg, tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
    
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'ema_model' in ckpt:
            print("@@@@@ Loading EMA model weights (from full checkpoint)...")
            state_dict = ckpt['ema_model']
        elif 'model' in ckpt:
            print("@@@@ Loading standard model weights (from full checkpoint)...")
            state_dict = ckpt['model']
        else:
            # Check filename for 'ema' to give a better message
            if 'ema' in args.checkpoint.lower():
                print(f"@@@@@ Loading EMA model weights from {args.checkpoint}...")
            else:
                print(f"@@@@ Loading model weights from {args.checkpoint}...")
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
    
    if args.dtype == 'bf16':
        print("Casting model to bfloat16...")
        model = model.to(torch.bfloat16)
    elif args.dtype == 'fp16':
        print("Casting model to float16...")
        model = model.half()
    
    model.eval()

    # Load Mimi
    from seedvox.modules.mimi import get_mimi_model
    mimi = get_mimi_model(device=device, checkpoint_path=cfg.get('mimi_checkpoint', 'pretrained_models/best_mimi.pt')).eval()
    
    # 1.5 Optional Compilation
    if args.compile:
        import torch._inductor.config as inductor_config
        # Enable persistent on-disk cache
        inductor_config.fx_graph_cache = True
        # Prioritize faster startup for cache hits
        inductor_config.compile_threads = 8 
        
        # Set a project-local cache directory
        cache_dir = os.path.join(os.getcwd(), ".torch_compile_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        
        print("\n\033[93m⚠ Optimizing model with torch.compile...\033[0m")
        print(f"\033[90m  (Cache: {cache_dir})\033[0m")
        comp_start = time.time()
        
        # Mode "reduce-overhead" uses CUDA graphs to eliminate CPU launch overhead
        # which is perfect for AR loops like our acoustic generator.
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        # Mimi is a standard CNN/Transformer, default compilation is usually best.
        mimi = torch.compile(mimi, dynamic=True)
        
        # Streamlined warmup pass
        print("  Verifying cache and warming up generation loop...")
        with torch.no_grad():
            try:
                # 1. Minimal phonetic + acoustic warmup (triggers most kernels)
                dummy_text = torch.zeros((1, 5), dtype=torch.long, device=device)
                dummy_lens = torch.tensor([5], device=device)
                _ = model.sample(dummy_text, dummy_lens, max_steps=4)
                
                # 2. Mimi decode warmup
                _ = mimi.decode(torch.zeros((1, 8, 4), dtype=torch.long, device=device))
            except Exception as e:
                print(f"\033[91m  Warmup notification: {e}\033[0m")
        
        comp_time = time.time() - comp_start
        print(f"  \033[92m✓\033[0m Ready in {comp_time:.2f}s\n")
        metrics['compilation_time_s'] = comp_time

    # 2. Text processing
    text = args.text
    print(f"Input Text: '{text}'")
    t_ids = torch.tensor([tokenizer.encode(text)], device=device)
    t_lens = torch.tensor([t_ids.shape[1]], device=device)
    
    total_start = time.time()
    
    # 3. Phonetic Planning (CoT step 1)
    ph_start = time.time()
    
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16 if args.dtype == 'fp16' else torch.float32
    autocast_enabled = args.dtype != 'fp32'

    bpe_ids, bpe_lens, char_to_bpe = None, None, None
    if args.overwrite_phonemes:
        print(f"Overwriting phonemes with: {args.overwrite_phonemes}")
        ph_tokenizer = PhonemeTokenizer()
        ph_ids_list = ph_tokenizer.encode(args.overwrite_phonemes)
        ph_ids = torch.tensor([[model.SOS_ID] + ph_ids_list + [model.EOS_ID]], device=device)
    elif args.use_external_g2p:
        print(f"Using external G2P ({args.g2p}) for phonemes...")
        generator = PhoneticGenerator(backend=args.g2p, phoneme_vocab_size=model.EOS_ID)
        ph_ids_list = generator.generate_targets(text)
        ph_ids = torch.tensor([ph_ids_list], device=device)
        ph_tokenizer = PhonemeTokenizer()
        clean_tokens = [t for t in ph_ids_list if t != model.SOS_ID and t != model.EOS_ID]
        print(f"G2P Phonemes: {ph_tokenizer.decode(clean_tokens)}")
    else:
        # Sample from phonetic_planner
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                if model.use_bpe_encoder:
                    from seedvox.utils.tokenizer import BPECharCollator
                    bpe_collator = BPECharCollator(model.bpe_encoder)
                    bpe_ids, bpe_lens, char_to_bpe = bpe_collator.process_batch_texts([text], t_lens, device)
                    decoded_bpe = model.bpe_encoder.bpe_tokenizer.decode(bpe_ids[0].tolist())
                    print(f"BPE Tokens: '{decoded_bpe}'")

                text_feat = model.get_enriched_text_feat(
                    t_ids, t_lens, raw_texts=[text],
                    bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe
                )
                ph_ids = model.phonetic_planner.sample(text_feat, temp=args.phoneme_temp, top_p=args.phoneme_top_p)
                if device.type == 'cuda': torch.cuda.synchronize()
                ph_tokenizer = PhonemeTokenizer()
                tokens = ph_ids[0].tolist()
                clean_tokens = [t for t in tokens if t != model.SOS_ID and t != model.EOS_ID]
                print(f"Planned Phonemes: {ph_tokenizer.decode(clean_tokens)}")
    
    if device.type == 'cuda': torch.cuda.synchronize()
    metrics['phonetic_planning_ms'] = (time.time() - ph_start) * 1000

    # 4. Speaker and Prosody Conditioning
    cond_start = time.time()
    
    ext_spk = None
    if args.ref_wav_speaker:
        print(f"Extracting speaker from {args.ref_wav_speaker}...")
        wav = load_audio(args.ref_wav_speaker, device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                mimi_toks = mimi.encode(wav.unsqueeze(0))
                ae = _sum_embeddings(model.audio_embs, mimi_toks, model.n_q)
                ae = model.audio_prenet(model.audio_norm(ae))
                ext_spk = model.speaker_encoder(ae)
    elif args.random_speaker:
        print("Sampling random speaker...")
        num_spk = model.cfg.get('num_speaker_latents', 16)
        ext_spk = torch.randn(1, num_spk, model.dim, device=device)

    ext_prs = None
    if args.ref_wav_prosody:
        print(f"Extracting prosody from {args.ref_wav_prosody}...")
        wav = load_audio(args.ref_wav_prosody, device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                mimi_toks = mimi.encode(wav.unsqueeze(0))
                ae = _sum_embeddings(model.audio_embs, mimi_toks, model.n_q)
                ae = model.audio_prenet(model.audio_norm(ae))
                ext_prs = model.prosody_encoder(ae)
    elif args.random_prosody:
        print("Sampling random prosody...")
        num_prs = model.cfg.get('num_prosody_tokens', 32)
        ext_prs = torch.randn(1, num_prs, model.dim, device=device)
    
    # NEW: Explicit Context Encoding (Timing Prosody Planning)
    # This ensures jepa_planner is timed here, not hidden in generation.
    print("Encoding context and planning prosody...")
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            context, ctx_mask, _, _, _ = model.encode_context(
                t_ids, t_lens, raw_texts=[text], 
                phoneme_ids=ph_ids,
                bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
                external_speaker=ext_spk,
                external_prosody=ext_prs
            )
            
    if device.type == 'cuda': torch.cuda.synchronize()
    cond_end = time.time()
    metrics['conditioning_total_ms'] = (cond_end - cond_start) * 1000
    # Prosody planning is effectively the entire context assembly here
    metrics['prosody_planning_ms'] = metrics['conditioning_total_ms']

    # 5. Acoustic Generation (CoT step 2)
    ac_start = time.time()
    
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            # We now pass the precomputed context to avoid redundant work
            # (Note: model.sample will need to be updated to support this, 
            # or it will just recompute it, which is fine for metrics accuracy)
            audio_tokens, _ = model.sample(
                t_ids, t_lens, 
                phoneme_ids=ph_ids,
                temp=args.temp,
                cfg_scale=args.cfg,
                bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
                external_speaker=ext_spk,
                external_prosody=ext_prs,
                precomputed_context=context,
                precomputed_mask=ctx_mask
            )
        if device.type == 'cuda': torch.cuda.synchronize()
        metrics['acoustic_gen_ms'] = (time.time() - ac_start) * 1000
        
        # Extract prosody embeddings for visualization separately if needed
        # (This avoids polluting the main sample timing)
        print("Extracting prosody embeddings for viz...")
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            context, _, _, _, _ = model.encode_context(
                t_ids, t_lens, audio_tokens, None, [text], 
                phoneme_ids=ph_ids,
                bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
                external_speaker=ext_spk,
                external_prosody=ext_prs
            )
        
        spk_len = model.cfg['num_speaker_latents'] if model.use_speaker else 0
        prs_len = model.cfg.get('num_prosody_tokens', 32)
        prosody_embs = context[:, spk_len : spk_len + prs_len, :]
        torch.save(prosody_embs.cpu(), "prosody_embs.pt")
    
    # 6. Decode with Mimi
    dec_start = time.time()
    with torch.no_grad():
        wav = mimi.decode(audio_tokens)
        if device.type == 'cuda': torch.cuda.synchronize()
    metrics['mimi_decode_ms'] = (time.time() - dec_start) * 1000
    
    total_time = time.time() - total_start
    audio_duration = wav.shape[2] / 24000
    rtf = total_time / audio_duration if audio_duration > 0 else 0
    
    torchaudio.save(args.output, wav[0].cpu(), 24000)
    print(f"Saved generated audio to {args.output}")

    if args.log_metrics:
        print("\n" + "="*30)
        print("Performance Metrics:")
        if 'compilation_time_s' in metrics:
            print(f"  Compilation Time:  {metrics['compilation_time_s']:.2f}s")
        
        # Combined Planner Metrics
        ph_ms = metrics.get('phonetic_planning_ms', 0)
        pr_ms = metrics.get('prosody_planning_ms', 0)
        print(f"  Planner:           Phonetics {ph_ms:.1f}ms; Prosody {pr_ms:.1f}ms")
        
        print(f"  Acoustic Gen:      {metrics['acoustic_gen_ms']:.1f}ms")
        print(f"  Mimi Decode:       {metrics['mimi_decode_ms']:.1f}ms")
        print("-" * 30)
        print(f"  Total Latency:     {total_time*1000:.1f}ms")
        print(f"  Audio Duration:    {audio_duration:.2f}s")
        print(f"  Real-time Factor:  {rtf:.4f}x")
        print("="*30 + "\n")

    if args.view_waveform:
        print_waveform(wav[0])
    
    if args.play:
        play_audio_v5(wav[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/seedvox_latest.pt")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--overwrite_phonemes", type=str, default=None)
    parser.add_argument("--use_external_g2p", action="store_true", help="Use external G2P instead of phonetic planner")
    parser.add_argument("--g2p", default="espeak", help="G2P backend to use with --use_external_g2p")
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=['fp32', 'fp16', 'bf16'], default='fp32', help="Inference precision")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducibility")
    
    # New speaker/prosody options
    parser.add_argument("--ref_wav_speaker", type=str, default=None)
    parser.add_argument("--ref_wav_prosody", type=str, default=None)
    parser.add_argument("--random_speaker", action="store_true")
    parser.add_argument("--random_prosody", action="store_true")
    
    # Phoneme sampling
    parser.add_argument("--phoneme_temp", type=float, default=1.0)
    parser.add_argument("--phoneme_top_p", type=float, default=0.9)

    # Visualization and playback
    parser.add_argument("--play", action="store_true", help="Play audio after generation")
    parser.add_argument("--view_waveform", action="store_true", help="View static waveform")
    parser.add_argument("--log_metrics", action="store_true", help="Log performance metrics (latency, RTF)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for faster inference")
    
    args = parser.parse_args()
    run_inference(args)

