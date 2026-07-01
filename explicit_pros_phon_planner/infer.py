import torch
import torch._dynamo
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
from explicit_pros_phon_planner.model_fusion import FusionPlannerModel
from explicit_pros_phon_planner.finetune_lora_fusion import inject_lora, LoRALinear, LoRAConv1d, LoRAConvTranspose1d

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
    
    audio_np = audio_tensor.squeeze().cpu().numpy()
    duration = len(audio_np) / sample_rate
    
    width = 80
    height = 4 # 4 rows above, 4 below = 32 dots total vertical
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
                        color = "\033[38;2;255;20;147m" if avg_h > 0.5 else "\033[38;5;205m"  #"\033[96m" "\033[36m"
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

def merge_lora_weights(model_with_lora):
    """
    Permanently merge LoRA weights into a clean model.
    Returns a new model with merged weights and no LoRA wrappers.
    Handles LoRALinear (weight-merged), LoRAConv1d, LoRAConvTranspose1d.
    """
    model_cls = type(model_with_lora)
    cfg = model_with_lora.cfg
    tokenizer = CharTokenizer()
    clean_model = model_cls(cfg, tokenizer.vocab_size, phoneme_vocab_size=128).to(
        next(model_with_lora.parameters()).device
    )
    clean_sd = clean_model.state_dict()
    loaded_sd = model_with_lora.state_dict()

    lora_linear = {}
    lora_conv1d = {}
    lora_convtr1d = {}
    for name, mod in model_with_lora.named_modules():
        if isinstance(mod, LoRALinear):
            lora_linear[name] = mod
        elif isinstance(mod, LoRAConv1d):
            lora_conv1d[name] = mod
        elif isinstance(mod, LoRAConvTranspose1d):
            lora_convtr1d[name] = mod

    merged_sd = {}
    for key in clean_sd:
        parts = key.rsplit('.', 1)
        mod_path = parts[0] if len(parts) == 2 else ''
        param_name = parts[-1] if len(parts) == 2 else parts[0]

        if mod_path in lora_linear:
            mod = lora_linear[mod_path]
            if param_name == 'weight':
                merged_sd[key] = mod.weight.detach().cpu()
            elif param_name == 'bias':
                merged_sd[key] = mod.bias.detach().cpu() if mod.bias is not None else clean_sd[key]
            else:
                merged_sd[key] = clean_sd[key]

        elif mod_path in lora_conv1d:
            mod = lora_conv1d[mod_path]
            if param_name == 'weight':
                merged = mod.original_conv.weight + (mod.lora_B @ mod.lora_A) * mod.scaling
                merged_sd[key] = merged.detach().cpu()
            elif param_name == 'bias':
                merged_sd[key] = mod.bias.detach().cpu() if mod.bias is not None else clean_sd[key]
            else:
                merged_sd[key] = clean_sd[key]

        elif mod_path in lora_convtr1d:
            mod = lora_convtr1d[mod_path]
            if param_name == 'weight':
                out_c, in_c_g = mod.original_convtr.weight.shape[:2]
                g = mod.original_convtr.groups
                lo = (mod.lora_B @ mod.lora_A.transpose(1, 2)).transpose(0, 1).contiguous()
                merged = mod.original_convtr.weight + lo.view_as(mod.original_convtr.weight) * mod.scaling
                merged_sd[key] = merged.detach().cpu()
            elif param_name == 'bias':
                merged_sd[key] = mod.bias.detach().cpu() if mod.bias is not None else clean_sd[key]
            else:
                merged_sd[key] = clean_sd[key]


        else:
            if key in loaded_sd:
                merged_sd[key] = loaded_sd[key].detach().cpu()
            else:
                merged_sd[key] = clean_sd[key]

    clean_model.load_state_dict(merged_sd, strict=False)
    return clean_model


def load_lora_and_merge(model, lora_checkpoint, rank=16, alpha=32, device='cpu'):
    """
    Inject LoRA into a loaded model, load LoRA weights, then merge them in.
    Returns a model with LoRA weights permanently merged (no LoRA wrappers).
    """
    model = inject_lora(model, rank=rank, alpha=alpha)
    lora_sd = torch.load(lora_checkpoint, map_location=device)
    model.load_state_dict(lora_sd, strict=False)
    merged_model = merge_lora_weights(model)
    return merged_model


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
    
    # Factory: Choose model class based on config (with CLI override & auto-detection from checkpoint)
    use_fusion = args.use_linguistic_fusion or cfg['model'].get('use_linguistic_fusion', False)
    
    if args.checkpoint:
        try:
            ckpt_temp = torch.load(args.checkpoint, map_location='cpu')
            state_dict_temp = ckpt_temp.get('model', ckpt_temp.get('ema_model', ckpt_temp))
            has_fusion_keys = any(k.startswith('linguistic_fusion') for k in state_dict_temp.keys())
            if has_fusion_keys and not use_fusion:
                print("\033[93m[Auto-detect]\033[0m Detected 'linguistic_fusion' keys in checkpoint. Automatically enabling FusionPlannerModel.")
                use_fusion = True
            elif not has_fusion_keys and use_fusion:
                print("\033[93m[Warning]\033[0m use_linguistic_fusion is enabled but checkpoint does not contain 'linguistic_fusion' keys. Bypassing fusion.")
                use_fusion = False
        except Exception as e:
            print(f"\033[91m[Auto-detect Error]\033[0m Failed to pre-scan checkpoint: {e}")
            
    if use_fusion:
        print("\033[94m[Inference]\033[0m Using FusionPlannerModel (Unified Linguistic Encoder)")
        model_cls = FusionPlannerModel
    else:
        print("\033[94m[Inference]\033[0m Using ExplicitPlannerModel")
        model_cls = ExplicitPlannerModel
        
    model = model_cls(cfg, tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
    
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'ema_model' in ckpt:
            print("@@@@@ Loading EMA model weights (from full checkpoint)...")
            state_dict = ckpt['ema_model']
        elif 'model' in ckpt:
            print("@@@@ Loading standard model weights (from full checkpoint)...")
            state_dict = ckpt['model']
        else:
            if 'ema' in args.checkpoint.lower():
                print(f"@@@@@ Loading EMA model weights from {args.checkpoint}...")
            else:
                print(f"@@@@ Loading model weights from {args.checkpoint}...")
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)

    # 1b. LoRA injection (if provided)
    if args.lora_checkpoint:
        print(f"\033[93m[LoRA]\033[0m Loading LoRA adapter from {args.lora_checkpoint}...")
        raw_lora = torch.load(args.lora_checkpoint, map_location=device)
        lora_sd = raw_lora.get('lora_state_dict', raw_lora)
        # Infer rank from the first lora_A tensor shape
        first_a = next(v for k, v in lora_sd.items() if 'lora_A' in k)
        inferred_rank = first_a.shape[0]
        inferred_alpha = raw_lora.get('lora_alpha', inferred_rank * 2)
        lora_rank = args.lora_rank if args.lora_rank is not None else inferred_rank
        lora_alpha = args.lora_alpha if args.lora_alpha is not None else inferred_alpha
        print(f"\033[93m[LoRA]\033[0m Detected rank={inferred_rank}, alpha={inferred_alpha} from checkpoint (using rank={lora_rank}, alpha={lora_alpha})")
        model = inject_lora(model, rank=lora_rank, alpha=lora_alpha)
        model.load_state_dict(lora_sd, strict=False)
        print(f"\033[92m[LoRA]\033[0m LoRA weights applied successfully.")

    if args.ph_checkpoint:
        print(f"@@@@@ Overloading phonetic weights from {args.ph_checkpoint}...")
        ph_ckpt = torch.load(args.ph_checkpoint, map_location=device)
        ph_sd = ph_ckpt['model_state'] if 'model_state' in ph_ckpt else ph_ckpt
        
        # Explicitly check and copy weights for ph_decoder_emb from phonetic_planner.phoneme_emb if possible
        # to ensure the acoustic model uses the same embeddings as the planner
        if 'phonetic_planner.phoneme_emb.weight' in ph_sd and hasattr(model, 'ph_decoder_emb'):
            with torch.no_grad():
                model.ph_decoder_emb.weight.copy_(ph_sd['phonetic_planner.phoneme_emb.weight'])
        
        msg = model.load_state_dict(ph_sd, strict=False)
        print(f"Loaded phonetic weights. Incompatible keys: {msg.unexpected_keys if msg.unexpected_keys else 'None'}")
    
    if args.dtype == 'bf16':
        print("Casting model to bfloat16...")
        model = model.to(torch.bfloat16)
    elif args.dtype == 'fp16':
        print("Casting model to float16...")
        model = model.half()
    
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16 if args.dtype == 'fp16' else torch.float32
    autocast_enabled = args.dtype != 'fp32'

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
        # Prevent recompiles from integer module attributes (e.g. streaming offset)
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        
        # Set a project-local cache directory
        cache_dir = os.path.join(os.getcwd(), ".torch_compile_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        
        print("\n\033[93m⚠ Optimizing model with torch.compile...\033[0m")
        print(f"\033[90m  (Cache: {cache_dir})\033[0m")
        comp_start = time.time()
        
        # Mode "reduce-overhead" uses CUDA graphs to eliminate CPU launch overhead
        # which is perfect for AR loops like our acoustic generator.
        print("  Compiling decoder layers...")
        for i in range(len(model.decoder_layers)):
            model.decoder_layers[i] = torch.compile(model.decoder_layers[i], mode="reduce-overhead", dynamic=True)
            
        print("  Compiling dependency transformer...")
        model.dep_transformer = torch.compile(model.dep_transformer, mode="reduce-overhead", dynamic=True)
        
        print("  Compiling audio prenet...")
        model.audio_prenet = torch.compile(model.audio_prenet, mode="reduce-overhead", dynamic=True)
        
        if hasattr(model, 'phonetic_planner'):
            print("  Compiling phonetic planner...")
            model.phonetic_planner.transformer = torch.compile(model.phonetic_planner.transformer, mode="reduce-overhead", dynamic=True)
            model.phonetic_planner = torch.compile(model.phonetic_planner, mode="reduce-overhead", dynamic=True)
            
        # Mimi is a standard CNN/Transformer, default compilation is usually best.
        mimi = torch.compile(mimi, dynamic=True)
        
        # Streamlined warmup pass
        print("  Verifying cache and warming up generation loop...")
        with torch.no_grad():
            try:
                # 1. Minimal phonetic + acoustic warmup (triggers most kernels)
                dummy_text = torch.zeros((1, 5), dtype=torch.long, device=device)
                dummy_lens = torch.tensor([5], device=device)
                
                if hasattr(model, 'phonetic_planner'):
                    dummy_text_feat = torch.randn(1, 5, model.dim, device=device, dtype=dtype)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                        _ = model.phonetic_planner.sample(dummy_text_feat, max_len=4)

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                    _ = model.sample(dummy_text, dummy_lens, max_steps=4)
                
                # 2. Mimi decode warmup (match model's n_q)
                _ = mimi.decode(torch.zeros((1, model.n_q, 4), dtype=torch.long, device=device))
            except Exception as e:
                print(f"\033[91m  Warmup notification: {e}\033[0m")
        
        comp_time = time.time() - comp_start
        print(f"  \033[92m✓\033[0m Ready in {comp_time:.2f}s\n")
        metrics['compilation_time_s'] = comp_time

    # 2. Text processing
    from seedvox.utils.text import normalize_text
    raw_text = args.text
    text = normalize_text(raw_text)
    print(f"Input Text (Raw): '{raw_text}'")
    print(f"Input Text (Norm): '{text}'")
    
    t_ids = torch.tensor([tokenizer.encode(text, normalize=False)], device=device)
    t_lens = torch.tensor([t_ids.shape[1]], device=device)
    
    total_start = time.time()
    
    # 3. Phonetic Planning (CoT step 1)
    ph_start = time.time()


    bpe_ids, bpe_lens, char_to_bpe = None, None, None
    if args.overwrite_phonemes:
        print(f"Overwriting phonemes with: {args.overwrite_phonemes}")
        ph_tokenizer = PhonemeTokenizer()
        ph_ids_list = ph_tokenizer.encode(args.overwrite_phonemes)
        ph_ids = torch.tensor([[model.SOS_ID] + ph_ids_list + [model.EOS_ID]], device=device)
    elif args.use_external_g2p:
        print(f"Using external G2P ({args.g2p}) for phonemes...")
        generator = PhoneticGenerator(backend=args.g2p, phoneme_vocab_size=model.EOS_ID)
        # Note: text is already normalized
        ph_ids_list = generator.generate_targets(text, normalize=False)
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

    # 4. Variant generation loop
    output_base = os.path.splitext(args.output)[0]
    output_ext = os.path.splitext(args.output)[1] or '.wav'

    cached_ext_spk = None
    cached_ext_prs = None

    for variant_idx in range(args.variant_n):
        if args.variant_n > 1:
            print(f"\n{'='*40}")
            print(f"  Variant {variant_idx + 1} / {args.variant_n}")
            print(f"{'='*40}")
            if args.seed is not None:
                set_seed(args.seed + variant_idx)

        cond_start = time.time()

        ext_spk = None
        if args.ref_wav_speaker:
            if cached_ext_spk is None:
                print(f"Extracting speaker from {args.ref_wav_speaker}...")
                wav = load_audio(args.ref_wav_speaker, device)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                        mimi_toks = mimi.encode(wav.unsqueeze(0))
                        ae = _sum_embeddings(model.audio_embs, mimi_toks, model.n_q)
                        ae = model.audio_prenet(model.audio_norm(ae))
                        cached_ext_spk = model.speaker_encoder(ae)
            ext_spk = cached_ext_spk
        elif args.random_speaker:
            print("Sampling random speaker...")
            num_spk = model.cfg.get('num_speaker_latents', 16)
            ext_spk = torch.randn(1, num_spk, model.dim, device=device)

        ext_prs = None
        if args.ref_wav_prosody:
            if cached_ext_prs is None:
                print(f"Extracting prosody from {args.ref_wav_prosody}...")
                wav = load_audio(args.ref_wav_prosody, device)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                        mimi_toks = mimi.encode(wav.unsqueeze(0))
                        ae = _sum_embeddings(model.audio_embs, mimi_toks, model.n_q)
                        ae = model.audio_prenet(model.audio_norm(ae))
                        cached_ext_prs = model.prosody_encoder(ae)
            ext_prs = cached_ext_prs
        elif args.random_prosody:
            print("Sampling random prosody...")
            num_prs = model.cfg.get('num_prosody_tokens', 32)
            ext_prs = torch.randn(1, num_prs, model.dim, device=device)

        # Auto-generate random embeddings for the varied axis if not already set
        if args.variant_axis == 'speaker' and ext_spk is None:
            print("Sampling random speaker (variant axis)...")
            num_spk = model.cfg.get('num_speaker_latents', 16)
            ext_spk = torch.randn(1, num_spk, model.dim, device=device)
        if args.variant_axis == 'pros' and ext_prs is None:
            print("Sampling random prosody (variant axis)...")
            num_prs = model.cfg.get('num_prosody_tokens', 32)
            ext_prs = torch.randn(1, num_prs, model.dim, device=device)

        # Explicit Context Encoding (Timing Prosody Planning)
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
        cond_time = (time.time() - cond_start) * 1000
        metrics['conditioning_total_ms'] = cond_time
        metrics['prosody_planning_ms'] = cond_time

        # 5. Acoustic Generation (CoT step 2)
        ac_start = time.time()

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
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

            if args.variant_n <= 1:
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
            wav = mimi.decode(audio_tokens.clamp(0, model.card - 1))
            # Trim to expected length: T frames at 12.5Hz → T * 24000/12.5 = T*1920 samples
            expected_len = audio_tokens.shape[-1] * 1920
            if wav.shape[-1] > expected_len:
                wav = wav[..., :expected_len]
            # Fade-in to mask causal convolution startup transient
            fade_len = min(240, wav.shape[-1] // 4)  # 10ms at 24kHz
            fade = torch.linspace(0.0, 1.0, fade_len, device=wav.device, dtype=wav.dtype)
            wav[..., :fade_len] *= fade
            wav[..., :fade_len] *= 0.90
            if device.type == 'cuda': torch.cuda.synchronize()
        metrics['mimi_decode_ms'] = (time.time() - dec_start) * 1000

        # 7. Save output
        if args.variant_n > 1:
            variant_output = f"{output_base}_{variant_idx}{output_ext}"
        else:
            variant_output = args.output

        torchaudio.save(variant_output, wav[0].cpu(), 24000)
        print(f"Saved generated audio to {variant_output}")

        if args.variant_n > 1 and args.log_metrics:
            ph_ms = metrics.get('phonetic_planning_ms', 0)
            print(f"  Variant {variant_idx + 1} | Conditioning: {cond_time:.1f}ms | "
                  f"Acoustic: {metrics['acoustic_gen_ms']:.1f}ms | "
                  f"Decode: {metrics['mimi_decode_ms']:.1f}ms")

        if args.view_waveform:
            print_waveform(wav[0])

        if args.play:
            play_audio_v5(wav[0])

    # 8. Save merged checkpoint (LoRA weights baked into model)
    if args.save_merged_checkpoint and args.lora_checkpoint:
        print(f"\033[93m[Merge]\033[0m Merging LoRA weights into full model...")
        merged_model = merge_lora_weights(model)
        torch.save(merged_model.state_dict(), args.save_merged_checkpoint)
        print(f"\033[92m[Merge]\033[0m Saved merged model to {args.save_merged_checkpoint}")

    if args.variant_n <= 1 and args.log_metrics:
        total_time = time.time() - total_start
        audio_duration = wav.shape[2] / 24000
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        print("\n" + "="*30)
        print("Performance Metrics:")
        if 'compilation_time_s' in metrics:
            print(f"  Compilation Time:  {metrics['compilation_time_s']:.2f}s")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/seedvox_latest.pt")
    parser.add_argument("--ph_checkpoint", type=str, default=None, help="Optional separate phonetic pre-train weights")
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
    parser.add_argument("--use_linguistic_fusion", action="store_true", help="Force use of FusionPlannerModel")
    
    # Phoneme sampling
    parser.add_argument("--phoneme_temp", type=float, default=1.0)
    parser.add_argument("--phoneme_top_p", type=float, default=0.9)

    # Visualization and playback
    parser.add_argument("--play", action="store_true", help="Play audio after generation")
    parser.add_argument("--view_waveform", action="store_true", help="View static waveform")
    parser.add_argument("--log_metrics", action="store_true", help="Log performance metrics (latency, RTF)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for faster inference")
    
    # Variant generation options
    parser.add_argument("--variant_n", type=int, default=1, help="Number of variants to generate")
    parser.add_argument("--variant_axis", choices=['pros', 'speaker'], default='pros',
                        help="Axis to vary: 'pros' (different prosody, same speaker) or 'speaker' (different speakers, same prosody)")

    # LoRA adapter options
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="LoRA adapter weights to apply on top of base checkpoint")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank (auto-detected from checkpoint if not set)")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (auto-detected from checkpoint if not set)")
    parser.add_argument("--save_merged_checkpoint", type=str, default=None,
                        help="Save a full model checkpoint with LoRA weights permanently merged (requires --lora_checkpoint)")

    args = parser.parse_args()
    run_inference(args)

