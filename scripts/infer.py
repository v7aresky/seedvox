import os
import sys
import json
import torch
import torchaudio
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvox.model import SeedVoxModel
from seedvox.modules.mimi import get_mimi_model
from seedvox.utils.tokenizer import CharTokenizer, PhonemeTokenizer
from seedvox.utils.text import normalize_text, PhoneticAligner
from seedvox.utils.text import collapse_duplicates

def load_audio(path, device):
    wav, sr = torchaudio.load(path)
    if sr != 24000: wav = torchaudio.transforms.Resample(sr, 24000)(wav)
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    return wav.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ref_wav", type=str, default=None)
    parser.add_argument("--ref_prosody", type=str, default=None)
    parser.add_argument("--random_speaker", action="store_true")
    parser.add_argument("--random_prosody", action="store_true")
    parser.add_argument("--ablation_mode", type=str, default="full", choices=["full", "null", "oracle"])
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--dtype", choices=['fp32', 'fp16', 'bf16'], default='fp32', help="Load model in specific precision")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r") as f: cfg = json.load(f)

    tokenizer = CharTokenizer()
    model = SeedVoxModel(cfg, tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
    
    if args.dtype == 'fp16':
        model = model.half()
    elif args.dtype == 'bf16':
        model = model.to(torch.bfloat16)

    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Load EMA weights if available, otherwise fallback to standard model weights
    if 'ema_model' in ckpt:
        print("Loading EMA model weights...")
        model.load_state_dict(ckpt['ema_model'], strict=False)
    elif 'model' in ckpt:
        print("Loading standard model weights...")
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    mimi = get_mimi_model(device=device, checkpoint_path=cfg.get('mimi_checkpoint', 'pretrained_models/best_mimi.pt')).eval()

    norm_text = normalize_text(args.text)
    print(f"DEBUG: Input Text: {args.text}")
    print(f"DEBUG: Char Tokens: {norm_text}")

    # Phonetics
    aligner = PhoneticAligner()
    ph_ids = aligner.align_text_to_phonemes(args.text)
    ph_str = aligner.tokenizer.decode(ph_ids)
    print(f"DEBUG: Predicted phonemes: {ph_str}")

    # BPE Processing
    bpe_ids, bpe_lens, char_to_bpe = None, None, None
    if model.use_bpe_encoder:
        from seedvox.bpe_char_encoder import BPECharCollator
        collator = BPECharCollator(model.bpe_encoder)
        bpe_ids, bpe_lens, char_to_bpe = collator.process_batch_texts([args.text], torch.tensor([len(norm_text)]), device=device)
        print("DEBUG: BPE tokens processed.")

    t_test = torch.tensor([tokenizer.encode(norm_text, normalize=False)], device=device)
    t_len = torch.tensor([len(t_test[0])], device=device)

    ref_audio, ref_len = None, None
    if args.ref_wav:
        wav_spk = load_audio(args.ref_wav, device)
        ref_audio = mimi.encode(wav_spk.unsqueeze(0))
        ref_len = torch.tensor([ref_audio.shape[2]], device=device)

    print("Generating...")
    with torch.no_grad():
        gen, _ = model.sample(
            t_test, t_len, 
            ref_audio=ref_audio, ref_lens=ref_len,
            cfg_scale=args.cfg_scale, temp=args.temp, 
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
            phoneme_ids=torch.tensor([ph_ids], device=device)
        )
    
    audio = mimi.decode(gen)[0].cpu()
    torchaudio.save(args.output, audio, 24000)
    print(f"Saved generated audio to {args.output}")

if __name__ == "__main__":
    main()
