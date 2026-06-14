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
from seedvox.utils.tokenizer import CharTokenizer, BPECharCollator
from seedvox.utils.text import normalize_text, PhoneticAligner

def load_audio(path, device):
    wav, sr = torchaudio.load(path)
    if sr != 24000: wav = torchaudio.transforms.Resample(sr, 24000)(wav)
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    return wav.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="The prosody planner determines the rhythm and melody of the voice.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ref_speaker", type=str, help="Wav file for voice identity")
    parser.add_argument("--ref_prosody", type=str, help="Wav file for oracle prosody transfer")
    parser.add_argument("--output_dir", type=str, default="ablation_results")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--temp", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.config, "r") as f: cfg = json.load(f)

    tokenizer = CharTokenizer()
    model = SeedVoxModel(cfg, tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
    model.eval()

    mimi = get_mimi_model(device=device, checkpoint_path=cfg.get('mimi_checkpoint', 'pretrained_models/best_mimi.pt')).eval()

    norm_text = normalize_text(args.text)
    print(f"\n--- Text Analysis ---")
    print(f"Original:  {args.text}")
    print(f"Normalized: {norm_text}")

    # Phonetics Logging
    aligner = PhoneticAligner()
    ph_ids = aligner.align_text_to_phonemes(args.text)
    ph_str = aligner.tokenizer.decode(ph_ids)
    print(f"Phonemes:   {ph_str}")

    # BPE Processing & Logging
    bpe_ids, bpe_lens, char_to_bpe = None, None, None
    if model.bpe_encoder:
        collator = BPECharCollator(model.bpe_encoder)
        bpe_ids, bpe_lens, char_to_bpe = collator.process_batch_texts([args.text], torch.tensor([len(norm_text)]), device=device)
        bpe_tokens = model.bpe_encoder.bpe_tokenizer.convert_ids_to_tokens(bpe_ids[0].tolist())
        print(f"BPE Tokens: {' | '.join(bpe_tokens[:bpe_lens[0]])}")
    print(f"---------------------\n")

    t_test = torch.tensor([tokenizer.encode(norm_text, normalize=False)], device=device)
    t_len = torch.tensor([len(t_test[0])], device=device)

    # Handle Speaker Reference
    ref_audio_toks, ref_audio_len = None, None
    if args.ref_speaker:
        wav_spk = load_audio(args.ref_speaker, device)
        ref_audio_toks = mimi.encode(wav_spk.unsqueeze(0))
        ref_audio_len = torch.tensor([ref_audio_toks.shape[2]], device=device)

    # Handle Prosody Reference (for Oracle/Transfer)
    mimi_latents_prosody = None
    if args.ref_prosody:
        wav_prs = load_audio(args.ref_prosody, device)
        mimi_latents_prosody = mimi.encoder(wav_prs.unsqueeze(0))

    modes = ["full", "null"]
    if mimi_latents_prosody is not None:
        modes.append("oracle")

    for mode in modes:
        print(f"Generating mode: {mode}...")
        gen, _ = model.sample(
            t_test, t_len, 
            ref_audio=ref_audio_toks, ref_lens=ref_audio_len, 
            mimi_latents=mimi_latents_prosody,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
            ablation_mode=mode, cfg_scale=args.cfg_scale, temp=args.temp
        )
        audio = mimi.decode(gen.clamp(0, 2047))[0]
        out_path = os.path.join(args.output_dir, f"seedvox_{mode}.wav")
        torchaudio.save(out_path, audio.cpu(), 24000)
        print(f"  Saved to {out_path}")

if __name__ == "__main__":
    main()
