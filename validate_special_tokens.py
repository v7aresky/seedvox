import torch
import json
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from explicit_pros_phon_planner.trainer import ExplicitTrainer
from seedvox.utils.tokenizer import PhonemeTokenizer

def validate_tokens():
    config_path = "configs/light.json"
    
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    cfg['training']['batch_size'] = 4
    cfg['training']['num_workers'] = 0
    
    trainer = ExplicitTrainer(
        config=cfg, 
        device=torch.device("cpu"), 
        g2p_backend="espeak"
    )
    
    tokenizer = PhonemeTokenizer()
    
    print(f"SOS_ID: {trainer.ph_generator.SOS_ID}")
    print(f"EOS_ID: {trainer.ph_generator.EOS_ID}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    print("\nChecking a few samples...")
    for i, batch in enumerate(trainer.loader):
        padded_text, padded_audio, t_lens, a_lens, raw_texts, ph_targets = batch
        
        for j in range(len(raw_texts)):
            text = raw_texts[j]
            targets = ph_targets[j]
            # Remove padding (0)
            valid_targets = targets[targets != 0]
            
            print(f"\nText: {text}")
            print(f"Target IDs: {valid_targets.tolist()}")
            
            # Check SOS
            if valid_targets[0] == trainer.ph_generator.SOS_ID:
                print("✅ SOS present at start")
            else:
                print(f"❌ SOS MISSING! Found {valid_targets[0]}")
                
            # Check EOS
            if valid_targets[-1] == trainer.ph_generator.EOS_ID:
                print("✅ EOS present at end")
            else:
                print(f"❌ EOS MISSING! Found {valid_targets[-1]}")
                
            # Check for UNK (1)
            unks = (valid_targets == 1).sum().item()
            if unks > 0:
                print(f"⚠️ Found {unks} UNK tokens!")
            
            # Decode for sanity
            # Note: PhonemeTokenizer.decode doesn't handle EOS index (128) by default
            # It only handles >= 3 by looking up id_to_ph. 
            # If EOS is 128 and not in id_to_ph, it will show "?"
            decoded = tokenizer.decode(valid_targets.tolist())
            print(f"Decoded: {decoded}")
            
        break

if __name__ == "__main__":
    validate_tokens()
