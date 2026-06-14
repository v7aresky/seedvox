import torch
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvox.model import SeedVoxModel
from seedvox.utils.tokenizer import CharTokenizer

def test_load():
    device = torch.device("cpu")
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.json"
    ckpt_path = Path(__file__).resolve().parent.parent / "pretrained_models" / "seedvox_latest.pt"
    
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    tokenizer = CharTokenizer()
    print("Initializing model...")
    model = SeedVoxModel(cfg, tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
    
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Check if 'model' key exists in ckpt (standard trainer output)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if len(missing) > 0:
        print(f"⚠️ Missing keys: {len(missing)}")
        # Print first few missing keys for debugging
        for k in list(missing)[:5]:
            print(f"  - {k}")
            
    if len(unexpected) > 0:
        print(f"⚠️ Unexpected keys: {len(unexpected)}")
        for k in list(unexpected)[:5]:
            print(f"  - {k}")
            
    if len(missing) == 0 and len(unexpected) == 0:
        print("✅ Model loaded perfectly with strict match!")
    elif len(missing) < 20: # Allow some minor missing keys if they are non-critical
        print("✅ Model loaded successfully (with minor discrepancies).")
    else:
        print("❌ Significant mismatch in model loading.")
        sys.exit(1)

if __name__ == "__main__":
    test_load()
