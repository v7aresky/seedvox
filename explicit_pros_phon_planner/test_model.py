import torch
import json
from pathlib import Path
import sys

# Add project root and src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from explicit_pros_phon_planner.model import ExplicitPlannerModel

from seedvox.utils.tokenizer import CharTokenizer

def test_model_forward():
    with open("configs/default.json", "r") as f:
        cfg = json.load(f)
    
    tokenizer = CharTokenizer()
    model = ExplicitPlannerModel(cfg, tokenizer.vocab_size, phoneme_vocab_size=128)
    
    B = 2
    T = 10
    text = torch.randint(0, tokenizer.vocab_size, (B, T))
    text_lens = torch.tensor([T, T-2])
    
    audio_tokens = torch.randint(0, 1024, (B, 16, 100))
    audio_lens = torch.tensor([100, 80])
    
    # Explicit phonemes (not aligned)
    phoneme_ids = torch.randint(0, 128, (B, 15))
    # Aligned phonemes (matched to text_feat length T+2)
    aligned_phoneme_ids = torch.randint(0, 128, (B, T + 2))
    
    # Optional BPE
    bpe_ids = torch.randint(0, 50257, (B, 5))
    bpe_lens = torch.tensor([5, 4])
    char_to_bpe = torch.randint(0, 5, (B, T))
    
    # Mimi latents for JEPA (B, 512, Ta_mimi)
    mimi_latents = torch.randn(B, 512, 50)
    audio_lens_mimi = torch.tensor([50, 40])
    
    logits, targets, ph_planner_logits, jepa_loss, contrastive_loss = model(
        text, audio_tokens, text_lens, audio_lens,
        phoneme_ids=phoneme_ids,
        mimi_latents=mimi_latents,
        bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe
    )
    
    print(f"Logits shape: {logits.shape}")
    print(f"Ph Planner Logits shape: {ph_planner_logits.shape}")
    print(f"JEPA Loss: {jepa_loss}")
    
    assert logits.shape[0] == 16
    assert ph_planner_logits.shape[0] == B
    assert ph_planner_logits.shape[1] == 14 # ph_ids[:, :-1]
    assert jepa_loss is not None
    
    print("Model forward test passed!")

def test_fusion_model_forward():
    from explicit_pros_phon_planner.model_fusion import FusionPlannerModel
    with open("configs/default.json", "r") as f:
        cfg = json.load(f)
    
    tokenizer = CharTokenizer()
    model = FusionPlannerModel(cfg, tokenizer.vocab_size, phoneme_vocab_size=128)
    
    B = 2
    T = 10
    text = torch.randint(0, tokenizer.vocab_size, (B, T))
    text_lens = torch.tensor([T, T-2])
    
    audio_tokens = torch.randint(0, 1024, (B, 16, 100))
    audio_lens = torch.tensor([100, 80])
    
    phoneme_ids = torch.randint(0, 128, (B, 15))
    
    bpe_ids = torch.randint(0, 50257, (B, 5))
    bpe_lens = torch.tensor([5, 4])
    char_to_bpe = torch.randint(0, 5, (B, T))
    
    mimi_latents = torch.randn(B, 512, 50)
    
    logits, targets, ph_planner_logits, jepa_loss, contrastive_loss = model(
        text, audio_tokens, text_lens, audio_lens,
        phoneme_ids=phoneme_ids,
        mimi_latents=mimi_latents,
        bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe
    )
    
    print(f"[Fusion] Logits shape: {logits.shape}")
    print(f"[Fusion] Ph Planner Logits shape: {ph_planner_logits.shape}")
    print(f"[Fusion] JEPA Loss: {jepa_loss}")
    
    assert logits.shape[0] == 16
    assert ph_planner_logits.shape[0] == B
    assert ph_planner_logits.shape[1] == 14
    assert jepa_loss is not None
    
    print("Fusion model forward test passed!")

if __name__ == "__main__":
    test_model_forward()
    test_fusion_model_forward()
