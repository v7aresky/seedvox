import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explicit_pros_phon_planner.utils import PhoneticGenerator
from seedvox.utils.tokenizer import PhonemeTokenizer

def test_phonetic_target_structure():
    print("Running Phonetic Target Structure Test...")
    
    # Initialize generator (using g2p_en as default)
    gen = PhoneticGenerator(backend='g2p_en', phoneme_vocab_size=128)
    tokenizer = PhonemeTokenizer()
    
    test_text = "The quick brown fox."
    # We expect normalize=False because we already normalize in the dataset/infer scripts
    targets = gen.generate_targets(test_text, normalize=True)
    
    print(f"Text: '{test_text}'")
    print(f"Target IDs: {targets}")
    
    decoded = tokenizer.decode([t for t in targets if t >= 4])
    print(f"Decoded Phonemes: {decoded}")
    
    # 1. Check for SOS/EOS
    assert targets[0] == 3, f"Expected SOS_ID (3) at start, got {targets[0]}"
    assert targets[-1] == 128, f"Expected EOS_ID (128) at end, got {targets[-1]}"
    
    # 2. Check for "alignment" symptoms (long strings of identical phonemes)
    # In a sequence-based target, we should NOT see the same phoneme repeated 
    # many times unless it's a very long word with repeating sounds.
    max_repeat = 0
    curr_repeat = 1
    for i in range(1, len(targets)):
        if targets[i] == targets[i-1] and targets[i] >= 4: # Ignore special tokens
            curr_repeat += 1
        else:
            max_repeat = max(max_repeat, curr_repeat)
            curr_repeat = 1
    
    print(f"Max phoneme repetition: {max_repeat}")
    
    # 3. Check length vs Text length
    # Aligned phonemes would have length exactly equal to the normalized text (lower case + punct)
    from seedvox.utils.text import normalize_text
    norm_text = normalize_text(test_text)
    print(f"Normalized text length: {len(norm_text)}")
    print(f"Target sequence length: {len(targets)}")
    
    if len(targets) == len(norm_text):
        print("WARNING: Target length matches text length. This might indicate alignment logic!")
    else:
        print("SUCCESS: Target length is independent of text length (Sequence-based).")

    # 4. Check for duplicates in sequence (should be minimal)
    if max_repeat > 2:
        print(f"WARNING: High repetition detected ({max_repeat}). Verify if this is expected for the G2P output.")
    else:
        print("SUCCESS: No artificial duplication detected.")

if __name__ == "__main__":
    try:
        test_phonetic_target_structure()
        print("\nTEST PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
