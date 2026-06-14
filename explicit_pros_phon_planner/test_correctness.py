import torch
from explicit_pros_phon_planner.utils import PhoneticGenerator
from seedvox.utils.tokenizer import PhonemeTokenizer

def test_g2p_correctness():
    generator = PhoneticGenerator(backend='g2p_en')
    tokenizer = PhonemeTokenizer()
    
    test_text = "Hello, world! This is a test."
    targets = generator.generate_targets(test_text)
    
    print(f"Text: {test_text}")
    print(f"Phoneme IDs: {targets}")
    
    # Decode to verify (ignoring EOS for now as it's new)
    decoded = []
    for tid in targets:
        if tid == generator.SOS_ID:
            decoded.append("<SOS>")
        elif tid == generator.EOS_ID:
            decoded.append("<EOS>")
        else:
            decoded.append(tokenizer.id_to_ph.get(tid, "?"))
            
    print(f"Decoded: {' '.join(decoded)}")
    
    # Verify punctuation is preserved
    has_punct = False
    for p in [',', '!', '.']:
        if p in decoded:
            has_punct = True
            print(f"Found punctuation: {p}")
            
    if not has_punct:
        print("WARNING: Punctuation not found in decoded phonemes!")

if __name__ == "__main__":
    test_g2p_correctness()
