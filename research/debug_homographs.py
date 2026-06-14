from seedvox.utils.g2p_factory import DeepPhonemizerGenerator
from seedvox.utils.tokenizer import PhonemeTokenizer

def test_homographs():
    try:
        g2p = DeepPhonemizerGenerator()
    except Exception as e:
        print(f"Could not initialize DeepPhonemizer: {e}")
        return

    # Test cases for homographs
    test_cases = [
        "Please record the record.",      # Verb vs Noun
        "The wind will wind the clock.",   # Noun vs Verb
        "I live near the live music.",     # Verb vs Adjective
        "The object will object to it."    # Noun vs Verb
    ]
    
    tokenizer = PhonemeTokenizer()

    for text in test_cases:
        phonemes = g2p(text)
        print(f"\nText: {text}")
        print(f"Phonemes: {' '.join(phonemes)}")

if __name__ == "__main__":
    test_homographs()
