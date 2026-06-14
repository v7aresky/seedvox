from seedvox.utils.g2p_factory import EspeakGenerator
from seedvox.utils.tokenizer import PhonemeTokenizer

def test_homographs():
    try:
        g2p = EspeakGenerator()
    except Exception as e:
        print(f"Could not initialize Espeak: {e}")
        return

    # Test cases for homographs
    test_cases = [
        "Please record the record.",      # Verb (rɪˈkɔːrd) vs Noun (ˈrɛkərd)
        "The wind will wind the clock.",   # Noun (wɪnd) vs Verb (waɪnd)
        "I live near the live music.",     # Verb (lɪv) vs Adjective (laɪv)
        "The object will object to it."    # Noun (ˈɒbdʒɪkt) vs Verb (əbˈdʒɛkt)
    ]
    
    tokenizer = PhonemeTokenizer()

    for text in test_cases:
        phonemes = g2p(text)
        print(f"\nText: {text}")
        print(f"Phonemes: {' '.join(phonemes)}")

if __name__ == "__main__":
    test_homographs()
