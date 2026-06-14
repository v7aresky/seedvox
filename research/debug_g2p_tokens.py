from seedvox.utils.g2p_factory import get_phoneme_generator
from explicit_pros_phon_planner.utils import PhoneticGenerator

def test_backends(text):
    print(f"Testing text: '{text}'")
    
    espeak_gen = get_phoneme_generator('espeak')
    g2p_en_gen = get_phoneme_generator('g2p_en')
    
    espeak_ph = espeak_gen(text)
    g2p_en_ph = g2p_en_gen(text)
    
    print(f"Espeak phonemes ({len(espeak_ph)}): {espeak_ph}")
    print(f"G2PEn phonemes ({len(g2p_en_ph)}): {g2p_en_ph}")
    
    # Check how PhoneticGenerator processes these
    gen_wrapper = PhoneticGenerator(backend='espeak') # Uses default for vocab
    
    espeak_targets = gen_wrapper.generate_targets_from_phonemes(espeak_ph)
    # Re-init wrapper for g2p_en as it wasn't the original, but the class handles it
    g2p_en_targets = gen_wrapper.generate_targets_from_phonemes(g2p_en_ph)
    
    print(f"Espeak target length: {len(espeak_targets)}")
    print(f"G2PEn target length: {len(g2p_en_targets)}")

if __name__ == "__main__":
    test_backends("Hello world, this is a test.")
