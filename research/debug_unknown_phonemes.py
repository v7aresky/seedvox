from seedvox.utils.tokenizer import PhonemeTokenizer
from seedvox.utils.g2p_factory import get_phoneme_generator
from seedvox.utils.text import PhoneticAligner

tokenizer = PhonemeTokenizer()
aligner = PhoneticAligner(generator=get_phoneme_generator('deep-phonemizer'))

text = "Testing half precision inference."
ph_ids = aligner.align_text_to_phonemes(text)

print(f"Phonemes: {aligner.tokenizer.decode(ph_ids)}")
print(f"IDs: {ph_ids}")

# Check for unknowns
unknowns = []
for p_str in aligner.g2p("Testing half precision inference."):
    if p_str.strip() and p_str not in tokenizer.ph_to_id:
        unknowns.append(p_str)
print(f"Unknown phonemes: {unknowns}")
