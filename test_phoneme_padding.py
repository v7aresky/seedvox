from seedvox.utils.text import PhoneticAligner
from seedvox.utils.tokenizer import PhonemeTokenizer

aligner = PhoneticAligner()
ph_ids = aligner.align_text_to_phonemes("testing half")
ph_str = aligner.tokenizer.decode(ph_ids)
print(f"Phonemes: {ph_str}")
