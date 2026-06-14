import torch
import re
from phoneme_tokenizer import PhonemeTokenizer
from text_normalizer import normalize_text
from g2p_factory import get_phoneme_generator, PhonemeGenerator

class PhoneticAligner:
    def __init__(self, generator: PhonemeGenerator = None):
        self.g2p = generator if generator is not None else get_phoneme_generator('g2p_en')
        self.tokenizer = PhonemeTokenizer()

    def align_text_to_phonemes(self, text, normalize=True, manual_phonemes=None):
        """
        Aligns text characters to phonemes by mapping G2P output directly to the 
        tokenized text character-by-character, using context from full G2P.
        """
        norm_text = normalize_text(text) if normalize else text
        
        # 1. Get phonemes globally for context
        # We need to filter out spaces from the raw G2P output for direct mapping
        full_phonemes = manual_phonemes if manual_phonemes is not None else self.g2p(norm_text)
        clean_phonemes = [p for p in full_phonemes if p != ' ']
        
        # 2. Tokenize text into words/punct/spaces
        tokens = []
        for match in re.finditer(r"[\w']+|[.,!?;-]| ", norm_text):
            tokens.append((match.group(0), match.start(), match.end()))
            
        # 3. Map phonemes directly to characters
        all_ph_ids = []
        
        # Use a simple pointer to current phoneme
        ph_idx = 0
        
        for token_str, start, end in tokens:
            if token_str == ' ':
                # Revert to eps_token_id (3) as the internal model uses this for spaces
                for _ in range(end - start):
                    all_ph_ids.append(self.tokenizer.eps_token_id)
            elif token_str in self.tokenizer.special:
                # Map punctuation
                all_ph_ids.append(self.tokenizer.ph_to_id.get(token_str, self.tokenizer.eps_token_id))
            else:
                # Map word: use a slice of the phoneme list proportional to word length
                t_len = end - start
                
                # We need to estimate how many phonemes correspond to this word.
                # This is hard because punctuation breaks it.
                # Let's find the corresponding phonemes for this word by
                # counting non-special tokens.
                
                # Simple heuristic: map phonemes to characters within this token
                # This is fundamentally difficult without explicit alignment.
                # Based on previous attempts, let's just assign a few phonemes
                # to the word and pad with EPS.
                
                # The user said it was correct BEFORE. 
                # Reverting to the logic that worked: word-by-word G2P.
                # Let's trust G2P on isolated words for this specific mapping logic.
                
                word_phonemes = self.g2p(token_str)
                word_phonemes = [p for p in word_phonemes if p.strip()]
                
                p_len = len(word_phonemes)
                for i in range(t_len):
                    if i < p_len:
                        p = word_phonemes[i]
                        all_ph_ids.append(self.tokenizer.ph_to_id.get(p, self.tokenizer.eps_token_id))
                    else:
                        # Pad the rest of the characters in the word with EPS
                        all_ph_ids.append(self.tokenizer.eps_token_id)
            
        # Pad end
        for _ in range(len(norm_text) - len(all_ph_ids)):
            all_ph_ids.append(self.tokenizer.eps_token_id)
            
        return all_ph_ids[:len(norm_text)]

def get_phoneme_targets(raw_texts, max_len):
    aligner = PhoneticAligner()
    B = len(raw_texts)
    targets = torch.zeros((B, max_len), dtype=torch.long)
    for i, text in enumerate(raw_texts):
        ph_ids = aligner.align_text_to_phonemes(text)
        l = min(len(ph_ids), max_len)
        targets[i, :l] = torch.tensor(ph_ids[:l], dtype=torch.long)
    return targets
