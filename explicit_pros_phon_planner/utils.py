import torch
import re
from seedvox.utils.tokenizer import PhonemeTokenizer
from seedvox.utils.text import normalize_text
from seedvox.utils.g2p_factory import get_phoneme_generator

class PhoneticGenerator:
    """
    Utility for generating phoneme targets for training the PhoneticPlanner.
    Uses an external G2P (e.g., deep-phonemizer or g2p_en) to create 
    ground-truth phoneme sequences from text.
    """
    def __init__(self, backend='espeak', phoneme_vocab_size=None):
        self.g2p = get_phoneme_generator(backend)
        self.tokenizer = PhonemeTokenizer()
        
        # SOS is usually <EPS> (3)
        self.SOS_ID = 3
        # EOS is the token after the last vocab token
        self.EOS_ID = phoneme_vocab_size if phoneme_vocab_size is not None else self.tokenizer.vocab_size
        
    def generate_targets(self, text, normalize=True):
        """
        Generates a sequence of phoneme IDs with SOS and EOS tokens.
        Includes punctuation as requested by the professor.
        """
        norm_text = normalize_text(text) if normalize else text
        
        # Get phonemes with punctuation
        # g2p_en typically handles punctuation by returning it as is.
        phonemes = self.g2p(norm_text)
        return self.generate_targets_from_phonemes(phonemes)

    def generate_targets_batch(self, texts, normalize=True):
        """
        Generates a batch of phoneme ID sequences with SOS and EOS tokens.
        Much more efficient for backends like espeak that support batching.
        """
        norm_texts = [normalize_text(t) if normalize else t for t in texts]
        
        # Batch G2P call
        phonemes_list = self.g2p(norm_texts)
        
        return [self.generate_targets_from_phonemes(ph) for ph in phonemes_list]

    def generate_targets_from_phonemes(self, phonemes):
        """Converts pre-generated phonemes to IDs with SOS and EOS."""
        # Convert to IDs
        ids = [self.SOS_ID]
        for ph in phonemes:
            # The PhonemeTokenizer.ph_to_id handles some special tokens like ' ', '.', etc.
            if ph.strip() == '':
                # Map space to ' ' if in vocab, else skip or map to a break token
                ph_id = self.tokenizer.ph_to_id.get(' ', self.SOS_ID) 
            else:
                ph_id = self.tokenizer.ph_to_id.get(ph, self.tokenizer.unk_token_id)
            ids.append(ph_id)
            
        ids.append(self.EOS_ID)
        return ids

def collate_phonemes(phoneme_id_list, pad_id=0):
    """Pads a batch of phoneme ID sequences."""
    max_len = max(len(ids) for ids in phoneme_id_list)
    B = len(phoneme_id_list)
    padded = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(phoneme_id_list):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded
