
class PhonemeTokenizer:
    """
    Tokenizer for ARPAbet-style phonemes.
    Includes standard ARPAbet symbols plus stress markers and special tokens.
    """
    def __init__(self):
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.mask_token_id = 2
        self.eps_token_id = 3
        
        # Standard ARPAbet phonemes (40)
        self.phonemes = [
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 
            'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 
            'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 
            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
        ]
        
        # Add stress markers (0, 1, 2) to vowels
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
        stressed_vowels = []
        for v in vowels:
            stressed_vowels.extend([v + '0', v + '1', v + '2'])
            
        # Punctuation and special markers
        self.special = [' ', '.', ',', '!', '?', '-', "'", '"', '(', ')', '<EPS>']
        
        self.vocab = self.phonemes + stressed_vowels + self.special
        self.id_to_ph = {i + 4: ph for i, ph in enumerate(self.vocab)}
        self.id_to_ph[self.eps_token_id] = "<EPS>"
        self.ph_to_id = {ph: i + 4 for i, ph in enumerate(self.vocab)}
        self.ph_to_id['<EPS>'] = self.eps_token_id # ensure <EPS> is 3
        self.vocab_size = len(self.vocab) + 4

    def encode(self, phoneme_string):
        """Encodes a string of space-separated phonemes."""
        tokens = []
        for ph in phoneme_string.split():
            tokens.append(self.ph_to_id.get(ph, self.unk_token_id))
        return tokens

    def decode(self, ids):
        """Decodes a list of phoneme IDs back to a space-separated string."""
        res = []
        for i in ids:
            if i >= 3:
                res.append(self.id_to_ph.get(i, "?"))
            elif i == 1:
                res.append("?")
            elif i == 2:
                res.append("[MASK]")
        return " ".join(res)
