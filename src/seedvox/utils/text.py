import re
import unicodedata
import torch
import math
from seedvox.utils.tokenizer import PhonemeTokenizer
from seedvox.utils.g2p_factory import get_phoneme_generator

# ==============================================================================
# NUMBER TO WORDS
# ==============================================================================
_ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
         'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
         'seventeen', 'eighteen', 'nineteen']
_tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

def _int_to_words(n):
    if n < 0: return 'minus ' + _int_to_words(-n)
    if n == 0: return 'zero'
    parts = []
    if n >= 1_000_000_000:
        parts.append(_int_to_words(n // 1_000_000_000) + ' billion')
        n %= 1_000_000_000
    if n >= 1_000_000:
        parts.append(_int_to_words(n // 1_000_000) + ' million')
        n %= 1_000_000
    if n >= 1000:
        parts.append(_int_to_words(n // 1000) + ' thousand')
        n %= 1000
    if n >= 100:
        parts.append(_ones[n // 100] + ' hundred')
        n %= 100
    if n >= 20:
        parts.append(_tens[n // 10])
        if n % 10: parts.append(_ones[n % 10])
    elif n > 0: parts.append(_ones[n])
    return ' '.join(parts)

def _number_to_words(match):
    text = match.group(0).replace(',', '')
    if '.' in text:
        parts = text.split('.')
        return _int_to_words(int(parts[0])) + ' point ' + ' '.join(_ones[int(d)] if d != '0' else 'zero' for d in parts[1])
    try: return _int_to_words(int(text))
    except ValueError: return text

def _expand_currency(text):
    def _dollar_match(m):
        dollars, cents = int(m.group(1)), int(m.group(2)) if m.group(2) else 0
        parts = []
        if dollars: parts.append(_int_to_words(dollars) + (' dollar' if dollars == 1 else ' dollars'))
        if cents: parts.append(_int_to_words(cents) + (' cent' if cents == 1 else ' cents'))
        return ' '.join(parts) if parts else 'zero dollars'
    text = re.sub(r'\$(\d+)\.(\d{2})', _dollar_match, text)
    text = re.sub(r'\$(\d+)', lambda m: _int_to_words(int(m.group(1))) + (' dollar' if m.group(1) == '1' else ' dollars'), text)
    text = re.sub(r'£(\d+)', lambda m: _int_to_words(int(m.group(1))) + (' pound' if m.group(1) == '1' else ' pounds'), text)
    text = re.sub(r'€(\d+)', lambda m: _int_to_words(int(m.group(1))) + (' euro' if m.group(1) == '1' else ' euros'), text)
    return text

_ordinal_map = {'1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth', '5th': 'fifth', '6th': 'sixth', '7th': 'seventh', '8th': 'eighth', '9th': 'ninth', '10th': 'tenth'}
def _expand_ordinals(text):
    return re.sub(r'\b\d{1,2}(?:st|nd|rd|th)\b', lambda m: _ordinal_map.get(m.group(0).lower(), m.group(0)), text, flags=re.IGNORECASE)

_abbreviations = [(re.compile(r'\bMr\.', re.IGNORECASE), 'Mister'), (re.compile(r'\bMrs\.', re.IGNORECASE), 'Missis'), (re.compile(r'\bMs\.', re.IGNORECASE), 'Miss'), (re.compile(r'\bDr\.', re.IGNORECASE), 'Doctor'), (re.compile(r'\betc\.', re.IGNORECASE), 'et cetera')]
def _expand_abbreviations(text):
    for regex, replacement in _abbreviations: text = regex.sub(replacement, text)
    return text

_letter_to_word = {
    'a': 'ay', 'b': 'bee', 'c': 'see', 'd': 'dee', 'e': 'ee', 'f': 'ef', 'g': 'jee',
    'h': 'aitch', 'i': 'eye', 'j': 'jay', 'k': 'kay', 'l': 'el', 'm': 'em', 'n': 'en',
    'o': 'oh', 'p': 'pee', 'q': 'cue', 'r': 'are', 's': 'ess', 't': 'tee', 'u': 'you',
    'v': 'vee', 'w': 'double u', 'x': 'ex', 'y': 'wye', 'z': 'zee'
}

def _expand_isolated_letters(text):
    def _replace_letter(m):
        l = m.group(1).lower()
        return _letter_to_word.get(l, l)
    return re.sub(r'\b([A-Za-z])\b', _replace_letter, text)

def normalize_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = _expand_abbreviations(text)
    text = _expand_currency(text)
    text = _expand_ordinals(text)
    text = _expand_isolated_letters(text)
    text = re.sub(r'\d+', _number_to_words, text)
    return re.sub(r'\s+', ' ', text).strip().lower()

class PhoneticAligner:
    def __init__(self, generator=None):
        self.g2p = generator if generator is not None else get_phoneme_generator('g2p_en')
        self.tokenizer = PhonemeTokenizer()

    def normalize(self, text):
        return normalize_text(text)

    def align_phonemes_to_text(self, norm_text, full_phonemes):
        # Clean phonemes for mapping
        clean_phonemes = [p for p in full_phonemes if p.strip()]
        
        # 2. Tokenize text into words/punct/spaces
        tokens = []
        for match in re.finditer(r"[\w']+|[.,!?;-]| ", norm_text):
            tokens.append((match.group(0), match.start(), match.end()))
            
        # 3. Map phonemes directly to characters
        all_ph_ids = []
        ph_idx = 0
        
        for token_str, start, end in tokens:
            t_len = end - start
            if token_str == ' ':
                for _ in range(t_len):
                    all_ph_ids.append(self.tokenizer.eps_token_id)
            elif token_str in ".,!?;-()\"'":
                for _ in range(t_len):
                    all_ph_ids.append(self.tokenizer.ph_to_id.get(token_str, self.tokenizer.eps_token_id))
            else:
                # Word: map characters to the pre-computed phoneme list
                for i in range(t_len):
                    if ph_idx < len(clean_phonemes):
                        p = clean_phonemes[ph_idx]
                        all_ph_ids.append(self.tokenizer.ph_to_id.get(p, self.tokenizer.eps_token_id))
                        ph_idx += 1
                    else:
                        all_ph_ids.append(self.tokenizer.eps_token_id)
            
        # Pad end
        for _ in range(len(norm_text) - len(all_ph_ids)):
            all_ph_ids.append(self.tokenizer.eps_token_id)
            
        return all_ph_ids[:len(norm_text)]

    def align_text_to_phonemes(self, text, normalize=True):
        norm_text = normalize_text(text) if normalize else text
        
        # 1. Global G2P call for context
        # Ensure we call with a string if it's a single text, 
        # but our generators now handle both.
        full_phonemes = self.g2p(norm_text)
        
        if not isinstance(full_phonemes, list) or (len(full_phonemes) > 0 and isinstance(full_phonemes[0], list)):
             # If it returned a batch, take the first one
             full_phonemes = full_phonemes[0]

        return self.align_phonemes_to_text(norm_text, full_phonemes)

def collapse_duplicates(ids):
    if not ids: return []
    collapsed = [ids[0]]
    for i in range(1, len(ids)):
        if ids[i] != ids[i-1]:
            collapsed.append(ids[i])
    return collapsed
