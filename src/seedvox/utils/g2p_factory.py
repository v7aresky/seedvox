import abc
from g2p_en import G2p
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
import re
import math
from seedvox.utils.tokenizer import PhonemeTokenizer

class PhonemeGenerator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, text) -> list:
        pass

class G2PEnGenerator(PhonemeGenerator):
    def __init__(self):
        self.g2p = G2p()

    def __call__(self, text) -> list:
        if isinstance(text, str):
            return self.g2p(text)
        # g2p_en's G2p object doesn't have a batch method, but we can loop
        # Actually it's pretty fast, but let's be consistent
        return [self.g2p(t) for t in text]

class EspeakGenerator(PhonemeGenerator):
    IPA_TO_ARPABET = {
        'ɑ': 'AA', 'æ': 'AE', 'ʌ': 'AH', 'ɔ': 'AO', 'aʊ': 'AW', 'aɪ': 'AY',
        'b': 'B', 'tʃ': 'CH', 'd': 'D', 'ð': 'DH', 'ɛ': 'EH', 'ɚ': 'ER', 'ɝ': 'ER',
        'eɪ': 'EY', 'f': 'F', 'ɡ': 'G', 'h': 'HH', 'ɪ': 'IH', 'i': 'IY', 'iː': 'IY',
        'dʒ': 'JH', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'ŋ': 'NG', 'oʊ': 'OW',
        'ɔɪ': 'OY', 'p': 'P', 'ɹ': 'R', 's': 'S', 'ʃ': 'SH', 't': 'T', 'θ': 'TH',
        'ʊ': 'UH', 'u': 'UW', 'uː': 'UW', 'v': 'V', 'w': 'W', 'j': 'Y', 'z': 'Z',
        'ʒ': 'ZH', 'ə': 'AH', 'ᵻ': 'IH',
        'ɐ': 'AH', 'ɜ': 'ER', 'a': 'AA', 'o': 'OW', 'e': 'EY', 'ɾ': 'T', 'ʔ': 'T'
    }

    VOWELS = {
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
    }

    def __init__(self, language='en-us'):
        self.language = language
        self.backend = EspeakBackend(
            language=language,
            punctuation_marks=None,
            preserve_punctuation=True,
            with_stress=True
        )
        self.separator = Separator(phone=None, word=' ')

    def __call__(self, text) -> list:
        if isinstance(text, str):
            texts = [text]
            is_single = True
        else:
            texts = text
            is_single = False
            
        phonemes_list = self.backend.phonemize(
            texts, 
            separator=self.separator,
            strip=True
        )
        
        results = [self._process_single_result(p) for p in phonemes_list]
        return results[0] if is_single else results

    def _process_single_result(self, phonemes_str):
        result = []
        words = phonemes_str.split(' ')
        for i, word in enumerate(words):
            result.extend(self._map_ipa_word_to_arpabet(word))
            if i < len(words) - 1:
                result.append(' ')
        return result

    def _map_ipa_word_to_arpabet(self, ipa_word: str) -> list[str]:
        arpabet_word = []
        i = 0
        current_stress = '0'
        
        while i < len(ipa_word):
            char = ipa_word[i]
            if char == 'ˈ':
                current_stress = '1'
                i += 1
                continue
            elif char == 'ˌ':
                current_stress = '2'
                i += 1
                continue
            
            found = False
            for length in [2, 1]:
                if i + length <= len(ipa_word):
                    sub = ipa_word[i:i+length]
                    if sub in self.IPA_TO_ARPABET:
                        ar = self.IPA_TO_ARPABET[sub]
                        if ar in self.VOWELS:
                            arpabet_word.append(ar + current_stress)
                            current_stress = '0'
                        else:
                            arpabet_word.append(ar)
                        i += length
                        found = True
                        break
            
            if not found:
                if char in ".,!?;-()\"'":
                    arpabet_word.append(char)
                i += 1
                
        return arpabet_word

class DeepPhonemizerGenerator(PhonemeGenerator):
    def __init__(self, model_path='pretrained_models/en_us_cmudict_forward.pt'):
        import torch
        import os
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DeepPhonemizer model not found at {model_path}")

        # Monkeypatch torch.load for PyTorch 2.6+ compatibility with old checkpoints
        original_load = torch.load
        try:
            torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
            from dp.phonemizer import Phonemizer
            self.phonemizer = Phonemizer.from_checkpoint(model_path)
        finally:
            torch.load = original_load
        
        from seedvox.utils.text import normalize_text
        self.normalize = normalize_text
        
        self.vowels = {
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
        }

    def __call__(self, text) -> list:
        if isinstance(text, str):
            return self._process_single(text)
        
        # Batch processing for DeepPhonemizer
        norm_texts = [self.normalize(t) for t in text]
        # DeepPhonemizer's self.phonemizer might support batching if it's the right version
        # But let's check if it returns a list if passed a list
        try:
            raw_outputs = self.phonemizer(norm_texts, lang='en_us')
            if isinstance(raw_outputs, str): raw_outputs = [raw_outputs]
            return [self._parse_raw_output(o) for o in raw_outputs]
        except:
            # Fallback to sequential if batching fails
            return [self._process_single(t) for t in text]

    def _process_single(self, text):
        norm_text = self.normalize(text)
        raw_output = self.phonemizer(norm_text, lang='en_us')
        return self._parse_raw_output(raw_output)

    def _parse_raw_output(self, raw_output):
        # Parse the output
        # Find all blocks like [PH] or spaces or punctuation
        tokens = re.findall(r'\[([^\]]+)\]|(\s+)|([^\s\[\]]+)', raw_output)
        
        result = []
        for ph, space, punct in tokens:
            if ph:
                # Add default stress '1' to vowels if missing
                if ph in self.vowels:
                    result.append(ph + '1')
                elif len(ph) > 2 and ph[:-1] in self.vowels and ph[-1] in '012':
                    result.append(ph)
                else:
                    result.append(ph)
            elif space:
                result.append(' ')
            elif punct:
                # Handle punctuation
                for char in punct:
                    result.append(char)
                    
        return result

def get_phoneme_generator(backend: str) -> PhonemeGenerator:
    if backend == 'g2p_en': return G2PEnGenerator()
    if backend == 'espeak': return EspeakGenerator()
    if backend == 'deep-phonemizer': return DeepPhonemizerGenerator()
    raise ValueError(f"Unknown backend: {backend}")
