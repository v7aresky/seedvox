import abc
from g2p_en import G2p
from phonemizer import phonemize
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
        return self.g2p(text)

class EspeakGenerator(PhonemeGenerator):
    IPA_TO_ARPABET = {
        'ɑ': 'AA', 'æ': 'AE', 'ʌ': 'AH', 'ɔ': 'AO', 'aʊ': 'AW', 'aɪ': 'AY',
        'b': 'B', 'tʃ': 'CH', 'd': 'D', 'ð': 'DH', 'ɛ': 'EH', 'ɚ': 'ER', 'ɝ': 'ER',
        'eɪ': 'EY', 'f': 'F', 'ɡ': 'G', 'h': 'HH', 'ɪ': 'IH', 'i': 'IY', 'iː': 'IY',
        'dʒ': 'JH', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'ŋ': 'NG', 'oʊ': 'OW',
        'ɔɪ': 'OY', 'p': 'P', 'ɹ': 'R', 's': 'S', 'ʃ': 'SH', 't': 'T', 'θ': 'TH',
        'ʊ': 'UH', 'u': 'UW', 'uː': 'UW', 'v': 'V', 'w': 'W', 'j': 'Y', 'z': 'Z',
        'ʒ': 'ZH', 'ə': 'AH', 'ᵻ': 'IH'
    }

    VOWELS = {
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
    }

    def __init__(self, language='en-us'):
        self.language = language

    def __call__(self, text) -> list:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        phonemes_list = phonemize(
            texts, 
            backend='espeak', 
            language=self.language, 
            strip=True, 
            preserve_punctuation=True, 
            with_stress=True,
            separator=Separator(phone=None, word=' ')
        )
        
        if isinstance(text, str):
            return self._process_single_result(phonemes_list[0])
        
        return [self._process_single_result(p) for p in phonemes_list]

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
        return [self._process_single(t) for t in text]

    def _process_single(self, text):
        norm_text = self.normalize(text)
        raw_output = self.phonemizer(norm_text, lang='en_us')
        
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
