from phonemizer import phonemize
from phonemizer.separator import Separator
import multiprocessing

class EspeakGenerator:
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
        return phonemes_list

class Worker:
    def __init__(self, generator):
        self.generator = generator
    
    def __call__(self, text):
        try:
            return self.generator(text)
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    gen = EspeakGenerator()
    worker_obj = Worker(gen)
    
    texts = ["Hello world", "This is a test", "Parallel phonemization"]
    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(worker_obj, texts)
    
    for t, r in zip(texts, results):
        print(f"Text: {t}")
        print(f"Result: {r}")
