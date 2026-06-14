import time
from seedvox.utils.g2p_factory import get_phoneme_generator

def benchmark_backend(name, texts):
    try:
        g2p = get_phoneme_generator(name)
        start = time.time()
        for text in texts:
            _ = g2p(text)
        end = time.time()
        return end - start
    except Exception as e:
        return f"Error: {e}"

texts = [
    "Hello, this is the phoneme AND prosody planner today especially brought to you by seedvox.",
    "The quick brown fox jumps over the lazy dog.",
    "I live near the live music.",
    "Please record the record."
] * 10 # 40 sentences total

print(f"{'Backend':<20} | {'Total Time (s)':<15} | {'Avg per sent (ms)':<20}")
print("-" * 60)

for backend in ['g2p_en', 'espeak', 'deep-phonemizer']:
    t = benchmark_backend(backend, texts)
    if isinstance(t, float):
        print(f"{backend:<20} | {t:<15.4f} | {t/len(texts)*1000:<20.2f}")
    else:
        print(f"{backend:<20} | {t}")
