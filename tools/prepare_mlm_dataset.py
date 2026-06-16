import torch
import argparse
import os
import json
import re
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing

# Import from current project structure
from src.seedvox.utils.tokenizer import CharTokenizer
from src.seedvox.utils.g2p_factory import get_phoneme_generator
from explicit_pros_phon_planner.utils import PhoneticGenerator
from text_normalizer import normalize_text

def get_sentences(text):
    """
    Robustly split text into sentences by first splitting paragraphs, 
    then splitting sentences by punctuation-followed-by-whitespace.
    """
    # Split paragraphs by one or more newlines
    paragraphs = re.split(r'\n+', text)
    all_sentences = []
    for p in paragraphs:
        # Improved regex to handle various whitespace including non-breaking spaces
        # and splitting on [.!?] followed by whitespace
        sents = re.split(r'(?<=[.!?])(?:\s+|\u00a0+)', p)
        all_sentences.extend(sents)
    
    # Filter for quality
    cleaned = []
    for s in all_sentences:
        s = s.strip()
        if len(s) > 10:
            cleaned.append(s)
    return cleaned

def process_and_align(example, g2p_backend='g2p_en', max_len=256):
    """
    Worker function: splits text into sentences using a robust two-stage approach
    and performs direct G2P generation.
    """
    # Need to re-initialize generator inside the worker for thread safety
    generator = PhoneticGenerator(backend=g2p_backend, phoneme_vocab_size=128)
    char_tokenizer = CharTokenizer()
    
    text = example.get('text', '')
    sentences = get_sentences(text)
    
    processed_results = []
    
    for sent in sentences:
        norm_sent = normalize_text(sent)
        if len(norm_sent) < 10: continue
        
        char_ids = char_tokenizer.encode(norm_sent)
        # Skip samples that are too long instead of truncating
        if len(char_ids) > max_len:
            continue
        
        # DIRECT G2P Prediction
        try:
            ph_ids = generator.generate_targets(norm_sent)
        except Exception:
            continue
            
        # Skip if phoneme length also exceeds limit (or doesn't match)
        if len(ph_ids) > max_len:
            continue
        
        # Ensure parity
        min_l = min(len(char_ids), len(ph_ids))
        if min_l < 10: continue
        
        processed_results.append({
            'text': sent,
            'norm_text': norm_sent,
            'ph_ids': ph_ids
        })
        
    return {'results': processed_results}

from functools import partial

# Helper for partial application
def worker_wrapper(example, g2p_backend='g2p_en', max_len=256):
    return process_and_align(example, g2p_backend=g2p_backend, max_len=max_len)

def prepare_dataset(corpus_name, output_path, num_samples=100000, max_len=256, g2p_backend='g2p_en'):
    CORPUS_REGISTRY = {
        "tinystories": ("karpathy/tinystories-gpt4-clean", {"split": "train"}),
        "wikipedia": ("wikimedia/wikipedia", {"name": "20231101.en", "split": "train"}),
    }
    
    if corpus_name not in CORPUS_REGISTRY:
        print(f"Unknown corpus: {corpus_name}")
        return

    ds_name, ds_kwargs = CORPUS_REGISTRY[corpus_name]
    # streaming=True to allow stopping early
    dataset = load_dataset(ds_name, **ds_kwargs, streaming=True)
    # Process in parallel
    print(f"Processing in parallel using {multiprocessing.cpu_count()} cores...")

    count = 0
    pbar = tqdm(total=num_samples, desc="Preparing sentences")

    # We use a pool to process paragraphs in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Map using partial to pass arguments
        func = partial(worker_wrapper, g2p_backend=g2p_backend, max_len=max_len)

        with open(output_path, 'w') as f:
            for result_dict in pool.imap(func, dataset):
                for entry in result_dict.get('results', []):
                    f.write(json.dumps(entry) + '\n')
                    count += 1
                    pbar.update(1)
                    if count >= num_samples:
                        print(f"\nReached target of {num_samples} sentences. Stopping.")
                        pool.terminate()
                        break
                if count >= num_samples:
                    break

    pbar.close()
    print(f"Saved {count} sentences to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="wikipedia")
    parser.add_argument("--output", default="dataset/preprocessed_data.jsonl")
    parser.add_argument("--num", type=int, default=50000)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--g2p_backend", type=str, default="g2p_en")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    prepare_dataset(args.corpus, args.output, num_samples=args.num, max_len=args.max_len, g2p_backend=args.g2p_backend)
