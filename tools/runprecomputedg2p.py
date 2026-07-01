import torch
from tqdm import tqdm
from explicit_pros_phon_planner.utils import PhoneticGenerator

# Initialize generator
ph_generator = PhoneticGenerator(backend='espeak', phoneme_vocab_size=128)

"""
datasets = [
 "../autovoc/dataset/train_tokens_libri_prosody_16q.pt",
"../autovoc/train_tokens_globe.pt"
 ]
 """


"""
datasets = [
  "../autovoc/dataset/hifitts/train_tokens_hifitts.pt",
  "../autovoc/dataset/train_tokens_prosody_16q.pt",
  "../autovoc/dataset/train_tokens_prosody_zp.pt"
 ]

"""

datasets = [
  "../autovoc/dataset/train_tokens_rf.pt"
 ]

for path in datasets:
    print(f"Precomputing phonemes for {path}...")
    dataset = torch.load(path, weights_only=False)

    # Handle list or dict format
    data_list = dataset['data'] if isinstance(dataset, dict) else dataset

    # Extract texts for batching
    texts = [item['text'] for item in data_list]

    # Batch G2P computation
    print("Running espeak...")
    ph_ids_list = ph_generator.generate_targets_batch(texts, normalize=True)

    for item, ph_ids in zip(data_list, ph_ids_list):
        item['ph_ids'] = torch.tensor(ph_ids, dtype=torch.long)

    torch.save(dataset, path)
    print(f"Saved {path} successfully.")



