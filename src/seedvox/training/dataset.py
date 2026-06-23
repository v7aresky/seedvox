import os
import torch
import math
import random
from torch.utils.data import Dataset, Sampler

class TokenizedSpeechDataset(Dataset):
    def __init__(self, token_paths, tokenizer):
        if isinstance(token_paths, str): token_paths = [token_paths]
        self.data = []
        for p in token_paths:
            if os.path.exists(p):
                raw = torch.load(p, weights_only=False, map_location='cpu')
                if isinstance(raw, dict) and 'data' in raw: self.data.extend(raw['data'])
                elif isinstance(raw, list): self.data.extend(raw)
        self.tokenizer = tokenizer
        # Pre-calculate lengths to avoid calling it in Sampler
        from tqdm import tqdm
        self.lengths = []
        for item in tqdm(self.data, desc="Pre-calculating dataset lengths"):
            self.lengths.append(item['audio_tokens'].shape[-1])
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        from seedvox.utils.text import normalize_text
        # Use pre-normalized text if available, otherwise normalize now
        norm_text = item.get('normalized_text', normalize_text(item['text']))
        text_ids = torch.tensor(self.tokenizer.encode(norm_text, normalize=False), dtype=torch.long)
        audio_tokens = item['audio_tokens'].squeeze(0)
        ph_ids = item.get('ph_ids')
        if ph_ids is not None:
            # Ensure it is a tensor
            if not isinstance(ph_ids, torch.Tensor):
                ph_ids = torch.tensor(ph_ids, dtype=torch.long)
        return text_ids, audio_tokens, norm_text, ph_ids

class LengthGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset, self.batch_size = dataset, batch_size
        self.indices = list(range(len(dataset)))
        if hasattr(dataset, 'lengths'):
            self.lengths = dataset.lengths
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'lengths'):
            # Handle Subset: map subset indices to original dataset indices
            self.lengths = [dataset.dataset.lengths[dataset.indices[i]] for i in self.indices]
        else:
            print("Warning: Dataset does not have pre-calculated lengths. This may be slow.")
            self.lengths = [item[1].shape[-1] for item in dataset]
    def __iter__(self):
        # Reduced noise factor from 0.1 to 0.02 for more stable length grouping
        indices_with_lengths = [(i, self.lengths[i] + self.lengths[i] * 0.02 * (random.random() - 0.5)) for i in self.indices]
        indices_with_lengths.sort(key=lambda x: x[1])
        sorted_indices = [x[0] for x in indices_with_lengths]
        batches = [sorted_indices[i : i + self.batch_size] for i in range(0, len(sorted_indices), self.batch_size)]
        random.shuffle(batches)
        for b in batches:
            for idx in b: yield idx
    def __len__(self): return len(self.indices)

def collate_fn(batch):
    text_ids, audio_tokens, raw_texts, ph_ids = zip(*batch)
    
    t_lens = torch.tensor([len(t) for t in text_ids], dtype=torch.long)
    t_max = ((t_lens.max().item() + 7) // 8) * 8
    padded_text = torch.full((len(text_ids), t_max), 0, dtype=torch.long)
    for i, t in enumerate(text_ids): padded_text[i, :len(t)] = t
    
    # Dynamically find max K (codebooks) and max T (sequence length)
    k_max = max(a.shape[0] for a in audio_tokens)
    a_max = max(a.shape[1] for a in audio_tokens)
    a_max = ((a_max + 7) // 8) * 8
    
    padded_audio = torch.zeros(len(audio_tokens), k_max, a_max, dtype=torch.long)
    for i, a in enumerate(audio_tokens): 
        padded_audio[i, :a.shape[0], :a.shape[1]] = a
    
    a_lens = torch.tensor([a.shape[1] for a in audio_tokens], dtype=torch.long)
    
    # Handle ph_ids (might be None)
    padded_ph = None
    if any(p is not None for p in ph_ids):
        # Filter out Nones for padding
        ph_list = [p if p is not None else torch.tensor([], dtype=torch.long) for p in ph_ids]
        ph_lens = torch.tensor([len(p) for p in ph_list], dtype=torch.long)
        p_max = ph_lens.max().item()
        padded_ph = torch.full((len(ph_list), p_max), 0, dtype=torch.long)
        for i, p in enumerate(ph_list): 
            if len(p) > 0: padded_ph[i, :len(p)] = p
            
    return padded_text, padded_audio, t_lens, a_lens, raw_texts, padded_ph
