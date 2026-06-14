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
                raw = torch.load(p, weights_only=False)
                if isinstance(raw, dict) and 'data' in raw: self.data.extend(raw['data'])
                elif isinstance(raw, list): self.data.extend(raw)
        self.tokenizer = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text_ids = torch.tensor(self.tokenizer.encode(item['text']), dtype=torch.long)
        audio_tokens = item['audio_tokens'].squeeze(0)
        return text_ids, audio_tokens, item['text']

class LengthGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset, self.batch_size = dataset, batch_size
        self.indices, self.lengths = list(range(len(dataset))), [item[1].shape[-1] for item in dataset]
    def __iter__(self):
        indices_with_lengths = [(i, self.lengths[i] + self.lengths[i] * 0.1 * (random.random() - 0.5)) for i in self.indices]
        indices_with_lengths.sort(key=lambda x: x[1])
        sorted_indices = [x[0] for x in indices_with_lengths]
        batches = [sorted_indices[i : i + self.batch_size] for i in range(0, len(sorted_indices), self.batch_size)]
        random.shuffle(batches)
        for b in batches:
            for idx in b: yield idx
    def __len__(self): return len(self.indices)

def collate_fn(batch):
    text_ids, audio_tokens, raw_texts = zip(*batch)
    t_lens = torch.tensor([len(t) for t in text_ids], dtype=torch.long)
    t_max = ((t_lens.max().item() + 7) // 8) * 8
    padded_text = torch.full((len(text_ids), t_max), 0, dtype=torch.long)
    for i, t in enumerate(text_ids): padded_text[i, :len(t)] = t
    a_lens = torch.tensor([a.shape[1] for a in audio_tokens], dtype=torch.long)
    a_max = ((a_lens.max().item() + 7) // 8) * 8
    padded_audio = torch.zeros(len(audio_tokens), audio_tokens[0].shape[0], a_max, dtype=torch.long)
    for i, a in enumerate(audio_tokens): padded_audio[i, :, :a.shape[1]] = a
    return padded_text, padded_audio, t_lens, a_lens, raw_texts
