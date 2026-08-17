import os
import torch
import math
import random
from torch.utils.data import Dataset, Sampler

class TokenizedSpeechDataset(Dataset):
    def __init__(self, token_paths, tokenizer, return_wav=False):
        if isinstance(token_paths, str): token_paths = [token_paths]
        self.data = []
        self.return_wav = return_wav
        for p in token_paths:
            if os.path.exists(p):
                raw = torch.load(p, weights_only=False, map_location='cpu')
                if isinstance(raw, dict) and 'data' in raw: self.data.extend(raw['data'])
                elif isinstance(raw, list): self.data.extend(raw)
        self.tokenizer = tokenizer
        if return_wav:
            before = len(self.data)
            self.data = [d for d in self.data if d.get('wav') is not None]
            if before != len(self.data):
                print(f"Filtered dataset: {before} -> {len(self.data)} (dropped {before - len(self.data)} items without wav)")
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

        wav = None
        if self.return_wav:
            wav = item.get('wav', None)
            if wav is None and 'wav_path' in item:
                import torchaudio
                wav, _ = torchaudio.load(item['wav_path'])
            if wav is not None:
                wav = wav.squeeze(0)  # [1, T] -> [T]

        # Prosody features (stage-1 codec input): [T, 3] (log_f0_center, e_center, voicing)
        prosody_feat = None
        if all(k in item for k in ('log_f0_center', 'e_center', 'voicing')):
            prosody_feat = torch.stack(
                [item['log_f0_center'], item['e_center'], item['voicing'].float()], dim=-1)

        return text_ids, audio_tokens, norm_text, ph_ids, wav, prosody_feat

class LengthGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size, max_len=None):
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
        if max_len is not None:
            before = len(self.indices)
            self.indices = [i for i in self.indices if self.lengths[i] <= max_len]
            dropped = before - len(self.indices)
            if dropped:
                print(f"LengthGroupedSampler: pruned {dropped} samples (audio_tokens > {max_len}) -> {len(self.indices)} remain")
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

def _collate(batch, return_feats):
    text_ids, audio_tokens, raw_texts, ph_ids, _wavs, feats = zip(*batch)

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

    # Handle ph_ids (might be None). Use stored phonemes ONLY if every item has them;
    # otherwise return None so callers regenerate G2P for the whole batch (mixing
    # datasets with and without ph_ids would otherwise leave zero rows that NaN the model).
    padded_ph = None
    if all(p is not None for p in ph_ids):
        ph_list = [p if p is not None else torch.tensor([], dtype=torch.long) for p in ph_ids]
        ph_lens = torch.tensor([len(p) for p in ph_list], dtype=torch.long)
        p_max = ph_lens.max().item()
        padded_ph = torch.full((len(ph_list), p_max), 0, dtype=torch.long)
        for i, p in enumerate(ph_list):
            if len(p) > 0: padded_ph[i, :len(p)] = p

    base = (padded_text, padded_audio, t_lens, a_lens, raw_texts, padded_ph)

    if not return_feats:
        return base

    # Prosody features padded to a multiple of num_blocks (32) so the codec's
    # adaptive pool bins are consistent with stage-1 training.
    if all(f is not None for f in feats):
        a_pros = ((a_max + 31) // 32) * 32
        padded_feat = torch.zeros(len(feats), a_pros, 3)
        for i, f in enumerate(feats):
            n = min(f.shape[0], a_pros)
            padded_feat[i, :n] = f[:n]
    else:
        padded_feat = None

    return base + (padded_feat,)

def collate_fn(batch):
    return _collate(batch, return_feats=False)

def collate_fn_prosody(batch):
    return _collate(batch, return_feats=True)
