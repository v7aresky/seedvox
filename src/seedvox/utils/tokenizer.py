import torch
import torch.nn as nn
import torch.nn.functional as F
from seedvox.modules.moshi.modules.transformer import StreamingTransformer, create_norm_fn

class CharTokenizer:
    def __init__(self):
        self.pad_token_id, self.unk_token_id = 0, 1
        self.chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-':;\"()[]/\\àâèéêü’“”—_*"
        self.char_to_id = {c: i + 2 for i, c in enumerate(self.chars)}
        self.id_to_char = {i + 2: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 2
    def encode(self, text, normalize=True):
        from seedvox.utils.text import normalize_text
        if normalize:
            text = normalize_text(text)
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]
    def decode(self, ids):
        res = []
        for i in ids:
            if i >= 2: res.append(self.id_to_char.get(i, "?"))
            elif i == 1: res.append("?")
        return "".join(res)

class PhonemeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.mask_token_id = 2
        self.eps_token_id = 3
        self.phonemes = [
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 
            'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 
            'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 
            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
        ]
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
        stressed_vowels = []
        for v in vowels:
            stressed_vowels.extend([v + '0', v + '1', v + '2'])
        self.special = [' ', '.', ',', '!', '?', '-', "'", '"', '(', ')', '<EPS>']
        self.vocab = self.phonemes + stressed_vowels + self.special
        self.ph_to_id = {ph: i + 4 for i, ph in enumerate(self.vocab)}
        self.ph_to_id['<EPS>'] = self.eps_token_id
        self.ph_to_id['<SOS>'] = self.eps_token_id
        self.ph_to_id['<EOS>'] = 128
        
        self.id_to_ph = {i + 4: ph for i, ph in enumerate(self.vocab)}
        self.id_to_ph[self.eps_token_id] = "<SOS>"
        self.id_to_ph[128] = "<EOS>"
        
        self.vocab_size = 129 # Ensure vocab size covers EOS
    def encode(self, phoneme_string):
        tokens = []
        for ph in phoneme_string.split():
            tokens.append(self.ph_to_id.get(ph, self.unk_token_id))
        return tokens
    def decode(self, ids):
        res = []
        for i in ids:
            if isinstance(i, torch.Tensor): i = i.item()
            if i in self.id_to_ph:
                res.append(self.id_to_ph[i])
            elif i == 1: res.append("?")
            elif i == 2: res.append("[MASK]")
            elif i == 0: continue # Skip padding in decode
            else: res.append(f"<{i}>")
        return " ".join(res)

class BPECharEncoder(nn.Module):
    def __init__(self, dim=512, bpe_layers=2, bpe_heads=8, bpe_hidden_scale=4.0,
                 use_pretrained_embeddings=True, freeze_bpe_embeddings=False,
                 combination="add"):
        super().__init__()
        self.dim = dim
        self.combination = combination
        from transformers import GPT2TokenizerFast
        self.bpe_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if self.bpe_tokenizer.pad_token is None:
            self.bpe_tokenizer.pad_token = self.bpe_tokenizer.eos_token
        self.bpe_vocab_size = self.bpe_tokenizer.vocab_size
        self.pad_token_id = self.bpe_tokenizer.pad_token_id
        if use_pretrained_embeddings:
            from transformers import GPT2Model
            gpt2 = GPT2Model.from_pretrained("gpt2")
            pretrained_emb = gpt2.wte.weight.data
            del gpt2
            self.bpe_emb = nn.Embedding(self.bpe_vocab_size, 768)
            self.bpe_emb.weight.data = pretrained_emb
            self.bpe_proj = nn.Linear(768, dim)
            if freeze_bpe_embeddings:
                self.bpe_emb.weight.requires_grad = False
        else:
            self.bpe_emb = nn.Embedding(self.bpe_vocab_size, dim)
            nn.init.normal_(self.bpe_emb.weight, std=0.02)
            self.bpe_proj = None
        hs = int(bpe_hidden_scale * dim)
        self.bpe_transformer = StreamingTransformer(
            dim, bpe_heads, bpe_layers, hs,
            causal=False, positional_embedding="sin_rope", context=512
        )
        self.bpe_norm = create_norm_fn("layer_norm", dim)
        if combination == "concat":
            self.combine_proj = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
        else:
            self.combine_proj = None
        self.intra_word_emb = nn.Embedding(20, dim)
        nn.init.normal_(self.intra_word_emb.weight, std=0.01)
    def tokenize_bpe_normalized(self, text, normalized_text):
        encoded = self.bpe_tokenizer(normalized_text, return_offsets_mapping=True)
        bpe_ids = encoded['input_ids']
        offsets = encoded['offset_mapping']
        char_to_bpe = [0] * len(normalized_text)
        for bpe_idx, (start, end) in enumerate(offsets):
            for char_pos in range(start, end):
                if char_pos < len(normalized_text): char_to_bpe[char_pos] = bpe_idx
        return bpe_ids, char_to_bpe
    def forward_bpe(self, bpe_ids, bpe_lens, device='cuda', length_scale=1.0):
        B, T_bpe = bpe_ids.shape
        x = self.bpe_emb(bpe_ids)
        if self.bpe_proj is not None: x = self.bpe_proj(x)
        indices = torch.arange(T_bpe, device=device).unsqueeze(0)
        attn_mask = (indices < bpe_lens.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
        return self.bpe_norm(self.bpe_transformer(x, attn_mask=attn_mask, length_scale=length_scale))
    def expand_to_chars(self, bpe_ctx, char_to_bpe_batch, char_lens, device='cuda'):
        B, T_char_max = char_to_bpe_batch.shape
        bpe_indices = char_to_bpe_batch.unsqueeze(-1).expand(-1, -1, self.dim)
        char_features = torch.gather(bpe_ctx, 1, bpe_indices.clamp(max=bpe_ctx.shape[1]-1))
        shifted = torch.cat([torch.zeros((B, 1), device=device, dtype=char_to_bpe_batch.dtype), char_to_bpe_batch[:, :-1]], dim=1)
        boundary = (char_to_bpe_batch != shifted).long()
        cumsum = torch.arange(T_char_max, device=device).unsqueeze(0).expand(B, -1)
        last_boundary = torch.zeros((B,), device=device, dtype=torch.long)
        intra_pos = torch.zeros((B, T_char_max), device=device, dtype=torch.long)
        for t in range(T_char_max):
            is_boundary = boundary[:, t]
            last_boundary = torch.where(is_boundary == 1, cumsum[:, t], last_boundary)
            intra_pos[:, t] = cumsum[:, t] - last_boundary
        return char_features + self.intra_word_emb(intra_pos.clamp(max=19))

class BPECharCollator:
    def __init__(self, bpe_encoder):
        self.bpe_encoder = bpe_encoder
        from seedvox.utils.text import normalize_text
        self.normalize_text = normalize_text
    def process_batch_texts(self, raw_texts, char_lens, device='cpu'):
        B, T_char_max = len(raw_texts), char_lens.max().item()
        bpe_ids_list, char_to_bpe_list = [], []
        for text in raw_texts:
            normalized = self.normalize_text(text)
            bpe_ids, c2b = self.bpe_encoder.tokenize_bpe_normalized(text, normalized)
            bpe_ids_list.append(bpe_ids)
            char_to_bpe_list.append(c2b[:T_char_max] + [0] * max(0, T_char_max - len(c2b)))
        max_bpe_len = max(len(ids) for ids in bpe_ids_list)
        pad_id = getattr(self.bpe_encoder, 'pad_token_id', 0)
        bpe_ids_padded = torch.full((B, max_bpe_len), pad_id, dtype=torch.long)
        bpe_lens = torch.zeros(B, dtype=torch.long)
        for i, ids in enumerate(bpe_ids_list):
            bpe_ids_padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            bpe_lens[i] = len(ids)
        return bpe_ids_padded.to(device), bpe_lens.to(device), torch.tensor(char_to_bpe_list, dtype=torch.long).to(device)

def get_bpe_encoder_from_config(cfg, device='cpu'):
    bpe_cfg = cfg.get('model', {}).get('bpe_encoder', None)
    if bpe_cfg is None or not bpe_cfg.get('enabled', False): return None
    encoder = BPECharEncoder(
        dim=cfg['model']['dim'],
        bpe_layers=bpe_cfg.get('bpe_layers', 2),
        bpe_heads=bpe_cfg.get('bpe_heads', 8),
        bpe_hidden_scale=bpe_cfg.get('bpe_hidden_scale', 4.0),
        use_pretrained_embeddings=bpe_cfg.get('use_pretrained_embeddings', True),
        freeze_bpe_embeddings=bpe_cfg.get('freeze_bpe_embeddings', False),
        combination=bpe_cfg.get('combination', 'add'),
    ).to(device)
    return encoder
