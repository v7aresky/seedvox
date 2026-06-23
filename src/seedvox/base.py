import torch
import torch.nn as nn
import torch.nn.functional as F
from seedvox.modules.moshi.modules.streaming import StreamingContainer
from seedvox.modules.moshi.modules.transformer import StreamingTransformer, create_norm_fn
from seedvox.modules.components import (
    ScaledEmbedding, TextTransformerEncoder, SpeakerEncoder, 
    ProsodyEncoder, MonotonicAttention
)

# Shared helper
def _sum_embeddings(emb_list, tokens, n_q):
    # Only iterate up to the number of codebooks present in tokens or n_q, whichever is smaller
    num_codebooks = min(tokens.shape[1], n_q)
    ae = emb_list[0](tokens[:, 0])
    for k in range(1, num_codebooks):
        ae = ae + emb_list[k](tokens[:, k])
    return ae

class CrossAttentionDecoderLayer(nn.Module):
    def __init__(self, dim, heads, hidden_scale=4.0, pre_norm=True):
        super().__init__()
        hs = int(hidden_scale * dim)
        self.pre_norm = pre_norm
        self.self_attn = StreamingTransformer(dim, heads, 1, hs, causal=True, positional_embedding="sin_rope", context=2048)
        self.cross_attn = MonotonicAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hs), nn.GELU(),
            nn.Linear(hs, dim), nn.Dropout(0.1)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, kv_ctx, kv_mask=None, positions=None):
        if self.pre_norm:
            x = x + self.self_attn(self.norm1(x), positions=positions)
            x = x + self.cross_attn(self.norm2(x), kv_input=kv_ctx, kv_mask=kv_mask)
            x = x + self.ff(self.norm3(x))
        else:
            x = self.norm1(x + self.self_attn(x, positions=positions))
            x = self.norm2(x + self.cross_attn(x, kv_input=kv_ctx, kv_mask=kv_mask))
            x = self.norm3(x + self.ff(x))
        return x

class JEPAProsodyBase(StreamingContainer):
    def __init__(self, config, tokenizer_vocab_size):
        super().__init__()
        cfg = config['model']
        self.dim, self.n_q, self.card = cfg['dim'], cfg['n_q'], cfg['card']
        
        self.SOT_ID = tokenizer_vocab_size + 1
        self.EOT_ID = tokenizer_vocab_size + 2
        self.SOA_ID = self.card
        self.EOA_ID = self.card + 1
        
        self.text_emb = ScaledEmbedding(tokenizer_vocab_size + 4, self.dim)
        self.text_encoder = TextTransformerEncoder(
            self.dim, cfg['num_heads'], cfg['text_encoder_layers'],
            cfg['hidden_scale'], causal=False
        )
        self.speaker_encoder = SpeakerEncoder(self.dim, num_latents=cfg['num_speaker_latents'])
        self.prosody_encoder = ProsodyEncoder(self.dim, num_latents=cfg['num_prosody_latents'])
        
        self.audio_embs = nn.ModuleList([ScaledEmbedding(self.card + 3, self.dim, zero_idx=-1) for _ in range(self.n_q)])
        self.audio_prenet = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Dropout(0.1))
        self.audio_norm = create_norm_fn("layer_norm", self.dim)
        
        use_pre_norm = cfg.get('use_pre_norm', True)
        self.decoder_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(self.dim, cfg['num_heads'], cfg['hidden_scale'], pre_norm=use_pre_norm) 
            for _ in range(cfg['dec_num_layers'])
        ])
        
        self.dep_in = nn.ModuleList([nn.Linear(self.dim, self.dim, bias=False) for _ in range(self.n_q)])
        self.dep_layers = nn.ModuleList([nn.Linear(self.dim, self.card + 3) for _ in range(self.n_q)])
        self.dep_emb = nn.ModuleList([ScaledEmbedding(self.card + 3, self.dim, zero_idx=-1) for _ in range(self.n_q - 1)])
        self.dep_transformer = StreamingTransformer(
            self.dim, self.dim // 64, cfg['depformer_num_layers'],
            int(4.0 * self.dim), causal=True, context=self.n_q, positional_embedding="rope"
        )
        self.cfg = cfg

    def encode_text(self, text, text_lens, raw_texts=None):
        B, device = text.shape[0], text.device
        text_in = torch.zeros((B, text.shape[1] + 2), device=device, dtype=torch.long)
        for b in range(B):
            l = text_lens[b].item()
            text_in[b, 0], text_in[b, 1:1+l], text_in[b, 1+l] = self.SOT_ID, text[b, :l], self.EOT_ID
        t_mask = torch.ones(B, 1, text_in.shape[1], text_in.shape[1], device=device, dtype=torch.bool)
        for b in range(B):
            l = text_lens[b].item() + 2
            t_mask[b, 0, :, l:], t_mask[b, 0, l:, :] = False, False
        return self.text_encoder(self.text_emb(text_in), mask=t_mask), text_in
    
    def forward_with_context(self, context, ctx_mask, audio_tokens, audio_lens):
        B, K, Ta = audio_tokens.shape
        device = audio_tokens.device
        a_in = torch.full((B, K, Ta + 1), self.SOA_ID, device=device, dtype=torch.long)
        a_in[:, :, 1:] = audio_tokens
        a_tgt = torch.full((B, K, Ta + 1), -1, device=device, dtype=torch.long)
        a_tgt[:, :, :Ta] = audio_tokens
        for b in range(B):
            a_tgt[b, :, audio_lens[b]] = self.EOA_ID
            
        a_emb = self.audio_prenet(self.audio_norm(
            _sum_embeddings(self.audio_embs, a_in, K)
        ))
        
        x = a_emb
        positions = torch.arange(a_emb.shape[1], device=device).unsqueeze(0)
        for layer in self.decoder_layers:
            x = layer(x, context, kv_mask=ctx_mask, positions=positions)
            
        T_a_in = x.shape[1]
        flat_out = x.reshape(B * T_a_in, 1, self.dim).transpose(0, 1)
        flat_tgt = a_tgt.transpose(1, 2).reshape(1, B * T_a_in, K).transpose(1, 2)
        
        dep_inputs = []
        for k in range(self.n_q):
            if k == 0:
                dep_inputs.append(self.dep_in[k](flat_out))
            else:
                dep_inputs.append(self.dep_in[k](flat_out) + self.dep_emb[k-1](flat_tgt[:, k-1]))
        
        d_out = self.dep_transformer(torch.stack(dep_inputs, 2).view(B * T_a_in, self.n_q, -1))
        logits = torch.stack([self.dep_layers[k](d_out[:, k, :]).view(B, T_a_in, self.card + 3) for k in range(self.n_q)], 0)
        return logits, a_tgt
