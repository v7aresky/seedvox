import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from seedvox.modules.moshi.modules.streaming import StreamingContainer
from seedvox.modules.moshi.modules.transformer import StreamingTransformer, create_norm_fn
from seedvox.modules.components import (
    ScaledEmbedding, TextTransformerEncoder, SpeakerEncoder, 
    ProsodyEncoder, MonotonicAttention, AdaLN
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
        # Speaker AdaLN: directly modulate hidden states with speaker vector
        self.speaker_adaLN = AdaLN(dim)
        # Initialize AdaLN to identity: gamma=0, beta=0 => x * (1+0) + 0 = x
        with torch.no_grad():
            nn.init.zeros_(self.speaker_adaLN.mlp[1].weight)
            nn.init.zeros_(self.speaker_adaLN.mlp[1].bias)

    def forward(self, x, kv_ctx, kv_mask=None, positions=None, speaker_emb=None):
        if self.pre_norm:
            x = x + self.self_attn(self.norm1(x), positions=positions)
            x = x + self.cross_attn(self.norm2(x), kv_input=kv_ctx, kv_mask=kv_mask)
            if speaker_emb is not None:
                spk = speaker_emb.mean(dim=1) if speaker_emb.dim() == 3 else speaker_emb
                x = self.speaker_adaLN(x, spk)
            x = x + self.ff(self.norm3(x))
        else:
            x = self.norm1(x + self.self_attn(x, positions=positions))
            x = self.norm2(x + self.cross_attn(x, kv_input=kv_ctx, kv_mask=kv_mask))
            if speaker_emb is not None:
                spk = speaker_emb.mean(dim=1) if speaker_emb.dim() == 3 else speaker_emb
                x = self.speaker_adaLN(x, spk)
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
        
        self.dep_level_emb = nn.Parameter(torch.randn(1, self.n_q, self.dim) * 0.02)
        self.dep_in = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(self.dim * 2), nn.Linear(self.dim * 2, self.dim), nn.GELU(), nn.Linear(self.dim, self.dim)
        ) for _ in range(self.n_q)])
        self.dep_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, self.card + 3)
        ) for _ in range(self.n_q)])
        self.dep_emb = nn.ModuleList([ScaledEmbedding(self.card + 3, self.dim, zero_idx=-1) for _ in range(self.n_q - 1)])
        self.dep_transformer = StreamingTransformer(
            self.dim, self.dim // 64, cfg['depformer_num_layers'],
            int(4.0 * self.dim), causal=True, context=self.n_q, positional_embedding="rope"
        )
        # Direct speaker conditioning for the NAR depformer (codebook predictor).
        # Voice/timbre lives in the residual codebooks, so give the depformer its
        # own speaker signal instead of relying only on the AR context tokens.
        # Zero-init => identity at load, so existing checkpoints stay valid.
        self.use_dep_speaker_cond = cfg.get('use_dep_speaker_cond', True)
        if self.use_dep_speaker_cond:
            self.dep_speaker_adaLN = AdaLN(self.dim)
            with torch.no_grad():
                nn.init.zeros_(self.dep_speaker_adaLN.mlp[1].weight)
                nn.init.zeros_(self.dep_speaker_adaLN.mlp[1].bias)
        # Direct prosody conditioning for the NAR depformer (codebook predictor).
        # Without this, prosody only reaches the depformer indirectly through the AR
        # decoder's hidden state (which the probe showed barely absorbs prosody).
        # Zero-init => identity at load, so existing checkpoints stay valid.
        self.use_dep_prosody_cond = cfg.get('use_dep_prosody_cond', True)
        if self.use_dep_prosody_cond:
            self.dep_prosody_adaLN = AdaLN(self.dim)
            with torch.no_grad():
                nn.init.zeros_(self.dep_prosody_adaLN.mlp[1].weight)
                nn.init.zeros_(self.dep_prosody_adaLN.mlp[1].bias)
        self.latent_regressor = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.cfg = cfg
        self.use_grad_ckpt = cfg.get('gradient_checkpointing', False)

    def encode_text(self, text, text_lens, raw_texts=None):
        B, T, device = text.shape[0], text.shape[1], text.device
        text_in = torch.zeros((B, T + 2), device=device, dtype=torch.long)
        text_in[:, 0] = self.SOT_ID
        text_in[:, 1:1+T] = text
        text_in[torch.arange(B, device=device), text_lens.long() + 1] = self.EOT_ID

        T_full = T + 2
        lens = (text_lens.long() + 2).unsqueeze(1)
        arange = torch.arange(T_full, device=device).unsqueeze(0)
        valid = arange < lens
        t_mask = (valid.unsqueeze(2) & valid.unsqueeze(1)).unsqueeze(1)
        return self.text_encoder(self.text_emb(text_in), mask=t_mask), text_in
    
    def forward_with_context(self, context, ctx_mask, audio_tokens, audio_lens, speaker_emb=None, prosody_emb=None):
        B, K, Ta = audio_tokens.shape
        device = audio_tokens.device
        a_in = torch.full((B, K, Ta + 1), self.SOA_ID, device=device, dtype=torch.long)
        a_in[:, :, 1:] = audio_tokens
        a_tgt = torch.full((B, K, Ta + 1), -1, device=device, dtype=torch.long)
        a_tgt[:, :, :Ta] = audio_tokens
        a_tgt[torch.arange(B, device=device), :, audio_lens] = self.EOA_ID
            
        a_emb = self.audio_prenet(self.audio_norm(
            _sum_embeddings(self.audio_embs, a_in, K)
        ))
        
        x = a_emb
        positions = torch.arange(a_emb.shape[1], device=device).unsqueeze(0)
        if self.use_grad_ckpt and self.training:
            for layer in self.decoder_layers:
                x = checkpoint(layer, x, context, ctx_mask, positions, speaker_emb, use_reentrant=False)
        else:
            for layer in self.decoder_layers:
                x = layer(x, context, kv_mask=ctx_mask, positions=positions, speaker_emb=speaker_emb)
            
        T_a_in = x.shape[1]
        flat_out = x.reshape(B * T_a_in, 1, self.dim).transpose(0, 1)
        flat_tgt = a_tgt.transpose(1, 2).reshape(1, B * T_a_in, K).transpose(1, 2)
        
        dep_spk = None
        if speaker_emb is not None and getattr(self, 'use_dep_speaker_cond', True):
            spk1d = speaker_emb.mean(dim=1) if speaker_emb.dim() == 3 else speaker_emb  # [B, D]
            dep_spk = spk1d[:, None, :].expand(B, T_a_in, -1).reshape(B * T_a_in, -1)  # [B*T, D]

        dep_prs = None
        if prosody_emb is not None and getattr(self, 'use_dep_prosody_cond', True):
            prs1d = prosody_emb.mean(dim=1) if prosody_emb.dim() == 3 else prosody_emb  # [B, D]
            dep_prs = prs1d[:, None, :].expand(B, T_a_in, -1).reshape(B * T_a_in, -1)  # [B*T, D]

        dep_inputs = []
        for k in range(self.n_q):
            level_ctx = torch.cat([flat_out, self.dep_level_emb[:, k:k+1, :].expand_as(flat_out)], dim=-1)
            if k == 0:
                dep_inputs.append(self.dep_in[k](level_ctx))
            else:
                dep_inputs.append(self.dep_in[k](level_ctx) + self.dep_emb[k-1](flat_tgt[:, k-1]))
        
        d_out = self.dep_transformer(torch.stack(dep_inputs, 2).view(B * T_a_in, self.n_q, -1))
        if dep_prs is not None:
            d_out = self.dep_prosody_adaLN(d_out, dep_prs)
        if dep_spk is not None:
            d_out = self.dep_speaker_adaLN(d_out, dep_spk)
        logits = torch.stack([self.dep_layers[k](d_out[:, k, :]).view(B, T_a_in, self.card + 3) for k in range(self.n_q)], 0)
        # Predict continuous acoustic latents (for auxiliary L1 loss against mimi.decode_latent output)
        latent_feat = d_out.view(B, T_a_in, self.n_q, -1).mean(dim=2)
        latent_pred = self.latent_regressor(latent_feat[:, 1:]).transpose(1, 2)
        return logits, a_tgt, latent_pred
