import torch
import torch.nn as nn
import torch.nn.functional as F
from seedvox.modules.moshi.modules.transformer import StreamingTransformer, create_norm_fn

class ScaledEmbedding(nn.Embedding):
    def __init__(self, *args, norm: bool = False, zero_idx: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = create_norm_fn("layer_norm", self.embedding_dim) if norm else None
        self.zero_idx = zero_idx
    def forward(self, input, *args, **kwargs):
        input_clamped = input.clamp(min=0)
        y = super().forward(input_clamped, *args, **kwargs)
        if self.norm is not None: y = self.norm(y)
        if self.zero_idx >= 0:
            is_zero = (input == self.zero_idx)
            y = y.masked_fill(is_zero.unsqueeze(-1), 0.0)
        return y

class ResidualConvPrenet(nn.Module):
    def __init__(self, dim, num_layers=3, kernel_size=5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2),
                nn.GELU(),
                nn.GroupNorm(1, dim),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
    def forward(self, x):
        x = x.transpose(1, 2)
        for f in self.layers: x = x + f(x)
        return x.transpose(1, 2)

class TextTransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, hidden_scale=4.0, causal=False):
        super().__init__()
        self.conv_prenet = ResidualConvPrenet(dim)
        hs = int(hidden_scale * dim)
        self.transformer = StreamingTransformer(dim, num_heads, num_layers, hs, causal=causal, positional_embedding="sin_rope", context=1024)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, mask=None, positions=None, length_scale=1.0):
        x = self.conv_prenet(x)
        out = self.transformer(x, attn_mask=mask, positions=positions, length_scale=length_scale)
        return self.norm(out)

class SpeakerEncoder(nn.Module):
    def __init__(self, dim, num_latents=16, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(0.1))
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, key_padding_mask=None):
        B = x.shape[0]
        latents = self.latents.expand(B, -1, -1)
        res, _ = self.attn(latents, x, x, key_padding_mask=key_padding_mask)
        latents = self.norm(latents + res)
        latents = self.norm2(latents + self.ff(latents))
        return self.proj(latents)

class ProsodyEncoder(nn.Module):
    def __init__(self, dim, num_latents=16, num_heads=8, stride=2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=3, padding=1, stride=stride), nn.GELU(), nn.GroupNorm(1, dim))
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(0.1))
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, key_padding_mask=None):
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        if key_padding_mask is not None:
            m = key_padding_mask.float().unsqueeze(1)
            m = F.interpolate(m, size=x.shape[1], mode='nearest').squeeze(1).bool()
            key_padding_mask = m
        latents = self.latents.expand(B, -1, -1)
        res, _ = self.attn(latents, x, x, key_padding_mask=key_padding_mask)
        latents = self.norm(latents + res)
        latents = self.norm2(latents + self.ff(latents))
        return self.proj(latents)

class MonotonicAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)

    def forward(self, x, mask=None, kv_input=None, kv_mask=None, mono_bias=None, precomputed_kv=None):
        B, T, C = x.shape
        if kv_input is not None:
            qkv_q = self.qkv(x)
            q = qkv_q.reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)[0]
            k_v = F.linear(kv_input, self.qkv.weight[C:], self.qkv.bias[C:]).reshape(B, kv_input.shape[1], 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = k_v[0], k_v[1]
        else:
            qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
        if q.dtype != torch.float32:
            # Clean SDPA path for BF16/FP16 (FlashAttention)
            attn_mask = None
            if mask is not None or kv_mask is not None or mono_bias is not None:
                # Fallback to manual if we have complex masks and want to avoid SDPA overhead
                # for single-token generation.
                if T == 1:
                    attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
                    if mono_bias is not None: attn = attn + mono_bias.unsqueeze(1)
                    if kv_mask is not None: attn = attn.masked_fill(kv_mask.view(B, 1, 1, -1), float('-inf'))
                    out = F.softmax(attn, dim=-1) @ v
                else:
                    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            else:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            # Fast manual path for FP32
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            if mono_bias is not None: 
                if mono_bias.shape[-1] < attn.shape[-1]:
                    pad_len = attn.shape[-1] - mono_bias.shape[-1]
                    mono_bias = F.pad(mono_bias, (pad_len, 0), value=-100.0)
                attn = attn + mono_bias.unsqueeze(1)
            if mask is not None: attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            if kv_mask is not None: attn = attn.masked_fill(kv_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            out = F.softmax(attn, dim=-1) @ v
            
        return self.proj(out.transpose(1, 2).reshape(B, T, C))

class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.dim = dim
    def forward(self, x, emb):
        gamma, beta = self.mlp(emb).chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

class JEPA_Block(nn.Module):
    def __init__(self, dim, heads, hidden_scale):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ada_ln = AdaLN(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * hidden_scale)), nn.GELU(), nn.Linear(int(dim * hidden_scale), dim)
        )
    def forward(self, x, cond):
        res = x
        x = self.norm1(x)
        x = self.attn(x, x, x, need_weights=False)[0] + res
        x = self.ada_ln(x, cond) 
        x = x + self.ff(self.norm2(x))
        return x

class DiTBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim, elementwise_affine=False), MonotonicAttention(dim, heads)
        self.norm_cross, self.cross_attn = nn.LayerNorm(dim, elementwise_affine=False), MonotonicAttention(dim, heads)
        self.norm2, self.mlp = nn.LayerNorm(dim, elementwise_affine=False), nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.constant_(self.adaLN_modulation[1].weight, 0); nn.init.constant_(self.adaLN_modulation[1].bias, 0)
        with torch.no_grad(): self.adaLN_modulation[1].bias[5 * dim : 6 * dim] = 1.0 
    def forward(self, x, cond_emb, kv_ctx=None, mask=None, kv_mask=None, mono_bias=None):
        mods = self.adaLN_modulation(cond_emb).chunk(9, dim=1)
        shift_msa, scale_msa, gate_msa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = mods
        x = x + gate_msa.unsqueeze(1) * self.attn(self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1), mask=mask)
        if kv_ctx is not None: x = x + gate_ca.unsqueeze(1) * self.cross_attn(self.norm_cross(x) * (1 + scale_ca.unsqueeze(1)) + shift_ca.unsqueeze(1), kv_input=kv_ctx, kv_mask=kv_mask, mono_bias=mono_bias)
        return x + gate_mlp.unsqueeze(1) * self.mlp(self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))

class DurationPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=3, padding=1), nn.GELU(), nn.Conv1d(dim, dim, kernel_size=3, padding=1), nn.GELU())
        self.linear = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
    def forward(self, x, mask_inv):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.linear((x * mask_inv).sum(dim=1) / mask_inv.sum(dim=1).clamp(min=1)).squeeze(-1)

class ProsodyBottleneck(nn.Module):
    def __init__(self, dim, num_prosody_tokens=32):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_prosody_tokens
        self.refiner = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, dim),
            nn.GELU()
        )
    def forward(self, x, mask=None):
        B, C, T = x.shape
        x = self.refiner(x)
        if mask is not None:
            valid_mask = (~mask).unsqueeze(1).float()
            m = (x * valid_mask).sum(dim=-1, keepdim=True) / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            x = x - m
        else:
            x = x - x.mean(dim=-1, keepdim=True)
        x = F.adaptive_avg_pool1d(x, self.num_tokens)
        return x.transpose(1, 2)

class SpeakerAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim * 2))
    def forward(self, speaker_vector):
        out = self.net(speaker_vector)
        scale, shift = out.chunk(2, dim=-1)
        return scale + 1.0, shift

class ContrastiveDurationPhonemeHead(nn.Module):
    def __init__(self, dim, phoneme_vocab_size, temperature=0.07):
        super().__init__()
        self.dim = dim
        self.temp = temperature
        self.phoneme_vocab_size = phoneme_vocab_size
        self.phoneme_protos = nn.Embedding(phoneme_vocab_size, dim)
        nn.init.normal_(self.phoneme_protos.weight, std=0.02)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dur_predictor = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dim, 1)
        )
    def forward(self, text_feat, text_lens, phoneme_targets=None):
        B, T, dim = text_feat.shape
        q = F.normalize(self.proj(text_feat), dim=-1)
        protos = F.normalize(self.phoneme_protos.weight, dim=-1)
        phoneme_logits = torch.matmul(q, protos.T) / self.temp
        contrastive_loss = None
        if phoneme_targets is not None:
            contrastive_loss = F.cross_entropy(phoneme_logits.view(-1, self.phoneme_vocab_size), phoneme_targets.view(-1), ignore_index=0)
        log_durations = self.dur_predictor(text_feat).squeeze(-1)
        return contrastive_loss, log_durations, phoneme_logits
    def upsample_to_audio(self, text_feat, log_durations, text_lens, target_audio_len):
        B, T_text, dim = text_feat.shape
        durations = torch.round(F.softplus(log_durations)).long().clamp(min=1)
        mask = torch.arange(T_text, device=text_feat.device).unsqueeze(0) < text_lens.unsqueeze(1)
        upsampled_list = []
        for b in range(B):
            d = durations[b][mask[b]]
            t = text_feat[b][mask[b]]
            upsampled = torch.repeat_interleave(t, d, dim=0)
            curr_len = upsampled.shape[0]
            if curr_len > target_audio_len:
                upsampled = upsampled[:target_audio_len]
            elif curr_len < target_audio_len:
                upsampled = F.pad(upsampled, (0, 0, 0, target_audio_len - curr_len))
            upsampled_list.append(upsampled)
        return torch.stack(upsampled_list, dim=0), durations

class PhoneticBypassHead(nn.Module):
    def __init__(self, dim, phoneme_vocab_size):
        super().__init__()
        self.dim = dim
        self.dur_predictor = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dim, 1)
        )
    def forward(self, text_feat, text_lens, phoneme_targets=None):
        B, T, dim = text_feat.shape
        log_durations = self.dur_predictor(text_feat).squeeze(-1)
        return torch.tensor(0.0, device=text_feat.device), log_durations, torch.zeros((B, T, 1), device=text_feat.device)
    def upsample_to_audio(self, text_feat, log_durations, text_lens, target_audio_len):
        B, T_text, dim = text_feat.shape
        durations = torch.round(F.softplus(log_durations)).long().clamp(min=1)
        mask = torch.arange(T_text, device=text_feat.device).unsqueeze(0) < text_lens.unsqueeze(1)
        upsampled_list = []
        for b in range(B):
            d = durations[b][mask[b]]
            t = text_feat[b][mask[b]]
            upsampled = torch.repeat_interleave(t, d, dim=0)
            curr_len = upsampled.shape[0]
            if curr_len > target_audio_len:
                upsampled = upsampled[:target_audio_len]
            elif curr_len < target_audio_len:
                upsampled = F.pad(upsampled, (0, 0, 0, target_audio_len - curr_len))
            upsampled_list.append(upsampled)
        return torch.stack(upsampled_list, dim=0), durations

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
