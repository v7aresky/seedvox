import math
import torch
import torch.nn as nn
from seedvox.modules.components import AdaLN


class JEPAProsodyPlanner(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=4, num_prosody_tokens=32, hidden_scale=4.0,
                 learned_std=True, std_init=0.3, std_min=0.05, std_max=2.0):
        super().__init__()
        self.dim, self.num_tokens = dim, num_prosody_tokens
        self.prosody_queries = nn.Parameter(torch.randn(1, num_prosody_tokens, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * hidden_scale),
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))

        # Stochastic prosody: a learned per-element scale around the mean. Sampling at
        # inference makes the latent carry information the decoder cannot get from text
        # alone, so the decoder must learn to follow it (breaks the text-redundancy).
        self.learned_std = learned_std
        self.std_init = std_init
        self.std_min = std_min
        self.std_max = std_max
        self.std_head = None
        if learned_std:
            self.std_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            with torch.no_grad():
                self.std_head[0].weight.mul_(0.01)
                self.std_head[-1].weight.mul_(0.01)
                nn.init.constant_(self.std_head[-1].bias, math.log(std_init))

        # Style conditioning (plan-level): a zero-init AdaLN applied to the plan mean,
        # so style starts as identity (existing checkpoints stay valid) and early
        # gradients flow only into the style modules, not the planner.
        self.style_adaLN = AdaLN(dim)
        with torch.no_grad():
            nn.init.zeros_(self.style_adaLN.mlp[1].weight)
            nn.init.zeros_(self.style_adaLN.mlp[1].bias)

    def _encode(self, text_feat, text_mask):
        B = text_feat.shape[0]
        queries = self.prosody_queries.expand(B, -1, -1)
        x = torch.cat([queries, text_feat], dim=1)
        if text_mask is not None:
            q_mask = torch.zeros((B, self.num_tokens), device=text_mask.device, dtype=torch.bool)
            full_mask = torch.cat([q_mask, text_mask], dim=1)
        else:
            full_mask = None
        out = self.transformer(x, src_key_padding_mask=full_mask)
        return self.head(out[:, :self.num_tokens, :])

    def std(self, mean):
        if self.std_head is None:
            return torch.full_like(mean, self.std_init)
        log_std = self.std_head(mean)
        log_std = log_std.clamp(math.log(self.std_min), math.log(self.std_max))
        return log_std.exp()

    def forward(self, text_feat, text_mask=None, temperature=1.0, sample=False, style_emb=None):
        mean = self._encode(text_feat, text_mask)
        if style_emb is not None:
            mean = self.style_adaLN(mean, style_emb)
        if not sample:
            return mean
        return mean + (self.std(mean) * temperature) * torch.randn_like(mean)

    def std_reg(self, mean):
        """Push the sampled scale toward std_init so it neither collapses nor explodes."""
        if self.std_head is None:
            return torch.zeros((), device=mean.device)
        return (self.std(mean) - self.std_init).pow(2).mean()
