import torch
import torch.nn as nn

class JEPAProsodyPlanner(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=4, num_prosody_tokens=32, hidden_scale=4.0):
        super().__init__()
        self.dim, self.num_tokens = dim, num_prosody_tokens
        self.prosody_queries = nn.Parameter(torch.randn(1, num_prosody_tokens, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * hidden_scale),
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))
    def forward(self, text_feat, text_mask=None):
        B = text_feat.shape[0]
        queries = self.prosody_queries.expand(B, -1, -1)
        x = torch.cat([queries, text_feat], dim=1)
        if text_mask is not None:
            q_mask = torch.zeros((B, self.num_tokens), device=text_mask.device, dtype=torch.bool)
            full_mask = torch.cat([q_mask, text_mask], dim=1)
        else: full_mask = None
        out = self.transformer(x, src_key_padding_mask=full_mask)
        return self.head(out[:, :self.num_tokens, :])
