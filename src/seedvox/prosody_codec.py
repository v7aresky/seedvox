import torch
import torch.nn as nn
import torch.nn.functional as F


class ProsodyCodec(nn.Module):
    """Waveform-derived F0/energy/voicing -> K fixed block vectors (stage 1 codec).

    Encodes per-frame prosody features (12.5 Hz, aligned to Mimi frames) into
    K fixed-length block vectors via masked mean pooling, and reconstructs the
    per-frame features from those blocks (reconstruction loss regularizer).
    Trained standalone; frozen as the prosody teacher for stage 2.

    Input:  feats [B, T, feat_dim] = (log_f0_center, e_center, voicing)
    Output: z [B, K, dim] block vectors; rec [B, T, feat_dim] reconstruction.
    """
    def __init__(self, dim=512, num_blocks=32, feat_dim=3, hidden=256):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.feat_dim = feat_dim
        self.register_buffer('feat_std', torch.ones(feat_dim))

        self.enc = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.pool_proj = nn.Linear(hidden, dim)
        self.dec = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, feat_dim),
        )

    def encode(self, feats, lens=None):
        """feats [B,T,feat_dim] -> z [B, num_blocks, dim].

        Masked adaptive mean pooling over the padded batch grid (K evenly
        spaced blocks). Note: pooling includes the zero-padded region, so bins
        of short utterances blend real signal with zeros; this is intentional
        (probe-proven: it keeps the block->frame mapping consistent and trains
        far better than valid-region-only pooling). Use batch padding that is
        a multiple of num_blocks for consistent bins.
        """
        B, T, _ = feats.shape
        x = feats / self.feat_std.clamp(min=1e-4)
        h = self.conv(self.enc(x).permute(0, 2, 1))          # [B, hidden, T]
        z = F.adaptive_avg_pool1d(h, self.num_blocks)        # [B, hidden, K]
        z = self.pool_proj(z.permute(0, 2, 1))               # [B, K, dim]
        return z

    def decode(self, z, T):
        """z [B,K,dim] -> rec [B,T,feat_dim] (linear interpolate blocks to frames)."""
        B, K, _ = z.shape
        up = F.interpolate(z.transpose(1, 2), size=T, mode='linear')      # [B, dim, T]
        rec = self.dec(up.transpose(1, 2))        # [B, T, feat_dim]
        return rec

    def forward(self, feats, lens=None):
        z = self.encode(feats, lens)
        rec = self.decode(z, feats.shape[1])
        return z, rec
