import torch
import torch.nn as nn
import torch.nn.functional as F

class ProsodyBottleneck(nn.Module):
    """
    Extracts coarse prosody from Mimi latents.
    Goal: High temporal compression to remove phonetic detail while preserving rhythm/melody.
    """
    def __init__(self, dim, num_prosody_tokens=32):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_prosody_tokens
        
        # Compression layers: refine features before pooling
        self.refiner = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, dim),
            nn.GELU()
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, dim, T_audio] - Mimi encoder outputs (pre-quantization)
            mask: [B, T_audio] - Boolean mask (True for padding)
        Returns:
            prosody_latents: [B, num_tokens, dim]
        """
        B, C, T = x.shape
        
        # 1. Refine features
        x = self.refiner(x) # [B, dim, T]
        
        # 2. Mean centering to remove speaker bias (global offset)
        if mask is not None:
            # Masked mean
            # mask: [B, T], True for padding
            valid_mask = (~mask).unsqueeze(1).float() # [B, 1, T]
            m = (x * valid_mask).sum(dim=-1, keepdim=True) / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            x = x - m
        else:
            x = x - x.mean(dim=-1, keepdim=True)
            
        # 3. Adaptive pooling to fixed block size
        # This is the "JEPA block" - independent of audio length
        x = F.adaptive_avg_pool1d(x, self.num_tokens) # [B, dim, num_tokens]
        
        return x.transpose(1, 2) # [B, num_tokens, dim]
