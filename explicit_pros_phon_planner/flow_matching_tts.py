"""Pitch-conditioned TTS via Flow Matching on Mimi continuous latents.

Architecture (inspired by Pocket-TTS/CALM):
    1. Mimi encoder (frozen) extracts continuous latents [B, 512, T]
    2. Text encoder embeds phoneme tokens [B, L, D]
    3. Pitch encoder embeds F0 values [B, T, D]
    4. Flow matching transformer generates latents conditioned on text + pitch
    5. Mimi decoder (frozen) converts latents back to audio

Key insight: Continuous latents preserve full information without quantization bottleneck.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    """Adaptive Layer Normalization with time conditioning."""
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
    def forward(self, x, emb):
        gamma, beta = self.mlp(emb).chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with self-attention and cross-attention."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)
        with torch.no_grad():
            self.adaLN_modulation[1].bias[5 * dim : 6 * dim] = 1.0
    
    def forward(self, x, cond_emb, kv_ctx=None, kv_mask=None):
        mods = self.adaLN_modulation(cond_emb).chunk(9, dim=1)
        shift_msa, scale_msa, gate_msa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = mods
        
        # Self-attention
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1),
            self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1),
            self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        )[0]
        
        # Cross-attention (if context provided)
        if kv_ctx is not None:
            x = x + gate_ca.unsqueeze(1) * self.cross_attn(
                self.norm_cross(x) * (1 + scale_ca.unsqueeze(1)) + shift_ca.unsqueeze(1),
                kv_ctx, kv_ctx, key_padding_mask=kv_mask
            )[0]
        
        # Feedforward
        x = x + gate_mlp.unsqueeze(1) * self.ff(
            self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        )
        return x


class PitchEncoder(nn.Module):
    """Encode F0 values into continuous embeddings."""
    def __init__(self, dim, fmin=50.0, fmax=8000.0):
        super().__init__()
        self.dim = dim
        self.fmin, self.fmax = fmin, fmax
        # Sinusoidal positional encoding for log-F0
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, f0_hz):
        """f0_hz [B, T] in Hz -> [B, T, D]"""
        # Convert to log scale and normalize
        log_f0 = torch.log(f0_hz.clamp(min=self.fmin, max=self.fmax))
        log_f0_norm = (log_f0 - math.log(self.fmin)) / (math.log(self.fmax) - math.log(self.fmin))
        
        # Sinusoidal encoding
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=f0_hz.device) * -emb)
        emb = log_f0_norm.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        
        return self.proj(emb)


class FlowMatchingTTS(nn.Module):
    """Pitch-conditioned TTS via Flow Matching on continuous latents.
    
    Args:
        dim: Model dimension
        num_layers: Number of transformer blocks
        heads: Number of attention heads
        dim_head: Dimension per head
        latent_dim: Dimension of Mimi continuous latents (512)
        text_vocab_size: Vocabulary size for text tokenizer
        text_dim: Dimension of text embeddings
        dropout: Dropout rate
        fmin: Minimum F0 for pitch encoding
        fmax: Maximum F0 for pitch encoding
    """
    def __init__(
        self,
        dim=512,
        num_layers=12,
        heads=8,
        dim_head=64,
        latent_dim=512,
        text_vocab_size=256,
        text_dim=256,
        dropout=0.1,
        fmin=50.0,
        fmax=8000.0
    ):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        
        # Text embedding
        self.text_emb = nn.Embedding(text_vocab_size, text_dim)
        self.text_proj = nn.Linear(text_dim, dim)
        
        # Pitch encoder
        self.pitch_encoder = PitchEncoder(dim, fmin, fmax)
        
        # Time embedding (for flow matching)
        self.time_emb = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, dim)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads, dim_head, dropout) for _ in range(num_layers)
        ])
        
        # Output projection
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, latent_dim)
        
        # Initialize output projection to zero (DiT-style)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def sinusoidal_embedding(self, t, dim):
        """Sinusoidal positional encoding for time step t."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
    
    def forward(self, x_t, t, text_tokens, text_mask, f0_hz, f0_mask):
        """
        Args:
            x_t: Noisy latents [B, T, D]
            t: Time step [B, 1] in [0, 1]
            text_tokens: Text token IDs [B, L]
            text_mask: Text padding mask [B, L] (True = padded)
            f0_hz: F0 in Hz [B, T]
            f0_mask: F0 padding mask [B, T] (True = padded)
        
        Returns:
            v_pred: Predicted velocity [B, T, D]
        """
        B, T, D = x_t.shape
        
        # Encode text
        text_emb = self.text_proj(self.text_emb(text_tokens))  # [B, L, dim]
        
        # Encode pitch
        pitch_emb = self.pitch_encoder(f0_hz)  # [B, T, dim]
        
        # Time embedding
        t_emb = self.time_emb(self.sinusoidal_embedding(t.squeeze(-1), self.dim))  # [B, dim]
        
        # Input projection
        h = self.input_proj(x_t) + pitch_emb  # [B, T, dim]
        
        # Process through DiT blocks
        for block in self.blocks:
            h = block(h, t_emb, kv_ctx=text_emb, kv_mask=text_mask)
        
        # Output projection
        h = self.out_norm(h)
        v_pred = self.out_proj(h)
        
        return v_pred
    
    def compute_loss(self, x_0, text_tokens, text_mask, f0_hz, f0_mask):
        """
        Compute flow matching loss.
        
        Args:
            x_0: Target latents [B, T, D]
            text_tokens: Text token IDs [B, L]
            text_mask: Text padding mask [B, L]
            f0_hz: F0 in Hz [B, T]
            f0_mask: F0 padding mask [B, T]
        
        Returns:
            loss: Scalar loss
            metrics: Dict of metrics
        """
        B, T, D = x_0.shape
        
        # Sample time step
        t = torch.rand(B, 1, device=x_0.device)
        
        # Sample noise
        x_1 = torch.randn_like(x_0)
        
        # Interpolate: x_t = (1 - t) * x_1 + t * x_0 (broadcast t over D)
        t_3d = t[:, :, None]  # [B, 1, 1]
        x_t = (1 - t_3d) * x_1 + t_3d * x_0
        
        # Target velocity: v = x_0 - x_1
        v_target = x_0 - x_1
        
        # Predict velocity
        v_pred = self.forward(x_t, t, text_tokens, text_mask, f0_hz, f0_mask)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss, {"flow_loss": loss.detach()}
    
    @torch.no_grad()
    def generate(
        self,
        text_tokens,
        text_mask,
        f0_hz,
        f0_mask,
        num_frames,
        n_steps=10,
        temp=0.7
    ):
        """
        Generate audio latents via flow matching sampling.
        
        Args:
            text_tokens: Text token IDs [B, L]
            text_mask: Text padding mask [B, L]
            f0_hz: Target F0 in Hz [B, T]
            f0_mask: F0 padding mask [B, T]
            num_frames: Number of frames to generate
            n_steps: Number of ODE steps
            temp: Temperature for noise sampling
        
        Returns:
            x_0: Generated latents [B, num_frames, D]
        """
        B = text_tokens.shape[0]
        device = text_tokens.device
        
        # Start from noise
        x_t = torch.randn(B, num_frames, self.latent_dim, device=device) * temp
        
        # Euler ODE solver
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B, 1), i * dt, device=device)
            v = self.forward(x_t, t, text_tokens, text_mask, f0_hz, f0_mask)
            x_t = x_t + v * dt
        
        return x_t
