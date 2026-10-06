"""Soft-mixture mimi decode: differentiable decoding of soft token logits.

Hard c0 (content anchor) + soft codebooks 1..15 -> waveform, with gradients
flowing to the logits. Used as the all-differentiable audio path for the token
operator training (identity CE anchors content; a frozen F0 estimator scores the
decoded audio against the warped prosody target).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMimi(nn.Module):
    def __init__(self, mimi):
        super().__init__()
        self.m = mimi
        for p in mimi.parameters():
            p.requires_grad = False
        rf, rr = mimi.quantizer.rvq_first, mimi.quantizer.rvq_rest
        self.rf, self.rr = rf, rr
        # The codec uses 16 of the 32 quantizer layers: 1 semantic + 15 acoustic.
        self.layers = [rf.vq.layers[0]] + list(rr.vq.layers)[:15]
        assert len(self.layers) == 16, f"expected 16 codebooks, got {len(self.layers)}"

    def _quantize(self, logits, c0_hard=None, ste=False):
        """logits [B, K, T, card] -> quantized latent [B, 512, T].

        ste=False: soft mixture (P = softmax), smooth gradients everywhere.
        ste=True:  hard forward (P = one-hot argmax, i.e. the real mimi decode)
                   with the softmax gradient (straight-through). The loss is then
                   scored on the true decoded audio, closing the soft/hard gap."""
        B, K, T, card = logits.shape
        z_first = z_rest = None
        for j in range(K):
            E = self.layers[j]._codebook.embedding            # [card, 256]
            if j == 0 and c0_hard is not None:
                P = F.one_hot(c0_hard.long(), card).float()   # [B, T, card]
            elif ste:
                P_soft = F.softmax(logits[:, j], dim=-1)
                P_hard = F.one_hot(logits[:, j].argmax(-1), card).float()
                P = P_soft + (P_hard - P_soft).detach()
            else:
                P = F.softmax(logits[:, j], dim=-1)
            q = (P @ E).transpose(1, 2)                       # [B, 256, T]
            if j == 0:
                z_first = q
            elif j == 1:
                z_rest = q
            else:
                z_rest = z_rest + q
        z = self.rf.output_proj(z_first)
        if z_rest is not None:
            z = z + self.rr.output_proj(z_rest)
        return z

    def _waveform(self, z):
        e = self.m._to_encoder_framerate(z)
        (e,) = self.m.decoder_transformer(e)
        return self.m.decoder(e)

    def soft_decode(self, logits, c0_hard=None, ste=False):
        """logits [B, K, T, card] -> wav [B, 1, T*1920]. Differentiable in logits."""
        z = self._quantize(logits, c0_hard, ste=ste)
        return self._waveform(z)

    @torch.no_grad()
    def hard_decode(self, codes):
        """codes [B, K, T] -> wav [B, 1, T*1920] (exact mimi decode)."""
        return self.m.decode(codes)
