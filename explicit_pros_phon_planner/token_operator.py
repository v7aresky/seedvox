import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenOperator(nn.Module):
    """Decoupled prosody knob.

    Rewrites a 16-codebook token sequence so the DECODED audio matches a target
    prosody latent (the frozen stage-1 codec space, [B, blocks, plan_dim]).

    Trained offline on (original_tokens, shifted_target_latent) -> shifted_tokens
    pairs generated from real pitch-shifted audio. It is a post-hoc edit model:
    it never participates in the AR model's graph and never fights its signal,
    so it can only ADD the prosody delta requested by the injected plan.

    Forward is teacher-forced (all codebooks, all frames in parallel):
      tokens [B, n_q, T], plan_latent [B, blocks, plan_dim] (or [B, plan_dim])
      -> logits [B, n_q, T, card]
    """
    def __init__(self, n_q=16, card=2048, d=128, plan_dim=512, plan_blocks=32,
                 hidden=128, blocks=6, kernel=5, dropout=0.0,
                 num_style_tokens=16, use_style=True, id_scale=3.0):
        super().__init__()
        self.n_q, self.card, self.d = n_q, card, d
        self.use_style = use_style
        self.id_scale = id_scale
        self.codebook_embs = nn.ModuleList([nn.Embedding(card, d) for _ in range(n_q)])

        # StyleTTS2/GST-style: the plan latent (our JEPA/prosody-codec space) attends
        # over a learned style-token bank; the pooled style FiLM-modulates the plan
        # control (gamma/beta). Zero/one-init so it is neutral at start.
        self.num_style_tokens = num_style_tokens
        self.style_tokens = nn.Parameter(torch.randn(num_style_tokens, d) * 0.02)
        self.style_query = nn.Sequential(nn.LayerNorm(plan_dim), nn.Linear(plan_dim, d))
        self.style_gamma = nn.Linear(d, d)
        self.style_beta = nn.Linear(d, d)
        with torch.no_grad():
            self.style_gamma.weight.zero_()
            self.style_gamma.bias.fill_(1.0)
            self.style_beta.weight.zero_()
            self.style_beta.bias.zero_()
        self.plan_proj = nn.Sequential(
            nn.LayerNorm(plan_dim), nn.Linear(plan_dim, d), nn.SiLU())
        self.plan_temporal = nn.Sequential(
            nn.Conv1d(d, d, 3, padding=1), nn.SiLU(),
            nn.Conv1d(d, d, 3, padding=1), nn.SiLU(),
        )
        self.in_proj = nn.Linear(d + d, hidden)
        self.hidden = hidden
        ch = n_q * hidden
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(nn.ModuleList([
                nn.Conv1d(ch, ch, kernel, padding=kernel // 2, groups=n_q),
                nn.GroupNorm(n_q, ch),
                nn.SiLU(),
                nn.Conv1d(ch, ch, 1),
                nn.Dropout(dropout),
            ]))
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, card)

        # Auxiliary pitch readout on the post-FiLM plan control: predicts the
        # target centered log-F0 trace from ctl. Gives the plan-conditioning path
        # (plan_proj/plan_temporal/FiLM/style bank) a strong direct gradient so it
        # actually learns the warp scale instead of letting the token path win by
        # default (the F0 gradient reaching the plan path is ~100x weaker than the
        # token path). Frozen at inference; only used for the aux training loss.
        self.plan_f0_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))

        # Direct plan->logits grip. Runs 1-6a showed the plan path (plan_proj ->
        # in_proj -> shared conv blocks -> out_proj) is ~100x weaker than the token
        # path, so every training trick either hit the weak-FiLM gradient ceiling or
        # depended on exemplar targets that never transferred. Here the plan control
        # gets a second, UNATTENUATED route straight into the output logits: a per-
        # frame bias b_t = MLP(ctl_t) in codebook-embedding space, applied as
        #     logits += <W_j, b_t>   (per codebook level j)
        # The gradient through this path has the SAME strength as the token path
        # (both land on logits -> CE), so the plan can actually push the distribution
        # toward tokens that realize the scaled pitch trace. Zero-init so it is
        # neutral at start and does not disturb the identity-CE anchor.
        self.plan_logit_bias = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        with torch.no_grad():
            self.plan_logit_bias[-1].weight.zero_()
            self.plan_logit_bias[-1].bias.zero_()

    def _plan_ctl(self, plan_latent, mask, T):
        """Per-frame control [B, T, d]. The 32 block vectors live in prosody time;
        the excursion SHAPE is carried by the block-to-block pattern (each block is
        centered, so a single pooled vector would throw the excursion away). We
        project each block then interpolate up to the T token frames."""
        if plan_latent.dim() == 2:
            v = plan_latent.unsqueeze(1).expand(-1, T, -1)
        else:
            v = plan_latent
        v = self.plan_proj(v)                                  # [B, blocks, d]
        v = v.transpose(1, 2)                                  # [B, d, blocks]
        v = self.plan_temporal(v)                              # temporal conv over blocks
        v = F.interpolate(v, size=T, mode='linear', align_corners=False)
        return v.transpose(1, 2)                               # [B, T, d]

    def _style(self, plan_latent):
        """GST style code [B, d] -> (gamma, beta) [B, d] FiLM on the plan control."""
        if not self.use_style:
            return None, None
        q = self.style_query(plan_latent).mean(dim=1)          # [B, d]
        attn = F.softmax(q @ self.style_tokens.T / (self.d ** 0.5), dim=-1)  # [B, n_style]
        style = attn @ self.style_tokens                       # [B, d]
        return self.style_gamma(style), self.style_beta(style)

    def forward(self, tokens, plan_latent, plan_mask=None, return_ctl=False,
                lock_levels=None):
        B, n_q, T = tokens.shape
        emb = torch.stack(
            [self.codebook_embs[j](tokens[:, j]) for j in range(n_q)], dim=1)
        ctl = self._plan_ctl(plan_latent, plan_mask, T)
        gamma, beta = self._style(plan_latent)
        if gamma is not None:
            ctl = ctl * gamma.unsqueeze(1) + beta.unsqueeze(1)
        ctl = ctl.unsqueeze(1).expand(B, n_q, T, self.d)
        x = self.in_proj(torch.cat([emb, ctl], dim=-1))          # [B,n_q,T,hidden]
        x = x.permute(0, 1, 3, 2).reshape(B, n_q * self.hidden, T)
        for (conv_dw, gn, act, conv_pw, drop) in self.blocks:
            h = x
            h = drop(act(gn(conv_dw(h))))
            h = conv_pw(h)
            x = x + h
        x = x.reshape(B, n_q, self.hidden, T).permute(0, 1, 3, 2)  # [B,n_q,T,hidden]
        x = self.norm(x)
        logits = self.out_proj(x)
        # Copy-prior (identity residual): a cosine-like kernel that peaks at the
        # INPUT codes. Runs 1-6b all converged to the same attractor: random-init
        # destroyed content in the first steps (ident -> 0) and the weak identity-CE
        # (0.01-0.05) could never pull the model back, so it froze in a bad basin
        # whose decoded excursion anti-scales (exc*s == const, slope == 0). This base
        # term makes the operator an EDIT model: at init it outputs the input tokens
        # and the plan delta must push a ~O(1) logit shift on top (same order as the
        # bias path), so content is preserved by construction while the plan edits
        # levels 1..15 to realize the scale. /d keeps the kernel magnitude ~1.
        logits = logits + self.id_scale * torch.stack(
            [emb[:, j] @ self.codebook_embs[j].weight.T / self.d
             for j in range(n_q)], dim=1)
        # Direct plan->logits grip (full-strength gradient, see plan_logit_bias).
        b = self.plan_logit_bias(ctl[:, 0])                       # [B, T, d]
        logits = logits + torch.stack(
            [b @ self.codebook_embs[j].weight.T for j in range(n_q)], dim=1)
        if lock_levels is not None:
            # Structurally pin locked codebook rows to the input token (copy kernel
            # only, no net/bias delta): the operator can only edit the allowed rows.
            # Prevents the artifact cheat of corrupting high residual levels (9..15)
            # to fool the frozen pitch estimator (run 6g was unintelligible).
            for j in lock_levels:
                logits[:, j] = self.id_scale * (
                    emb[:, j] @ self.codebook_embs[j].weight.T / self.d)
        if return_ctl:
            return logits, ctl[:, 0]                              # [B, T, d]
        return logits

    @torch.no_grad()
    def edit(self, tokens, plan_latent, plan_mask=None, lock_c0=True,
             edit_levels=None):
        """Greedy argmax edit: tokens [B, n_q, T] -> edited tokens [B, n_q, T].
        With lock_c0, the semantic codebook (row 0) is copied from the input so
        content is preserved by construction (prosody lives in codebooks 1..15).
        edit_levels: restrict which codebook rows may change (others are copied
        verbatim from the input). Diagnostic for localizing quality-corrupting
        levels: artifacts come from the operator drifting too many acoustic rows."""
        self.eval()
        locked = [j for j in range(self.n_q)
                  if (edit_levels is not None and j not in edit_levels)]
        logits = self.forward(tokens, plan_latent, plan_mask,
                              lock_levels=(locked or None))
        out = logits.argmax(-1)
        if lock_c0:
            out[:, 0] = tokens[:, 0]
        if edit_levels is not None:
            for j in [j for j in range(self.n_q) if j not in edit_levels]:
                out[:, j] = tokens[:, j]
        return out
