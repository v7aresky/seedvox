import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
import os
from .base import JEPAProsodyBase, _sum_embeddings
from .planner import JEPAProsodyPlanner
from .prosody_utils import ProsodyBottleneck
from .prosody_codec import ProsodyCodec


def jepa_contrastive_loss(pred, gt, margin=0.2):
    """Batch-matched contrastive term: pull each pred toward its OWN gt latent and
    push it away from every OTHER text's gt latent (hard negatives from the batch).

    Fights the planner's mean-collapse: matched-vs-mismatched cosine gap grows, so
    plans become text-discriminative instead of hugging the average prosody direction.
    """
    B = pred.shape[0]
    if B < 2:
        return torch.zeros((), device=pred.device)
    S = F.cosine_similarity(pred.unsqueeze(1), gt.unsqueeze(0), dim=-1).mean(-1)  # [B, B]
    matched = S.diag().view(B, 1)
    hinge = F.relu(S - matched + margin)
    eye = torch.eye(B, dtype=torch.bool, device=pred.device)
    hinge = hinge.masked_fill(eye, 0.0)
    return hinge.sum() / (B * (B - 1))


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

class SpeakerAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim * 2))
    def forward(self, speaker_vector):
        out = self.net(speaker_vector)
        scale, shift = out.chunk(2, dim=-1)
        return scale + 1.0, shift

class JEPAProsodyHybridModel(JEPAProsodyBase):
    """
    SOTA Hybrid Model: JEPA-based Prosody Planning + v3 AR Generation.
    """
    def __init__(self, config, tokenizer_vocab_size, phoneme_vocab_size=128):
        super().__init__(config, tokenizer_vocab_size)
        
        # Override phoneme_vocab_size to 128 as required by the checkpoint
        self.phoneme_vocab_size = 128
        self.use_phonetic = config['model'].get('use_phonetic', True)
        if self.use_phonetic:
            self.phoneme_head = ContrastiveDurationPhonemeHead(self.dim, 128, temperature=config['model'].get('phoneme_temperature', 0.07))
        else:
            self.phoneme_head = PhoneticBypassHead(self.dim, 128)
        
        self.use_speaker = config['model'].get('use_speaker', True)
        self.use_prosody = config['model'].get('use_prosody', True)
        self.use_bpe_encoder = config['model'].get('bpe_encoder', {}).get('enabled', False)
        if self.use_bpe_encoder:
            from seedvox.bpe_char_encoder import get_bpe_encoder_from_config
            self.bpe_encoder = get_bpe_encoder_from_config(config)
        self._use_contrastive_duration = config['model'].get('use_contrastive_duration', True)
        
        cfg = config['model']
        self.jepa_planner = JEPAProsodyPlanner(
            dim=self.dim,
            num_heads=cfg.get('jepa_heads', 8),
            num_layers=cfg.get('jepa_layers', 4),
            num_prosody_tokens=cfg.get('num_prosody_tokens', 32),
            learned_std=cfg.get('planner_learned_std', True),
            std_init=cfg.get('planner_std_init', 0.3),
            std_min=cfg.get('planner_std_min', 0.05),
            std_max=cfg.get('planner_std_max', 2.0),
        )
        self.prosody_bottleneck = ProsodyBottleneck(
            dim=self.dim,
            num_prosody_tokens=cfg.get('num_prosody_tokens', 32)
        )
        # Stage-1 prosody codec (waveform-derived F0/E/voicing teacher). Frozen.
        self.prosody_codec = ProsodyCodec(
            dim=self.dim,
            num_blocks=cfg.get('num_prosody_tokens', 32)
        )
        self._load_prosody_codec(cfg)
        self.speaker_adapter = SpeakerAdapter(self.dim)
        self.jepa_weight = config['training'].get('jepa_weight', 1.0)
        self.MASK_ID = tokenizer_vocab_size + 3
        self.bpe_gate = nn.Parameter(torch.tensor([1.0]))
        
        # Null conditioning for CFG and robustness
        self.null_speaker = nn.Parameter(torch.randn(1, cfg.get('num_speaker_latents', 16), self.dim) * 0.02)
        self.null_prosody = nn.Parameter(torch.randn(1, cfg.get('num_prosody_tokens', 32), self.dim) * 0.02)
        self.null_text_feat = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)

        # ---- Discrete style tokens (PLAN-level conditioning) ----
        # Style now conditions the JEPA PROSODY PLAN (via a zero-init AdaLN inside
        # the planner), not the decoder: the decoder provably follows the plan, so a
        # style-specific plan is audible by construction. `style_bank` is a trainable
        # codebook used ONLY to pseudo-label the GT prosody with its nearest style id
        # (EM-style clustering); `style_head` predicts the style id from text so that
        # inference can run with no extra inputs (or a forced `external_style`).
        self.num_style_tokens = cfg.get('num_style_tokens', 16)
        self.use_style = cfg.get('use_style', True)
        self.style_drop = cfg.get('style_drop', 0.2)
        self.style_gt_prob = cfg.get('style_gt_prob', 0.85)
        self.style_anchor_weight = cfg.get('style_anchor_weight', 0.0)
        self.style_diversity_weight = cfg.get('style_diversity_weight', 0.0)
        self.style_emb = nn.Embedding(self.num_style_tokens, self.dim)
        nn.init.normal_(self.style_emb.weight, std=0.02)
        self.style_bank = nn.Parameter(torch.randn(self.num_style_tokens, self.dim) * 0.02)
        self.style_head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim),
            nn.GELU(), nn.Linear(self.dim, self.num_style_tokens))
        self._last_style_logits = None
        self._last_style_ids = None
        self._last_gt_style_ids = None
        self._last_style_anchor = None
        # Set by the trainer's eval loop so style metrics (CE / anchor) are also
        # stashed during eval forward passes (otherwise they only exist in train()).
        self.eval_stats = False

    def _load_prosody_codec(self, cfg):
        path = cfg.get('prosody_codec_path', 'checkpoints/prosody_codec.pt')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            self.prosody_codec.load_state_dict(sd, strict=False)
            print(f"Loaded prosody codec from {path}")
        else:
            print(f"WARNING: prosody codec checkpoint not found at {path}; codec stays untrained (random).")
        self.prosody_codec.eval()
        for p in self.prosody_codec.parameters():
            p.requires_grad = False

    def _style_id(self, style_ids, external_style, style_logits, B, device):
        if external_style is not None:
            if torch.is_tensor(external_style):
                return external_style.long().reshape(B).to(device)
            return torch.full((B,), int(external_style), device=device, dtype=torch.long)
        if style_ids is not None:
            return style_ids.long().reshape(B).to(device)
        if style_logits is not None:
            return style_logits.argmax(-1)
        return torch.zeros(B, dtype=torch.long, device=device)

    def _add_style_losses(self, jepa_loss, pred_prosody, gt_style_ids, skeep):
        """Anchor (plan mean -> style_bank centroid, masked by style dropout) +
        codebook diversity penalty. Both fold into jepa_loss; the anchor is also
        stashed (unweighted) so the trainer can log it. Both are self-gating."""
        if self.style_anchor_weight > 0 and gt_style_ids is not None:
            sim = F.cosine_similarity(pred_prosody.mean(dim=1), self.style_bank[gt_style_ids], dim=-1)
            loss1 = 1 - sim
            if skeep is not None:
                anchor = (loss1 * skeep.view(-1)).sum() / max(skeep.sum(), 1.0)
            else:
                anchor = loss1.mean()
            jepa_loss = jepa_loss + self.style_anchor_weight * anchor
            if self.training or self.eval_stats:
                self._last_style_anchor = anchor.detach()
        if self.style_diversity_weight > 0:
            # Scale-invariant (cosine) diversity: push style_bank rows toward
            # mutual orthogonality so the EM labeling keeps all codes distinct.
            b = F.normalize(self.style_bank, dim=-1)
            off = b @ b.T - torch.eye(self.num_style_tokens, device=self.style_bank.device)
            jepa_loss = jepa_loss + self.style_diversity_weight * off.pow(2).mean()
        return jepa_loss

    def encode_context(self, 
                       text: torch.Tensor, 
                       text_lens: torch.Tensor, 
                       audio_tokens: tp.Optional[torch.Tensor] = None, 
                       audio_lens: tp.Optional[torch.Tensor] = None, 
                       raw_texts: tp.Optional[tp.List[str]] = None, 
                       use_speaker: tp.Optional[bool] = None, 
                       use_prosody: tp.Optional[bool] = None, 
                       phoneme_ids: tp.Optional[torch.Tensor] = None, 
                       mimi_latents: tp.Optional[torch.Tensor] = None,
                       prosody_feats: tp.Optional[torch.Tensor] = None,
                       bpe_ids: tp.Optional[torch.Tensor] = None, 
                       bpe_lens: tp.Optional[torch.Tensor] = None, 
                       char_to_bpe: tp.Optional[torch.Tensor] = None, 
                       char_lens: tp.Optional[torch.Tensor] = None,
                       drop_prob: float = 0.0, 
                       external_speaker: tp.Optional[torch.Tensor] = None, 
                       external_prosody: tp.Optional[torch.Tensor] = None,
                       style_ids: tp.Optional[torch.Tensor] = None,
                       external_style: tp.Optional[torch.Tensor] = None,
                       text_feat: tp.Optional[torch.Tensor] = None,
                       exagg: float = 1.0,
                       prosody_temperature: tp.Optional[float] = None) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.Optional[torch.Tensor]]:
        
        B, device = text.shape[0], text.device
        if audio_tokens is not None: audio_tokens = audio_tokens[:, :self.n_q]
        
        if text_feat is None:
            # 1. Base Text Encoding
            text_feat, _ = self.encode_text(text, text_lens, raw_texts)
            
            # 2. BPE Enrichment
            if self.use_bpe_encoder and bpe_ids is not None:
                B_bpe, T_char = char_to_bpe.shape
                padded_c2b = torch.zeros((B, T_char + 2), dtype=char_to_bpe.dtype, device=device)
                padded_c2b[:, 1:1+T_char] = char_to_bpe
                wrapped_char_lens = (char_lens if char_lens is not None else text_lens) + 2
                
                bpe_ctx = self.bpe_encoder.forward_bpe(bpe_ids, bpe_lens, device=device)
                bpe_expanded = self.bpe_encoder.expand_to_chars(bpe_ctx, padded_c2b, wrapped_char_lens, device=device)
                
                if bpe_expanded.shape[1] < text_feat.shape[1]:
                    bpe_expanded = F.pad(bpe_expanded, (0, 0, 0, text_feat.shape[1] - bpe_expanded.shape[1]))
                elif bpe_expanded.shape[1] > text_feat.shape[1]:
                    bpe_expanded = bpe_expanded[:, :text_feat.shape[1]]
                text_feat = text_feat + torch.sigmoid(self.bpe_gate) * bpe_expanded
        else:
            text_feat = text_feat

        text_lens_wrapped = text_lens + 2

        # 3. Speaker / Prosody Extraction
        spk = None
        prs = None
        use_speaker = use_speaker if use_speaker is not None else self.use_speaker
        use_prosody = use_prosody if use_prosody is not None else self.use_prosody
        
        if external_speaker is not None:
            spk = external_speaker
        elif (use_speaker or use_prosody) and audio_tokens is not None and audio_lens is not None:
            ae = _sum_embeddings(self.audio_embs, audio_tokens, self.n_q)
            ae = self.audio_prenet(self.audio_norm(ae))
            mask = torch.arange(audio_tokens.shape[2], device=device).unsqueeze(0) >= audio_lens.unsqueeze(1)
            if use_speaker: spk = self.speaker_encoder(ae, key_padding_mask=mask)
        else:
            if use_speaker: spk = self.null_speaker.expand(B, -1, -1)

        # 4. JEPA Prosody Planning
        text_feat_text_rate = text_feat
        t_mask = torch.arange(text_feat_text_rate.shape[1], device=device).unsqueeze(0) >= text_lens_wrapped.unsqueeze(1)
        temp = self.cfg.get('prosody_temperature', 1.0) if prosody_temperature is None else prosody_temperature

        # 4a. GT prosody latent (stage-1 codec teacher / Mimi bottleneck). STRICTLY
        # NO GRAD / DETACHED (both are frozen). Also yields the discrete GT style id.
        jepa_loss = None
        gt_prosody = None
        if external_prosody is not None:
            current_prosody = external_prosody
        elif prosody_feats is not None:
            with torch.no_grad():
                gt_prosody = self.prosody_codec.encode(prosody_feats).detach()
        elif mimi_latents is not None:
            with torch.no_grad():
                a_mask = torch.arange(mimi_latents.shape[2], device=device).unsqueeze(0) >= audio_lens.unsqueeze(1)
                gt_prosody = self.prosody_bottleneck(mimi_latents, mask=a_mask).detach()
        else:
            gt_prosody = None

        # 4b. Style selection (PLAN-level conditioning). The style head always runs so
        # the style-CE loss can be scored against the GT style ids in the trainer; the
        # GT prosody is labeled with its nearest style_bank codebook id (EM-style), and
        # that id is teacher-forced with prob style_gt_prob during training. The chosen
        # id embeds into a [B, D] vector that conditions the planner's output.
        style_vec = None
        gt_style_ids = None
        skeep = None
        if self.use_style:
            style_logits = self.style_head(text_feat.mean(dim=1))     # [B, K]
            gt_style_ids = None
            if gt_prosody is not None:
                with torch.no_grad():
                    g = gt_prosody.mean(dim=1)                        # [B, D]
                    gt_style_ids = (g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6) @ self.style_bank.T).argmax(-1)
            if self.training and gt_style_ids is not None:
                use_gt = torch.rand(B, 1, device=device) < self.style_gt_prob
                style_id = torch.where(use_gt.squeeze(-1), gt_style_ids, style_logits.argmax(-1))
            else:
                style_id = self._style_id(style_ids, external_style, style_logits, B, device)
            style_vec = self.style_emb(style_id)                      # [B, D]
            skeep = None
            if self.training and self.style_drop > 0:
                skeep = (torch.rand(B, 1, device=device) > self.style_drop).float()
                style_vec = skeep * style_vec
            if self.training or self.eval_stats:
                # NOT detached: the trainer's style-CE backward must reach style_head
                # (style_id is chosen by non-diff argmax/teacher-forcing, so CE on the
                # logits is the only training signal for the text->style head).
                self._last_style_logits = style_logits
                self._last_style_ids = style_id.detach()
                self._last_gt_style_ids = gt_style_ids.detach() if gt_style_ids is not None else None
                self._last_style_anchor = None

        # 4c. Style-conditioned deterministic plan mean (JEPA loss + planning metrics).
        pred_prosody = self.jepa_planner(text_feat_text_rate, text_mask=t_mask, style_emb=style_vec)

        if external_prosody is not None:
            current_prosody = external_prosody
        elif prosody_feats is not None:
            # Stage-1 codec teacher: GT prosody = codec.encode(waveform-derived
            # F0/E/voicing).
            jepa_loss = 1 - F.cosine_similarity(pred_prosody, gt_prosody, dim=-1).mean()
            cw = self.cfg.get('planner_contrastive_weight', 0.0)
            if cw > 0:
                jepa_loss = jepa_loss + cw * jepa_contrastive_loss(
                    pred_prosody, gt_prosody, self.cfg.get('planner_contrastive_margin', 0.2))
            aw = self.cfg.get('planner_anchor_weight', 0.0)
            if aw > 0:
                # Magnitude anchor: cosine ignores scale, so anchor pred to the GT latent
                # magnitude to fight mean-collapse (preds hugging the average direction).
                jepa_loss = jepa_loss + aw * F.mse_loss(pred_prosody, gt_prosody)
            sw = self.cfg.get('planner_std_weight', 0.0)
            if sw > 0:
                jepa_loss = jepa_loss + sw * self.jepa_planner.std_reg(pred_prosody)
            jepa_loss = self._add_style_losses(jepa_loss, pred_prosody, gt_style_ids, skeep)
            # Teacher-force the decoder with GT prosody so it LEARNS to use the latent
            # (GT is not a deterministic function of text -> not redundant). See
            # _decoder_prosody for gt/mix/pred modes.
            current_prosody = self._decoder_prosody(gt_prosody, pred_prosody, text_feat_text_rate, t_mask, temp, style_emb=style_vec)
        elif mimi_latents is not None:
            # Compute JEPA Loss (Cosine similarity — focuses on prosody pattern, not magnitude)
            jepa_loss = 1 - F.cosine_similarity(pred_prosody, gt_prosody, dim=-1).mean()
            cw = self.cfg.get('planner_contrastive_weight', 0.0)
            if cw > 0:
                jepa_loss = jepa_loss + cw * jepa_contrastive_loss(
                    pred_prosody, gt_prosody, self.cfg.get('planner_contrastive_margin', 0.2))
            aw = self.cfg.get('planner_anchor_weight', 0.0)
            if aw > 0:
                jepa_loss = jepa_loss + aw * F.mse_loss(pred_prosody, gt_prosody)
            sw = self.cfg.get('planner_std_weight', 0.0)
            if sw > 0:
                jepa_loss = jepa_loss + sw * self.jepa_planner.std_reg(pred_prosody)
            jepa_loss = self._add_style_losses(jepa_loss, pred_prosody, gt_style_ids, skeep)
            current_prosody = self._decoder_prosody(gt_prosody, pred_prosody, text_feat_text_rate, t_mask, temp, style_emb=style_vec)
        else:
            # Inference mode: SAMPLE the plan distribution so the latent carries
            # information the decoder cannot get from text alone, then apply the
            # exagg dial around the null latent.
            sampled_prosody = self.jepa_planner(text_feat_text_rate, text_mask=t_mask, temperature=temp, sample=True, style_emb=style_vec)
            if exagg != 1.0:
                current_prosody = self.null_prosody.expand(B, -1, -1) + exagg * (sampled_prosody - self.null_prosody.expand(B, -1, -1))
            else:
                current_prosody = sampled_prosody

        # 5. Conditioning Dropout
        if drop_prob > 0 and self.training:
            keep = (torch.rand(B, 1, 1, device=device) > drop_prob).float()
            if spk is not None:
                spk = keep * spk + (1 - keep) * self.null_speaker.expand(B, -1, -1)
            
            current_prosody = keep * current_prosody + (1 - keep) * self.null_prosody.expand(B, -1, -1)
            
            if torch.rand(1).item() < 0.2:
                text_feat = keep * text_feat + (1 - keep) * self.null_text_feat.expand(B, text_feat.shape[1], -1)

        # Stash the decoder's prosody conditioning (post-dropout) and the GT prosody
        # latent for the plan-follow (cycle) loss in trainer_fusion: it compares the
        # prosody of what the model GENERATES against the plan it was conditioned on.
        if self.training or self.eval_stats:
            self._last_plan_prosody = current_prosody.detach() if current_prosody is not None else None
            self._last_gt_prosody = gt_prosody

        # 6. Speaker Adaptation (FiLM) - modulate prosody features with speaker info
        # Done after dropout so spk_vec (used for AdaLN in decoder) also gets nulled during training
        speaker_vector = spk.mean(dim=1, keepdim=True) if spk is not None else torch.zeros(B, 1, self.dim, device=device)
        scale, shift = self.speaker_adapter(speaker_vector)
        adapted_prosody = current_prosody * scale + shift

        # 7. Final Context Assembly (no style token: style lives in the plan).
        ctx_parts = []
        if spk is not None: ctx_parts.append(spk)
        ctx_parts.append(adapted_prosody)
        ctx_parts.append(text_feat)
        context = torch.cat(ctx_parts, dim=1)
        
        ctx_mask = torch.zeros(B, context.shape[1], device=device, dtype=torch.bool)
        offset = (spk.shape[1] if spk is not None else 0) + adapted_prosody.shape[1]
        remaining = context.shape[1] - offset
        arange = torch.arange(remaining, device=device).unsqueeze(0)
        ctx_mask[:, offset:] = arange >= text_lens_wrapped.unsqueeze(1)
            
        # 8. Phonetic Head
        ph_logits = None
        contrastive_loss = None
        if self.use_phonetic:
            if phoneme_ids is not None:
                if phoneme_ids.shape[1] == text.shape[1]:
                    B_ph, T_ph = phoneme_ids.shape
                    padded_ph_ids = torch.zeros((B_ph, T_ph + 2), dtype=phoneme_ids.dtype, device=device)
                    padded_ph_ids[:, 1:1+T_ph] = phoneme_ids
                    phoneme_ids = padded_ph_ids
            
            contrastive_loss, _, ph_logits = self.phoneme_head(text_feat, text_lens_wrapped, phoneme_ids)
            
        return context, ctx_mask, ph_logits, jepa_loss, contrastive_loss, speaker_vector, adapted_prosody.mean(dim=1)

    def _decoder_prosody(self, gt_prosody, pred_prosody, text_feat, t_mask, temp, style_emb=None):
        """Which prosody latent the AR decoder is conditioned on during TRAINING.

        Feeding `pred` makes the latent text-redundant (pred is a deterministic
        function of text) so the decoder learns to ignore it. Feeding `gt` (from the
        frozen codec, not derivable from text) teaches the decoder the latent->audio
        mapping. `mix` schedules GT most of the time with a sampled-plan fraction so
        inference plans (imperfect) remain in-distribution.
        """
        source = self.cfg.get('decoder_prosody_source', 'pred')
        if source == 'gt':
            return gt_prosody
        if source == 'mix':
            B, device = gt_prosody.shape[0], gt_prosody.device
            p_gt = self.cfg.get('decoder_gt_prob', 0.85)
            use_gt = torch.rand(B, 1, 1, device=device) < p_gt
            sampled_pred = self.jepa_planner(text_feat, text_mask=t_mask, temperature=temp, sample=True, style_emb=style_emb)
            return torch.where(use_gt, gt_prosody, sampled_pred)
        return pred_prosody

    def mlm_forward(self, masked_text, text_lens, raw_texts=None, phoneme_ids=None,
                    bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None):
        context, ctx_mask, ph_logits, jepa_loss, contrastive_loss, _, _ = self.encode_context(
            masked_text, text_lens, raw_texts=raw_texts, phoneme_ids=phoneme_ids,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens
        )
        return None, ph_logits

    def forward(self, text, audio_tokens, text_lens, audio_lens, raw_texts=None, 
                use_speaker=None, use_prosody=None, phoneme_ids=None, mimi_latents=None,
                prosody_feats=None,
                bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None,
                drop_prob=0.0, style_ids=None, external_style=None):
        audio_tokens = audio_tokens[:, :self.n_q]
        context, ctx_mask, ph_logits, jepa_loss, contrastive_loss, spk_vec, prs_emb = self.encode_context(
            text, text_lens, audio_tokens, audio_lens, raw_texts, 
            use_speaker, use_prosody, phoneme_ids, mimi_latents=mimi_latents,
            prosody_feats=prosody_feats,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens,
            drop_prob=drop_prob, style_ids=style_ids, external_style=external_style
        )
        logits, targets, latent_pred = self.forward_with_context(context, ctx_mask, audio_tokens, audio_lens, speaker_emb=spk_vec, prosody_emb=prs_emb)
        return logits, targets, ph_logits, jepa_loss, contrastive_loss, latent_pred

    @torch.no_grad()
    def sample(self, text, text_lens, ref_audio=None, ref_lens=None, max_steps=1000, temp=0.1, curr_n_q=None, raw_texts=None, top_k=0, top_p=0.9, use_speaker=None, use_prosody=None, cfg_scale=1.0,
               bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None, phoneme_ids=None, drop_prob=0.0,
               external_speaker=None, external_prosody=None,
               precomputed_context=None, precomputed_mask=None,
               offset=None, spk_vec=None, min_p=0.0, rep_penalty=1.0, exagg=1.0, mono_slack=0.0,
               prosody_emb=None, prosody_temperature=None,
               style_ids=None, external_style=None):
        B, device = text.shape[0], text.device
        if curr_n_q is None: curr_n_q = self.n_q
        
        # 1. Encode Context
        if precomputed_context is not None:
            context, ctx_mask = precomputed_context, precomputed_mask
        else:
            context, ctx_mask, _, _, _, spk_vec, prs_emb = self.encode_context(
                text, text_lens, audio_tokens=ref_audio, audio_lens=ref_lens, 
                raw_texts=raw_texts, use_speaker=use_speaker, use_prosody=use_prosody,
                bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens,
                phoneme_ids=phoneme_ids,
                drop_prob=drop_prob,
                external_speaker=external_speaker,
                external_prosody=external_prosody,
                style_ids=style_ids,
                external_style=external_style,
                exagg=exagg,
                prosody_temperature=prosody_temperature
            )
            if prosody_emb is None:
                prosody_emb = prs_emb
        
        # Calculate text offset for CFG masking
        if offset is None:
            offset = 0
            if (use_speaker if use_speaker is not None else self.use_speaker):
                offset += self.cfg['num_speaker_latents']
            # JEPA prosody tokens are ALWAYS present in this model
            offset += self.cfg.get('num_prosody_tokens', 32)

        if cfg_scale != 1.0:
            uncond_mask = ctx_mask.clone()
            uncond_mask[:, offset:] = True
            context = torch.cat([context, context], dim=0)
            ctx_mask = torch.cat([ctx_mask, uncond_mask], dim=0)
            # Double speaker vector: real for cond, zeros for uncond
            if spk_vec is not None:
                spk_vec = torch.cat([spk_vec, torch.zeros_like(spk_vec)], dim=0)
            else:
                spk_vec = torch.zeros(B, 1, self.dim, device=device).repeat(2, 1, 1)
            B_eff = 2 * B
        else:
            B_eff = B

        # Direct speaker conditioning for the depformer (NAR codebook predictor).
        # CFG doubles spk_vec; use the conditional (first) half.
        dep_spk = None
        if spk_vec is not None and getattr(self, 'use_dep_speaker_cond', True):
            s = spk_vec.mean(dim=1) if spk_vec.dim() == 3 else spk_vec
            dep_spk = s[:B]

        # Direct prosody conditioning for the depformer. prosody_emb is already
        # (B, dim) from encode_context (post-FiLM, post-dropout, post-exagg).
        dep_prs = None
        if prosody_emb is not None and getattr(self, 'use_dep_prosody_cond', True):
            p = prosody_emb.mean(dim=1) if prosody_emb.dim() == 3 else prosody_emb
            dep_prs = p[:B]

        generated = []
        curr_step_toks = torch.full((B, self.n_q, 1), self.SOA_ID, device=device, dtype=torch.long)
        
        streams = [layer.self_attn.streaming(B_eff) for layer in self.decoder_layers]
        is_pre_norm = self.decoder_layers[0].pre_norm if len(self.decoder_layers) > 0 else True
        
        # Monotone attention pointer: (B,) relative text-frame index, non-decreasing.
        mono_center = None
        
        pos_tensor = torch.empty((B_eff, 1), device=device, dtype=torch.long)
        try:
            for layer_stream in streams: layer_stream.__enter__()
            for t in range(max_steps):
                pos_tensor.fill_(t)
                in_toks = curr_step_toks if cfg_scale == 1.0 else curr_step_toks.repeat(2, 1, 1)
                step_emb = self.audio_prenet(self.audio_norm(
                    _sum_embeddings(self.audio_embs, in_toks, self.n_q)
                ))
                
                x = step_emb
                spk_for_adaln = spk_vec.mean(dim=1) if spk_vec is not None and spk_vec.dim() == 3 else spk_vec

                # Monotone window bias for this step (rolled-forward text pointer)
                kv_len = context.shape[1]
                mono_bias = None
                if mono_slack > 0:
                    if mono_center is not None:
                        center_b = mono_center.unsqueeze(1) if B_eff == B else mono_center.repeat(2, 1)
                        pos = torch.arange(kv_len, device=device).unsqueeze(0)
                        rel = pos - offset
                        allowed = (rel >= center_b - mono_slack) & (rel <= center_b + mono_slack) & (rel >= 0)
                        allowed = allowed | (pos < offset)
                        mono_bias = torch.where(
                            allowed,
                            torch.zeros((), device=device, dtype=x.dtype),
                            torch.full((), -1e9, device=device, dtype=x.dtype),
                        )
                    else:
                        mono_bias = torch.zeros(B_eff, kv_len, device=device, dtype=x.dtype)
                    mono_bias = mono_bias.unsqueeze(1)

                attn_weights_0 = None
                for layer_i, layer in enumerate(self.decoder_layers):
                    if is_pre_norm:
                        x = x + layer.self_attn(layer.norm1(x), positions=pos_tensor)
                        ca = layer.cross_attn(layer.norm2(x), kv_input=context, kv_mask=ctx_mask, mono_bias=mono_bias, return_attn=(layer_i == 0 and mono_slack > 0))
                        if layer_i == 0 and mono_slack > 0:
                            x = x + ca[0]; attn_weights_0 = ca[1]
                        else:
                            x = x + ca
                        if spk_for_adaln is not None:
                            x = layer.speaker_adaLN(x, spk_for_adaln)
                        x = x + layer.ff(layer.norm3(x))
                    else:
                        x_self = layer.self_attn(x, positions=pos_tensor)
                        x = layer.norm1(x + x_self)
                        ca = layer.cross_attn(x, kv_input=context, kv_mask=ctx_mask, mono_bias=mono_bias, return_attn=(layer_i == 0 and mono_slack > 0))
                        if layer_i == 0 and mono_slack > 0:
                            x = layer.norm2(x + ca[0]); attn_weights_0 = ca[1]
                        else:
                            x = layer.norm2(x + ca)
                        if spk_for_adaln is not None:
                            x = layer.speaker_adaLN(x, spk_for_adaln)
                        x = layer.norm3(x + layer.ff(x))

                # Advance monotone pointer from cond-row attention center (non-decreasing).
                # Argmax (not mean): lets the window slide forward only when the model
                # actually looks ahead, and dwell freely otherwise.
                if mono_slack > 0 and attn_weights_0 is not None:
                    text_len = kv_len - offset
                    if text_len > 0:
                        cond_w = attn_weights_0[:B].mean(dim=1)[:, 0, offset:]  # (B, text_len)
                        denom = cond_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                        center = (cond_w / denom).argmax(dim=-1).to(cond_w.dtype)
                        mono_center = center if mono_center is None else torch.maximum(mono_center, center)
                
                if cfg_scale != 1.0:
                    x_cond, x_uncond = x.chunk(2, dim=0)
                    x = x_uncond + cfg_scale * (x_cond - x_uncond)
                
                t_in_base = x
                step_toks = []
                prev_tok = None
                
                # Fix: Initialize streaming with the actual batch size of t_in_base (B)
                current_batch_size = t_in_base.shape[0]
                hist = torch.cat(generated, dim=-1) if generated else None
                with self.dep_transformer.streaming(current_batch_size):
                    for k in range(self.n_q):
                        if k < curr_n_q:
                            level_ctx = torch.cat([t_in_base, self.dep_level_emb[:, k:k+1, :].expand(t_in_base.shape[0], -1, -1)], dim=-1)
                            dep_input = self.dep_in[k](level_ctx) if k == 0 or prev_tok is None else self.dep_in[k](level_ctx) + self.dep_emb[k-1](prev_tok)
                            dep_out = self.dep_transformer(dep_input)
                            if dep_prs is not None:
                                dep_out = self.dep_prosody_adaLN(dep_out, dep_prs)
                            if dep_spk is not None:
                                dep_out = self.dep_speaker_adaLN(dep_out, dep_spk)
                            l = self.dep_layers[k](dep_out) / max(temp, 1e-6)
                            # Use the model's card + 3 as the vocabulary size for the view
                            l = l.view(current_batch_size, self.card + 3)
                            if k > 0: l[:, self.SOA_ID:self.EOA_ID+1] = -float('inf')
                            if rep_penalty > 1.0 and hist is not None and hist.shape[-1] > 0:
                                pen = torch.zeros_like(l)
                                pen.scatter_add_(1, hist[:, k, :], torch.ones((current_batch_size, hist.shape[-1]), device=l.device, dtype=l.dtype))
                                l = torch.where(pen > 0, l / rep_penalty, l)
                            if top_k > 0:
                                v, _ = torch.topk(l, min(top_k, l.size(-1)))
                                l[l < v[:, [-1]]] = -float('inf')
                            if top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(l, descending=True)
                                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                remove_mask = torch.zeros_like(l)
                                remove_mask.scatter_(1, sorted_indices, sorted_indices_to_remove.to(l.dtype))
                                l[remove_mask.bool()] = float('-inf')
                            if min_p > 0.0:
                                probs = F.softmax(l, dim=-1)
                                min_thresh = min_p * probs.max(dim=-1, keepdim=True).values
                                l = torch.where(probs < min_thresh, torch.full_like(l, -float('inf')), l)
                            next_tok = torch.multinomial(F.softmax(l, dim=-1), 1)
                            step_toks.append(next_tok)
                            prev_tok = next_tok
                        else:
                            step_toks.append(torch.zeros((current_batch_size, 1), device=device, dtype=torch.long))
                
                curr_step_toks = torch.stack(step_toks, 1)
                if curr_step_toks[0, 0, 0] == self.EOA_ID: break
                generated.append(curr_step_toks)
        finally:
            for layer_stream in reversed(streams): layer_stream.__exit__(None, None, None)
        return torch.cat(generated, -1) if generated else None, None

# Backward compatibility alias
SeedVoxModel = JEPAProsodyHybridModel
