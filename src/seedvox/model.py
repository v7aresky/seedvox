import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from .base import JEPAProsodyBase, _sum_embeddings
from .planner import JEPAProsodyPlanner
from .prosody_utils import ProsodyBottleneck


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
            num_prosody_tokens=cfg.get('num_prosody_tokens', 32)
        )
        self.prosody_bottleneck = ProsodyBottleneck(
            dim=self.dim,
            num_prosody_tokens=cfg.get('num_prosody_tokens', 32)
        )
        self.speaker_adapter = SpeakerAdapter(self.dim)
        self.jepa_weight = config['training'].get('jepa_weight', 1.0)
        self.MASK_ID = tokenizer_vocab_size + 3
        self.bpe_gate = nn.Parameter(torch.tensor([1.0]))
        
        # Null conditioning for CFG and robustness
        self.null_speaker = nn.Parameter(torch.randn(1, cfg.get('num_speaker_latents', 16), self.dim) * 0.02)
        self.null_prosody = nn.Parameter(torch.randn(1, cfg.get('num_prosody_tokens', 32), self.dim) * 0.02)
        self.null_text_feat = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)

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
                       bpe_ids: tp.Optional[torch.Tensor] = None, 
                       bpe_lens: tp.Optional[torch.Tensor] = None, 
                       char_to_bpe: tp.Optional[torch.Tensor] = None, 
                       char_lens: tp.Optional[torch.Tensor] = None,
                       drop_prob: float = 0.0, 
                       external_speaker: tp.Optional[torch.Tensor] = None, 
                       external_prosody: tp.Optional[torch.Tensor] = None) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.Optional[torch.Tensor]]:
        
        B, device = text.shape[0], text.device
        if audio_tokens is not None: audio_tokens = audio_tokens[:, :self.n_q]
        
        # 1. Base Text Encoding
        text_feat, text_in = self.encode_text(text, text_lens, raw_texts)
        text_lens_wrapped = text_lens + 2
        
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
            if use_prosody: prs = self.prosody_encoder(ae, key_padding_mask=mask)
        else:
            if use_speaker: spk = self.null_speaker.expand(B, -1, -1)
            if use_prosody: prs = self.null_prosody.expand(B, -1, -1)

        # 4. JEPA Prosody Planning
        text_feat_text_rate = text_feat
        t_mask = torch.arange(text_feat_text_rate.shape[1], device=device).unsqueeze(0) >= text_lens_wrapped.unsqueeze(1)
        
        pred_prosody = self.jepa_planner(text_feat_text_rate, text_mask=t_mask)
        
        jepa_loss = None
        if external_prosody is not None:
            current_prosody = external_prosody
        elif mimi_latents is not None:
            # Extract GT Prosody (Teacher) - STRICTLY NO GRAD and DETACHED
            with torch.no_grad():
                a_mask = torch.arange(mimi_latents.shape[2], device=device).unsqueeze(0) >= audio_lens.unsqueeze(1)
                gt_prosody = self.prosody_bottleneck(mimi_latents, mask=a_mask).detach()
            
            # Compute JEPA Loss (Targeting the teacher)
            jepa_loss = F.mse_loss(pred_prosody, gt_prosody)
            current_prosody = pred_prosody
        else:
            # Inference mode
            current_prosody = pred_prosody

        # 5. Speaker Adaptation (FiLM)
        speaker_vector = spk.mean(dim=1, keepdim=True) if spk is not None else torch.zeros(B, 1, self.dim, device=device)
        scale, shift = self.speaker_adapter(speaker_vector)
        adapted_prosody = current_prosody * scale + shift
        
        # 6. Conditioning Dropout
        if drop_prob > 0 and self.training:
            keep = (torch.rand(B, 1, 1, device=device) > drop_prob).float()
            if spk is not None:
                spk = keep * spk + (1 - keep) * self.null_speaker.expand(B, -1, -1)
            
            adapted_prosody = keep * adapted_prosody + (1 - keep) * self.null_prosody.expand(B, -1, -1)
            
            if torch.rand(1).item() < 0.2:
                text_feat = keep * text_feat + (1 - keep) * self.null_text_feat.expand(B, text_feat.shape[1], -1)

        # 7. Final Context Assembly
        ctx_parts = []
        if spk is not None: ctx_parts.append(spk)
        ctx_parts.append(adapted_prosody)
        ctx_parts.append(text_feat)
        context = torch.cat(ctx_parts, dim=1)
        
        ctx_mask = torch.zeros(B, context.shape[1], device=device, dtype=torch.bool)
        offset = (spk.shape[1] if spk is not None else 0) + adapted_prosody.shape[1]
        for b in range(B):
            ctx_mask[b, offset + text_lens_wrapped[b]:] = True
            
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
            
        return context, ctx_mask, ph_logits, jepa_loss, contrastive_loss

    def mlm_forward(self, masked_text, text_lens, raw_texts=None, phoneme_ids=None,
                    bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None):
        context, ctx_mask, ph_logits, jepa_loss, contrastive_loss = self.encode_context(
            masked_text, text_lens, raw_texts=raw_texts, phoneme_ids=phoneme_ids,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens
        )
        return None, ph_logits

    def forward(self, text, audio_tokens, text_lens, audio_lens, raw_texts=None, 
                use_speaker=None, use_prosody=None, phoneme_ids=None, mimi_latents=None,
                bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None,
                drop_prob=0.0):
        audio_tokens = audio_tokens[:, :self.n_q]
        context, ctx_mask, ph_logits, jepa_loss, contrastive_loss = self.encode_context(
            text, text_lens, audio_tokens, audio_lens, raw_texts, 
            use_speaker, use_prosody, phoneme_ids, mimi_latents=mimi_latents,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens,
            drop_prob=drop_prob
        )
        logits, targets = self.forward_with_context(context, ctx_mask, audio_tokens, audio_lens)
        return logits, targets, ph_logits, jepa_loss, contrastive_loss

    @torch.no_grad()
    def sample(self, text, text_lens, ref_audio=None, ref_lens=None, max_steps=1000, temp=0.1, curr_n_q=None, raw_texts=None, top_k=0, top_p=0.9, use_speaker=None, use_prosody=None, cfg_scale=1.0,
               bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None, phoneme_ids=None, drop_prob=0.0,
               external_speaker=None, external_prosody=None,
               precomputed_context=None, precomputed_mask=None):
        B, device = text.shape[0], text.device
        if curr_n_q is None: curr_n_q = self.n_q
        
        # 1. Encode Context
        if precomputed_context is not None:
            context, ctx_mask = precomputed_context, precomputed_mask
        else:
            context, ctx_mask, _, _, _ = self.encode_context(
                text, text_lens, audio_tokens=ref_audio, audio_lens=ref_lens, 
                raw_texts=raw_texts, use_speaker=use_speaker, use_prosody=use_prosody,
                bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens,
                phoneme_ids=phoneme_ids,
                drop_prob=drop_prob,
                external_speaker=external_speaker,
                external_prosody=external_prosody
            )
        
        # Calculate text offset for CFG masking
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
            B_eff = 2 * B
        else:
            B_eff = B

        generated = []
        curr_step_toks = torch.full((B, self.n_q, 1), self.SOA_ID, device=device, dtype=torch.long)
        
        streams = [layer.self_attn.streaming(B_eff) for layer in self.decoder_layers]
        is_pre_norm = self.decoder_layers[0].pre_norm if len(self.decoder_layers) > 0 else True
        
        try:
            for layer_stream in streams: layer_stream.__enter__()
            for t in range(max_steps):
                in_toks = curr_step_toks if cfg_scale == 1.0 else curr_step_toks.repeat(2, 1, 1)
                step_emb = self.audio_prenet(self.audio_norm(
                    _sum_embeddings(self.audio_embs, in_toks, self.n_q)
                ))
                
                x = step_emb
                for layer in self.decoder_layers:
                    if is_pre_norm:
                        x = x + layer.self_attn(layer.norm1(x), positions=torch.full((B_eff, 1), t, device=device))
                        x = x + layer.cross_attn(layer.norm2(x), kv_input=context, kv_mask=ctx_mask)
                        x = x + layer.ff(layer.norm3(x))
                    else:
                        x_self = layer.self_attn(x, positions=torch.full((B_eff, 1), t, device=device))
                        x = layer.norm1(x + x_self)
                        x = layer.norm2(x + layer.cross_attn(x, kv_input=context, kv_mask=ctx_mask))
                        x = layer.norm3(x + layer.ff(x))
                
                if cfg_scale != 1.0:
                    x_cond, x_uncond = x.chunk(2, dim=0)
                    x = x_uncond + cfg_scale * (x_cond - x_uncond)
                
                t_in_base = x
                step_toks = []
                prev_tok = None
                with self.dep_transformer.streaming(B_eff):
                    for k in range(self.n_q):
                        if k < curr_n_q:
                            dep_input = self.dep_in[k](t_in_base) if k == 0 or prev_tok is None else self.dep_in[k](t_in_base) + self.dep_emb[k-1](prev_tok)
                            l = self.dep_layers[k](self.dep_transformer(dep_input)) / max(temp, 1e-6)
                            l = l.view(B_eff, -1)
                            if k > 0: l[:, self.SOA_ID:self.EOA_ID+1] = -float('inf')
                            if top_k > 0:
                                v, _ = torch.topk(l, min(top_k, l.size(-1)))
                                l[l < v[:, [-1]]] = -float('inf')
                            if top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(l, descending=True)
                                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                for b_idx in range(B_eff):
                                    indices_to_remove = sorted_indices[b_idx][sorted_indices_to_remove[b_idx]]
                                    l[b_idx, indices_to_remove] = -float('inf')
                            next_tok = torch.multinomial(F.softmax(l, dim=-1), 1)
                            step_toks.append(next_tok)
                            prev_tok = next_tok
                        else:
                            step_toks.append(torch.zeros((B_eff, 1), device=device, dtype=torch.long))
                
                curr_step_toks = torch.stack(step_toks, 1)
                if curr_step_toks[0, 0, 0] == self.EOA_ID: break
                generated.append(curr_step_toks)
        finally:
            for layer_stream in reversed(streams): layer_stream.__exit__(None, None, None)
        return torch.cat(generated, -1) if generated else None, None

# Backward compatibility alias
SeedVoxModel = JEPAProsodyHybridModel
