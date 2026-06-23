import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import ExplicitPlannerModel

class LinguisticFusion(nn.Module):
    """
    Unified Linguistic Encoder: Fuses Text/BPE and Explicit Phonemes using Cross-Attention.
    Text backbone acts as Queries, Phonemes act as Keys/Values.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        # Initial gate at -5.0 so sigmoid(-5.0) ≈ 0.006 (trusting text backbone initially)
        self.gate = nn.Parameter(torch.tensor([-5.0]))
    def forward(self, text_feat, ph_feat, ph_mask=None):
        # Cross-Attention: Text queries Phonemes
        # res has the same length as text_feat
        res, _ = self.attn(query=text_feat, key=ph_feat, value=ph_feat, key_padding_mask=ph_mask)
        
        # Gated Residual: Allows model to rely on Text backbone and 'dial in' Phonetic details (Sigmoid allows non-zero gradients initially)
        return self.norm(text_feat + torch.sigmoid(self.gate) * res)

class FusionPlannerModel(ExplicitPlannerModel):
    """
    Extends ExplicitPlannerModel to use the Unified Linguistic Encoder.
    Fuses Phonemes back onto the Text backbone to avoid Tri-Alignment issues
    in the Acoustic Decoder's cross-attention.
    """
    def __init__(self, config, tokenizer_vocab_size, phoneme_vocab_size=128):
        super().__init__(config, tokenizer_vocab_size, phoneme_vocab_size)
        
        self.linguistic_fusion = LinguisticFusion(
            dim=self.dim,
            num_heads=config['model'].get('fusion_heads', 8)
        )
        
    def encode_context(self, text, text_lens, audio_tokens=None, audio_lens=None, raw_texts=None, 
                       use_speaker=None, use_prosody=None, phoneme_ids=None, mimi_latents=None,
                       bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None,
                       drop_prob=0.0, external_speaker=None, external_prosody=None):
        
        # 1. Get Base Context from ExplicitPlannerModel (which calls JEPAProsodyHybridModel)
        # We call super(ExplicitPlannerModel, self) to get the standard JEPA context 
        # [Spk, Prosody, Text] before ExplicitPlannerModel adds its own concatenation.
        # But wait, ExplicitPlannerModel.encode_context is what adds the markers.
        # Let's get the standard context first.
        
        from seedvox.model import JEPAProsodyHybridModel
        context, ctx_mask, ph_logits, jepa_loss, contrastive_loss = JEPAProsodyHybridModel.encode_context(
            self, text, text_lens, audio_tokens, audio_lens, raw_texts,
            use_speaker, use_prosody, phoneme_ids=None, mimi_latents=mimi_latents,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens,
            drop_prob=drop_prob, external_speaker=external_speaker, external_prosody=external_prosody
        )
        
        # 2. Extract Speaker Latents for FiLM (consistent with ExplicitPlannerModel)
        spk_len = self.cfg['num_speaker_latents'] if (use_speaker if use_speaker is not None else self.use_speaker) else 0
        if spk_len > 0:
            speaker_latents = context[:, :spk_len, :].mean(dim=1)
        else:
            speaker_latents = torch.zeros(text.shape[0], self.dim, device=text.device)

        # 3. Fusion Logic
        if phoneme_ids is not None:
            B = text.shape[0]
            ph_feat = self.ph_decoder_emb(phoneme_ids)
            ph_feat = self.ph_projection(ph_feat) 
            
            # Extract Prosody tokens
            prs_len = self.cfg.get('num_prosody_tokens', 32)
            prs_feat = context[:, spk_len : spk_len + prs_len, :]
            
            # Apply FiLM Modulation
            if self.cfg.get('use_film', True):
                ph_feat = self.film_phn(ph_feat, speaker_latents)
                prs_feat = self.film_prs(prs_feat, speaker_latents)
                
            # Extract Text backbone
            txt = context[:, spk_len + prs_len:, :]
            
            # UNIFIED LINGUISTIC FUSION
            ph_mask = (phoneme_ids == 0) # True for padding
            unified_txt = self.linguistic_fusion(txt, ph_feat, ph_mask=ph_mask)
            
            # Context Assembly: [Marker_Spk, Spk, Marker_Prs, Prs, Marker_Txt, UnifiedTxt]
            # (Keeping markers for consistency with ExplicitPlannerModel's philosophy)
            spk = context[:, :spk_len, :]
            new_context = torch.cat([
                self.marker_spk.expand(B, 1, -1), spk,
                self.marker_prs.expand(B, 1, -1), prs_feat,
                self.marker_txt.expand(B, 1, -1), unified_txt
            ], dim=1)
            
            # Mask construction
            B, T = text.shape[0], new_context.shape[1]
            new_mask = torch.zeros(B, T, device=text.device, dtype=torch.bool)
            
            def mk_marker(B, device): return torch.zeros(B, 1, device=device, dtype=torch.bool)
            
            curr = 0
            new_mask[:, curr : curr + 1 + spk_len] = torch.cat([mk_marker(B, text.device), ctx_mask[:, :spk_len]], dim=1)
            curr += 1 + spk_len
            new_mask[:, curr : curr + 1 + prs_len] = torch.cat([mk_marker(B, text.device), ctx_mask[:, spk_len : spk_len + prs_len]], dim=1)
            curr += 1 + prs_len
            new_mask[:, curr : curr + 1 + txt.shape[1]] = torch.cat([mk_marker(B, text.device), ctx_mask[:, spk_len + prs_len:]], dim=1)
            
            return new_context, new_mask, ph_logits, jepa_loss, contrastive_loss
            
        return context, ctx_mask, ph_logits, jepa_loss, contrastive_loss

    @torch.no_grad()
    def sample(self, text, text_lens, ref_audio=None, ref_lens=None, max_steps=1000, temp=0.1, curr_n_q=None, raw_texts=None, top_k=0, top_p=0.9, use_speaker=None, use_prosody=None, cfg_scale=1.0,
               bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None, phoneme_ids=None, drop_prob=0.0,
               external_speaker=None, external_prosody=None,
               precomputed_context=None, precomputed_mask=None):
        """
        Overrides sample to handle correct CFG offsets for the Fusion context structure.
        """
        if precomputed_context is None:
            context, ctx_mask, _, _, _ = self.encode_context(
                text, text_lens, audio_tokens=ref_audio, audio_lens=ref_lens, 
                raw_texts=raw_texts, use_speaker=use_speaker, use_prosody=use_prosody,
                bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens,
                phoneme_ids=phoneme_ids,
                drop_prob=drop_prob,
                external_speaker=external_speaker,
                external_prosody=external_prosody
            )
        else:
            context, ctx_mask = precomputed_context, precomputed_mask

        # Calculate CFG offset for [M_spk, Spk, M_prs, Prs, M_txt, UnifiedTxt]
        spk_len = self.cfg['num_speaker_latents'] if (use_speaker if use_speaker is not None else self.use_speaker) else 0
        prs_len = self.cfg.get('num_prosody_tokens', 32)
        
        if phoneme_ids is not None:
            # Offset = 1 (M_spk) + spk_len + 1 (M_prs) + prs_len = 2 + spk_len + prs_len
            offset = 2 + spk_len + prs_len
        else:
            offset = spk_len + prs_len

        # Pass precomputed context to avoid re-encoding
        return super(ExplicitPlannerModel, self).sample(
            text, text_lens, ref_audio, ref_lens, max_steps, temp, curr_n_q, raw_texts, top_k, top_p, use_speaker, use_prosody, cfg_scale,
            bpe_ids, bpe_lens, char_to_bpe, char_lens, phoneme_ids, drop_prob,
            external_speaker, external_prosody,
            precomputed_context=context, precomputed_mask=ctx_mask,
            offset=offset
        )
