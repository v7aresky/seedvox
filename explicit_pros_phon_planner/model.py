import torch
import torch.nn as nn
import torch.nn.functional as F
from seedvox.model import JEPAProsodyHybridModel, _sum_embeddings
from .planner import PhoneticPlanner

class FiLMAdapter(nn.Module):
    def __init__(self, speaker_dim, target_dim):
        super().__init__()
        self.norm = nn.LayerNorm(target_dim)
        self.proj = nn.Linear(speaker_dim, target_dim * 2)
        # Initialize: gamma proj to ~0 (so output scale ~1), beta proj to ~0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # Bias for gamma starts at 1.0 (so scale is 1.0 initially)
        self.proj.bias.data[:target_dim] = 1.0 
        
    def forward(self, x, speaker_latents):
        # x is [B, T, D]
        # speaker_latents is [B, D]
        params = self.proj(speaker_latents).unsqueeze(1) # [B, 1, D*2]
        gamma, beta = params.chunk(2, dim=-1)
        return self.norm(x) * gamma + beta

class ExplicitPlannerModel(JEPAProsodyHybridModel):
    """
    Extends JEPAProsodyHybridModel to include an explicit PhoneticPlanner.
    Flow: Text -> Chars + BPE -> text_feat
          text_feat -> pred_phonemes
          text_feat -> pred_prosody
          Context = [Speaker, Prosody, Phonemes, Text] -> Acoustic tokens
    """
    def __init__(self, config, tokenizer_vocab_size, phoneme_vocab_size=128):
        super().__init__(config, tokenizer_vocab_size, phoneme_vocab_size)
        
        # Consistent SOS/EOS
        self.SOS_ID = 3
        self.EOS_ID = phoneme_vocab_size # Using the passed size as EOS
        
        # 1. Phonetic Planner (AR Transformer)
        # The vocab size should be phoneme_vocab_size + 1 (to accommodate the EOS index)
        self.phonetic_planner = PhoneticPlanner(
            dim=self.dim,
            phoneme_vocab_size=phoneme_vocab_size + 1,
            SOS_ID=self.SOS_ID,
            EOS_ID=self.EOS_ID,
            num_layers=config['model'].get('phonetic_layers', 6),
            num_heads=config['model'].get('phonetic_heads', 8)
        )
        
        # 2. Phoneme embedding for acoustic decoder
        self.ph_decoder_emb = nn.Embedding(phoneme_vocab_size + 1, self.dim)
        nn.init.normal_(self.ph_decoder_emb.weight, std=0.01) # Low variance start
        
        # 3. Projection layer to normalize/project phonetic features
        self.ph_projection = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.LayerNorm(self.dim)
        )
        
        # 4. Context Boundary Markers
        self.marker_spk = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.marker_prs = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.marker_phn = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.marker_txt = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        
        # 5. FiLM Adapters for Prosody and Phonemes
        speaker_dim = self.cfg.get('speaker_dim', self.dim) # Assuming speaker latent dim matches
        self.film_prs = FiLMAdapter(speaker_dim, self.dim)
        self.film_phn = FiLMAdapter(speaker_dim, self.dim)
        
    def get_enriched_text_feat(self, text, text_lens, raw_texts=None, bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None):
        text_feat, _ = self.encode_text(text, text_lens, raw_texts)
        if self.use_bpe_encoder and bpe_ids is not None:
            B, device = text.shape[0], text.device
            T_char = char_to_bpe.shape[1]
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
        return text_feat

    def encode_context(self, text, text_lens, audio_tokens=None, audio_lens=None, raw_texts=None, 
                       use_speaker=None, use_prosody=None, phoneme_ids=None, mimi_latents=None,
                       bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None,
                       drop_prob=0.0, external_speaker=None, external_prosody=None):
        
        # 1. Get Base Context (Speaker, Prosody, Text)
        context, ctx_mask, ph_logits, jepa_loss, contrastive_loss = super().encode_context(
            text, text_lens, audio_tokens, audio_lens, raw_texts,
            use_speaker, use_prosody, phoneme_ids=None, mimi_latents=mimi_latents,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens,
            drop_prob=drop_prob, external_speaker=external_speaker, external_prosody=external_prosody
        )
        
        # 2. Extract Speaker Latents for FiLM
        # We need a single vector representing the speaker. 
        # In JEPAProsodyHybridModel, context = [spk, adapted_prosody, text_feat]
        # spk is [B, spk_len, dim]
        spk_len = self.cfg['num_speaker_latents'] if (use_speaker if use_speaker is not None else self.use_speaker) else 0
        if spk_len > 0:
            speaker_latents = context[:, :spk_len, :].mean(dim=1)
        else:
            speaker_latents = torch.zeros(text.shape[0], self.dim, device=text.device)

        # 3. Augmented with Explicit Phonemes
        if phoneme_ids is not None:
            B = text.shape[0]
            ph_feat = self.ph_decoder_emb(phoneme_ids)
            ph_feat = self.ph_projection(ph_feat) # Normalize/Project
            
            # Extract Prosody tokens
            prs_len = self.cfg.get('num_prosody_tokens', 32)
            prs_feat = context[:, spk_len : spk_len + prs_len, :]
            
            # Apply FiLM Modulation to phonemes and prosody if enabled
            if self.cfg.get('use_film', True):
                ph_feat = self.film_phn(ph_feat, speaker_latents)
                prs_feat = self.film_prs(prs_feat, speaker_latents)
                
            # Reconstruct context with markers
            spk = context[:, :spk_len, :]
            txt = context[:, spk_len + prs_len:, :]
            
            # Context Assembly: [Marker_Spk, Spk, Marker_Prs, Prs, Marker_Phn, Phn, Marker_Txt, Txt]
            new_context = torch.cat([
                self.marker_spk.expand(B, 1, -1), spk,
                self.marker_prs.expand(B, 1, -1), prs_feat,
                self.marker_phn.expand(B, 1, -1), ph_feat,
                self.marker_txt.expand(B, 1, -1), txt
            ], dim=1)
            
            # Update mask (new structure: [1, spk_len, 1, prs_len, 1, ph_len, 1, txt_len])
            # Original spk_prs_len = spk_len + prs_len
            ph_mask = (phoneme_ids == 0) # True for padding
            
            # Correcting mask construction for the new layout
            B, T = text.shape[0], new_context.shape[1]
            new_mask = torch.zeros(B, T, device=text.device, dtype=torch.bool)
            
            # Helper to create a False (valid) mask for a marker
            def mk_marker(B, device): return torch.zeros(B, 1, device=device, dtype=torch.bool)
            
            # 1. Spk
            curr = 0
            new_mask[:, curr : curr + 1 + spk_len] = torch.cat([mk_marker(B, text.device), ctx_mask[:, :spk_len]], dim=1)
            curr += 1 + spk_len
            
            # 2. Prs
            new_mask[:, curr : curr + 1 + prs_len] = torch.cat([mk_marker(B, text.device), ctx_mask[:, spk_len : spk_len + prs_len]], dim=1)
            curr += 1 + prs_len
            
            # 3. Phn
            new_mask[:, curr : curr + 1 + ph_feat.shape[1]] = torch.cat([mk_marker(B, text.device), ph_mask], dim=1)
            curr += 1 + ph_feat.shape[1]
            
            # 4. Txt
            new_mask[:, curr : curr + 1 + txt.shape[1]] = torch.cat([mk_marker(B, text.device), ctx_mask[:, spk_len + prs_len:]], dim=1)
            
            return new_context, new_mask, ph_logits, jepa_loss, contrastive_loss
            
        return context, ctx_mask, ph_logits, jepa_loss, contrastive_loss

    def forward(self, text, audio_tokens, text_lens, audio_lens, raw_texts=None, 
                use_speaker=None, use_prosody=None, phoneme_ids=None, mimi_latents=None,
                bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None,
                drop_prob=0.0):
        
        # 1. Extract enriched text features for the phonetic planner
        text_feat = self.get_enriched_text_feat(
            text, text_lens, raw_texts, bpe_ids, bpe_lens, char_to_bpe, char_lens
        )
        
        # Compute text mask (KEEP mask: True for valid tokens)
        # text_feat length is text.shape[1] + 2 due to SOT/EOT
        B, T_feat, _ = text_feat.shape
        device = text_feat.device
        text_mask = torch.arange(T_feat, device=device).unsqueeze(0) < (text_lens.unsqueeze(1) + 2)
        
        # 2. Phonetic Planner Forward (AR Loss)
        ph_planner_logits = None
        if phoneme_ids is not None:
            # Compute phoneme mask (assuming 0 is pad)
            ph_mask = (phoneme_ids != 0)
            ph_planner_logits = self.phonetic_planner(text_feat, phoneme_ids, text_mask=text_mask, phoneme_mask=ph_mask)
        
        # 3. Acoustic Forward (using augmented context)
        if audio_tokens is not None:
            audio_tokens = audio_tokens[:, :self.n_q]
            
        context, ctx_mask, _, jepa_loss, _ = self.encode_context(
            text, text_lens, audio_tokens, audio_lens, raw_texts, 
            use_speaker, use_prosody, phoneme_ids, mimi_latents=mimi_latents,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens,
            drop_prob=drop_prob
        )
        
        logits, targets, latent_pred = self.forward_with_context(context, ctx_mask, audio_tokens, audio_lens)
        
        return logits, targets, ph_planner_logits, jepa_loss, None, latent_pred

    @torch.no_grad()
    def sample(self, text, text_lens, ref_audio=None, ref_lens=None, max_steps=1000, temp=0.1, curr_n_q=None, raw_texts=None, top_k=0, top_p=0.9, use_speaker=None, use_prosody=None, cfg_scale=1.0,
               bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None, phoneme_ids=None, drop_prob=0.0,
               external_speaker=None, external_prosody=None,
               precomputed_context=None, precomputed_mask=None):
        
        # Calculate CFG offset for [M_spk, Spk, M_prs, Prs, M_phn, Phn, M_txt, Txt]
        spk_len = self.cfg['num_speaker_latents'] if (use_speaker if use_speaker is not None else self.use_speaker) else 0
        prs_len = self.cfg.get('num_prosody_tokens', 32)
        
        if phoneme_ids is not None:
            # Offset = 1 (M_spk) + spk_len + 1 (M_prs) + prs_len = 2 + spk_len + prs_len
            offset = 2 + spk_len + prs_len
        else:
            offset = spk_len + prs_len

        return super().sample(
            text, text_lens, ref_audio, ref_lens, max_steps, temp, curr_n_q, raw_texts, top_k, top_p, use_speaker, use_prosody, cfg_scale,
            bpe_ids, bpe_lens, char_to_bpe, char_lens, phoneme_ids, drop_prob,
            external_speaker, external_prosody,
            precomputed_context=precomputed_context, precomputed_mask=precomputed_mask,
            offset=offset
        )
