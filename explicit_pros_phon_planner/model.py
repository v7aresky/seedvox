import torch
import torch.nn as nn
import torch.nn.functional as F
from seedvox.model import JEPAProsodyHybridModel, _sum_embeddings
from .planner import PhoneticPlanner

class FiLMAdapter(nn.Module):
    def __init__(self, speaker_dim, target_dim):
        super().__init__()
        # Predict both gamma and beta in one pass
        self.proj = nn.Linear(speaker_dim, target_dim * 2)
        
    def forward(self, speaker_latents):
        # speaker_latents: [B, speaker_dim]
        # output: [B, target_dim * 2]
        params = self.proj(speaker_latents)
        gamma, beta = params.chunk(2, dim=-1)
        # Scale gamma to be near 0 initially so the modulation is small at the start
        return torch.tanh(gamma), beta

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
        
        # 3. FiLM Adapters for Prosody and Phonemes
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
        context, ctx_mask, _ = super().encode_context(
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
            
            # Extract Prosody tokens
            prs_len = self.cfg.get('num_prosody_tokens', 32)
            prs_feat = context[:, spk_len : spk_len + prs_len, :]
            
            # Apply FiLM Modulation
            gamma_phn, beta_phn = self.film_phn(speaker_latents)
            ph_feat = (1 + gamma_phn.unsqueeze(1)) * ph_feat + beta_phn.unsqueeze(1)
            
            gamma_prs, beta_prs = self.film_prs(speaker_latents)
            prs_feat = (1 + gamma_prs.unsqueeze(1)) * prs_feat + beta_prs.unsqueeze(1)
            
            # Reconstruct context
            spk = context[:, :spk_len, :]
            txt = context[:, spk_len + prs_len:, :]
            
            new_context = torch.cat([spk, prs_feat, ph_feat, txt], dim=1)
            
            # Update mask (assume phoneme_ids padding is 0)
            ph_mask = (phoneme_ids == 0) # True for padding
            new_mask = torch.cat([ctx_mask[:, :spk_len + prs_len], ph_mask, ctx_mask[:, spk_len + prs_len:]], dim=1)
            
            return new_context, new_mask, None
            
        return context, ctx_mask, None

    def forward(self, text, audio_tokens, text_lens, audio_lens, raw_texts=None, 
                use_speaker=None, use_prosody=None, phoneme_ids=None, mimi_latents=None,
                bpe_ids=None, bpe_lens=None, char_to_bpe=None, char_lens=None,
                drop_prob=0.0):
        
        # 1. Extract enriched text features for the phonetic planner
        text_feat = self.get_enriched_text_feat(
            text, text_lens, raw_texts, bpe_ids, bpe_lens, char_to_bpe, char_lens
        )
        
        # 2. Phonetic Planner Forward (AR Loss)
        ph_planner_logits = None
        if phoneme_ids is not None:
            ph_planner_logits = self.phonetic_planner(text_feat, phoneme_ids)
        
        # 3. Acoustic Forward (using augmented context)
        if audio_tokens is not None:
            audio_tokens = audio_tokens[:, :self.n_q]
            
        context, ctx_mask, _ = self.encode_context(
            text, text_lens, audio_tokens, audio_lens, raw_texts, 
            use_speaker, use_prosody, phoneme_ids, mimi_latents=mimi_latents,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe, char_lens=char_lens,
            drop_prob=drop_prob
        )
        
        logits, targets = self.forward_with_context(context, ctx_mask, audio_tokens, audio_lens)
        
        return logits, targets, ph_planner_logits, self._jepa_loss
