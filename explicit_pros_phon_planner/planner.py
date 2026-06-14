import torch
import torch.nn as nn
import torch.nn.functional as F
from seedvox.modules.moshi.modules.transformer import StreamingTransformer, create_norm_fn

class PhoneticPlanner(nn.Module):
    """
    Explicit Phonetic Planner: An AR Transformer that predicts Phoneme tokens
    from joint Char and BPE features.
    """
    def __init__(self, dim, phoneme_vocab_size, SOS_ID=3, EOS_ID=4, num_layers=6, num_heads=8, hidden_scale=4.0):
        super().__init__()
        self.dim = dim
        self.phoneme_vocab_size = phoneme_vocab_size
        self.SOS_ID = SOS_ID
        self.EOS_ID = EOS_ID
        
        # Phoneme embedding for the AR decoder part
        self.phoneme_emb = nn.Embedding(phoneme_vocab_size, dim)
        
        hs = int(dim * hidden_scale)
        self.transformer = StreamingTransformer(
            dim, num_heads, num_layers, hs,
            causal=True, positional_embedding="sin_rope", context=1024
        )
        self.norm = create_norm_fn("layer_norm", dim)
        
        # Output head
        self.head = nn.Linear(dim, phoneme_vocab_size)
        
    def forward(self, text_feat, phoneme_ids=None, text_mask=None, phoneme_mask=None):
        """
        text_feat: [B, T_text, dim] - joint Char and BPE features
        phoneme_ids: [B, T_ph] - target phoneme tokens for training (includes SOS and EOS)
        """
        B, T_text, _ = text_feat.shape
        device = text_feat.device
        
        if phoneme_ids is not None:
            # Training mode: AR with teacher forcing
            # We shift the phonemes: Input is ph[:, :-1], Target is ph[:, 1:]
            ph_in = phoneme_ids[:, :-1]
            ph_emb = self.phoneme_emb(ph_in)
            
            # Combine text and phoneme inputs
            x = torch.cat([text_feat, ph_emb], dim=1)
            
            logits = self.head(self.norm(self.transformer(x)))
            
            # Slice logits to match the targets (phoneme_ids[:, 1:])
            # Index T_text predicts phoneme_ids[:, 1] (the first token after SOS)
            # The indices T_text to the end correspond to targets p1, p2, ..., EOS
            ph_logits = logits[:, T_text:, :]
            return ph_logits
        else:
            # Inference mode: AR sampling
            return self.sample(text_feat, text_mask)

    @torch.no_grad()
    def sample(self, text_feat, text_mask=None, max_len=512, temp=1.0, top_p=0.9):
        B, T_text, _ = text_feat.shape
        device = text_feat.device
        
        generated = []
        # SOS is our first token
        curr_tok = torch.full((B, 1), self.SOS_ID, device=device, dtype=torch.long)
        
        with self.transformer.streaming(B):
            # 1. Process text features prefix (updates KV cache)
            _ = self.transformer(text_feat, attn_mask=text_mask)
            
            # 2. AR Sampling loop
            for i in range(max_len):
                ph_emb = self.phoneme_emb(curr_tok)
                
                # Transformer call with 1 token + streaming state
                logits = self.head(self.norm(self.transformer(ph_emb)))
                last_logits = logits[:, -1, :] / max(temp, 1e-6)
                
                # Sample
                probs = F.softmax(last_logits, dim=-1)
                # Apply top_p if needed
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    for b_idx in range(B):
                        indices_to_remove = sorted_indices[b_idx][sorted_indices_to_remove[b_idx]]
                        last_logits[b_idx, indices_to_remove] = -float('inf')
                    probs = F.softmax(last_logits, dim=-1)

                next_tok = torch.multinomial(probs, 1)
                
                generated.append(next_tok)
                curr_tok = next_tok
                
                if (next_tok == self.EOS_ID).all():
                    break
                    
        return torch.cat(generated, dim=1)
