import torch
import re
from seedvox.utils.tokenizer import PhonemeTokenizer
from seedvox.utils.text import normalize_text
from seedvox.utils.g2p_factory import get_phoneme_generator


def extract_ref_prosody_latent(wav_path, model, device):
    """Encode a reference wav's prosody into the stage-1 codec latent space
    ([1, num_blocks, dim]) that the planner/decoder are trained against.

    wav -> F0/energy/voicing (12.5 Hz, 3ch, center-normalized) via the same
    extractor used for training data, zero-padded to a multiple of num_blocks
    (codec pooling convention, matching train_prosody_codec.collate), then
    ProsodyCodec.encode (the frozen stage-1 teacher). The random prosody_encoder
    is NOT used; it never appears in the training graph.
    """
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.prepare_prosody_codec_data import extract_prosody
    import numpy as np

    pr = extract_prosody(wav_path)
    if pr is None or '_error' in pr:
        raise RuntimeError(f"Prosody extraction failed for {wav_path}: {pr}")

    T = pr['log_f0_center'].shape[0]
    nb = int(model.prosody_codec.num_blocks)
    T_pad = max(nb, ((T + nb - 1) // nb) * nb)
    feats = np.zeros((T_pad, 3), dtype=np.float32)
    feats[:T, 0] = pr['log_f0_center']
    feats[:T, 1] = pr['e_center']
    feats[:T, 2] = pr['voicing'].astype(np.float32)
    dtype = next(model.prosody_codec.parameters()).dtype
    ft = torch.from_numpy(feats).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        return model.prosody_codec.encode(ft)


def filter_state_dict(model, state_dict):
    """Drop keys whose shapes don't match the model. load_state_dict(strict=False)
    still raises on size mismatches, so pre-filter them out (resized layers such as
    the 16->32 prosody planner re-initialize)."""
    msd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in msd and msd[k].shape == v.shape}
    dropped = sorted(set(state_dict.keys()) - set(filtered.keys()))
    if dropped:
        print(f"[filter_state_dict] Dropped {len(dropped)} keys (shape mismatch / not in model):")
        for k in dropped[:10]:
            print(f"  - {k}  (ckpt {tuple(state_dict[k].shape) if state_dict[k].dim() else 'scalar'} vs model "
                  f"{tuple(msd[k].shape) if k in msd else 'missing'})")
    return filtered

class PhoneticGenerator:
    """
    Utility for generating phoneme targets for training the PhoneticPlanner.
    Uses an external G2P (e.g., deep-phonemizer or g2p_en) to create 
    ground-truth phoneme sequences from text.
    """
    def __init__(self, backend='espeak', phoneme_vocab_size=None):
        self.g2p = get_phoneme_generator(backend)
        self.tokenizer = PhonemeTokenizer()
        
        # SOS is usually <EPS> (3)
        self.SOS_ID = 3
        # EOS is the token after the last vocab token
        self.EOS_ID = phoneme_vocab_size if phoneme_vocab_size is not None else self.tokenizer.vocab_size
        
    def generate_targets(self, text, normalize=True):
        """
        Generates a sequence of phoneme IDs with SOS and EOS tokens.
        Includes punctuation as requested by the professor.
        """
        norm_text = normalize_text(text) if normalize else text
        
        # Get phonemes with punctuation
        # g2p_en typically handles punctuation by returning it as is.
        phonemes = self.g2p(norm_text)
        return self.generate_targets_from_phonemes(phonemes)

    def generate_targets_batch(self, texts, normalize=True):
        """
        Generates a batch of phoneme ID sequences with SOS and EOS tokens.
        Much more efficient for backends like espeak that support batching.
        """
        norm_texts = [normalize_text(t) if normalize else t for t in texts]
        
        # Batch G2P call
        phonemes_list = self.g2p(norm_texts)
        
        return [self.generate_targets_from_phonemes(ph) for ph in phonemes_list]

    def generate_targets_from_phonemes(self, phonemes):
        """Converts pre-generated phonemes to IDs with SOS and EOS."""
        # Convert to IDs
        ids = [self.SOS_ID]
        for ph in phonemes:
            # The PhonemeTokenizer.ph_to_id handles some special tokens like ' ', '.', etc.
            if ph.strip() == '':
                # Map space to ' ' if in vocab, else skip or map to a break token
                ph_id = self.tokenizer.ph_to_id.get(' ', self.SOS_ID) 
            else:
                ph_id = self.tokenizer.ph_to_id.get(ph, self.tokenizer.unk_token_id)
            ids.append(ph_id)
            
        ids.append(self.EOS_ID)
        return ids

def collate_phonemes(phoneme_id_list, pad_id=0):
    """Pads a batch of phoneme ID sequences."""
    max_len = max(len(ids) for ids in phoneme_id_list)
    B = len(phoneme_id_list)
    padded = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(phoneme_id_list):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded
