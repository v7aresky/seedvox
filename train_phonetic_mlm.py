
import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from hybrid_TTS import CharTokenizer
from phoneme_tokenizer import PhonemeTokenizer
from hybrid_TTS_phonetic import PhoneticHybridTTSModelV3
from phonetic_utils import PhoneticAligner
from g2p_factory import get_phoneme_generator
import random

def train_phonetic_mlm(config, device, max_steps=15000, lr=1e-4, batch_size=64, save_path='pretrained_models/text_encoder_phonetic_mlm.pt', g2p_backend='g2p_en'):
    """
    Dual Masked Language Modeling (MLM) pretraining.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    char_tokenizer = CharTokenizer()
    ph_tokenizer = PhonemeTokenizer()
    
    # Initialize G2P backend
    ph_generator = get_phoneme_generator(g2p_backend)
    aligner = PhoneticAligner(generator=ph_generator)
    
    model = PhoneticHybridTTSModelV3(config, char_tokenizer.vocab_size, ph_tokenizer.vocab_size).to(device)
    
    # Try to load pre-processed data first
    preprocessed_path = config['training'].get('mlm_data_path', 'dataset/mlm_pretrain_data.pt')
    if os.path.exists(preprocessed_path):
        print(f"Loading pre-processed MLM data from {preprocessed_path}")
        preprocessed_data = torch.load(preprocessed_path, weights_only=False)
        use_preprocessed = True
    else:
        print("No pre-processed data found. Using live streaming (slow).")
        print("Tip: Run 'python prepare_mlm_dataset.py' first for 10x speedup.")
        use_preprocessed = False
        
        # Load corpus for live mode
        corpus_list = config['training'].get('pretrain_corpus', ["librispeech_lm", "wikitext", "tinystories"])
        CORPUS_REGISTRY = {
            "librispeech_lm": ("librispeech_lm", {"split": "train", "streaming": True}),
            "wikitext": ("wikitext", {"name": "wikitext-103-raw-v1", "split": "train", "streaming": True}),
            "tinystories": ("karpathy/tinystories-gpt4-clean", {"split": "train", "streaming": True}),
        }
        dataset = None
        for corpus_key in corpus_list:
            if corpus_key in CORPUS_REGISTRY:
                ds_name, ds_kwargs = CORPUS_REGISTRY[corpus_key]
                try:
                    dataset = load_dataset(ds_name, **ds_kwargs)
                    print(f"Loaded corpus: {corpus_key}")
                    break
                except Exception as e:
                    print(f"Failed to load {corpus_key}: {e}")
        if dataset is None:
            print("Error: No text corpus loaded.")
            return

    # Optimize only text-related parameters
    text_params = [p for n, p in model.named_parameters() if any(x in n for x in ['text_emb', 'text_encoder', 'phoneme_emb', 'phoneme_predictor', 'mlm_head', 'bpe_encoder', 'bpe_gate']) and p.requires_grad]
    optimizer = torch.optim.AdamW(text_params, lr=lr)
    char_criterion = nn.CrossEntropyLoss(ignore_index=0)
    ph_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    step = 0
    pbar = tqdm(total=max_steps, desc="Phonetic MLM")

    def mask_tokens(tokens, tokenizer):
        masked = tokens.clone()
        labels = torch.full(tokens.shape, 0, device=tokens.device)
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                if tokens[i, j] == 0: continue
                if random.random() < 0.15:
                    labels[i, j] = tokens[i, j]
                    r = random.random()
                    if r < 0.8: masked[i, j] = tokenizer.unk_token_id
                    elif r < 0.9: masked[i, j] = random.randint(2, tokenizer.vocab_size - 1)
        return masked, labels
    
    def save_checkpoint(step_num):
        torch.save({
            'text_encoder_state': {k: v for k, v in model.state_dict().items() if any(x in k for x in ['text_emb', 'text_encoder', 'phoneme_emb', 'phoneme_predictor', 'mlm_head', 'bpe_encoder', 'bpe_gate'])},
            'steps': step_num,
            'config': config
        }, save_path)
        print(f"\nSaved checkpoint to {save_path} at step {step_num}")

    try:
        if use_preprocessed:
            while step < max_steps:
                batch = random.sample(preprocessed_data, min(batch_size, len(preprocessed_data)))
                T_max = max(len(b['char_ids']) for b in batch)
                char_ids = torch.zeros((len(batch), T_max), dtype=torch.long, device=device)
                ph_targets = torch.zeros((len(batch), T_max), dtype=torch.long, device=device)
                char_lens = torch.zeros(len(batch), dtype=torch.long, device=device)
                raw_texts = [b['norm_text'] for b in batch]
                for i, b in enumerate(batch):
                    l_char = len(b['char_ids'])
                    l_ph = len(b['ph_ids'])
                    l = min(l_char, l_ph) # Use minimum to avoid index errors
                    char_ids[i, :l] = b['char_ids'][:l].long()
                    ph_targets[i, :l] = b['ph_ids'][:l].long()
                    char_lens[i] = l
                masked_ids, char_labels = mask_tokens(char_ids, char_tokenizer)
                mlm_logits, ph_logits = model.mlm_forward(masked_ids, char_lens, raw_texts=raw_texts)
                mlm_labels_wrapped = torch.zeros((len(batch), T_max + 2), dtype=torch.long, device=device)
                ph_labels_wrapped = torch.zeros((len(batch), T_max + 2), dtype=torch.long, device=device)
                for i in range(len(batch)):
                    l = char_lens[i].item()
                    mlm_labels_wrapped[i, 1:1+l] = char_labels[i, :l]
                    ph_labels_wrapped[i, 1:1+l] = ph_targets[i, :l]
                loss_mlm = char_criterion(mlm_logits.view(-1, mlm_logits.shape[-1]), mlm_labels_wrapped.view(-1))
                loss_ph = ph_criterion(ph_logits.view(-1, ph_logits.shape[-1]), ph_labels_wrapped.view(-1))
                loss = loss_mlm + 0.5 * loss_ph
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                step += 1; pbar.update(1); pbar.set_postfix(mlm=f"{loss_mlm.item():.4f}", ph=f"{loss_ph.item():.4f}")
                
                if step % 1000 == 0:
                    save_checkpoint(step)
        else:
            buffer = []
            for example in dataset:
                text = example.get('text', '')
                if len(text) < 10: continue
                ids = char_tokenizer.encode(text)
                if len(ids) > 10: buffer.append((text, ids[:256]))
                if len(buffer) >= batch_size:
                    batch_texts = [b[0] for b in buffer[:batch_size]]
                    batch_ids = [b[1] for b in buffer[:batch_size]]
                    T_max = max(len(ids) for ids in batch_ids)
                    char_ids = torch.zeros((batch_size, T_max), dtype=torch.long, device=device)
                    char_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
                    for i, ids in enumerate(batch_ids):
                        char_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                        char_lens[i] = len(ids)
                    masked_ids, char_labels = mask_tokens(char_ids, char_tokenizer)
                    ph_targets = torch.zeros((batch_size, T_max), dtype=torch.long, device=device)
                    for i, txt in enumerate(batch_texts):
                        aligned_ph = aligner.align_text_to_phonemes(txt)
                        l = min(len(aligned_ph), T_max)
                        ph_targets[i, :l] = torch.tensor(aligned_ph[:l], dtype=torch.long)
                    mlm_logits, ph_logits = model.mlm_forward(masked_ids, char_lens, raw_texts=batch_texts)
                    mlm_labels_wrapped = torch.zeros((batch_size, T_max + 2), dtype=torch.long, device=device)
                    ph_labels_wrapped = torch.zeros((batch_size, T_max + 2), dtype=torch.long, device=device)
                    for i in range(batch_size):
                        l = char_lens[i].item()
                        mlm_labels_wrapped[i, 1:1+l] = char_labels[i, :l]
                        ph_labels_wrapped[i, 1:1+l] = ph_targets[i, :l]
                    loss_mlm = char_criterion(mlm_logits.view(-1, mlm_logits.shape[-1]), mlm_labels_wrapped.view(-1))
                    loss_ph = ph_criterion(ph_logits.view(-1, ph_logits.shape[-1]), ph_labels_wrapped.view(-1))
                    loss = loss_mlm + 0.5 * loss_ph
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    step += 1; pbar.update(1); pbar.set_postfix(mlm=f"{loss_mlm.item():.4f}", ph=f"{loss_ph.item():.4f}")
                    buffer = buffer[batch_size:]
                    
                    if step % 1000 == 0:
                        save_checkpoint(step)
                    if step >= max_steps: break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    save_checkpoint(step)
    print(f"Pretraining session complete at step {step}.")

if __name__ == "__main__":
    with open("configs/phonetic_train.json", "r") as f:
        cfg = json.load(f)
    train_phonetic_mlm(cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
