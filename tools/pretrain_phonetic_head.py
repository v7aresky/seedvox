import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
import random

# Import from current project structure
from src.seedvox.utils.tokenizer import CharTokenizer, PhonemeTokenizer
from explicit_pros_phon_planner.model import ExplicitPlannerModel
from explicit_pros_phon_planner.utils import collate_phonemes

MAX_LEN = 512

class PhoneticStreamingDataset(IterableDataset):
    def __init__(self, dataset_path, char_tokenizer, max_len=512):
        self.dataset_path = dataset_path
        self.char_tokenizer = char_tokenizer
        self.max_len = max_len
        # Using streaming=True for large datasets
        self.dataset = load_dataset("json", data_files=dataset_path, split="train", streaming=True)

    def __iter__(self):
        for example in self.dataset:
            ph_ids = example.get('ph_ids')
            norm_text = example.get('norm_text')
            
            if not ph_ids or not norm_text or len(norm_text) < 10:
                continue
            
            # Use normalize=False because norm_text is already normalized in preprocessed_data.jsonl
            ids = self.char_tokenizer.encode(norm_text, normalize=False)
            
            if len(ids) > 10:
                # Yield as tensors to avoid conversion in the main loop
                yield {
                    "char_ids": torch.tensor(ids[:self.max_len], dtype=torch.long),
                    "ph_targets": torch.tensor(ph_ids[:self.max_len], dtype=torch.long)
                }

def collate_fn(batch):
    char_ids_list = [item["char_ids"] for item in batch]
    ph_targets_list = [item["ph_targets"] for item in batch]
    
    # Pad sequences efficiently
    char_ids = torch.nn.utils.rnn.pad_sequence(char_ids_list, batch_first=True, padding_value=0)
    ph_targets = torch.nn.utils.rnn.pad_sequence(ph_targets_list, batch_first=True, padding_value=0)
    
    # Calculate lenses (min of char and ph targets as in original script)
    batch_size = len(batch)
    char_lens = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        char_lens[i] = min(len(char_ids_list[i]), len(ph_targets_list[i]))
        # Also ensure we don't exceed the padded size
        char_lens[i] = min(char_lens[i], char_ids.shape[1])
        
    return char_ids, ph_targets, char_lens

def pretrain_phonetic_head(config, device, max_steps=15000, lr=1e-4, batch_size=32, save_path='pretrained_models/phonetic_head_pretrain.pt', resume=False):
    """
    Optimized pretraining for ExplicitPlannerModel phonetic components.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    char_tokenizer = CharTokenizer()
    
    # Initialize ExplicitPlannerModel
    model = ExplicitPlannerModel(config, char_tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
    
    # Optimize text-related parameters
    text_params = [p for n, p in model.named_parameters() if any(x in n for x in ['text_emb', 'text_encoder', 'phoneme_head', 'phonetic_planner', 'bpe_encoder', 'bpe_gate']) and p.requires_grad]
    optimizer = torch.optim.AdamW(text_params, lr=lr)
    ph_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    start_step = 0
    if resume and os.path.exists(save_path):
        print(f"Resuming from checkpoint: {save_path}")
        checkpoint = torch.load(save_path, map_location=device)
        # Handle both compiled and non-compiled state dicts
        model_state = checkpoint['model_state']
        # If the current model is NOT compiled but the checkpoint IS, we need to strip prefixes
        # If the current model IS compiled, we'll compile first then load
        model.load_state_dict(model_state, strict=False)
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scaler_state' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state'])
        start_step = checkpoint.get('steps', 0)
        print(f"Resumed from step {start_step}")

    # Use torch.compile for a significant speedup on modern GPUs
    try:
        print("Compiling model for faster training...")
        # Compile the key parts used in this pretraining
        model.text_encoder = torch.compile(model.text_encoder)
        model.phonetic_planner = torch.compile(model.phonetic_planner)
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}. Proceeding without compilation.")
    
    model.train()
    step = start_step
    target_steps = start_step + max_steps
    pbar = tqdm(total=target_steps, desc="Phonetic Head Pretraining", initial=step)
        
    # Load dataset with DataLoader for prefetching
    dataset = PhoneticStreamingDataset("dataset/preprocessed_data.jsonl", char_tokenizer, max_len=MAX_LEN)
    # Using num_workers=1 because the dataset has only 1 shard (single file)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1)
    print(f"Loaded preprocessed dataset from dataset/preprocessed_data.jsonl")
            
    def save_checkpoint(step_num):
        # Save only the relevant phonetic/text encoder parts
        # We need to handle compiled models which prefix keys with '_orig_mod.'
        raw_sd = model.state_dict()
        state_dict = {}
        for k, v in raw_sd.items():
            clean_k = k.replace('_orig_mod.', '')
            if any(x in clean_k for x in ['text_emb', 'text_encoder', 'phonetic_planner', 'bpe_encoder', 'bpe_gate']):
                state_dict[clean_k] = v
                
        torch.save({
            'model_state': state_dict,
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'steps': step_num,
            'config': config
        }, save_path)
        print(f"\nSaved checkpoint to {save_path} at step {step_num}")

    # Counter for samples in the dataloader
    batch_idx = 0
    try:
        for char_ids, ph_targets, char_lens in dataloader:
            batch_idx += 1
            if batch_idx <= start_step:
                if batch_idx % 100 == 0:
                    pbar.set_description(f"Fast-forwarding dataset: {batch_idx}/{start_step}")
                continue

            char_ids = char_ids.to(device)
            ph_targets = ph_targets.to(device)
            char_lens = char_lens.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Get enriched features
                text_feat = model.get_enriched_text_feat(char_ids, char_lens)
                
                # Forward pass: Phonetic Planner
                ph_logits = model.phonetic_planner(text_feat, ph_targets)
                
                # Loss: Shift targets to match logits (predicts ph_targets[:, 1:])
                shifted_targets = ph_targets[:, 1:]
                
                # Ensure ph_logits and shifted_targets match in length
                T_logit = ph_logits.shape[1]
                T_target = shifted_targets.shape[1]
                T_min = min(T_logit, T_target)
                
                loss = ph_criterion(
                    ph_logits[:, :T_min, :].reshape(-1, ph_logits.shape[-1]), 
                    shifted_targets[:, :T_min].reshape(-1)
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            step += 1
            pbar.update(1)
            pbar.set_description("Phonetic Head Pretraining")
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            if step % 1000 == 0:
                save_checkpoint(step)
            if step >= target_steps: break
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    save_checkpoint(step)
    print(f"Pretraining session complete at step {step}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pretrain Phonetic Head")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config file")
    parser.add_argument("--max_steps", type=int, default=15000, help="Maximum number of steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--save_path", type=str, default='pretrained_models/phonetic_head_pretrain.pt', help="Path to save checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_phonetic_head(
        cfg, 
        device, 
        max_steps=args.max_steps, 
        lr=args.lr, 
        batch_size=args.batch_size, 
        save_path=args.save_path,
        resume=args.resume
    )
