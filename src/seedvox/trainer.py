import os
import sys
import json
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

# SeedVox internal imports
from seedvox.model import JEPAProsodyHybridModel
from seedvox.modules.mimi import get_mimi_model
from seedvox.bpe_char_encoder import BPECharCollator
from seedvox.utils.tokenizer import PhonemeTokenizer, CharTokenizer
from seedvox.utils.text import PhoneticAligner, get_phoneme_generator, collapse_duplicates
from seedvox.training.dataset import TokenizedSpeechDataset, collate_fn, LengthGroupedSampler

class SeedVoxTrainer:
    def __init__(self, config, device, resume_path=None, ref_wav=None, g2p_backend='g2p_en'):
        print("DEBUG: Initializing SeedVoxTrainer...")
        self.cfg, self.device = config, device
        self.tokenizer = CharTokenizer()
        self.ph_tokenizer = PhonemeTokenizer()
        self.ref_wav_path = ref_wav
        
        self.ph_generator = get_phoneme_generator(g2p_backend)
        self.aligner = PhoneticAligner(generator=self.ph_generator)
        
        # 1. Initialize SOTA Hybrid Model
        self.model = JEPAProsodyHybridModel(config, self.tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
        
        # EMA initialization
        self.ema_decay = self.cfg['training'].get('ema_decay', 0.999)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False
        
        # BPE Collator
        self.bpe_collator = None
        if getattr(self.model, 'bpe_encoder', None) is not None:
            self.bpe_collator = BPECharCollator(self.model.bpe_encoder)
        
        # 2. Mimi Teacher
        self.mimi = get_mimi_model(device=device, checkpoint_path=config.get('mimi_checkpoint', 'pretrained_models/best_mimi.pt')).eval()
        for p in self.mimi.parameters(): p.requires_grad = False

        # 3. Data
        train_paths = self.cfg['training']['train_tokens_path']
        if isinstance(train_paths, str): train_paths = [train_paths]
        full_ds = TokenizedSpeechDataset(train_paths, self.tokenizer)
        val_ratio = self.cfg['training'].get('val_ratio', 0.05)
        n_val = max(1, int(len(full_ds) * val_ratio))
        n_train = len(full_ds) - n_val
        
        indices = torch.randperm(len(full_ds)).tolist()
        train_ds = Subset(full_ds, indices[:n_train])
        val_ds = Subset(full_ds, indices[n_train:])
        
        self.loader = DataLoader(
            train_ds, batch_size=self.cfg['training']['batch_size'], 
            sampler=LengthGroupedSampler(train_ds, self.cfg['training']['batch_size']), 
            collate_fn=collate_fn, pin_memory=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.cfg['training']['batch_size'],
            shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=1
        )

        # 4. Optimizer
        lr = self.cfg['training'].get('lr', 1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, self.cfg['training'].get('warmup_steps', 100), 
            len(self.loader) * self.cfg['training'].get('epochs', 100)
        )
        
        self.scaler = torch.amp.GradScaler("cuda")
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.ph_criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        self.global_step = 0
        self.start_epoch = 0
        self.writer = SummaryWriter(log_dir=f"logs/seedvox_{time.strftime('%Y%m%d-%H%M%S')}")

        if resume_path and os.path.exists(resume_path):
            print(f"Resuming model weights from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            
            # Robust state_dict extraction
            state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            
            # Load only model weights, skip optimizer state to avoid mismatches
            self.model.load_state_dict(state_dict, strict=False)
            
            # Load EMA if available, else initialize from model
            if 'ema_model' in ckpt:
                self.ema_model.load_state_dict(ckpt['ema_model'])
            else:
                self.ema_model.load_state_dict(state_dict)

            if isinstance(ckpt, dict):
                self.global_step = ckpt.get('step', 0)
                self.start_epoch = ckpt.get('epoch', 0)
            print("Successfully loaded model weights. Optimizer will be initialized from scratch.")

    def _compute_loss(self, batch):
        t_ids, a_toks, t_lens, a_lens, raw_texts = batch
        t_ids, a_toks = t_ids.to(self.device), a_toks.to(self.device)
        t_lens, a_lens = t_lens.to(self.device), a_lens.to(self.device)
        
        B = t_ids.shape[0]
        T_max = t_ids.shape[1]
        
        # 1. Phoneme alignment
        ph_targets = torch.zeros((B, T_max), dtype=torch.long, device=self.device)
        if self.model.use_phonetic:
            for i, text in enumerate(raw_texts):
                aligned_ph = self.aligner.align_text_to_phonemes(text)
                l = min(len(aligned_ph), T_max)
                ph_targets[i, :l] = torch.tensor(aligned_ph[:l], dtype=torch.long)
        
        # 1.5. BPE Encoding
        bpe_ids, bpe_lens, char_to_bpe = None, None, None
        if self.bpe_collator:
            bpe_ids, bpe_lens, char_to_bpe = self.bpe_collator.process_batch_texts(raw_texts, t_lens, self.device)
            
        # 2. Extract Mimi Latents (The "Teacher" for JEPA)
        with torch.no_grad():
            mimi_latents = self.mimi.decode_latent(a_toks[:, :self.model.n_q//2])
            
        # 3. Model Forward
        logits, targets, ph_logits, jepa_loss, phonetic_loss = self.model(
            t_ids, a_toks[:, :self.model.n_q], t_lens, a_lens, raw_texts=raw_texts,
            phoneme_ids=ph_targets, mimi_latents=mimi_latents,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
            drop_prob=0.1
        )
        
        # 4. Losses
        # Audio Loss (AR)
        loss_ar = 0
        for k in range(self.model.n_q):
            loss_ar += self.criterion(logits[k].view(-1, logits.shape[-1]), targets[:, k].reshape(-1))
        loss_ar /= self.model.n_q
        
        # Phonetic Loss
        loss_ph = phonetic_loss if phonetic_loss is not None else torch.tensor(0.0, device=self.device)
        
        # JEPA Loss
        loss_jepa = jepa_loss if jepa_loss is not None else torch.tensor(0.0, device=self.device)
        
        total_loss = loss_ar + self.cfg['training'].get('ph_weight', 0.1) * loss_ph + self.cfg['training'].get('jepa_weight', 2.0) * loss_jepa
        
        return total_loss, loss_ar, loss_jepa, loss_ph

    def train(self):
        print(f"🚀 Starting SeedVox Training")
        output_prefix = self.cfg['training'].get('output_prefix', 'seedvox')
        os.makedirs("checkpoints", exist_ok=True)
        
        for epoch in range(self.start_epoch, self.cfg['training'].get('epochs', 100)):
            self.model.train()
            pbar = tqdm(self.loader, desc=f"Epoch {epoch}")
            
            for batch in pbar:
                step_start = time.time()
                self.optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    loss, loss_ar, loss_jepa, loss_ph = self._compute_loss(batch)
                
                loss_time = time.time() - step_start
                    
                self.scaler.scale(loss).backward()
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
                # EMA Update
                ema_start = time.time()
                with torch.no_grad():
                    for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                        ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
                ema_time = time.time() - ema_start
                
                self.global_step += 1
                pbar.set_postfix(ar=f"{loss_ar.item():.3f}", jepa=f"{loss_jepa.item():.3f}", ph=f"{loss_ph.item():.3f}", 
                                 total=f"{loss.item():.3f}")
                
                if self.global_step % self.cfg['training'].get('log_every', 10) == 0:
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

            # Save Checkpoint after epoch completes
            ckpt_path = f"checkpoints/{output_prefix}_epoch_{epoch}.pt"
            print(f"DEBUG: Attempting to save checkpoint to {ckpt_path} (CWD: {os.getcwd()})")
            try:
                torch.save({
                    'model': self.model.state_dict(),
                    'ema_model': self.ema_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'step': self.global_step,
                    'epoch': epoch + 1,
                    'config': self.cfg
                }, ckpt_path)
                print(f"DEBUG: Successfully called torch.save for {ckpt_path}")
            except Exception as e:
                print(f"DEBUG: ERROR SAVING CHECKPOINT: {e}")
            
            torch.save(self.model.state_dict(), f"checkpoints/{output_prefix}_latest.pt")
            torch.save(self.ema_model.state_dict(), f"checkpoints/{output_prefix}_latest_ema.pt")
            print(f"Epoch {epoch} finished. Checkpoint saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--g2p", default="g2p_en")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = json.load(f)
        
    trainer = SeedVoxTrainer(cfg, torch.device(args.device), resume_path=args.resume, g2p_backend=args.g2p)
    trainer.train()
