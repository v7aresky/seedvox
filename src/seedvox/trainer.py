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
from explicit_pros_phon_planner.model import ExplicitPlannerModel
from seedvox.modules.mimi import get_mimi_model
from seedvox.bpe_char_encoder import BPECharCollator
from seedvox.utils.tokenizer import PhonemeTokenizer, CharTokenizer
from explicit_pros_phon_planner.utils import PhoneticGenerator
from seedvox.utils.text import PhoneticAligner
from seedvox.training.dataset import TokenizedSpeechDataset, collate_fn, LengthGroupedSampler

class SeedVoxTrainer:
    def __init__(self, config, device, resume_path=None, ref_wav=None, g2p_backend='g2p_en'):
    
        self.cfg, self.device = config, device
        self.tokenizer = CharTokenizer()
        self.ph_tokenizer = PhonemeTokenizer()
        self.ref_wav_path = ref_wav
        
        self.ph_generator = PhoneticGenerator(backend=g2p_backend, phoneme_vocab_size=128)
        self.aligner = PhoneticAligner(generator=self.ph_generator)
        
        # 1. Initialize Model (Toggle between Implicit and Explicit)
        self.use_explicit = config['model'].get('use_explicit_planner', True)
        if self.use_explicit:
            self.model = ExplicitPlannerModel(config, self.tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
        else:
            self.model = JEPAProsodyHybridModel(config, self.tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
        
        # Load phonetic pretrain if available
        ph_pretrain_path = self.cfg['training'].get('ph_pretrain_path')
        
        if ph_pretrain_path and os.path.exists(ph_pretrain_path):
            # Load phonetic pretrain if available
            ph_ckpt = torch.load(ph_pretrain_path, map_location=device, weights_only=True)
            ph_sd = ph_ckpt.get('model_state', ph_ckpt)

            # Explicitly initialize ph_decoder_emb from phonetic_planner.phoneme_emb if possible
            if 'phonetic_planner.phoneme_emb.weight' in ph_sd and hasattr(self.model, 'ph_decoder_emb'):
                with torch.no_grad():
                    self.model.ph_decoder_emb.weight.copy_(ph_sd['phonetic_planner.phoneme_emb.weight'])

            incompatible_keys = self.model.load_state_dict(ph_sd, strict=False)

        # EMA initialization
        self.ema_decay = self.cfg['training'].get('ema_decay', 0.999)
        self.jepa_n_q = config['model'].get('jepa_n_q', self.model.n_q//2)
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

        # 4. Optimizer with configuration-driven parameter grouping
        param_configs = self.cfg['training'].get('param_groups', [])
        default_lr = self.cfg['training'].get('lr', 1e-4)
        
        assigned_params = {} # Mapping id(p) -> param
        param_groups = []
        
        # Track parameters to ensure each is assigned to only one group
        all_params = {id(p): p for p in self.model.parameters()}
        processed_param_ids = set()
        
        # Create groups based on config
        for pg_config in param_configs:
            keywords = pg_config.get('keywords', [])
            lr = pg_config.get('lr', default_lr)
            group_params = []
            
            for n, p in self.model.named_parameters():
                if id(p) not in processed_param_ids and any(k in n for k in keywords):
                    group_params.append(p)
                    processed_param_ids.add(id(p))
            
            if group_params:
                print(f"DEBUG: Added {len(group_params)} params to group with LR {lr} (keywords: {keywords})")
                param_groups.append({'params': group_params, 'lr': lr})
                
        # Remaining parameters (default group)
        remaining_params = [p for p in all_params.values() if id(p) not in processed_param_ids]
        if remaining_params:
            print(f"DEBUG: Added {len(remaining_params)} params to default group with LR {default_lr}")
            param_groups.append({'params': remaining_params, 'lr': default_lr})
            
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
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
            print(f"Resuming full state from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            
            # Robust state_dict extraction
            state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=True)
            
            if 'ema_model' in ckpt:
                self.ema_model.load_state_dict(ckpt['ema_model'])
            
            if isinstance(ckpt, dict):
                if 'optimizer' in ckpt:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                if 'scheduler' in ckpt:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                if 'scaler' in ckpt:
                    self.scaler.load_state_dict(ckpt['scaler'])
                self.global_step = ckpt.get('step', 0)
                self.start_epoch = ckpt.get('epoch', 0)
            print(f"Successfully resumed at Epoch {self.start_epoch}, Step {self.global_step}")

    def _compute_loss(self, batch):
        t_ids, a_toks, t_lens, a_lens, raw_texts, ph_ids = batch
        t_ids, a_toks = t_ids.to(self.device), a_toks.to(self.device)
        t_lens, a_lens = t_lens.to(self.device), a_lens.to(self.device)
        if ph_ids is not None:
            ph_ids = ph_ids.to(self.device)
        
        B = t_ids.shape[0]
        T_max = t_ids.shape[1]
        
        # 1. Phoneme generation
        use_precomputed = self.cfg['training'].get('use_precomputed_phonemes', True)
        
        if self.use_explicit:
            if ph_ids is not None and use_precomputed:
                ph_targets = ph_ids
            else:
                # Use raw sequences with SOS/EOS (matching pretraining) - Fallback
                raw_ph_ids = self.ph_generator.generate_targets_batch(raw_texts)
                max_ph_len = max(len(ids) for ids in raw_ph_ids)
                ph_targets = torch.zeros((B, max_ph_len), dtype=torch.long, device=self.device)
                for i, ids in enumerate(raw_ph_ids):
                    ph_targets[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        else:
            # Use Aligner style (char-aligned) for implicit model
            ph_targets = torch.zeros((B, T_max), dtype=torch.long, device=self.device)
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
            mimi_latents = self.mimi.decode_latent(a_toks[:, :self.jepa_n_q])
            
        # 3. Model Forward
        logits, targets, ph_logits, jepa_loss, phonetic_loss = self.model(
            t_ids, a_toks[:, :self.model.n_q], t_lens, a_lens, raw_texts=raw_texts,
            phoneme_ids=ph_targets, mimi_latents=mimi_latents,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
            drop_prob=0.1
        )
        
        # 4. Losses
        # Audio Loss (AR)
        curriculum = self.cfg['training'].get('curriculum_n_q', {})
        if curriculum.get('enabled', False):
            start = curriculum.get('start_codebooks', 4)
            ramp = curriculum.get('ramp_steps', 50000)
            num_enabled = min(self.model.n_q, start + int(self.global_step * (self.model.n_q - start) / ramp))
        else:
            num_enabled = self.model.n_q

        loss_ar = 0
        for k in range(num_enabled):
            loss_ar += self.criterion(logits[k].view(-1, logits.shape[-1]), targets[:, k].reshape(-1))
        loss_ar /= num_enabled
        
        # Phonetic Loss
        loss_ph = torch.tensor(0.0, device=self.device)
        if ph_logits is not None:
            # Match shapes for AR loss (predicts ph_targets[:, 1:])
            shifted_ph = ph_targets[:, 1:]
            T_logit = ph_logits.shape[1]
            T_target = shifted_ph.shape[1]
            T_min = min(T_logit, T_target)
            
            loss_ph = self.ph_criterion(
                ph_logits[:, :T_min, :].reshape(-1, ph_logits.shape[-1]),
                shifted_ph[:, :T_min].reshape(-1)
            )
        elif phonetic_loss is not None:
            loss_ph = phonetic_loss

        
        # JEPA Loss
        loss_jepa = jepa_loss if jepa_loss is not None else torch.tensor(0.0, device=self.device)
        
        jepa_weight = self.cfg['training'].get('jepa_weight', 2.0)
        ph_weight = self.cfg['training'].get('ph_weight', 0.1)
        
        total_loss = loss_ar + ph_weight * loss_ph + jepa_weight * loss_jepa
        
        return total_loss, loss_ar, loss_jepa, loss_ph

    def train(self):
        print(f"🚀 Starting SeedVox Training")
        output_prefix = self.cfg['training'].get('output_prefix', 'seedvox')
        os.makedirs("checkpoints", exist_ok=True)
        
        for epoch in range(self.start_epoch, self.cfg['training'].get('epochs', 100)):
            self.model.train()
            pbar = tqdm(self.loader, desc=f"Epoch {epoch}")
            
            for batch in pbar:
                self.optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    total_loss, loss_ar, loss_jepa, loss_ph = self._compute_loss(batch)

                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                # EMA Update - Optimized with lerp_
                with torch.no_grad():
                    for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                        ema_param.lerp_(param, 1 - self.ema_decay)

                self.global_step += 1

                # Detach losses for logging to ensure graph is cleared
                loss_val = total_loss.item()
                loss_ar_val = loss_ar.item()
                loss_jepa_val = loss_jepa.item() if isinstance(loss_jepa, torch.Tensor) else loss_jepa
                loss_ph_val = loss_ph.item() if isinstance(loss_ph, torch.Tensor) else loss_ph

                pbar.set_postfix(ar=f"{loss_ar_val:.3f}", jepa=f"{loss_jepa_val:.3f}", ph=f"{loss_ph_val:.3f}", 
                                 total=f"{loss_val:.3f}")
                if self.global_step % self.cfg['training'].get('log_every', 10) == 0:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
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
                    'scaler': self.scaler.state_dict(),
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
            
            # Clear cache to prevent fragmentation buildup
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--g2p", default="g2p_en")
    parser.add_argument("--logdir", default=None, help="Optional log directory (for CLI compatibility)")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = json.load(f)
        
    trainer = SeedVoxTrainer(cfg, torch.device(args.device), resume_path=args.resume, g2p_backend=args.g2p)
    trainer.train()
