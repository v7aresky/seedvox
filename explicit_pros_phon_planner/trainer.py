import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from seedvox.trainer import SeedVoxTrainer
from seedvox.training.dataset import TokenizedSpeechDataset, collate_fn, LengthGroupedSampler
from .model import ExplicitPlannerModel
from .utils import PhoneticGenerator, collate_phonemes

class ExplicitCollate:
    def __init__(self, ph_generator):
        self.ph_generator = ph_generator
        self.failure_count = 0
        self.total_batches = 0
        self.failure_threshold = 0.1 # Stop if more than 10% of batches fail

    def __call__(self, batch):
        self.total_batches += 1
        # 1. Base Collate (Pads audio/text)
        # raw_texts is now the normalized text from Dataset.__getitem__
        padded_text, padded_audio, t_lens, a_lens, raw_texts, ph_ids_from_ds = collate_fn(batch)
        
        # 2. Use precomputed phonemes if available
        if ph_ids_from_ds is not None:
             return padded_text, padded_audio, t_lens, a_lens, raw_texts, ph_ids_from_ds

        # 3. Parallel G2P (Fallback)
        try:
            # Note: raw_texts is already normalized, so we pass normalize=False
            ph_target_list = self.ph_generator.generate_targets_batch(raw_texts, normalize=False)
            ph_targets = collate_phonemes(ph_target_list)
        except Exception as e:
            self.failure_count += 1
            failure_rate = self.failure_count / self.total_batches
            print(f"Error in Parallel G2P worker: {e}. Failure rate: {failure_rate:.2%}")
            
            if failure_rate > self.failure_threshold:
                raise RuntimeError(f"G2P failure rate exceeded threshold!")

            try:
                # Direct fallback (sequential if batch fails)
                ph_target_list = [self.ph_generator.generate_targets(text, normalize=False) for text in raw_texts]
                ph_targets = collate_phonemes(ph_target_list)
            except Exception as e2:
                ph_targets = torch.zeros((len(raw_texts), 2), dtype=torch.long)
        
        return padded_text, padded_audio, t_lens, a_lens, raw_texts, ph_targets

class ExplicitTrainer(SeedVoxTrainer):
    """
    Trainer for the ExplicitPlannerModel.
    Handles ground-truth phoneme generation for the planner loss in parallel.
    """
    def __init__(self, config, device, resume_path=None, ref_wav=None, g2p_backend='espeak', num_workers=None):
        # We call super().__init__ first
        super().__init__(config, device, resume_path=resume_path, ref_wav=ref_wav, g2p_backend=g2p_backend)
        
        if num_workers is not None:
            self.cfg['training']['num_workers'] = num_workers
        
        # Replace the model with the Explicit version
        self.model = ExplicitPlannerModel(config, self.tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
        self.ema_model = ExplicitPlannerModel(config, self.tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
        self.ema_model.eval()
        for p in self.ema_model.parameters(): p.requires_grad = False

        # Load phonetic pretrain if available (Critical fix: super().__init__ loads it but we overwrite model above)
        ph_pretrain_path = self.cfg['training'].get('ph_pretrain_path')
        if ph_pretrain_path and os.path.exists(ph_pretrain_path):
            print(f"Loading pre-trained phonetic weights from {ph_pretrain_path}...")
            ph_ckpt = torch.load(ph_pretrain_path, map_location=device, weights_only=False)
            ph_sd = ph_ckpt.get('model_state', ph_ckpt)
            
            # Synchronize ph_decoder_emb with phonetic_planner.phoneme_emb
            if 'phonetic_planner.phoneme_emb.weight' in ph_sd and hasattr(self.model, 'ph_decoder_emb'):
                with torch.no_grad():
                    self.model.ph_decoder_emb.weight.copy_(ph_sd['phonetic_planner.phoneme_emb.weight'])
                    self.ema_model.ph_decoder_emb.weight.copy_(ph_sd['phonetic_planner.phoneme_emb.weight'])
            
            self.model.load_state_dict(ph_sd, strict=False)
            self.ema_model.load_state_dict(ph_sd, strict=False)
        
        # Initialize the wrapper generator correctly
        self.ph_generator = PhoneticGenerator(backend=g2p_backend, phoneme_vocab_size=128)
        
        # Override the DataLoader to use the Parallel G2P Collator
        self._reinit_dataloader()

        # Define weights to prioritize EOS token (index 128)
        # Assuming phoneme_vocab_size + 1 = 129
        weights = torch.ones(129, device=device)
        weights[128] = 2.0 
        self.ph_planner_criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)

        # Load weights
        if resume_path and os.path.exists(resume_path):
            print(f"Loading weights from {resume_path} into ExplicitPlannerModel...")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            self.ema_model.load_state_dict(state_dict, strict=False)
            if isinstance(ckpt, dict):
                self.global_step = ckpt.get('step', 0)
                self.start_epoch = ckpt.get('epoch', 0)

        # Re-initialize Optimizer for new parameters
        from transformers import get_cosine_schedule_with_warmup
        lr = self.cfg['training'].get('lr', 1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, self.cfg['training'].get('warmup_steps', 100), 
            len(self.loader) * self.cfg['training'].get('epochs', 100)
        )

    def _reinit_dataloader(self):
        """Re-initializes the DataLoader with the parallel G2P collator."""
        # Reuse dataset from super() if available to avoid double loading
        if hasattr(self, 'loader'):
            train_ds = self.loader.dataset
        else:
            train_paths = self.cfg['training']['train_tokens_path']
            if isinstance(train_paths, str): train_paths = [train_paths]
            for p in train_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Training data not found: {p} (from config['training']['train_tokens_path'])")
            full_ds = TokenizedSpeechDataset(train_paths, self.tokenizer)
            val_ratio = self.cfg['training'].get('val_ratio', 0.05)
            n_val = max(1, int(len(full_ds) * val_ratio))
            n_train = len(full_ds) - n_val
            indices = torch.randperm(len(full_ds)).tolist()
            train_ds = Subset(full_ds, indices[:n_train])
        
        # Use our custom collator
        collate = ExplicitCollate(self.ph_generator)
        
        self.loader = DataLoader(
            train_ds, 
            batch_size=self.cfg['training']['batch_size'], 
            sampler=LengthGroupedSampler(train_ds, self.cfg['training']['batch_size']), 
            collate_fn=collate, 
            pin_memory=True, 
            num_workers=self.cfg['training'].get('num_workers', 4)
        )

    def _compute_loss(self, batch):
        padded_text, padded_audio, t_lens, a_lens, raw_texts, ph_targets = batch
        
        padded_text = padded_text.to(self.device)
        padded_audio = padded_audio.to(self.device)
        t_lens, a_lens = t_lens.to(self.device), a_lens.to(self.device)
        ph_targets = ph_targets.to(self.device)
        
        bpe_ids, bpe_lens, char_to_bpe = None, None, None
        if self.bpe_collator:
            bpe_ids, bpe_lens, char_to_bpe = self.bpe_collator.process_batch_texts(raw_texts, t_lens, self.device)
            
        with torch.no_grad():
            mimi_latents = self.mimi.decode_latent(padded_audio[:, :self.model.n_q//2])
            
        logits, targets, ph_planner_logits, jepa_loss, *_ = self.model(
            padded_text, padded_audio[:, :self.model.n_q], t_lens, a_lens, raw_texts=raw_texts,
            phoneme_ids=ph_targets, mimi_latents=mimi_latents,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
            drop_prob=0.1
        )
        
        loss_ar = 0
        for k in range(self.model.n_q):
            loss_ar += self.criterion(logits[k].reshape(-1, logits.shape[-1]), targets[:, k].reshape(-1))
        loss_ar /= self.model.n_q
        
        loss_ph_planner = self.ph_planner_criterion(
            ph_planner_logits.reshape(-1, ph_planner_logits.shape[-1]),
            ph_targets[:, 1:].reshape(-1)
        )
        
        loss_jepa = jepa_loss if jepa_loss is not None else torch.tensor(0.0, device=self.device)
        
        total_loss = (loss_ar + 
                      self.cfg['training'].get('ph_planner_weight', 1.0) * loss_ph_planner + 
                      self.cfg['training'].get('jepa_weight', 2.0) * loss_jepa)
        
        return total_loss, loss_ar, loss_jepa, loss_ph_planner

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--g2p", default="espeak")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = json.load(f)
        
    trainer = ExplicitTrainer(cfg, torch.device(args.device), resume_path=args.resume, 
                              g2p_backend=args.g2p, num_workers=args.num_workers)
    trainer.train()
