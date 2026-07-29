import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from seedvox.modules.mimi import get_mimi_model
from seedvox.bpe_char_encoder import BPECharCollator
from seedvox.utils.tokenizer import CharTokenizer
from seedvox.training.dataset import TokenizedSpeechDataset, LengthGroupedSampler
from .model_fusion import FusionPlannerModel
from .utils import PhoneticGenerator, collate_phonemes
from .trainer import ExplicitCollate


class FusionTrainer:
    """
    Standalone trainer for FusionPlannerModel.
    Does its own clean initialization — no inheritance chain overhead.
    """
    def __init__(self, config, device, resume_path=None, ref_wav=None,
                 g2p_backend='espeak', num_workers=None):
        self.cfg, self.device = config, device
        self.tokenizer = CharTokenizer()
        self.ema_decay = config['training'].get('ema_decay', 0.999)
        self.global_step = 0
        self.start_epoch = 0

        # 1. Model
        print("[FusionTrainer] Creating FusionPlannerModel...")
        self.model = FusionPlannerModel(config, self.tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
        self.ema_model = FusionPlannerModel(config, self.tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
        self.ema_model.eval()
        for p in self.ema_model.parameters(): p.requires_grad = False

        # 2. Phonetic pretrain
        ph_pretrain_path = config['training'].get('ph_pretrain_path')
        if ph_pretrain_path and os.path.exists(ph_pretrain_path):
            print(f"  Loading phonetic pretrain from {ph_pretrain_path}...")
            ph_sd = torch.load(ph_pretrain_path, map_location=device, weights_only=True)
            ph_sd = ph_sd.get('model_state', ph_sd)
            if 'phonetic_planner.phoneme_emb.weight' in ph_sd and hasattr(self.model, 'ph_decoder_emb'):
                with torch.no_grad():
                    self.model.ph_decoder_emb.weight.copy_(ph_sd['phonetic_planner.phoneme_emb.weight'])
                    self.ema_model.ph_decoder_emb.weight.copy_(ph_sd['phonetic_planner.phoneme_emb.weight'])
            self.model.load_state_dict(ph_sd, strict=False)
            self.ema_model.load_state_dict(ph_sd, strict=False)

        # 3. Mimi
        self.mimi = get_mimi_model(device=device, checkpoint_path=config.get('mimi_checkpoint', 'pretrained_models/best_mimi.pt')).eval()
        for p in self.mimi.parameters(): p.requires_grad = False

        # 4. BPE collator
        self.bpe_collator = BPECharCollator(self.model.bpe_encoder) if getattr(self.model, 'bpe_encoder', None) else None

        # 5. G2P + collate
        self.ph_generator = PhoneticGenerator(backend=g2p_backend, phoneme_vocab_size=128)
        collate = ExplicitCollate(self.ph_generator)

        # 6. Dataset
        train_paths = config['training']['train_tokens_path']
        if isinstance(train_paths, str): train_paths = [train_paths]
        for p in train_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Training data not found: {p}")
        full_ds = TokenizedSpeechDataset(train_paths, self.tokenizer)
        val_ratio = config['training'].get('val_ratio', 0.05)
        n_val = max(1, int(len(full_ds) * val_ratio))
        n_train = len(full_ds) - n_val
        indices = torch.randperm(len(full_ds)).tolist()
        train_ds = Subset(full_ds, indices[:n_train])

        num_workers = num_workers if num_workers is not None else config['training'].get('num_workers', 4)
        self.loader = DataLoader(
            train_ds, batch_size=config['training']['batch_size'],
            sampler=LengthGroupedSampler(train_ds, config['training']['batch_size']),
            collate_fn=collate, pin_memory=True, num_workers=num_workers
        )

        # 7. Optimizer
        lr = config['training'].get('lr', 1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, config['training'].get('warmup_steps', 100),
            len(self.loader) * config['training'].get('epochs', 100),
            num_cycles=0.5
        )
        self.scaler = torch.amp.GradScaler("cuda")
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        weights = torch.ones(129, device=device)
        weights[128] = 2.0
        self.ph_planner_criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)

        nq = config['model']['n_q']
        w = torch.ones(nq)
        w[:nq // 2] = 1.5
        w[nq // 2:] = 0.5
        self.level_weights = (w / w.sum() * nq).to(device)

        # 8. Resume checkpoint
        if resume_path and os.path.exists(resume_path):
            print(f"  Resuming from {resume_path}...")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            self.ema_model.load_state_dict(state_dict, strict=False)
            if isinstance(ckpt, dict):
                self.global_step = ckpt.get('step', 0)
                self.start_epoch = ckpt.get('epoch', 0)

        if config['training'].get('compile', False):
            print("[FusionTrainer] Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # 9. Tensorboard
        run_name = f"fusion_{time.strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=f"logs/{run_name}")
        print(f"[FusionTrainer] Ready. Resumed at epoch={self.start_epoch}, step={self.global_step}")

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
            mimi_latents = self.mimi.decode_latent(padded_audio[:, :self.model.n_q // 2])

        logits, targets, ph_planner_logits, jepa_loss, _, latent_pred = self.model(
            padded_text, padded_audio[:, :self.model.n_q], t_lens, a_lens, raw_texts=raw_texts,
            phoneme_ids=ph_targets, mimi_latents=mimi_latents,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
            drop_prob=0.1
        )

        # Level-weighted CE (batched across all codebooks) with n_q curriculum
        B = targets.shape[0]
        n_q = self.model.n_q
        T_audio = targets.shape[2]
        curriculum = self.cfg['training'].get('curriculum_n_q', {})
        if curriculum.get('enabled', False):
            start = curriculum.get('start_codebooks', 4)
            ramp = curriculum.get('ramp_steps', 50000)
            num_enabled = min(n_q, start + int(self.global_step * (n_q - start) / ramp))
        else:
            num_enabled = n_q
        logits_btk = logits.permute(1, 2, 0, 3).reshape(B * T_audio * n_q, logits.shape[-1])
        targets_btk = targets.permute(0, 2, 1).reshape(B * T_audio * n_q)
        weights_ar = torch.zeros(n_q, device=self.device)
        weights_ar[:num_enabled] = self.level_weights[:num_enabled]
        weights_flat = weights_ar[None, None, :].expand(B, T_audio, n_q).reshape(B * T_audio * n_q)
        loss_ar = (F.cross_entropy(logits_btk, targets_btk, reduction='none', ignore_index=-1) * weights_flat).sum() / max(B * T_audio * num_enabled, 1)

        loss_ph = self.ph_planner_criterion(
            ph_planner_logits.reshape(-1, ph_planner_logits.shape[-1]),
            ph_targets[:, 1:].reshape(-1)
        )

        loss_jepa = jepa_loss if jepa_loss is not None else torch.tensor(0.0, device=self.device)

        total_loss = (loss_ar +
                      self.cfg['training'].get('ph_planner_weight', 1.0) * loss_ph +
                      self.cfg['training'].get('jepa_weight', 2.0) * loss_jepa)

        return total_loss, loss_ar, loss_jepa, loss_ph

    def train(self):
        output_prefix = self.cfg['training'].get('output_prefix', 'seedvox_fusion')
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(self.start_epoch, self.cfg['training'].get('epochs', 100)):
            self.model.train()
            pbar = tqdm(self.loader, desc=f"Epoch {epoch} (Fusion)")

            for batch in pbar:
                self.optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    total_loss, loss_ar, loss_jepa, loss_ph = self._compute_loss(batch)

                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                ema_every = self.cfg['training'].get('ema_update_every', 10)
                if self.global_step % ema_every == 0:
                    with torch.no_grad():
                        ema_decay_per_step = self.ema_decay ** ema_every
                        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                            ema_param.lerp_(param, 1 - ema_decay_per_step)

                self.global_step += 1

                pbar.set_postfix(
                    ar=f"{loss_ar.item():.3f}",
                    jepa=f"{loss_jepa.item() if isinstance(loss_jepa, torch.Tensor) else loss_jepa:.3f}",
                    ph=f"{loss_ph.item() if isinstance(loss_ph, torch.Tensor) else loss_ph:.3f}",
                    total=f"{total_loss.item():.3f}"
                )

                if self.global_step % self.cfg['training'].get('log_every', 10) == 0:
                    self.writer.add_scalar("train/loss", total_loss.item(), self.global_step)
                    self.writer.add_scalar("train/loss_jepa", loss_jepa.item() if isinstance(loss_jepa, torch.Tensor) else loss_jepa, self.global_step)
                    self.writer.add_scalar("train/loss_ph", loss_ph.item() if isinstance(loss_ph, torch.Tensor) else loss_ph, self.global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

            ckpt_path = f"checkpoints/{output_prefix}_epoch_{epoch}.pt"
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
            torch.save(self.model.state_dict(), f"checkpoints/{output_prefix}_latest.pt")
            print(f"Epoch {epoch} finished. Checkpoint: {ckpt_path}")
            torch.cuda.empty_cache()


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

    trainer = FusionTrainer(cfg, torch.device(args.device), resume_path=args.resume,
                            g2p_backend=args.g2p, num_workers=args.num_workers)
    trainer.train()
