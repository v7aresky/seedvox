import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .trainer import ExplicitTrainer, ExplicitCollate
from .model_fusion import FusionPlannerModel

class FusionTrainer(ExplicitTrainer):
    """
    Dedicated Trainer for the FusionPlannerModel.
    Uses the Unified Linguistic Encoder (LinguisticFusion) path.
    """
    def __init__(self, config, device, resume_path=None, ref_wav=None, g2p_backend='espeak', num_workers=None):
        # We call super().__init__ which initializes the base trainer and then ExplicitTrainer
        # Note: ExplicitTrainer.__init__ will initialize self.model as ExplicitPlannerModel,
        # so we will overwrite it with FusionPlannerModel.
        super().__init__(config, device, resume_path=resume_path, ref_wav=ref_wav, g2p_backend=g2p_backend, num_workers=num_workers)
        
        print("\033[94m[FusionTrainer]\033[0m Initializing with FusionPlannerModel (Unified Linguistic Encoder)")
        
        # Replace the model with the Fusion version
        self.model = FusionPlannerModel(config, self.tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
        self.ema_model = FusionPlannerModel(config, self.tokenizer.vocab_size, phoneme_vocab_size=128).to(device)
        self.ema_model.eval()
        for p in self.ema_model.parameters(): p.requires_grad = False

        # Reload phonetic pretrain weights if needed (since we just overrode the model)
        ph_pretrain_path = self.cfg['training'].get('ph_pretrain_path')
        if ph_pretrain_path and os.path.exists(ph_pretrain_path):
            print(f"  -> Re-loading pre-trained phonetic weights into Fusion model...")
            ph_ckpt = torch.load(ph_pretrain_path, map_location=device, weights_only=False)
            ph_sd = ph_ckpt.get('model_state', ph_ckpt)
            
            # Synchronize ph_decoder_emb with phonetic_planner.phoneme_emb
            if 'phonetic_planner.phoneme_emb.weight' in ph_sd and hasattr(self.model, 'ph_decoder_emb'):
                with torch.no_grad():
                    self.model.ph_decoder_emb.weight.copy_(ph_sd['phonetic_planner.phoneme_emb.weight'])
                    self.ema_model.ph_decoder_emb.weight.copy_(ph_sd['phonetic_planner.phoneme_emb.weight'])
            
            self.model.load_state_dict(ph_sd, strict=False)
            self.ema_model.load_state_dict(ph_sd, strict=False)

        # Reload resume weights if needed
        if resume_path and os.path.exists(resume_path):
            print(f"  -> Re-loading checkpoint weights from {resume_path}...")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            self.ema_model.load_state_dict(state_dict, strict=False)

        # Re-initialize Optimizer for the new Fusion parameters (specifically self.linguistic_fusion)
        from transformers import get_cosine_schedule_with_warmup
        lr = self.cfg['training'].get('lr', 1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, self.cfg['training'].get('warmup_steps', 100), 
            len(self.loader) * self.cfg['training'].get('epochs', 100)
        )

    def train(self):
        """
        Custom train loop for FusionTrainer to ensure all losses (AR, JEPA, PH) 
        are visible in the progress bar.
        """
        print(f"\033[94m[FusionTrainer]\033[0m Starting Training loop")
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

                # EMA Update
                with torch.no_grad():
                    for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                        ema_param.lerp_(param, 1 - self.ema_decay)

                self.global_step += 1

                # Logging to Progress Bar
                loss_val = total_loss.item()
                loss_ar_val = loss_ar.item()
                loss_jepa_val = loss_jepa.item() if isinstance(loss_jepa, torch.Tensor) else loss_jepa
                loss_ph_val = loss_ph.item() if isinstance(loss_ph, torch.Tensor) else loss_ph

                pbar.set_postfix(
                    ar=f"{loss_ar_val:.3f}", 
                    jepa=f"{loss_jepa_val:.3f}", 
                    ph=f"{loss_ph_val:.3f}", 
                    total=f"{loss_val:.3f}"
                )
                
                if self.global_step % self.cfg['training'].get('log_every', 10) == 0:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/loss_jepa", loss_jepa_val, self.global_step)
                    self.writer.add_scalar("train/loss_ph", loss_ph_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

            # Save Checkpoints
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
    
    print("\033[92m[READY]\033[0m Starting Fusion Training loop...")
    trainer.train()
