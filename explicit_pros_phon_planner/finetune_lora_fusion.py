import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import torchaudio
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from seedvox.training.dataset import TokenizedSpeechDataset, collate_fn as base_collate_fn, LengthGroupedSampler
from seedvox.modules.mimi import get_mimi_model
from .trainer_fusion import FusionTrainer
from .model_fusion import FusionPlannerModel
from .utils import PhoneticGenerator, collate_phonemes
from .trainer import ExplicitCollate


def collate_fn_with_wav(batch):
    """Like base collate_fn but also returns padded_wav for reconstruction loss."""
    text_ids, audio_tokens, raw_texts, ph_ids, wavs = zip(*batch)

    t_lens = torch.tensor([len(t) for t in text_ids], dtype=torch.long)
    t_max = ((t_lens.max().item() + 7) // 8) * 8
    padded_text = torch.full((len(text_ids), t_max), 0, dtype=torch.long)
    for i, t in enumerate(text_ids): padded_text[i, :len(t)] = t

    k_max = max(a.shape[0] for a in audio_tokens)
    a_max = max(a.shape[1] for a in audio_tokens)
    a_max = ((a_max + 7) // 8) * 8
    padded_audio = torch.zeros(len(audio_tokens), k_max, a_max, dtype=torch.long)
    for i, a in enumerate(audio_tokens):
        padded_audio[i, :a.shape[0], :a.shape[1]] = a
    a_lens = torch.tensor([a.shape[1] for a in audio_tokens], dtype=torch.long)

    padded_ph = None
    if any(p is not None for p in ph_ids):
        ph_list = [p if p is not None else torch.tensor([], dtype=torch.long) for p in ph_ids]
        ph_lens = torch.tensor([len(p) for p in ph_list], dtype=torch.long)
        p_max = ph_lens.max().item()
        padded_ph = torch.full((len(ph_list), p_max), 0, dtype=torch.long)
        for i, p in enumerate(ph_list):
            if len(p) > 0: padded_ph[i, :len(p)] = p

    padded_wav = None
    if any(w is not None for w in wavs):
        wav_list = [w if w is not None else torch.tensor([], dtype=torch.float32) for w in wavs]
        wav_lens = torch.tensor([w.shape[0] for w in wav_list], dtype=torch.long)
        w_max = wav_lens.max().item()
        padded_wav = torch.zeros(len(wav_list), w_max, dtype=torch.float32)
        for i, w in enumerate(wav_list):
            if w.shape[0] > 0: padded_wav[i, :w.shape[0]] = w

    return padded_text, padded_audio, t_lens, a_lens, raw_texts, padded_ph, padded_wav


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros((r, in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r), device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(p=dropout)

        for p in self.original_linear.parameters():
            p.requires_grad = False

    @property
    def weight(self):
        return self.original_linear.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.original_linear.bias

    @property
    def in_features(self):
        return self.original_linear.in_features

    @property
    def out_features(self):
        return self.original_linear.out_features

    def forward(self, x):
        original_output = self.original_linear(x)
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        lora_output = (self.lora_dropout(x) @ lora_A.t() @ lora_B.t()) * self.scaling
        return original_output + lora_output


class LoRAConv1d(nn.Module):
    def __init__(self, original_conv, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_conv = original_conv
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size[0]
        device = original_conv.weight.device
        dtype = original_conv.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros((r, in_channels, kernel_size), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_channels, r, 1), device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(p=dropout)

        for p in self.original_conv.parameters():
            p.requires_grad = False

    @property
    def weight(self): return self.original_conv.weight
    @property
    def bias(self): return self.original_conv.bias
    @property
    def in_channels(self): return self.original_conv.in_channels
    @property
    def out_channels(self): return self.original_conv.out_channels
    @property
    def kernel_size(self): return self.original_conv.kernel_size
    @property
    def stride(self): return self.original_conv.stride
    @property
    def padding(self): return self.original_conv.padding
    @property
    def dilation(self): return self.original_conv.dilation
    @property
    def groups(self): return self.original_conv.groups

    def forward(self, x):
        original_output = self.original_conv(x)
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        x_lora = F.conv1d(
            self.lora_dropout(x), lora_A,
            stride=self.original_conv.stride,
            padding=self.original_conv.padding,
            dilation=self.original_conv.dilation,
            groups=self.original_conv.groups
        )
        x_lora = F.conv1d(x_lora, lora_B)
        return original_output + x_lora * self.scaling


class LoRAConvTranspose1d(nn.Module):
    def __init__(self, original_convtr, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_convtr = original_convtr
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_channels = original_convtr.in_channels
        out_channels = original_convtr.out_channels
        kernel_size = original_convtr.kernel_size[0]
        groups = original_convtr.groups
        device = original_convtr.weight.device
        dtype = original_convtr.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros((r, in_channels, 1), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((r, out_channels // groups, kernel_size), device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(p=dropout)

        for p in self.original_convtr.parameters():
            p.requires_grad = False

    @property
    def weight(self): return self.original_convtr.weight
    @property
    def bias(self): return self.original_convtr.bias
    @property
    def in_channels(self): return self.original_convtr.in_channels
    @property
    def out_channels(self): return self.original_convtr.out_channels
    @property
    def kernel_size(self): return self.original_convtr.kernel_size
    @property
    def stride(self): return self.original_convtr.stride
    @property
    def padding(self): return self.original_convtr.padding
    @property
    def dilation(self): return self.original_convtr.dilation
    @property
    def groups(self): return self.original_convtr.groups

    def forward(self, x):
        original_output = self.original_convtr(x)
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        x_lora = F.conv1d(self.lora_dropout(x), lora_A)
        x_lora = F.conv_transpose1d(
            x_lora, lora_B,
            stride=self.original_convtr.stride,
            padding=self.original_convtr.padding,
            dilation=self.original_convtr.dilation,
            groups=self.original_convtr.groups if self.original_convtr.groups == 1 or self.r % self.original_convtr.groups == 0 else 1
        )
        if x_lora.shape[-1] > original_output.shape[-1]:
            x_lora = x_lora[..., :original_output.shape[-1]]
        return original_output + x_lora * self.scaling


def inject_lora(model, rank=8, alpha=16, skip_if_lora=True, targets=None):
    print(f"Injecting LoRA (rank={rank}, alpha={alpha}) into model...")

    count = 0
    target_list = targets or [
        "text_encoder", "decoder_layers", "dep_transformer",
        "dep_in", "dep_layers", "speaker_encoder",
        "prosody_encoder", "audio_prenet",
        "jepa_planner", "phonetic_planner",
        "linguistic_fusion", "ph_projection",
        "film_prs", "film_phn", "speaker_adapter",
        "bpe_encoder"
    ]
    replacements = []

    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            full_child_name = f"{name}.{child_name}" if name else child_name

            if any(t in full_child_name for t in target_list):
                if isinstance(child_module, nn.Linear) and not isinstance(child_module, (LoRALinear, LoRAConv1d, LoRAConvTranspose1d)):
                    replacements.append((module, child_name, child_module, "linear"))
                elif isinstance(child_module, nn.Conv1d) and not isinstance(child_module, (LoRAConv1d, LoRAConvTranspose1d)):
                    replacements.append((module, child_name, child_module, "conv1d"))
                elif isinstance(child_module, nn.ConvTranspose1d) and not isinstance(child_module, (LoRAConv1d, LoRAConvTranspose1d)):
                    replacements.append((module, child_name, child_module, "convtr1d"))

    for parent, child_name, child_module, mtype in replacements:
        curr = getattr(parent, child_name)
        if mtype == "linear":
            if isinstance(curr, nn.Linear) and not (skip_if_lora and isinstance(curr, LoRALinear)):
                setattr(parent, child_name, LoRALinear(child_module, r=rank, alpha=alpha))
                count += 1
        elif mtype == "conv1d":
            if isinstance(curr, nn.Conv1d) and not (skip_if_lora and isinstance(curr, LoRAConv1d)):
                setattr(parent, child_name, LoRAConv1d(child_module, r=rank, alpha=alpha))
                count += 1
        elif mtype == "convtr1d":
            if isinstance(curr, nn.ConvTranspose1d) and not (skip_if_lora and isinstance(curr, LoRAConvTranspose1d)):
                setattr(parent, child_name, LoRAConvTranspose1d(child_module, r=rank, alpha=alpha))
                count += 1
    print(f"Successfully injected LoRA into {count} layers.")
    return model


MIMI_DECODER_TARGETS = [
    "decoder_transformer",
    "decoder",
    "upsample",
]


class FusionLoRATrainer:
    def __init__(self, config, device, base_checkpoint, g2p_backend='espeak',
                 resume_lora=None, ref_wav=None, num_workers=None,
                 lora_rank=16, lora_alpha=32, adapt_mimi=False,
                 mimi_lora_rank=8, mimi_lora_alpha=16):
        self.cfg, self.device = config, device
        self.tokenizer = None
        self.ref_wav_path = ref_wav
        self.adapt_mimi = adapt_mimi
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.mimi_lora_rank = mimi_lora_rank
        self.mimi_lora_alpha = mimi_lora_alpha

        # 1. Initialize tokenizer first to know real vocab size
        if self.tokenizer is None:
            from seedvox.utils.tokenizer import CharTokenizer
            self.tokenizer = CharTokenizer()

        # 2. Initialize Base Model with actual tokenizer vocab size
        self.model = FusionPlannerModel(config, self.tokenizer.vocab_size, phoneme_vocab_size=128).to(device)

        print(f"Loading base checkpoint: {base_checkpoint}")
        ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)

        # 3. Freeze all model parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # 3. Setup Mimi
        self.mimi = get_mimi_model(device=device, checkpoint_path=config.get('mimi_checkpoint', 'pretrained_models/best_mimi.pt'))
        for p in self.mimi.parameters():
            p.requires_grad = False

        # 4. Inject LoRA into model
        self.model = inject_lora(self.model, rank=lora_rank, alpha=lora_alpha)

        # Optionally inject into Mimi decoder (with MIMI_DECODER_TARGETS)
        if adapt_mimi:
            print("Enabling LoRA adaptation for Mimi decoder...")
            self.mimi = inject_lora(self.mimi, rank=mimi_lora_rank, alpha=mimi_lora_alpha,
                                    targets=MIMI_DECODER_TARGETS)

        self.model.to(device)
        self.mimi.to(device)

        # 5. Phoneme generator & collator
        self.ph_generator = PhoneticGenerator(backend=g2p_backend, phoneme_vocab_size=128)

        # 6. Dataset (with wav loading for reconstruction loss)
        train_paths = self.cfg['training']['train_tokens_path']
        if isinstance(train_paths, str):
            train_paths = [train_paths]
        for p in train_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Training data not found: {p} (from config['training']['train_tokens_path'])")
        full_ds = TokenizedSpeechDataset(train_paths, self.tokenizer, return_wav=adapt_mimi)
        # Debug: check if wavs are present
        if adapt_mimi and len(full_ds) > 0:
            sample = full_ds[0]
            has_wav = sample[-1] is not None if len(sample) >= 5 else False
            print(f"[DEBUG] Dataset has_wav={has_wav} (item keys: {list(full_ds.data[0].keys())})")
        from torch.utils.data import Subset
        val_ratio = self.cfg['training'].get('val_ratio', 0.05)
        n_val = max(1, int(len(full_ds) * val_ratio))
        n_train = len(full_ds) - n_val
        indices = torch.randperm(len(full_ds)).tolist()
        train_ds = Subset(full_ds, indices[:n_train])

        collate = ExplicitCollate(self.ph_generator)
        self.loader = DataLoader(
            train_ds, batch_size=self.cfg['training']['batch_size'],
            sampler=LengthGroupedSampler(train_ds, self.cfg['training']['batch_size'],
                                         max_len=self.cfg['training'].get('max_audio_len_tokens', None)),
            collate_fn=collate, pin_memory=True,
            num_workers=num_workers if num_workers is not None else self.cfg['training'].get('num_workers', 4)
        )

        # 7. Optimizer (LoRA parameters only)
        lora_params = [p for n, p in self.model.named_parameters() if "lora_" in n]
        if adapt_mimi:
            lora_params += [p for n, p in self.mimi.named_parameters() if "lora_" in n]

        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.cfg['training'].get('lr', 1e-4),
            weight_decay=0.01
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            self.cfg['training'].get('warmup_steps', 100),
            len(self.loader) * self.cfg['training'].get('epochs', 100)
        )
        self.scaler = torch.amp.GradScaler("cuda")
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.ph_planner_criterion = nn.CrossEntropyLoss(ignore_index=0)
        nq = self.model.n_q
        w = torch.ones(nq)
        w[:nq // 2] = 1.5
        w[nq // 2:] = 0.5
        self.level_weights = (w / w.sum() * nq).to(device)

        # 8. Mimi decoder reconstruction loss (BigVGAN-style)
        self.recon_loss_fn = None
        if adapt_mimi:
            from .losses import MimiDecoderReconLoss
            self.recon_loss_fn = MimiDecoderReconLoss(
                sample_rate=24000,
                mel_weight=self.cfg['training'].get('mel_weight', 45.0),
                stft_weight=self.cfg['training'].get('stft_weight', 0.0),
            ).to(device)
            self.recon_weight = self.cfg['training'].get('recon_weight', 0.1)

        self.global_step = 0
        self.start_epoch = 0

        run_name = f"lora_fusion_{time.strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=f"logs/{run_name}")

        # Load LoRA resume if provided
        if resume_lora and os.path.exists(resume_lora):
            raw = torch.load(resume_lora, map_location=device)
            lora_state = raw.get('lora_state_dict', raw)
            # Validate rank/alpha from checkpoint against CLI args
            ckpt_rank = raw.get('lora_rank')
            ckpt_alpha = raw.get('lora_alpha')
            if ckpt_rank is not None and ckpt_rank != self.lora_rank:
                print(f"\033[93m[Warning]\033[0m Checkpoint lora_rank={ckpt_rank} != CLI lora_rank={self.lora_rank}, re-injecting LoRA with checkpoint rank.")
                self.lora_rank = ckpt_rank
                self.model = inject_lora(self.model, rank=self.lora_rank, alpha=self.lora_alpha)
            if ckpt_alpha is not None and ckpt_alpha != self.lora_alpha:
                self.lora_alpha = ckpt_alpha
            model_lora = {k: v for k, v in lora_state.items() if not k.startswith("mimi.")}
            mimi_lora = {k[5:]: v for k, v in lora_state.items() if k.startswith("mimi.")}
            self.model.load_state_dict(model_lora, strict=False)
            if mimi_lora:
                ckpt_mimi_rank = raw.get('mimi_lora_rank')
                ckpt_mimi_alpha = raw.get('mimi_lora_alpha')
                if ckpt_mimi_rank is not None and ckpt_mimi_rank != self.mimi_lora_rank:
                    print(f"\033[93m[Warning]\033[0m Checkpoint mimi_lora_rank={ckpt_mimi_rank} != CLI mimi_lora_rank={self.mimi_lora_rank}, re-injecting Mimi LoRA with checkpoint rank.")
                    self.mimi_lora_rank = ckpt_mimi_rank
                    self.mimi = inject_lora(self.mimi, rank=self.mimi_lora_rank, alpha=self.mimi_lora_alpha, targets=MIMI_DECODER_TARGETS)
                if ckpt_mimi_alpha is not None and ckpt_mimi_alpha != self.mimi_lora_alpha:
                    self.mimi_lora_alpha = ckpt_mimi_alpha
                self.mimi.load_state_dict(mimi_lora, strict=False)
            # Restore optimizer, scheduler, scaler, epoch, global_step
            if 'optimizer' in raw:
                self.optimizer.load_state_dict(raw['optimizer'])
            if 'scheduler' in raw:
                self.scheduler.load_state_dict(raw['scheduler'])
            if 'scaler' in raw:
                self.scaler.load_state_dict(raw['scaler'])
            if 'epoch' in raw:
                self.start_epoch = raw['epoch']
            if 'global_step' in raw:
                self.global_step = raw['global_step']
            print(f"Resumed LoRA weights from {resume_lora} (epoch={self.start_epoch}, step={self.global_step})")

    def _compute_loss(self, batch):
        padded_text, padded_audio, t_lens, a_lens, raw_texts, ph_targets, padded_wav = batch

        padded_text = padded_text.to(self.device)
        padded_audio = padded_audio.to(self.device)
        t_lens, a_lens = t_lens.to(self.device), a_lens.to(self.device)
        ph_targets = ph_targets.to(self.device)
        if padded_wav is not None:
            padded_wav = padded_wav.to(self.device)

        with torch.no_grad():
            mimi_latents = self.mimi.decode_latent(padded_audio[:, :self.model.n_q // 2])

        logits, targets, ph_planner_logits, jepa_loss, _, latent_pred = self.model(
            padded_text, padded_audio[:, :self.model.n_q], t_lens, a_lens,
            raw_texts=raw_texts, phoneme_ids=ph_targets, mimi_latents=mimi_latents,
            drop_prob=0.1
        )

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

        loss_ph_planner = self.ph_planner_criterion(
            ph_planner_logits.reshape(-1, ph_planner_logits.shape[-1]),
            ph_targets[:, 1:].reshape(-1)
        ) if ph_planner_logits is not None else torch.tensor(0.0, device=self.device)

        loss_jepa = jepa_loss if jepa_loss is not None else torch.tensor(0.0, device=self.device)

        # Style loss (CE between the text->style head and the GT style cluster id),
        # stashed by the model during forward().
        loss_style = torch.tensor(0.0, device=self.device)
        style_logits = getattr(self.model, '_last_style_logits', None)
        gt_style_ids = getattr(self.model, '_last_gt_style_ids', None)
        if style_logits is not None and gt_style_ids is not None:
            loss_style = self.criterion(style_logits.view(-1, style_logits.shape[-1]), gt_style_ids.reshape(-1))

        total_loss = (loss_ar +
                      self.cfg['training'].get('ph_planner_weight', 1.0) * loss_ph_planner +
                      self.cfg['training'].get('jepa_weight', 2.0) * loss_jepa +
                      self.cfg['training'].get('style_weight', 0.1) * loss_style)

        # Mimi decoder reconstruction loss (BigVGAN-style)
        loss_recon = torch.tensor(0.0, device=self.device)
        if self.adapt_mimi and self.recon_loss_fn is not None and padded_wav is not None:
            with torch.no_grad():
                pred_tokens = torch.stack(
                    [logits[k].argmax(dim=-1) for k in range(self.model.n_q)], dim=1
                )
            pred_tokens_clamped = pred_tokens.clamp(0, self.model.card - 1)
            recon_audio = self.mimi.decode(pred_tokens_clamped)
            recon_audio = torch.clamp(recon_audio, -1.0, 1.0)
            loss_recon = self.recon_weight * self.recon_loss_fn(recon_audio, padded_wav.unsqueeze(1))
            if not torch.isfinite(loss_recon):
                loss_recon = torch.tensor(0.0, device=self.device)
            total_loss = total_loss + loss_recon

        return total_loss, loss_ar, loss_jepa, loss_ph_planner, loss_recon

    def train(self):
        print("Starting Fusion LoRA training...")
        output_prefix = self.cfg['training'].get('output_prefix', 'seedvox_fusion_lora')
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(self.start_epoch, self.cfg['training'].get('epochs', 100)):
            self.model.train()
            if self.adapt_mimi:
                self.mimi.train()
            else:
                self.mimi.eval()

            pbar = tqdm(self.loader, desc=f"Epoch {epoch} (LoRA)")

            for batch in pbar:
                self.optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    total_loss, loss_ar, loss_jepa, loss_ph, loss_recon = self._compute_loss(batch)

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.adapt_mimi:
                    torch.nn.utils.clip_grad_norm_(self.mimi.parameters(), 1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                self.global_step += 1
                pbar.set_postfix(
                    ar=f"{loss_ar.item():.3f}",
                    jepa=f"{loss_jepa.item() if isinstance(loss_jepa, torch.Tensor) else loss_jepa:.3f}",
                    ph=f"{loss_ph.item() if isinstance(loss_ph, torch.Tensor) else loss_ph:.3f}",
                    recon=f"{loss_recon.item():.3f}",
                    total=f"{total_loss.item():.3f}"
                )

                if self.global_step % self.cfg['training'].get('log_every', 10) == 0:
                    self.writer.add_scalar("lora/total_loss", total_loss.item(), self.global_step)
                    self.writer.add_scalar("lora/loss_ar", loss_ar.item(), self.global_step)
                    self.writer.add_scalar("lora/loss_jepa", loss_jepa.item() if isinstance(loss_jepa, torch.Tensor) else loss_jepa, self.global_step)
                    self.writer.add_scalar("lora/loss_ph", loss_ph.item() if isinstance(loss_ph, torch.Tensor) else loss_ph, self.global_step)
                    self.writer.add_scalar("lora/loss_recon", loss_recon.item(), self.global_step)

            # Validation
            self.model.eval()
            self.mimi.eval()
            with torch.no_grad():
                test_text = "Testing the LoRA adaptation on this specific speaker."
                t_test = torch.tensor([self.tokenizer.encode(test_text, normalize=False)], device=self.device)
                t_len = torch.tensor([len(t_test[0])], device=self.device)

                ref_audio, ref_len = None, None
                if self.ref_wav_path and os.path.exists(self.ref_wav_path):
                    wav, sr = torchaudio.load(self.ref_wav_path)
                    wav = torchaudio.transforms.Resample(sr, 24000)(wav) if sr != 24000 else wav
                    if wav.shape[0] > 1:
                        wav = wav.mean(0, keepdim=True)
                    ref_audio = self.mimi.encode(wav.to(self.device).unsqueeze(0))
                    ref_len = torch.tensor([ref_audio.shape[2]], device=self.device)

                gen, *_ = self.model.sample(
                    t_test, t_len, ref_audio=ref_audio, ref_lens=ref_len,
                    curr_n_q=self.model.n_q, raw_texts=[test_text]
                )
                if gen is not None:
                    audio = self.mimi.decode(gen.clamp(0, self.model.card - 1))[0]
                    torchaudio.save(f"lora_fusion_epoch_{epoch}.wav", audio.cpu(), 24000)
                    self.writer.add_audio("Audio/val", audio, epoch, sample_rate=24000)

            # Save ONLY LoRA weights (with metadata)
            lora_dict = {n: p for n, p in self.model.named_parameters() if "lora_" in n}
            if self.adapt_mimi:
                mimi_lora = {f"mimi.{n}": p for n, p in self.mimi.named_parameters() if "lora_" in n}
                lora_dict.update(mimi_lora)

            save_payload = {
                'lora_state_dict': lora_dict,
                'lora_rank': self.lora_rank,
                'lora_alpha': self.lora_alpha,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'epoch': epoch + 1,
                'global_step': self.global_step,
            }
            if self.adapt_mimi:
                save_payload['mimi_lora_rank'] = self.mimi_lora_rank
                save_payload['mimi_lora_alpha'] = self.mimi_lora_alpha
            torch.save(save_payload, f"checkpoints/{output_prefix}_lora_epoch_{epoch}.pt")
            print(f"Saved LoRA adapter: checkpoints/{output_prefix}_lora_epoch_{epoch}.pt")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--resume_lora", type=str, default=None)
    parser.add_argument("--ref_wav", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--g2p", default="espeak")
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--adapt_mimi", action="store_true")
    parser.add_argument("--mimi_lora_rank", type=int, default=8)
    parser.add_argument("--mimi_lora_alpha", type=int, default=16)

    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    if args.train_data:
        cfg['training']['train_tokens_path'] = args.train_data
    if args.batch_size:
        cfg['training']['batch_size'] = args.batch_size
    if args.lr:
        cfg['training']['lr'] = args.lr
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
    if args.exp_name:
        cfg['training']['output_prefix'] = args.exp_name

    trainer = FusionLoRATrainer(
        cfg, torch.device(args.device),
        base_checkpoint=args.base_checkpoint,
        g2p_backend=args.g2p,
        resume_lora=args.resume_lora,
        ref_wav=args.ref_wav,
        num_workers=args.num_workers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        adapt_mimi=args.adapt_mimi,
        mimi_lora_rank=args.mimi_lora_rank,
        mimi_lora_alpha=args.mimi_lora_alpha,
    )
    trainer.train()
