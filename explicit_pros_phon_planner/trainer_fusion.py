import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from seedvox.bpe_char_encoder import BPECharCollator
from seedvox.utils.tokenizer import CharTokenizer
from seedvox.training.dataset import TokenizedSpeechDataset, LengthGroupedSampler
from .model_fusion import FusionPlannerModel
from .utils import PhoneticGenerator, collate_phonemes, filter_state_dict
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

        # 4. BPE collator
        self.bpe_collator = BPECharCollator(self.model.bpe_encoder) if getattr(self.model, 'bpe_encoder', None) else None

        # 5. G2P + collate
        self.ph_generator = PhoneticGenerator(backend=g2p_backend, phoneme_vocab_size=128)
        collate = ExplicitCollate(self.ph_generator)

        # 5b. Frozen Mimi decoder for the plan-follow (cycle) loss anchor: provides the
        # GT acoustic latent (decode_latent) whose prosody must match the frozen codec's.
        if config['training'].get('planner_cycle_weight', 0.0) > 0:
            print("[FusionTrainer] Loading frozen Mimi for cycle-loss anchor...")
            from seedvox.modules.mimi import get_mimi_model
            self.mimi = get_mimi_model(device=device, checkpoint_path='pretrained_models/best_mimi.pt').eval()
            for p in self.mimi.parameters(): p.requires_grad = False

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
            sampler=LengthGroupedSampler(train_ds, config['training']['batch_size'],
                                         max_len=config['training'].get('max_audio_len_tokens', None)),
            collate_fn=collate, pin_memory=True, num_workers=num_workers
        )
        self.eval_loader = None
        if n_val > 0:
            self.eval_loader = DataLoader(
                Subset(full_ds, indices[n_train:]), batch_size=config['training']['batch_size'],
                collate_fn=collate, shuffle=False, num_workers=0, pin_memory=True
            )

        # 7. Optimizer with configuration-driven parameter grouping
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
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

        # 8. Resume checkpoint (warm-start). Shapes differ for resized layers
        # (num_prosody_tokens 16->32), so pre-filter mismatched keys.
        if resume_path and os.path.exists(resume_path):
            print(f"  Resuming from {resume_path}...")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                self.model.load_state_dict(filter_state_dict(self.model, ckpt['model']), strict=False)
                ema_sd = filter_state_dict(self.ema_model, ckpt.get('ema_model', ckpt['model']))
                self.ema_model.load_state_dict(ema_sd, strict=False)
                self.global_step = ckpt.get('step', 0)
                self.start_epoch = ckpt.get('epoch', 0)
                # Restore optimizer/scheduler/scaler so Adam moments, LR position and
                # amp scaling survive restarts. Only safe when the architecture is
                # unchanged (it is: these resumes change no model hyperparams).
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                    if not self._validate_optimizer_state():
                        raise RuntimeError("optimizer state has shape-mismatched moments")
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                    self.scaler.load_state_dict(ckpt['scaler'])
                    print("  Restored optimizer + scheduler + scaler state.")
                except Exception as e:
                    # Group structure changed between checkpoints (e.g. r4's 6
                    # groups vs this config's 7). Rebuild the optimizer layout and
                    # transfer per-param Adam moments by NAME where the params still
                    # exist; new params (style modules) get fresh moments.
                    try:
                        self._rebuild_optimizer_state(ckpt['optimizer'], ckpt['model'])
                        self.scaler.load_state_dict(ckpt['scaler'])
                        print(f"  Optimizer group layout differed ({e.__class__.__name__}); "
                              "rebuilt + restored per-param moments. Scheduler/warmup restarted.")
                    except Exception as e2:
                        # Moment transfer is unreliable (e.g. the checkpoint saved its
                        # optimizer against a different param ordering). Weights were
                        # already loaded above; restart the optimizer/scheduler fresh
                        # rather than risk silently-corrupted Adam moments.
                        self.optimizer = self._build_optimizer()
                        self.scheduler = self._build_scheduler()
                        print(f"  [WARN] optimizer state NOT restorable ({e2}); "
                              "starting optimizer fresh (weights transferred, "
                              "scheduler/warmup restarted).")
                # Resume restores the SAVED param-group LRs; re-apply the
                # config's groups so an LR change (e.g. r4 hardening) takes
                # effect while keeping the Adam moments and schedule position.
                self._apply_config_lrs()
            else:
                state_dict = filter_state_dict(self.model, ckpt)
                self.model.load_state_dict(state_dict, strict=False)
                self.ema_model.load_state_dict(state_dict, strict=False)

        if config['training'].get('compile', False):
            print("[FusionTrainer] Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # 9. Tensorboard
        run_name = f"fusion_{time.strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=f"logs/{run_name}")
        print(f"[FusionTrainer] Ready. Resumed at epoch={self.start_epoch}, step={self.global_step}")

    def _apply_config_lrs(self):
        lr = self.cfg['training'].get('lr', 1e-4)
        lr_by_id = {}
        for pg_config in self.cfg['training'].get('param_groups', []):
            pg_lr = pg_config.get('lr', lr)
            for n, p in self.model.named_parameters():
                if any(k in n for k in pg_config.get('keywords', [])):
                    lr_by_id[id(p)] = pg_lr
        for g in self.optimizer.param_groups:
            target = lr_by_id.get(id(g['params'][0]), lr)
            g['lr'] = target
            g['initial_lr'] = target
        self.scheduler.base_lrs = [g['initial_lr'] for g in self.optimizer.param_groups]
        self.scheduler._last_lr = [g['initial_lr'] for g in self.optimizer.param_groups]


    def _build_optimizer(self):
        """Build AdamW with the config's parameter groups (used at init and to
        reset the optimizer fresh when checkpoint state can't be trusted)."""
        lr = self.cfg['training'].get('lr', 1e-4)
        param_configs = self.cfg['training'].get('param_groups', [])
        processed_param_ids = set()
        param_groups = []
        for pg_config in param_configs:
            keywords = pg_config.get('keywords', [])
            pg_lr = pg_config.get('lr', lr)
            group_params = []
            for n, p in self.model.named_parameters():
                if id(p) not in processed_param_ids and any(k in n for k in keywords):
                    group_params.append(p)
                    processed_param_ids.add(id(p))
            if group_params:
                param_groups.append({'params': group_params, 'lr': pg_lr})
        remaining = [p for p in self.model.parameters() if id(p) not in processed_param_ids]
        if remaining:
            param_groups.append({'params': remaining, 'lr': lr})
        return torch.optim.AdamW(param_groups, weight_decay=0.01)

    def _build_scheduler(self):
        return get_cosine_schedule_with_warmup(
            self.optimizer, self.cfg['training'].get('warmup_steps', 100),
            len(self.loader) * self.cfg['training'].get('epochs', 100),
            num_cycles=0.5
        )

    def _validate_optimizer_state(self):
        """True iff every param with state has correctly-shaped moments."""
        for p, st in self.optimizer.state.items():
            for k in ('exp_avg', 'exp_avg_sq'):
                if k in st and st[k].shape != p.shape:
                    return False
            if 'step' in st and not torch.is_tensor(st['step']):
                return False
        return True

    def _rebuild_optimizer_state(self, ckpt_opt, ckpt_model_sd):
        """Transfer Adam moments across a param-group layout change.

        optimizer.load_state_dict requires identical group counts; when the config's
        groups differ from the checkpoint's (or the checkpoint's optimizer was saved
        against a different parameter ordering), rebuild the group layout for the
        CURRENT optimizer and map each checkpoint parameter's Adam state to the
        current optimizer by NAME (state_dict key order == parameter order, and
        filtering buffer keys recovers the exact parameter order).

        Raises RuntimeError if the mapping can't be validated (a moment whose shape
        doesn't match its target param => the checkpoint's index->param order differs
        from its state_dict order, so ANY name transfer would silently corrupt
        training). Callers must then fall back to a fresh optimizer; the model
        WEIGHTS are still loaded correctly by name regardless.
        """
        old_buffers = {n for n, _ in self.model.named_buffers()}
        old_param_names = [k for k in ckpt_model_sd.keys() if k not in old_buffers]
        if not ckpt_opt.get('state'):
            raise RuntimeError("checkpoint optimizer has no per-param state")
        max_old_idx = max(ckpt_opt['state'].keys())
        if len(old_param_names) <= max_old_idx:
            raise RuntimeError("cannot map checkpoint optimizer indices to param names")
        name2idx = {n: i for i, (n, _) in enumerate(self.model.named_parameters())}
        new_state = {}
        for old_idx, st in ckpt_opt['state'].items():
            name = old_param_names[old_idx]
            if name not in name2idx:
                continue
            target = dict(self.model.named_parameters())[name]
            for k in ('exp_avg', 'exp_avg_sq'):
                if k in st and st[k].shape != target.shape:
                    raise RuntimeError(
                        f"checkpoint optimizer index {old_idx} -> '{name}' has "
                        f"{k} shape {tuple(st[k].shape)} vs param {tuple(target.shape)}; "
                        "index->param order is unreliable, refusing to transfer moments")
            new_state[name2idx[name]] = st
        new_params = list(self.model.parameters())
        for i, p in enumerate(new_params):
            if i not in new_state:
                new_state[i] = {
                    'step': torch.tensor(0.0),
                    'exp_avg': torch.zeros_like(p),
                    'exp_avg_sq': torch.zeros_like(p),
                }
        idx_by_id = {id(p): i for i, p in enumerate(new_params)}
        new_groups = []
        for g in self.optimizer.param_groups:
            new_groups.append({
                'params': [idx_by_id[id(p)] for p in g['params']],
                'lr': g['lr'],
                'betas': g['betas'],
                'eps': g['eps'],
                'weight_decay': g['weight_decay'],
                'amsgrad': g.get('amsgrad', False),
            })
        self.optimizer.load_state_dict({'state': new_state, 'param_groups': new_groups})
        if not self._validate_optimizer_state():
            raise RuntimeError("optimizer state invalid after rebuild")


    def _compute_loss(self, batch):
        padded_text, padded_audio, t_lens, a_lens, raw_texts, ph_targets, prosody_feat = batch

        padded_text = padded_text.to(self.device)
        padded_audio = padded_audio.to(self.device)
        t_lens, a_lens = t_lens.to(self.device), a_lens.to(self.device)
        ph_targets = ph_targets.to(self.device)
        if prosody_feat is not None:
            prosody_feat = prosody_feat.to(self.device)

        bpe_ids, bpe_lens, char_to_bpe = None, None, None
        if self.bpe_collator:
            bpe_ids, bpe_lens, char_to_bpe = self.bpe_collator.process_batch_texts(raw_texts, t_lens, self.device)

        logits, targets, ph_planner_logits, jepa_loss, _, latent_pred = self.model(
            padded_text, padded_audio[:, :self.model.n_q], t_lens, a_lens, raw_texts=raw_texts,
            phoneme_ids=ph_targets, prosody_feats=prosody_feat,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
            drop_prob=0.1 if self.model.training else 0.0
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

        # Plan-follow (cycle) loss: force the decoder to USE the injected prosody.
        # pred_prosody = prosody_bottleneck(latent_pred) is the prosody latent of what
        # the model GENERATES; match it against the plan it was conditioned on. Gradient
        # flows through latent_pred -> latent_regressor -> depformer output -> decoder,
        # so the only way to reduce it is for the decoder's tokens to carry the plan.
        # The anchor term distills prosody_bottleneck(gt_mimi) onto the frozen codec's
        # GT latent so the bottleneck stays an honest audio->prosody readout.
        loss_cycle = torch.tensor(0.0, device=self.device)
        cyc_w = self.cfg['training'].get('planner_cycle_weight', 0.0)
        anch_w = self.cfg['training'].get('cycle_anchor_weight', 0.0)
        if cyc_w > 0 and latent_pred is not None:
            plan = getattr(self.model, '_last_plan_prosody', None)
            gt_prs = getattr(self.model, '_last_gt_prosody', None)
            if plan is not None and plan.shape[0] == latent_pred.shape[0]:
                plan = plan.detach()
                a_mask = torch.arange(latent_pred.shape[-1], device=self.device).unsqueeze(0) >= a_lens.unsqueeze(1)
                pred_prosody = self.model.prosody_bottleneck(latent_pred, mask=a_mask)
                loss_cycle = cyc_w * (1 - F.cosine_similarity(pred_prosody, plan, dim=-1)).mean()
                if anch_w > 0 and gt_prs is not None:
                    with torch.no_grad():
                        gt_mimi = self.mimi.decode_latent(padded_audio[:, :self.model.n_q // 2])
                    anchor_prosody = self.model.prosody_bottleneck(gt_mimi, mask=a_mask)
                    loss_cycle = loss_cycle + anch_w * (1 - F.cosine_similarity(anchor_prosody, gt_prs.detach(), dim=-1)).mean()

        # Style loss (CE between the text->style head and the GT style cluster id),
        # stashed by the model during forward().
        loss_style = torch.tensor(0.0, device=self.device)
        style_logits = getattr(self.model, '_last_style_logits', None)
        gt_style_ids = getattr(self.model, '_last_gt_style_ids', None)
        if style_logits is not None and gt_style_ids is not None:
            loss_style = self.criterion(style_logits.view(-1, style_logits.shape[-1]), gt_style_ids.reshape(-1))

        # Style anchor loss: (1 - cos(plan_mean, style_bank[gt_id])) stashed by the
        # model. NOTE: this is already folded into jepa_loss inside encode_context
        # (weighted by model.style_anchor_weight, then scaled by jepa_weight below),
        # so it is NOT added to total_loss again -- it is returned/kept only for
        # logging (pbar anc / tensorboard loss_style_anchor).
        style_anchor = getattr(self.model, '_last_style_anchor', None)
        loss_anchor = torch.tensor(0.0, device=self.device)
        if style_anchor is not None and torch.isfinite(style_anchor):
            loss_anchor = style_anchor

        total_loss = (loss_ar +
                      self.cfg['training'].get('ph_planner_weight', 1.0) * loss_ph +
                      self.cfg['training'].get('jepa_weight', 2.0) * loss_jepa +
                      loss_cycle +
                      self.cfg['training'].get('style_weight', 0.1) * loss_style)

        return total_loss, loss_ar, loss_jepa, loss_ph, loss_cycle, loss_style, loss_anchor

    def _compute_gen_metrics(self, batch):
        """Generative eval: SAMPLE the plan from the JEPA planner with NO GT
        prosody and NO teacher forcing (style predicted from text), decode under
        that plan, and score the GENERATED prosody against the GT latent.

        This is what the model actually does at inference; _compute_loss instead
        feeds GT prosody in and measures reconstruction, so it over-estimates
        quality. Stochastically sampled (like real inference)."""
        padded_text, padded_audio, t_lens, a_lens, raw_texts, ph_targets, prosody_feat = batch
        padded_text = padded_text.to(self.device)
        padded_audio = padded_audio.to(self.device)
        t_lens, a_lens = t_lens.to(self.device), a_lens.to(self.device)
        ph_targets = ph_targets.to(self.device)
        if prosody_feat is not None:
            prosody_feat = prosody_feat.to(self.device)

        bpe_ids, bpe_lens, char_to_bpe = None, None, None
        if self.bpe_collator:
            bpe_ids, bpe_lens, char_to_bpe = self.bpe_collator.process_batch_texts(raw_texts, t_lens, self.device)

        with torch.no_grad():
            gt_prosody = self.model.prosody_codec.encode(prosody_feat).detach()

        B = padded_audio.shape[0]

        logits, targets, _, _, _, latent_pred = self.model(
            padded_text, padded_audio[:, :self.model.n_q], t_lens, a_lens, raw_texts=raw_texts,
            phoneme_ids=ph_targets, prosody_feats=None,
            bpe_ids=bpe_ids, bpe_lens=bpe_lens, char_to_bpe=char_to_bpe,
            drop_prob=0.0
        )

        a_mask = torch.arange(latent_pred.shape[-1], device=self.device).unsqueeze(0) >= a_lens.unsqueeze(1)
        n_q = self.model.n_q
        T_audio = targets.shape[2]
        weights_ar = torch.zeros(n_q, device=self.device)
        weights_ar[:n_q] = self.level_weights[:n_q]
        weights_flat = weights_ar[None, None, :].expand(B, T_audio, n_q).reshape(B * T_audio * n_q)
        gen_ar = (F.cross_entropy(
            logits.permute(1, 2, 0, 3).reshape(B * T_audio * n_q, logits.shape[-1]),
            targets.permute(0, 2, 1).reshape(B * T_audio * n_q),
            reduction='none', ignore_index=-1) * weights_flat).sum() / max(B * T_audio * n_q, 1)

        gen_prosody = self.model.prosody_bottleneck(latent_pred, mask=a_mask)
        gen_cos = 1 - F.cosine_similarity(gen_prosody, gt_prosody, dim=-1).mean()
        gen_cycle = torch.tensor(0.0, device=self.device)
        plan = getattr(self.model, '_last_plan_prosody', None)
        if plan is not None and plan.shape[0] == B:
            gen_cycle = 1 - F.cosine_similarity(gen_prosody, plan.detach(), dim=-1).mean()
        return gen_ar, gen_cos, gen_cycle

    def evaluate(self, max_batches=200):
        """Hold-out evaluation of the same loss terms on the val split. Style CE /
        anchor are scored with the model's PREDICTED style ids (no teacher forcing,
        no dropout). Logs eval/* to tensorboard. Also runs the generative eval
        (eval/gen_*): sample the plan, decode under it, score vs GT prosody."""
        if self.eval_loader is None or len(self.eval_loader) == 0:
            return None
        self.model.eval()
        self.model.eval_stats = True
        # Drop any plan/GT stashed by the last training batch so a batch whose
        # forward fails to re-stash it can't reuse a stale one (shape mismatch).
        self.model._last_plan_prosody = None
        self.model._last_gt_prosody = None
        sums = {'total': 0.0, 'ar': 0.0, 'jepa': 0.0, 'ph': 0.0, 'cycle': 0.0, 'style': 0.0, 'anchor': 0.0,
                'gen_ar': 0.0, 'gen_cos': 0.0, 'gen_cycle': 0.0}
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(0)
        n = 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for i, batch in enumerate(self.eval_loader):
                if i >= max_batches:
                    break
                tl, ar, jp, ph, cyc, sty, anc = self._compute_loss(batch)
                gar, gcos, gcyc = self._compute_gen_metrics(batch)
                sums['total'] += float(tl); sums['ar'] += float(ar); sums['jepa'] += float(jp)
                sums['ph'] += float(ph); sums['cycle'] += float(cyc)
                sums['style'] += float(sty); sums['anchor'] += float(anc)
                sums['gen_ar'] += float(gar); sums['gen_cos'] += float(gcos); sums['gen_cycle'] += float(gcyc)
                n += 1
        torch.random.set_rng_state(rng_state)
        self.model.eval_stats = False
        self.model.train()
        n = max(n, 1)
        for k, v in sums.items():
            self.writer.add_scalar(f"eval/{k}", v / n, self.global_step)
        print(f"  [EVAL] batches={n} total={sums['total']/n:.4f} ar={sums['ar']/n:.4f} "
              f"jepa={sums['jepa']/n:.4f} ph={sums['ph']/n:.4f} cycle={sums['cycle']/n:.4f} "
              f"style={sums['style']/n:.4f} anchor={sums['anchor']/n:.4f}")
        print(f"  [GEN ] batches={n} gen_ar={sums['gen_ar']/n:.4f} "
              f"gen_cos={sums['gen_cos']/n:.4f} gen_cycle={sums['gen_cycle']/n:.4f}")
        return sums

    def train(self):
        output_prefix = self.cfg['training'].get('output_prefix', 'seedvox_fusion')
        os.makedirs("checkpoints", exist_ok=True)
        eval_every = self.cfg['training'].get('eval_every', 5)
        eval_max_batches = self.cfg['training'].get('eval_max_batches', 200)

        for epoch in range(self.start_epoch, self.cfg['training'].get('epochs', 100)):
            self.model.train()
            pbar = tqdm(self.loader, desc=f"Epoch {epoch} (Fusion)")

            for batch in pbar:
                self.optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    total_loss, loss_ar, loss_jepa, loss_ph, loss_cycle, loss_style, style_anchor = self._compute_loss(batch)

                if not torch.isfinite(total_loss):
                    # Skip non-finite steps: never let nan/inf grads reach the
                    # optimizer (GradScaler would otherwise hit its min scale and
                    # "give up", poisoning every weight). A nan plan/prediction is
                    # usually transient and self-corrects next step.
                    self.scheduler.step()
                    self.global_step += 1
                    continue

                self.scaler.scale(total_loss).backward()
                if self.cfg['training'].get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['training']['grad_clip'])
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
                    cyc=f"{loss_cycle.item():.3f}" if isinstance(loss_cycle, torch.Tensor) else "0.000",
                    sty=f"{loss_style.item():.3f}" if isinstance(loss_style, torch.Tensor) else "0.000",
                    anc=f"{style_anchor.item():.3f}" if style_anchor is not None else "---",
                    total=f"{total_loss.item():.3f}"
                )

                if self.global_step % self.cfg['training'].get('log_every', 10) == 0:
                    self.writer.add_scalar("train/loss", total_loss.item(), self.global_step)
                    self.writer.add_scalar("train/loss_jepa", loss_jepa.item() if isinstance(loss_jepa, torch.Tensor) else loss_jepa, self.global_step)
                    self.writer.add_scalar("train/loss_ph", loss_ph.item() if isinstance(loss_ph, torch.Tensor) else loss_ph, self.global_step)
                    self.writer.add_scalar("train/loss_cycle", loss_cycle.item() if isinstance(loss_cycle, torch.Tensor) else 0.0, self.global_step)
                    self.writer.add_scalar("train/loss_style", loss_style.item() if isinstance(loss_style, torch.Tensor) else 0.0, self.global_step)
                    if style_anchor is not None:
                        self.writer.add_scalar("train/loss_style_anchor", style_anchor.item(), self.global_step)
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

            if eval_every > 0 and epoch % eval_every == 0:
                self.evaluate(max_batches=eval_max_batches)


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
