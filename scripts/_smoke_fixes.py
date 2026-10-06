import sys, json, torch
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

from explicit_pros_phon_planner.model_fusion import FusionPlannerModel
from explicit_pros_phon_planner.utils import filter_state_dict
from seedvox.utils.tokenizer import CharTokenizer

torch.manual_seed(0)
device = torch.device("cpu")

with open(root / "configs/light_fusion_r3.json") as f:
    cfg = json.load(f)

model = FusionPlannerModel(cfg, CharTokenizer().vocab_size, phoneme_vocab_size=128).to(device)
model.eval()

B, T_text, T_audio = 4, 8, 16
text = torch.randint(5, CharTokenizer().vocab_size, (B, T_text))
t_lens = torch.full((B,), T_text)
audio = torch.randint(0, 2048, (B, model.n_q, T_audio))
a_lens = torch.full((B,), T_audio)
prosody_feats = torch.randn(B, 96, 3)

with torch.no_grad():
    # 1. Training-like path with prosody_feats (GT teacher + JEPA loss)
    out = model.encode_context(
        text, t_lens, audio_tokens=audio, audio_lens=a_lens,
        prosody_feats=prosody_feats,
    )
    assert len(out) == 7, f"encode_context returned {len(out)} values"
    context, ctx_mask, ph_logits, jepa_loss, contrastive_loss, spk_vec, prs_emb = out
    assert jepa_loss is not None and torch.isfinite(jepa_loss), jepa_loss
    print(f"[OK] train path: jepa_loss={jepa_loss.item():.4f} context={tuple(context.shape)}")

    # 2. Inference path (sampling) — two calls differ (stochastic planner active)
    c1, _, _, _, _, _, e1 = model.encode_context(text, t_lens, exagg=1.0)
    c2, _, _, _, _, _, e2 = model.encode_context(text, t_lens, exagg=1.0)
    assert not torch.allclose(c1, c2), "inference sampling should vary between draws"
    # Planner mean (sample=False) is deterministic
    tf = model.get_enriched_text_feat(text, t_lens)
    t_mask = torch.arange(tf.shape[1], device=device).unsqueeze(0) >= (t_lens.unsqueeze(1) + 2)
    m1 = model.jepa_planner(tf, text_mask=t_mask)
    m2 = model.jepa_planner(tf, text_mask=t_mask)
    assert torch.allclose(m1, m2), "deterministic mean path broken"
    s1 = model.jepa_planner(tf, text_mask=t_mask, sample=True)
    assert not torch.allclose(s1, m1), "sample=True should deviate from mean"
    print(f"[OK] inference stochastic (plan mean {e1.mean().item():.4f}, dev {((s1-m1).abs().mean()).item():.4f})")

    # 3. Random control still overridable via external_prosody (FiLM-adapted, so != raw ext)
    ext = model.null_prosody.expand(B, -1, -1) + torch.randn_like(model.null_prosody) * 0.3
    spk_len = model.cfg['num_speaker_latents']
    prs_len = model.cfg.get('num_prosody_tokens', 32)
    c3, _, _, _, _, _, e3 = model.encode_context(text, t_lens, external_prosody=ext)
    ext_block = c3[:, spk_len:spk_len+prs_len, :]
    assert not torch.allclose(ext_block, c1[:, spk_len:spk_len+prs_len, :]), "external_prosody not overriding plan"
    print("[OK] external_prosody (random control) overrides the sampled plan")

    # 4. Forward pass with prosody_feats (full training step)
    logits, targets, ph, jepa, _, _ = model(
        text, audio, t_lens, a_lens, prosody_feats=prosody_feats, phoneme_ids=torch.randint(1, 120, (B, T_text + 2))
    )
    assert torch.isfinite(logits).all()
    print(f"[OK] forward: logits={tuple(logits.shape)} jepa={jepa.item():.4f}")

# 5. Resume-safety: load epoch_102 via filter_state_dict
ckpt_path = root / "checkpoints/seedvox_light_fusion_epoch_102.pt"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("ema_model", ckpt))
    has_std_head = any("jepa_planner.std_head" in k for k in sd.keys())
    sd2 = filter_state_dict(model, sd)
    missing, unexpected = model.load_state_dict(sd2, strict=False)
    std_head_missing = [k for k in missing if "std_head" in k]
    style_missing = [k for k in missing if "style" in k or k == "marker_style"]
    other_missing = [k for k in missing if "std_head" not in k and "style" not in k and k != "marker_style"]
    print(f"[OK] loaded epoch_102. ckpt_had_std_head={has_std_head}")
    print(f"     unexpected={len(unexpected)} non-std_head/style missing={len(other_missing)} std_head missing (expected)= {len(std_head_missing)} style missing (expected, new feature)= {len(style_missing)}")
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not other_missing, f"unexpected non-std_head/style missing keys: {other_missing[:5]}"
    with torch.no_grad():
        out = model.encode_context(text, t_lens, prosody_feats=prosody_feats)
    print(f"[OK] post-resume train path: jepa_loss={out[3].item():.4f}")
else:
    print("[SKIP] epoch_102 checkpoint not found")

print("ALL_SMOKE_TESTS_PASSED")
