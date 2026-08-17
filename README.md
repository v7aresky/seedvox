# 🌱 SeedVox

**Speech that plans before it speaks.**

Smarter Speech AI — bigger data, smarter architecture, better prosody.

Most modern speech engines treat human expression like a brute-force problem: a single, bloated
autoregressive model guesses emotion token by token. But emotion isn't a split-second dice roll —
it's a **global state of mind** that shapes an entire sentence before we even open our mouths.

**SeedVox** splits the workload into three specialized components, so each one learns a
clean job instead of juggling everything at once:

| Component | Role |
|---|---|
| **AR Phonetic Planner** | The "what" — stable, hallucination-free phoneme planning and acoustic anchoring. |
| **JEPA World Model** | The "how" — reads the whole sentence at once and predicts a global prosody latent before generation begins. |
| **AR Acoustic Decoder** | The voice — turns text and the prosody plan into audio tokens, guided by the JEPA latent. |

---

## The results

The JEPA world model plans prosody *before* the decoder runs. The decoder then conditions on that
plan and generates speech that actually follows it. Measured across 200 evaluation batches (2,000 sentences):

| Metric | Value | What it means |
|---|---|---|
| `ar` | **1.070** | Audio quality with *ground-truth* prosody — the best-case ceiling. |
| `gen_ar` | **1.070** | Audio quality with a *JEPA-sampled* plan — same as the ceiling. |
| `gen_cos` | **0.237** | How close the sampled plan lands to ground truth (0 = identical, 1 = unrelated). |
| `gen_cycle` | **0.317** | How faithfully the decoder followed the plan it was given (0 = perfect). |

> **Key result:** `gen_ar == ar` (both 1.070). The decoder follows a sampled plan as accurately as ground truth. No text-only shortcut collapse. The prosody latent is *informative*, not decorative.

---

## Quick start

### Installation

```bash
git clone https://github.com/v7aresky/seedvox.git
cd seedvox
pip install -e .
```

### Inference

```bash
# Basic inference
python -m explicit_pros_phon_planner.infer \
    --config ./configs/light_fusion_r3.json \
    --checkpoint ./checkpoints/seedvox_light_fusion_epoch_103.pt \
    --text "The quick brown fox jumps over the lazy dog." \
    --output output.wav

# Optimized inference with compilation and metrics
python -m explicit_pros_phon_planner.infer \
    --config ./configs/light_fusion_r3.json \
    --checkpoint ./checkpoints/seedvox_light_fusion_epoch_103.pt \
    --text "The quick brown fox jumps over the lazy dog." \
    --compile --log_metrics --play --output output.wav
```

### Features

- **Deterministic synthesis:** `--seed <INT>` for reproducible results.
- **Optimized inference:** Gradient checkpointing, Fused AdamW, `torch.compile` — latencies under 400ms on consumer GPUs.
- **N-variant generation:** `--variant_axis pros` or `--variant_axis speaker` to explore the latent space.
- **LoRA fine-tuning:** Adapt to new voices with lightweight low-rank adapters (`--lora_checkpoint`).
- **CFG scale tuning:** `--cfg_scale` to balance prosody diversity vs. reference fidelity.
- **Temperature control:** `--phoneme_temperature`, `--prosody_temperature`, `--acoustic_temperature`.

---

## Demo

**[Listen to the demo →](https://v7aresky.github.io/seedvox/)**

A 10-sentence narrative generated on a consumer laptop GPU, using a single reference speaker. All prosody is sampled from the JEPA world model — no manual tuning, no style tags.

> **Note on the reference voice:** The speaker is **LJ Speech** — a single-speaker audiobook corpus. LJ Speech is *read speech*: monotone, low expressiveness, no conversational dynamics. Expressive conversational training data will sound considerably more natural.

---

## Architecture

Instead of one giant network juggling everything at once, SeedVox **combines an AR LLM with a JEPA World Model**:

- **JEPA World Model** — reads the full sentence up front and predicts a global expression latent that shapes how the whole utterance should feel.
- **AR LLM** — handles sequential token generation across text, phonemes, and audio. The JEPA's expression latent guides audio generation.
- **Frozen Prosody Teacher** — a codec trained from expressive speech that supervises the JEPA planner.

### Key design choices

- **JEPA prosody planning:** Prosody is a global state of mind, not a token-level dice roll. The planner emits a learned `(B, 32, dim)` latent that shapes the entire utterance before generation begins.
- **Latent, not features:** Raw F0/Energy/Voicing only feed the frozen codec as a training target. Generation conditions exclusively on the learned latent.
- **Unified linguistic fusion:** A gated cross-attention layer folds phonemes into the text backbone before acoustic generation.
- **Dual-FiLM disentanglement:** Two speaker-modulated FiLM adapters separate articulation from rhythm/emotion.
- **Intervenable control:** Phonemes are planned explicitly — researchers can overwrite the phoneme string to fix pronunciations at runtime.

---

## License

Apache License 2.0. See `LICENSE` for details.

For a full technical breakdown, see [`intro_seedvox.md`](intro_seedvox.md).
