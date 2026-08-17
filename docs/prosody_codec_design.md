# Prosody Codec — Design Spec

Status: proposal, not yet implemented.
Related: `dual-FILM.md`, `plan_explicit_phonemes.md`.

## 1. Motivation

The current JEPA prosody teacher is derived from `mimi.decode_latent(tokens[:, :n_q//2])`
(codebooks 0–7) — the exact stream the AR loss supervises hardest (level weights 1.5
vs 0.5), and the bottleneck mean-pools it to 16 global blocks. Consequences:

- The teacher overlaps the AR target set (segmental-content confound).
- It is global/coarse (~400 ms/block at 16 blocks), cannot carry pitch *level*
  (bottleneck mean-centers), and the decoder barely turns it into pitch.
- Frame-wise F0 is the wrong yardstick for such a signal.

Fix: **decouple the prosody signal entirely from the token stream** by building a
standalone *prosody codec*, trained like Mimi but reconstructing prosody features
(F0 / energy / voicing) instead of the waveform. TTS then plans into and conditions
on that codec's frozen tokens.

## 2. Architecture (two stages, split problem)

```
STAGE 1 (pretrain, standalone):
  waveform --[prosody encoder]--> z_cont --[FSQ]--> prosody tokens --[decoder]--> (F0, E, v)
  loss = L1(log-F0) + L1(log-E) + CE(voicing) + L2(level scalars)   [FSQ needs no commitment]

STAGE 2 (TTS, codec frozen):
  text --[planner]--> predicted prosody tokens (per-dim CE vs teacher codec tokens)
  AR decoder conditions on predicted prosody stream (aligned per audio frame)
```

## 3. Preprocessing (one-time, offline)

Per utterance (24 kHz audio):

- **F0**: log-F0 per frame, extracted with pyin (fallback pyworld). Frame rate = mimi
  frame rate (12.5 Hz, hop 1920 samples) to align with audio tokens. Median-filtered.
- **Voicing** `v`: voiced/unvoiced per frame (flag from pyin, prob > threshold).
- **Energy**: log-RMS per frame, same hop.

### 3.1 Normalization (decision)

- **Center, don't standardize.** Subtract per-utterance median log-F0 (over voiced
  frames only) and per-utterance mean log-energy. Preserve variance: excursion
  magnitude is prosody (expressiveness); z-scoring by std would destroy it.
- Unvoiced log-F0 set to a neutral constant (0 after centering).
- **Level scalars**: keep `mu_logF0`, `mu_logE` (uncentered per-utterance means) as
  two auxiliary targets — absolute register/loudness is not lost, just moved to
  explicit channels.
- Optional refinement: low-pass baseline subtraction instead of a single constant.

Storage: one `.pt` per utterance (or one sharded file): `{log_f0_center[N], e_center[N],
voicing[N], mu_logF0, mu_logE, dur_sec}`.

## 4. Prosody codec (stage 1)

### 4.1 FSQ quantizer (instead of RVQ)

- `D=8` dims, `L=5` levels → 5^8 = 390,625 codes per frame. Configurable (D=6..8, L=5..7).
- Why FSQ: no codebook learning / EMA / collapse; natural for low-dim prosody;
  per-dim prediction (`D` × `L`-way) keeps the planner head cheap; clean continuous
  interpolation space for exagg.
- Straight-through gradient on rounding; bounded input (tanh).

### 4.2 Encoder / decoder

- Encoder: small conv stack (e.g., 4 conv blocks, 512 ch, stride matching frame rate)
  waveform → `z_cont[B, T, D]` → FSQ.
- Decoder: from code indices (embedding lookup) → MLP → predicts centered log-F0,
  centered log-E, voicing logits, and the two level scalars (mean-pooled head).
- Reconstruction is of *prosody features*, not audio — timbre/noise never enter.

### 4.3 Losses & training

- `L1(log-F0_center)` weighted on voiced frames; `L1(log-E_center)`; `CE(voicing)`;
  `L2(mu_logF0, mu_logE)`; optional slight temporal smoothing loss on F0.
- No commitment/EMA (FSQ). Standard optimizer (AdamW, lr ~3e-4, cosine).
- Data: all training audio (278k utterances). Small model → hours on one GPU.
- Checkpoint to freeze for stage 2.

## 5. TTS integration (stage 2)

- **Teacher**: `tokens = codec.encode(audio)` — frozen, zero AR overlap.
- **Planner**: text → per-dim FSQ digit logits. Head = `D` parallel `L`-way classifiers
  per predicted position (+ level-scalar heads). Loss = CE + scalar regression.
  Replaces the cosine JEPA loss; keep the JEPA framing (text → plan) but the target
  space is now real prosody.
- **Decoder conditioning**: per audio frame, embed the predicted code index
  (`nn.Embedding`) or use the decoder-side continuous features; join the fusion context.
- **exagg**: interpolate `null ↔ pred` in the continuous pre-quantization space
  (`null + e·(pred − null)`), then round. Becomes a true contour amplifier.
  With center-only normalization, exagg scales excursions, not levels.
- Level scalars condition the decoder via the speaker/register path so absolute
  pitch/loudness is controllable too.

## 6. Temporal resolution & alignment (fork — needs decision)

- **Recommended (v1): per-frame @ 12.5 Hz, planner at block rate.** Codec emits one
  token per mimi frame (aligned trivially with audio tokens). The planner predicts at
  a reduced rate (e.g., one token per ~8 frames ≈ 16 blocks/utterance as today), and
  the predicted tokens are repeated/interpolated across audio frames by the decoder.
  Keeps the planner cheap while conditioning is time-structured and per-frame for the
  decoder. Fixes "decoder can't do local shaping" without a full duration model.
- **v2: true per-frame planning.** Planner predicts a token every audio frame; requires
  text↔duration alignment (duration predictor or AR joint prediction). More capacity,
  more complexity. Only after v1 proves the signal.

## 7. Data prep job

- Tooling: pyin on CPU is ~10–20× realtime; parallelize over workers (num_workers ≫ 8)
  or GPU F0 (torch-based). Estimate ~400 h audio → single-machine wall clock
  ~a day with parallel workers; one-time.
- Output: sharded `.pt` files keyed by utterance id, consumed by the stage-2 dataset.

## 8. Timeline (single GPU, ~7.5 it/s ≈ 1.3 h/epoch for stage 2)

| step | effort |
|---|---|
| F0/energy/voicing preprocessing | ~1 day (parallel), one-time |
| Prosody codec pretrain (FSQ) | hours – ~1 day |
| Stage-2 planner head + conditioning changes | ~1–2 days code |
| Stage-2 retrain (warm start from epoch_93, `strict=False`) | ~10–20 epochs to re-settle, 30–60 to see prosody effect (~40–80 h) |
| Validate: planning cos (matched vs null) + plan-follow + exagg contour | reuse `validate_prosody_planning.py` / `prosody_audio_metrics.py` |

## 9. Open questions

- Frame rate: 12.5 Hz (aligned) vs 25 Hz (finer contour, needs downsampling for conditioning).
- FSQ D/L: default D=8, L=5; tune on codec reconstruction quality (L1 log-F0).
- Keep the level scalars out of FSQ (regressed separately) vs quantize them too.
- v1 vs v2 alignment (Section 6) — needs a decision before stage-2 coding.
