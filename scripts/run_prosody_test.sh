#!/usr/bin/env bash
# run_prosody_test.sh — does the JEPA prosody PLANNER actually drive the audio?
#
# The JEPA prosody signal is a GLOBAL style latent (num_prosody_tokens pooled
# blocks, mean-centered), so it is validated in that same global,
# frame-independent space — NOT on frame-wise F0 (which is dominated by
# sampling noise and has no pitch-level content after mean-centering).
#
# The test has two phases:
#
#   1. PLANNING VALIDATION (validate_prosody_planning.py): does the planner
#      predict a TEXT-SPECIFIC prosody plan? For real (text, audio) pairs the
#      planner's latent `pred` must match the audio's own teacher latent `gt`
#      far better than a mismatched text or the null baseline, and be the best
#      match for its own audio (discrimination).
#
#   2. AUDIO-LEVEL METRICS (prosody_audio_metrics.py): on the synthesized
#      samples (N seeds per exagg condition, speaker locked via REF_SPK):
#        a. PLAN-FOLLOW: cos(gen_latent, pred) - cos(gen_latent, null) per
#           condition. >0 means the output carries the plan; growth with exagg
#           means the dial is realized in the audio.
#        b. MFCC within vs across: sampling-noise (different seeds, same cond)
#           vs prosody effect (matched seeds, different cond). ratio > 1 means
#           exagg changes the audio reliably above noise.
#
# `--exagg e` dials the plan injection as  null + e*(pred - null):
#   e=0  -> flat/neutral prosody
#   e=1  -> natural predicted prosody
#   e>1  -> exaggerated (enforced) prosody
# `--random_prosody` replaces it with noise (sanity check).
# `--ref_wav_prosody` (condition `ref`) bypasses the planner and injects a REAL
# prosody latent extracted from REF_PROS via the stage-1 codec (wav -> F0/E/V
# -> codec latent), testing whether the decoder follows an injected plan.
# NOTE: external prosody bypasses the exagg dial (model.encode_context).
#
# IMPORTANT: pass REF_SPK=<ref_speaker.wav> so every condition uses the SAME
# voice; otherwise differences are contaminated by voice changes.
#
# Usage:
#   REF_SPK=ref.wav ./scripts/run_prosody_test.sh
#   DEMO_FILE=my_lines.txt CHECKPOINT=... N_SEEDS=3 ./scripts/run_prosody_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEMO_FILE="${DEMO_FILE:-$SCRIPT_DIR/demo_prosody.txt}"
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/light_fusion_r3.json}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/checkpoints/seedvox_light_fusion_epoch_102.pt}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/demos/prosody_test}"
SEED="${SEED:-430}"
N_SEEDS="${N_SEEDS:-3}"
TEMP="${TEMP:-0.15}"
CFG="${CFG:-1.15}"
DTYPE="${DTYPE:-bf16}"
SR="${SR:-24000}"
REF_SPK="${REF_SPK:-../autovoc/dataset/wavs/LJ002-0321.wav}"
REF_PROS="${REF_PROS:-$REF_SPK}"
MIN_P="${MIN_P:-0.01}"
MANIFEST="${MANIFEST:-../autovoc/dataset/train_manifest_jepa.jsonl}"
N_PLAN="${N_PLAN:-16}"


if [[ ! -f "$DEMO_FILE" ]]; then
  echo "Demo file not found: $DEMO_FILE"
  exit 1
fi
rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

if [[ -n "$REF_SPK" ]]; then
  if [[ ! -f "$REF_SPK" ]]; then
    echo "Reference speaker not found: $REF_SPK"
    exit 1
  fi
  SPK_ARGS=(--ref_wav_speaker "$REF_SPK")
  echo "Locking speaker across all conditions: $REF_SPK"
else
  SPK_ARGS=()
  echo "WARNING: no REF_SPK set — voice is NOT locked across conditions,"
  echo "         comparisons may reflect voice changes, not prosody."
fi

# cond_name -> batch_infer prosody flags
declare -A PROS_FLAGS=(
  [exagg00_0]="--exagg 0"
  [exagg00_5]="--exagg 0.5"
  [exagg01_0]="--exagg 1"
  [exagg01_5]="--exagg 1.5"
  [exagg02_0]="--exagg 2"
  [exagg04_0]="--exagg 4"
  [exagg10_0]="--exagg 100"
  [random]="--random_prosody"
)

if [[ -n "$REF_PROS" && -f "$REF_PROS" ]]; then
  PROS_FLAGS[ref]="--ref_wav_prosody $REF_PROS"
  echo "Adding 'ref' condition: injects real prosody latent from $REF_PROS"
else
  echo "WARNING: REF_PROS not set / not found ($REF_PROS) — skipping 'ref' condition"
fi

CONDS=("${!PROS_FLAGS[@]}")
IFS=$'\n' CONDS=($(sort <<<"${CONDS[*]}")); unset IFS

# ── Phase 1: generation, N seeds per condition (seeds SEED .. SEED+N_SEEDS-1,
#    matched across conditions so within/across decomposition is valid)
for COND in "${CONDS[@]}"; do
  OUT_DIR="$OUT_ROOT/$COND"
  mkdir -p "$OUT_DIR"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  condition: $COND   ($(eval echo ${PROS_FLAGS[$COND]}), $N_SEEDS seed(s))"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  eval python "$SCRIPT_DIR/scripts/batch_infer.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --demo_file "$DEMO_FILE" \
    --output_dir "$OUT_DIR" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --variant_n "$N_SEEDS" \
    --temp "$TEMP" \
    --cfg "$CFG" \
    --min_p "$MIN_P" \
    "${SPK_ARGS[@]}" \
    ${PROS_FLAGS[$COND]} \
    --log_metrics
done

# ── Phase 2: planning validation (planner predicts text-specific prosody)
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Phase 2 — Planning validation"
echo "═══════════════════════════════════════════════════"
python "$SCRIPT_DIR/scripts/validate_prosody_planning.py" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --num "$N_PLAN"

# ── Phase 3: audio-level metrics (plan-follow + within/across MFCC)
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Phase 3 — Audio-level metrics"
echo "═══════════════════════════════════════════════════"
python "$SCRIPT_DIR/scripts/prosody_audio_metrics.py" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --demo_file "$DEMO_FILE" \
  --output_dir "$OUT_ROOT" \
  --conditions "${CONDS[@]}" \
  --num_seeds "$N_SEEDS" \
  --sr "$SR" \
  ${REF_PROS:+--ref_wav_prosody "$REF_PROS"}

echo ""
echo "Interpretation:"
echo "  - Phase 2: matched cos >> null/mismatched + 100% self-match  -> planner works"
echo "  - Phase 3a: plan-follow gap (cos(gen,pred) - cos(gen,null)) > 0 and grows with exagg"
echo "              -> decoder realizes the plan in the audio"
echo "  - Phase 3b: across/within MFCC ratio > 1 -> exagg changes audio reliably"
echo "              (a global style needs only a modest ratio, NOT frame-wise F0 moves)"
echo "  - Outputs in: $OUT_ROOT/"
