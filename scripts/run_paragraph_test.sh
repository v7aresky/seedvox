#!/usr/bin/env bash
# run_paragraph_test.sh — LISTENING test: does the JEPA planner produce a
# COHESIVE multi-sentence paragraph, vs. a non-JEPA baseline?
#
# Key design decision: each paragraph is generated in a SINGLE PASS (one plan
# over the whole passage). Per-sentence generation gives each sentence an
# independent plan, so cross-sentence coherence can only be judged in
# single-pass mode — that is exactly what this test generates.
#
# Because there is no "non-JEPA" checkpoint, the non-JEPA baseline is the
# SAME model with the plan zeroed (exagg=0 -> null prosody; decoder falls back
# to text/speaker defaults). The full condition set isolates the planner's
# contribution:
#
#   jepa    --exagg 1      the JEPA plan drives prosody (default inference)
#   null    --exagg 0      NO prosody plan injected (non-JEPA baseline)
#   random  --random_prosody  noise prosody (negative control)
#   ref     --ref_wav_prosody  ONE real prosody latent applied to the WHOLE
#                              paragraph (forced-global-style control)
#   emph    --exagg 1.5    amplified JEPA plan (does more prosody help?)
#
# Speaker is locked across ALL conditions (REF_SPK) so differences are prosody,
# not voice. N matched seeds per condition for later metrics.
#
# LISTEN for (per paragraph):
#   1. Naturalness: jepa vs null — does plan-driven prosody sound more natural
#      (phrase rises/falls, pauses, emphasis) than flat text-driven defaults?
#   2. Cohesion: does pitch/rate/pause structure stay coherent ACROSS the
#      sentence boundaries in jepa, or drift/grid in null?
#   3. Sanity: random should sound erratic/broken; ref should sound the most
#      uniformly styled; emph should sound over-emphatic but not broken.
#   4. Preference: which single condition would you ship as a paragraph?
#
# Usage:
#   CHECKPOINT=checkpoints/seedvox_light_fusion_epoch_111.pt ./scripts/run_paragraph_test.sh
#   OUT_ROOT=... N_SEEDS=2 TEMP=0.15 ./scripts/run_paragraph_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/light_fusion_r3.json}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/checkpoints/seedvox_light_fusion_latest.pt}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/demos/paragraph_test}"
SEED="${SEED:-430}"
N_SEEDS="${N_SEEDS:-2}"
TEMP="${TEMP:-0.15}"
CFG="${CFG:-1.15}"
DTYPE="${DTYPE:-bf16}"
MIN_P="${MIN_P:-0.01}"
REF_SPK="${REF_SPK:-../autovoc/dataset/wavs/LJ002-0321.wav}"

DEMO_FILES=(
  "$SCRIPT_DIR/demos/paragraph_narrative.txt"
  "$SCRIPT_DIR/demos/paragraph_emotional.txt"
)

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi
for f in "${DEMO_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Demo file not found: $f"
    exit 1
  fi
done

rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

if [[ -n "$REF_SPK" && ! -f "$REF_SPK" ]]; then
  echo "Reference speaker not found: $REF_SPK"
  exit 1
fi

for DEMO in "${DEMO_FILES[@]}"; do
  PARA_NAME="$(basename "$DEMO" .txt)"
  OUT_DIR="$OUT_ROOT/$PARA_NAME"
  mkdir -p "$OUT_DIR"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  paragraph: $PARA_NAME   ($(wc -l < "$DEMO") conditions x $N_SEEDS seeds)"
  echo "  checkpoint: $(basename "$CHECKPOINT")"
  echo "  speaker locked: $REF_SPK"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python "$SCRIPT_DIR/scripts/batch_infer.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --demo_file "$DEMO" \
    --output_dir "$OUT_DIR" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --variant_n "$N_SEEDS" \
    --temp "$TEMP" \
    --cfg "$CFG" \
    --min_p "$MIN_P" \
    --ref_wav_speaker "$REF_SPK" \
    --log_metrics
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Outputs in: $OUT_ROOT/"
echo "  Per paragraph dir: {narrative,emotional}/{jepa,null,random,ref,emph}_[0..N].wav"
echo ""
echo "  LISTEN GUIDE:"
echo "    - jepa   : natural plan-driven prosody (the candidate)"
echo "    - null   : NO prosody plan — non-JEPA baseline"
echo "    - random : noise prosody (sanity: should sound broken)"
echo "    - ref    : one real prosody forced over the whole paragraph"
echo "    - emph   : exagg 1.5 — amplified plan"
echo "  Listen A/B within a paragraph (same seed index), same speaker."
echo "═══════════════════════════════════════════════════"
