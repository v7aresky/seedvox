#!/usr/bin/env bash
# run_mono_test.sh — A/B/C test of the monotone cross-attention window (--mono_slack)
#
# Runs the same demo lines at mono_slack = 0 (off), 2, 4 with a fixed seed and
# reports per-line durations so you can spot loop/stutter (inflated length) vs
# prosody compression (shortened length).
#
# Usage:
#   ./scripts/run_mono_test.sh
#   DEMO_FILE=my_lines.txt CHECKPOINT=... ./scripts/run_mono_test.sh
#
# Per-line syntax (same as run_demos.sh):  Text || outname.wav
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEMO_FILE="${DEMO_FILE:-$SCRIPT_DIR/demo_align.txt}"
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/light_fusion_r3.json}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/checkpoints/seedvox_light_fusion_epoch_86.pt}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/demos/mono_test}"
SEED="${SEED:-0}"
TEMP="${TEMP:-0.1}"
CFG="${CFG:-1.5}"
DTYPE="${DTYPE:-bf16}"
SLACKS="${SLACKS:-0 2 4}"
REF_SPK="${REF_SPK:-}"

if [[ ! -f "$DEMO_FILE" ]]; then
  echo "Demo file not found: $DEMO_FILE"
  echo "Create it with one 'Text || outname.wav' per line, or set DEMO_FILE=..."
  exit 1
fi

mkdir -p "$OUT_ROOT"
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
fi

# 1. Run each slack variant
for SLACK in $SLACKS; do
  OUT_DIR="$OUT_ROOT/slack_$SLACK"
  mkdir -p "$OUT_DIR"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  mono_slack = $SLACK"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python "$SCRIPT_DIR/scripts/batch_infer.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --demo_file "$DEMO_FILE" \
    --output_dir "$OUT_DIR" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --temp "$TEMP" \
    --cfg "$CFG" \
    --mono_slack "$SLACK" \
    "${SPK_ARGS[@]}" \
    --log_metrics
done

# 2. Compare durations per line across variants
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Duration comparison (seconds) per demo line"
echo "═══════════════════════════════════════════════════"

read -r -a SLACK_ARR <<< "$SLACKS"
BASE_DIR="$OUT_ROOT/slack_${SLACK_ARR[0]}"
FIRST=1

for wav in "$BASE_DIR"/*.wav; do
  [[ -e "$wav" ]] || continue
  NAME="$(basename "$wav")"
  ROW=""
  for SLACK in "${SLACK_ARR[@]}"; do
    D="$OUT_ROOT/slack_$SLACK/$NAME"
    if [[ -f "$D" ]]; then
      DUR="$(python -c "import torchaudio,sys; print(f'{torchaudio.load(sys.argv[1])[0].shape[1]/24000:.2f}')" "$D" 2>/dev/null)"
    else
      DUR="--"
    fi
    ROW="$ROW $(printf '%7s' "$DUR")"
  done
  if [[ $FIRST -eq 1 ]]; then
    HDR="$(printf '  %-22s' 'line')"
    for SLACK in "${SLACK_ARR[@]}"; do
      HDR="$HDR $(printf '%7s' "slack=$SLACK")"
    done
    echo "$HDR"
    echo "  $(printf '─%.0s' {1..50})"
    FIRST=0
  fi
  printf '  %-22s%s\n' "$NAME" "$ROW"
done

echo ""
echo "Look for:"
echo "  - slack=0 much longer => loop/stutter (window fixes it)"
echo "  - slack>0 much shorter => prosody compressed (window too tight; try slack=4)"
echo "  Outputs in: $OUT_ROOT/"
