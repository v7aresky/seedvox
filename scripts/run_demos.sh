#!/usr/bin/env bash
set -euo pipefail

# run_demos.sh — Run inference on each line of demo.txt
#
# Usage:
#   ./scripts/run_demos.sh [options]
#
# Each line in demo.txt is treated as the input text.
# Use the delimiter ' || ' for per-line overrides:
#
#   Text to speak || output_name.wav || --ref_wav_speaker path.wav --play
#
# Global defaults (can be overridden per-line via the third field):
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEMO_FILE="${DEMO_FILE:-$SCRIPT_DIR/demo.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/demos}"
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/light_fusion_r3.json}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/checkpoints/seedvox_light_fusion_epoch_86.pt}"
LORA="${LORA:-}"
BASE_FLAGS=""

mkdir -p "$OUTPUT_DIR"

# Parse named flags passed to this script (they are applied to EVERY line)
EXTRA_GLOBAL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora_checkpoint) LORA="$2"; shift 2 ;;
    --ref_wav_speaker|--ref_wav_prosody|--cfg_scale|--variant_axis|--variant_n|--seed)
      EXTRA_GLOBAL+=" $1 $2"; shift 2 ;;
    --play|--view_waveform|--log_metrics|--compile|--random_prosody|--random_speaker)
      EXTRA_GLOBAL+=" $1"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

LINECOUNT=0
SUCCESS=0
FAIL=0

while IFS= read -r line || [[ -n "$line" ]]; do
  # Skip empty lines and comments
  [[ -z "${line// /}" ]] && continue
  [[ "$line" == \#* ]] && continue

  LINECOUNT=$((LINECOUNT + 1))
  NUM=$(printf "%03d" "$LINECOUNT")

  # Parse per-line fields separated by ' || '
  TEXT="$line"
  OUTFILE="demo_${NUM}.wav"
  EXTRA_LINE=""

  if [[ "$line" == *" || "* ]]; then
    IFS=' || ' read -r TEXT OUTFILE EXTRA_LINE <<< "$line"
  fi

  # Build command
  CMD=("python3" "-m" "explicit_pros_phon_planner.infer"
    "--config" "$CONFIG"
    "--checkpoint" "$CHECKPOINT"
    "--text" "$TEXT"
    "--output" "$OUTPUT_DIR/$OUTFILE"
  )

  [[ -n "$LORA" ]] && CMD+=("--lora_checkpoint" "$LORA")
  [[ -n "$EXTRA_GLOBAL" ]] && CMD+=($EXTRA_GLOBAL)
  [[ -n "$EXTRA_LINE" ]] && CMD+=($EXTRA_LINE)

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [$NUM] Generating: $OUTFILE"
  echo "  Text: ${TEXT:0:80}..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if "${CMD[@]}" 2>&1; then
    SUCCESS=$((SUCCESS + 1))
  else
    FAIL=$((FAIL + 1))
    echo "  ✗ FAILED on line $LINECOUNT" >&2
  fi
done < "$DEMO_FILE"

echo ""
echo "═══════════════════════════════════════════"
echo "  Done: $SUCCESS succeeded, $FAIL failed out of $LINECOUNT lines"
echo "  Outputs in: $OUTPUT_DIR/"
echo "═══════════════════════════════════════════"
