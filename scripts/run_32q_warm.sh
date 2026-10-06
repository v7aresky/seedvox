#!/usr/bin/env bash
# Warm-start 32q test train: resume the trained 16q model at 32 codebooks.
# Self-respawns on external kills / freezes (watchdog kills a stale trainer
# and the outer loop restarts from the latest resume.pt).
set -u
cd /home/vpollet/proj/seedvox
LOG=logs/train_32q_warm.log
CFG=configs/light_fusion_r6_32q_nostyle.json
CKPT=checkpoints/seedvox_light_fusion_r6_32q_nostyle_epoch_159.pt

while :; do
  echo "[watchdog] $(date +%T) spawn" >> "$LOG"
  # inner runner: spawn python, watchdog-stall-kill, wait for exit
  bash -c '
    LOG='"$LOG"'
    cd /home/vpollet/proj/seedvox
    python -m explicit_pros_phon_planner.trainer_fusion \
        --config '"$CFG"' --resume '"$CKPT"' \
        >> "$LOG" 2>&1 &
    CHILD=$!
    while kill -0 $CHILD 2>/dev/null; do
      if [ $(( $(date +%s) - $(stat -c %Y "$LOG") )) -gt 900 ]; then
        kill -9 $CHILD
        echo "[watchdog] killed stale child $CHILD at $(date +%T)" >> "$LOG"
      fi
      sleep 300
    done
    echo "[watchdog] child exited rc=$? at $(date +%T)" >> "$LOG"
  '
  echo "[watchdog] $(date +%T) respawn in 20s" >> "$LOG"
  sleep 20
done