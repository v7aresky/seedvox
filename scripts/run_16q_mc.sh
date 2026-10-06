#!/usr/bin/env bash
# 16q multi-corpus run: LJ + DailyTalk + LibriTTS + HiFiTTS + GLOBES (276,529 utts),
# warm-started from the 16q looped2 e135 checkpoint; respawns continue from the
# run's own resume.pt (leading-silence runway active: config leading_silence).
# Self-respawns on external kills / freezes (watchdog kills a stale trainer
# and the outer loop restarts from the latest resume.pt).
set -u
cd /home/vpollet/proj/seedvox
LOG=logs/train_16q_mc.log
CFG=configs/light_fusion_r6_16q_multicorpus.json
CKPT=checkpoints/seedvox_light_fusion_r6_16q_multicorpus_resume.pt

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