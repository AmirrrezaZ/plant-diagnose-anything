#!/bin/bash
# usage: q_tft.sh GPU TAG "ARGS"  -> runs folds 0-9 seed 0 (skips finished)
cd "$(dirname "$0")/.."
GPU=$1; TAG=$2; shift 2
for f in 0 1 2 3 4 5 6 7 8 9; do
  [ -f results/v3/v6tft_${TAG}_fold${f}.json ] && continue
  CUDA_VISIBLE_DEVICES=$GPU timeout 1500 venv/bin/python scripts/v6_text_ft.py --fold $f --tag $TAG "$@" > logs/v6tft_${TAG}_f$f.log 2>&1
done
touch logs/${TAG}_done.flag
