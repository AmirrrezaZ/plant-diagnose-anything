#!/bin/bash
# usage: q_joint.sh GPU TAG "folds" [extra args]
cd "$(dirname "$0")/.."; GPU=$1; TAG=$2; FOLDS=$3; shift 3
for f in $FOLDS; do
  ls results/v3/v6joint_${TAG}_fold${f}_s*.json >/dev/null 2>&1 && continue
  CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false timeout 3000 venv/bin/python scripts/v6_joint.py --fold $f --tag $TAG --seed 0 --steps 600 "$@" > logs/v6joint_${TAG}_f$f.log 2>&1
done
touch logs/joint_${TAG}_${GPU}_done.flag
