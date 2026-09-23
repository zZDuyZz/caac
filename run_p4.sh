#!/usr/bin/env bash
# P4 training-tree collection.
# ==> RUN ONLY AFTER P3's Gate 2 PASSES (or is rescued). <==
# This is where the main GPU budget gets committed; the gate exists to decide
# whether to spend it. Do not run this if P3 Gate 2 = FAIL and rescue failed.
#
# Config: nr=2, depth 3, 150 problems, fresh tree. Per the determinism finding
# (see report note), nr=2 is informationally equivalent to nr=4 while ~11x
# cheaper per problem; the saving is reallocated to more problems (150) for
# state diversity, which the estimator benefits from most.
#
# Usage:  bash run_p4.sh
# Watch:  tail -f p4_run.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

grep -q 'setup_logging("INFO")' p3_collect.py || \
  sed -i '/from caac.types import CostWeights/a\    from caac.utils.logging import setup_logging\n\n    setup_logging("INFO")' p3_collect.py

echo "Launching P4 training-tree (nr=2, 150 problems, depth 3) in background..."
nohup python p3_collect.py \
  --n-problems 150 --n-rollouts 2 --max-depth 3 \
  --out outputs/tree_p4.json \
  > p4_run.log 2>&1 &
disown
echo "P4 job started (PID $!)."
echo "  Watch:  tail -f p4_run.log"
echo "  Expect: ~1.5-2h. Output tree: outputs/tree_p4.json (feeds estimator training)."
