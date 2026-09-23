#!/usr/bin/env bash
# P3 in one shot: env fix + robust install + FRESH baseline re-measure (matched n)
# + oracle tree collection (nr=2) + Gate 2 — all logged, survives disconnect.
#
# Usage:  bash run_p3.sh
# Watch:  tail -f p3_run.log
set -euo pipefail

echo "== 1. GPU index fix =="
export CUDA_VISIBLE_DEVICES=0
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "== 2. Install (pin heavy deps FIRST so pip doesn't backtrack over vllm) =="
pip install --break-system-packages -q "torch==2.5.1"
pip install --break-system-packages -q "vllm==0.7.3" "transformers==4.49.0"
pip install -e ".[gpu,dev]" --break-system-packages -q --no-deps
pip install -e ".[gpu,dev]" --break-system-packages -q
pip install --break-system-packages -q "datasets>=2.14" scipy scikit-learn

echo "== 3. Ensure progress logging (idempotent) =="
grep -q 'setup_logging("INFO")' p3_collect.py || \
  sed -i '/from caac.types import CostWeights/a\    from caac.utils.logging import setup_logging\n\n    setup_logging("INFO")' p3_collect.py

echo "== 4. Sanity =="
python -c "import torch,vllm,transformers,datasets; print('cuda:',torch.cuda.is_available(),'| vllm',vllm.__version__)"

echo "== 5. Launch background job: baseline -> P3 oracle tree -> Gate 2 =="
nohup bash -c '
set -e
echo "[1/3] Re-measuring baseline greedy_cot on the SAME 40 problems (n=40)..."
python experiments/run_e2_pareto.py --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct --n-problems 40 --out outputs/e2_pareto.json

echo "[2/3] Choosing baseline-cost from the fresh frontier..."
BASELINE=$(python extract_baseline.py)
echo "    baseline-cost = $BASELINE"

echo "[3/3] Collecting oracle tree (nr=2, 40 problems, depth 3) + Gate 2..."
python p3_collect.py --n-problems 40 --n-rollouts 2 --max-depth 3 \
  --baseline-cost "$BASELINE" --out outputs/tree.json

echo "=== P3 COMPLETE: oracle frontier + Gate 2 PASS/FAIL are above ==="
' > p3_run.log 2>&1 &
disown
echo
echo "P3 job started in background (PID $!)."
echo "  Watch:  tail -f p3_run.log"
echo "  Expect: ~10 min baseline + ~45 min tree = ~1h total"
