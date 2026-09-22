#!/usr/bin/env bash
# Setup for P3 (compute-tree collection + E0 oracle gate).
# Run once per fresh GPU machine, from the repo root.
set -euo pipefail

echo "== 1. Fix CUDA_VISIBLE_DEVICES if it's a GPU UUID instead of an index =="
# Some cloud GPU providers (RunPod/Vast) export CUDA_VISIBLE_DEVICES as a UUID
# like "GPU-4edaf659-...", which crashes vLLM 0.7.x
# (ValueError: invalid literal for int() with base 10: 'GPU-...').
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    echo "  Detected UUID-style CUDA_VISIBLE_DEVICES ('$CUDA_VISIBLE_DEVICES') -> resetting to 0"
    unset CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=0
fi
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset, will default to 0>}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "== 2. Install package (CPU core + GPU extras + dev) =="
pip install -e ".[gpu,dev]" --break-system-packages -q

echo "== 3. Pin versions known-good from P1 (see reports/P1_baselines.md) =="
pip install --break-system-packages -q \
    "vllm==0.7.3" \
    "transformers==4.49.0" \
    "torch==2.5.1"

echo "== 4. Install missing runtime deps not declared in pyproject.toml =="
# benchmarks.py::load_gsm8k() imports `datasets`, which pyproject.toml does not list.
# Without this, p3_collect.py crashes at benchmark-loading time, not at model-loading time.
pip install --break-system-packages -q "datasets>=2.14" scipy scikit-learn

echo "== 5. Sanity check =="
python -c "import torch, vllm, transformers, datasets, scipy, sklearn; \
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available()); \
print('vllm', vllm.__version__); print('transformers', transformers.__version__); \
print('datasets', datasets.__version__)"

echo "== 6. Run CPU-only test suite (should be green before touching GPU) =="
pytest -q

echo
echo "Setup done. Next:"
echo "  python p3_pilot.py     # measure real throughput before committing GPU-hours"
echo "  python p3_collect.py   # collect the compute tree (the expensive step)"
