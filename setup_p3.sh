#!/bin/bash
# setup_p3.sh — one-command environment for CAAC Phase 3.
# Pre-empts every version/env error hit in P1/P2. Run with:  bash setup_p3.sh
set -e

# --- must be inside the repo ---
cd /workspace/caac 2>/dev/null || {
  echo "ERROR: /workspace/caac not found."
  echo "Run first:  cd /workspace && git clone https://github.com/zZDuyZz/caac.git"
  exit 1
}

# --- git identity (fixes 'unable to auto-detect email') ---
git config --global user.name  "zZDuyZz"
git config --global user.email "minhduytmd06@gmail.com"

echo "=== [1/5] Pinned deps (order matters; use python -m pip to avoid conda/venv split) ==="
python -m pip install numpy==2.1.0 scipy scikit-learn pyyaml --break-system-packages -q
python -m pip install torch==2.5.1 --break-system-packages -q
python -m pip install vllm==0.7.3 --break-system-packages -q
python -m pip install transformers==4.49.0 --break-system-packages -q
python -m pip install datasets pytest --break-system-packages -q
python -m pip install -e . --break-system-packages -q

echo "=== [2/5] Safety net: np.trapezoid -> np.trapz ==="
grep -rl "np.trapezoid" src/ 2>/dev/null | xargs -r sed -i 's/np\.trapezoid/np.trapz/g' || true

echo "=== [3/5] GPU env fix (UUID device -> integer) ==="
export CUDA_VISIBLE_DEVICES=0

echo "=== [4/5] Verify torch + GPU ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== [5/5] Tests ==="
python -m pytest -q

echo ""
echo "=== READY.  In THIS shell also run:  export CUDA_VISIBLE_DEVICES=0 ==="
echo "=== (the p3_*.py scripts set it themselves too, so you are covered) ==="
