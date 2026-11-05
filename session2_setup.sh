#!/usr/bin/env bash
# =============================================================================
# PHIÊN GPU 2 — PHẦN 1/2: môi trường + tải model/dataset + qids A3. Chạy MỘT lần.
#   tmux new -s caac
#   bash session2_setup.sh 2>&1 | tee /workspace/s2_setup.log
# Không cần GPU cho các bước sau bước 0 -> làm xong rồi mới chạy session2_run.sh.
# =============================================================================
set -euo pipefail
export HF_HOME=/workspace/hf_cache REPO=/workspace/caac
REPO_URL=https://github.com/zZDuyZz/caac.git
mkdir -p "$HF_HOME"

echo "=== 0. GPU (phải là A100-SXM4-40GB như A2 — pool tái lập cần cùng loại card) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== 1. Repo ==="
[ -d "$REPO/.git" ] || git clone "$REPO_URL" "$REPO"
cd "$REPO" && git pull --ff-only && git log --oneline -1

echo "=== 2. Gói (PIN đúng môi trường Day 1 / A2) ==="
pip install -q vllm==0.7.3 transformers==4.52.4 "huggingface_hub[cli]<1.0" \
               datasets==3.6.0 math-verify==0.9.0 pytest
pip install -q -e .
python -c "import vllm,transformers,torch;print('vllm',vllm.__version__,'tf',transformers.__version__,'torch',torch.__version__)"

echo "=== 3. Test CPU (phải pass hết) ==="
python -m pytest -q -p no:warnings

echo "=== 4. Tải model + dataset vào cache ==="
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct >/dev/null
huggingface-cli download Qwen/Qwen2.5-7B-Instruct  >/dev/null
PYTHONPATH=src python - <<'PY'
from caac.data.benchmarks import load_benchmark as L
for name, split in [("math500","test"),("gsm8k","test"),("gsm8k","train"),
                    ("math_train","train"),("math_test_ex500","test")]:
    print(f"  {name:16} {split:5} {len(L(name, split=split))}")
PY

echo "=== 5. qids A3 (chọn ngẫu nhiên phân tầng, seed 0; tất định) ==="
mkdir -p qids
[ -f qids/a3_math_train.txt ]      || python make_qids.py --dataset math_train      --n 1000 --out qids/a3_math_train.txt
[ -f qids/a3_math_test_ex500.txt ] || python make_qids.py --dataset math_test_ex500 --split test --n 1000 --out qids/a3_math_test_ex500.txt
[ -f qids/a3_gsm8k_train.txt ]     || python make_qids.py --dataset gsm8k --split train --n 1500 --out qids/a3_gsm8k_train.txt
wc -l qids/*.txt
pip freeze | grep -iE "^(vllm|torch|transformers|datasets|math-verify|huggingface)" > /workspace/env_s2.txt
nvidia-smi --query-gpu=name,driver_version --format=csv >> /workspace/env_s2.txt
echo "=== SETUP XONG -> bash session2_run.sh ==="
