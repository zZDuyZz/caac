#!/usr/bin/env bash
# =============================================================================
# NGÀY 1 — PHẦN 1/2: cài môi trường + tải model/dataset (chạy MỘT LẦN đầu phiên)
# Chạy trong tmux. Không tự xóa gì. Tải xong mới sang day1_run.sh.
#
#   tmux new -s caac
#   bash day1_setup.sh
# =============================================================================
set -euo pipefail

# --- đường dẫn (sửa nếu ổ của bạn khác) ------------------------------------
export HF_HOME=/workspace/hf_cache
export POOLS=/workspace/pools
export REPO=/workspace/caac
REPO_URL=https://github.com/zZDuyZz/caac.git
mkdir -p "$HF_HOME" "$POOLS"

echo "=== 0. GPU phải thấy được ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== 1. Clone repo ==="
if [ ! -d "$REPO/.git" ]; then
  git clone "$REPO_URL" "$REPO"
fi
cd "$REPO"
git pull --ff-only
# build_pool.py phải là bản gom batch; nếu dòng này in 0 => bản cũ, DỪNG LẠI
echo -n "build_pool batched? (phải 1): "; grep -c "llm.generate(prompts, sps)" build_pool.py || true

echo "=== 2. Cài gói ==="
pip install -q vllm==0.7.3 math-verify datasets "huggingface_hub[cli]"
pip install -q -e .
python -c "import vllm; print('vllm', vllm.__version__)"

echo "=== 3. Test CPU (phải 8/8) ==="
PYTHONPATH=src python tests/test_pool_replay.py | tail -2

echo "=== 4. Tải model + dataset ==="
# Model tải qua HF hub. Dataset KHÔNG đặt offline (nhẹ, để datasets tự resolve).
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
python - <<'PY'
# Materialize dataset vào cache đúng cách loader sẽ gọi (bắt lỗi tên GSM8K sớm).
from datasets import load_dataset
m = load_dataset("HuggingFaceH4/MATH-500", split="test"); print("MATH-500:", len(m))
g = load_dataset("gsm8k", "main", split="test");          print("GSM8K:", len(g))
PY

echo
echo "=== XONG SETUP. Model weights -> dùng offline cho chắc ở day1_run.sh ==="
echo "Nếu bước GSM8K ở trên LỖI, chạy bản vá rồi làm lại mục 4:"
echo "    sed -i 's/load_dataset(\"gsm8k\", \"main\"/load_dataset(\"openai\\/gsm8k\", \"main\"/' src/caac/data/benchmarks.py"
