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

echo "=== 0. GPU (mọi pool của lần chạy này phải cùng một loại card) ==="
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

echo "=== 4. Tải model vào cache ==="
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct >/dev/null
huggingface-cli download Qwen/Qwen2.5-7B-Instruct  >/dev/null

echo "=== 5. qids A3 (chọn ngẫu nhiên phân tầng, seed 0; tất định) ==="
mkdir -p qids
[ -f qids/a3_math_train.txt ]      || python make_qids.py --dataset math_train      --n 1000 --out qids/a3_math_train.txt
[ -f qids/a3_math_test_ex500.txt ] || python make_qids.py --dataset math_test_ex500 --split test --n 1000 --out qids/a3_math_test_ex500.txt
[ -f qids/a3_gsm8k_train.txt ]     || python make_qids.py --dataset gsm8k --split train --n 1500 --out qids/a3_gsm8k_train.txt
wc -l qids/*.txt
PYTHONPATH=src python - <<'PY'
# kiểm số câu VÀ mọi qid của A3 có trong dataset; sai -> dừng (set -e) trước khi tốn giờ GPU
from caac.data.benchmarks import load_benchmark as L
EXPECT = {("math500","test"): 500, ("gsm8k","test"): 1319, ("gsm8k","train"): 7473,
          ("math_train","train"): 7499, ("math_test_ex500","test"): 4501}
ids = {}
for (name, split), n in EXPECT.items():
    items = L(name, split=split); ids[(name, split)] = {it["id"] for it in items}
    print(f"  {name:16} {split:5} {len(items)} (cần {n})")
    assert len(items) == n, f"{name}/{split}: {len(items)} != {n}"
for (name, split), f, n in [(("gsm8k","train"),"qids/a3_gsm8k_train.txt",1500),
                            (("math_train","train"),"qids/a3_math_train.txt",1000),
                            (("math_test_ex500","test"),"qids/a3_math_test_ex500.txt",1000)]:
    q = [l.strip() for l in open(f) if l.strip()]
    assert len(q) == n and len(set(q)) == n, f"{f}: {len(q)} dòng"
    miss = [x for x in q if x not in ids[(name, split)]]
    assert not miss, f"{f}: {len(miss)} qid không có trong dataset (vd {miss[:3]})"
    print(f"  {f}: {n} qid hợp lệ")
PY
pip freeze | grep -iE "^(vllm|torch|transformers|datasets|math-verify|huggingface)" > /workspace/env_s2.txt
nvidia-smi --query-gpu=name,driver_version --format=csv >> /workspace/env_s2.txt
echo "=== SETUP XONG -> bash session2_run.sh ==="
