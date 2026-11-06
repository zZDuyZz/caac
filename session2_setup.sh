#!/usr/bin/env bash
# =============================================================================
# PHIÊN GPU 2 — PHẦN 1/2: môi trường + tải model/dataset + qids A3. Chạy MỘT lần.
#   tmux new -s caac
#   bash session2_setup.sh 2>&1 | tee /workspace/s2_setup.log
# Không cần GPU cho các bước sau bước 0 -> làm xong rồi mới chạy session2_run.sh.
# =============================================================================
set -euo pipefail
export HF_HOME=/workspace/hf_cache REPO=/workspace/caac VENV=/workspace/venv
REPO_URL=https://github.com/zZDuyZz/caac.git
# 2026-11-06: model + engine refresh (CAAC_MASTER §5, §13.10). Đổi ở ĐÂY thì đổi cả session2_run.sh.
VLLM_VERSION=0.32.0
QS=Qwen/Qwen3.5-2B; QB=Qwen/Qwen3.5-9B
mkdir -p "$HF_HOME"

echo "=== 0. GPU (mọi pool của lần chạy này phải cùng một loại card) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== 1. Repo ==="
[ -d "$REPO/.git" ] || git clone "$REPO_URL" "$REPO"
cd "$REPO" && git pull --ff-only && git log --oneline -1

echo "=== 2. Môi trường: venv Python 3.12 + vLLM $VLLM_VERSION (engine V1) ==="
# vLLM >= 0.32 cần Python >= 3.11; Ubuntu 22.04 mặc định 3.10 -> dựng venv riêng bằng uv.
command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -p 3.12 "$VENV"
# shellcheck disable=SC1091
set +u; source "$VENV/bin/activate"; set -u   # +u: script activate có thể đọc biến chưa đặt
# --torch-backend=auto: uv chọn bánh xe torch khớp driver CUDA; bản uv cũ không có cờ này -> thử lại không cờ
uv pip install "vllm==$VLLM_VERSION" --torch-backend=auto || uv pip install "vllm==$VLLM_VERSION"
uv pip install datasets math-verify==0.9.0 pytest "huggingface_hub"
uv pip install -e .
python - <<PY
import sys, vllm, transformers, torch
print("python", sys.version.split()[0], "vllm", vllm.__version__, "tf", transformers.__version__,
      "torch", torch.__version__, "cuda", torch.version.cuda)
assert vllm.__version__.startswith("$VLLM_VERSION"), "sai phiên bản vLLM"
assert torch.cuda.is_available(), "torch không thấy GPU (driver/bánh xe CUDA không khớp)"
PY

echo "=== 3. Test CPU (phải pass hết) ==="
python -m pytest -q -p no:warnings

echo "=== 4. Tải model vào cache (sai tên repo -> dừng ở đây, chưa tốn giờ sinh dữ liệu) ==="
python - <<PY
from huggingface_hub import snapshot_download
for m in ("$QS", "$QB"):
    p = snapshot_download(m)
    print("  ok", m, "->", p)
PY

echo "=== 4b. Chat template: thinking TẮT tường minh (không có <think> mở trong prompt) ==="
PYTHONPATH=src python - <<PY
from transformers import AutoTokenizer
from caac.data.prompts import build_chat_prompt
for m in ("$QS", "$QB"):
    tok = AutoTokenizer.from_pretrained(m)
    off = build_chat_prompt(tok, "1+1?", thinking=False)
    on = build_chat_prompt(tok, "1+1?", thinking=True)
    tail = off[-80:].replace(chr(10), "\\n")
    print(f"  {m}: prompt tail (thinking off) = ...{tail}")
    # một <think> mở mà không đóng ở cuối prompt nghĩa là model sẽ suy nghĩ dài -> sai giao thức
    assert not off.rstrip().endswith("<think>"), "thinking vẫn bật trong prompt"
    print(f"  {m}: thinking on/off khác nhau: {on != off}")
PY

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
{ python --version; uv pip freeze | grep -iE "^(vllm|torch|transformers|datasets|math-verify|huggingface|flashinfer|triton)"; } > /workspace/env_s2.txt
nvidia-smi --query-gpu=name,driver_version --format=csv >> /workspace/env_s2.txt
echo "=== SETUP XONG -> bash session2_run.sh ==="
