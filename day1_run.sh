#!/usr/bin/env bash
# =============================================================================
# NGÀY 1 — PHẦN 2/2: A0 smoke test -> A1 pilot -> go/no-go
# Chạy SAU day1_setup.sh, trong cùng tmux.
#
#   bash day1_run.sh 2>&1 | tee /workspace/day1.log
#
# Dừng ngay nếu một bước báo lỗi (set -e). Mỗi build_pool có `time` để ghi giờ.
# =============================================================================
set -euo pipefail

export HF_HOME=/workspace/hf_cache
export HF_HUB_OFFLINE=1          # model đã tải -> chặn tải lại 15GB giữa chừng
export POOLS=/workspace/pools
export REPO=/workspace/caac
cd "$REPO"
export PYTHONPATH=src
Q15=Qwen/Qwen2.5-1.5B-Instruct
Q7=Qwen/Qwen2.5-7B-Instruct

echo "############ A0.1 — seed sanity (3 câu PHẢI khác nhau) ############"
python sanity_check_seed.py
echo ">>> Nếu 3 câu giống hệt: DỪNG, gửi log cho mình. Ctrl-C trong 10s để dừng."
sleep 10

echo "############ A0.2 — smoke 5 câu GSM8K (bắt lỗi logprobs + tên dataset) ############"
rm -f "$POOLS/smoke.jsonl.gz" "$POOLS/smoke.jsonl.gz.manifest.json"
time python build_pool.py --model "$Q15" --dataset gsm8k \
  --n-questions 5 --n-samples 4 --max-tokens 512 --out "$POOLS/smoke.jsonl.gz"
echo "--- 1 bản ghi mẫu: ---"
zcat "$POOLS/smoke.jsonl.gz" | head -c 1200; echo

echo "############ A0.3 — greedy có khớp khi chạy lại không ############"
zcat "$POOLS/smoke.jsonl.gz" | python -c "import sys,json;print('greedy#1:',[json.loads(l)['greedy']['answer'] for l in sys.stdin])"
rm -f "$POOLS/smoke.jsonl.gz" "$POOLS/smoke.jsonl.gz.manifest.json"
python build_pool.py --model "$Q15" --dataset gsm8k \
  --n-questions 5 --n-samples 4 --max-tokens 512 --out "$POOLS/smoke.jsonl.gz" >/dev/null
zcat "$POOLS/smoke.jsonl.gz" | python -c "import sys,json;print('greedy#2:',[json.loads(l)['greedy']['answer'] for l in sys.stdin])"
echo ">>> greedy#1 và greedy#2 phải GIỐNG nhau (greedy tất định)."

echo "############ A1 — PILOT 200 câu (dừng xét go/no-go sau bước này) ############"
# MATH-500: 1.5B 1+32, 7B 1+8 | GSM8K: 1.5B 1+16, 7B 1+8
time python build_pool.py --model "$Q15" --dataset math500 \
  --n-questions 200 --n-samples 32 --max-tokens 1024 --out "$POOLS/math500_q1.5b.jsonl.gz"
time python build_pool.py --model "$Q7"  --dataset math500 \
  --n-questions 200 --n-samples 8  --max-tokens 1024 --out "$POOLS/math500_q7b.jsonl.gz"
time python build_pool.py --model "$Q15" --dataset gsm8k \
  --n-questions 200 --n-samples 16 --max-tokens 512  --out "$POOLS/gsm8k_q1.5b.jsonl.gz"
time python build_pool.py --model "$Q7"  --dataset gsm8k \
  --n-questions 200 --n-samples 8  --max-tokens 512  --out "$POOLS/gsm8k_q7b.jsonl.gz"

echo "############ GO/NO-GO — đọc số, KHÔNG đổi ngưỡng ############"
python check_pilot.py \
  --math15 "$POOLS/math500_q1.5b.jsonl.gz" --math7 "$POOLS/math500_q7b.jsonl.gz" \
  --gsm15  "$POOLS/gsm8k_q1.5b.jsonl.gz"   --gsm7  "$POOLS/gsm8k_q7b.jsonl.gz"

echo
echo ">>> TẢI KHO VỀ NGAY (từ terminal laptop), rồi mới quyết định chạy A2:"
echo "    scp -P <PORT> root@<IP>:$POOLS/*.jsonl.gz ./pools/"
