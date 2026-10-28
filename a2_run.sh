#!/usr/bin/env bash
# A2: hoàn tất pool test đầy đủ. Tham số y hệt pilot, chỉ bỏ --n-questions.
eval "$(grep -E '^(export )?(Q15|Q7|POOLS|HF_HOME|HF_HUB_OFFLINE|CUDA_VISIBLE_DEVICES)=' day1_run.sh)"
export HF_HOME HF_HUB_OFFLINE CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src
echo "Q15=$Q15 Q7=$Q7 POOLS=$POOLS"

time python build_pool.py --model "$Q15" --dataset math500 \
  --n-samples 32 --max-tokens 1024 --out "$POOLS/math500_q1.5b.jsonl.gz"
time python build_pool.py --model "$Q7"  --dataset math500 \
  --n-samples 8  --max-tokens 1024 --out "$POOLS/math500_q7b.jsonl.gz"
time python build_pool.py --model "$Q15" --dataset gsm8k \
  --n-samples 16 --max-tokens 512  --out "$POOLS/gsm8k_q1.5b.jsonl.gz"
time python build_pool.py --model "$Q7"  --dataset gsm8k \
  --n-samples 8  --max-tokens 512  --out "$POOLS/gsm8k_q7b.jsonl.gz"
echo "A2 DONE"
