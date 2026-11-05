#!/usr/bin/env bash
# =============================================================================
# PHIÊN GPU 2 — PHẦN 2/2 (queue, ~3.5–4h A100-40GB). Chạy SAU session2_setup.sh:
#   bash session2_run.sh 2>&1 | tee /workspace/s2.log
# Resume được: chạy lại lệnh y hệt -> build_pool bỏ qua câu đã xong.
# Theo dõi:  grep -E "####|done,.*to go|wrote|real|compare|VERDICT" /workspace/s2.log
# =============================================================================
set -euo pipefail
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 PYTHONPATH=src
REPO=/workspace/caac; cd "$REPO"
Q15=Qwen/Qwen2.5-1.5B-Instruct; Q7=Qwen/Qwen2.5-7B-Instruct
P=/workspace/pools_v2; H=/workspace/hidden; OUT=outputs/s2
mkdir -p "$P" "$H" "$OUT"
BP="python build_pool.py"   # cấu hình MẶC ĐỊNH (batch 150, max_num_seqs 256) = y hệt A2

echo "#### S2.0 smoke: 5 câu GSM8K + 5 câu MATH-500, 1.5B, n=4 ####"
rm -f "$P"/smoke_*
$BP --model $Q15 --dataset gsm8k   --n-questions 5 --n-samples 4 --max-tokens 512  --out "$P/smoke_gsm8k.jsonl.gz"
$BP --model $Q15 --dataset math500 --n-questions 5 --n-samples 4 --max-tokens 1024 --out "$P/smoke_math.jsonl.gz"
python - <<'PY'
import gzip, json
P = "/workspace/pools_v2"
for n in ("smoke_gsm8k", "smoke_math"):
    pool = [json.loads(l) for l in gzip.open(f"{P}/{n}.jsonl.gz", "rt")]
    text = [json.loads(l) for l in gzip.open(f"{P}/{n}.text.jsonl.gz", "rt")]
    tok  = [json.loads(l) for l in gzip.open(f"{P}/{n}.tok.jsonl.gz", "rt")]
    assert [r["query_id"] for r in pool] == [t["query_id"] for t in text] == [t["query_id"] for t in tok]
    for r, t in zip(pool, tok):
        g = r["greedy"]
        assert g["conf_prefix32"] is not None and len(t["greedy"]["lp"]) == g["decode_tokens"], "token sidecar lệch"
        print(f"  {r['query_id']:12} gold={r['gold']!r:28} raw={g['answer_raw']!r:28} "
              f"MV={g['correct']} legacy={g['correct_legacy']} conf32={g['conf_prefix32']:.3f} "
              f"samples={sum(s['correct'] for s in r['samples'])}/{len(r['samples'])}")
print("  SMOKE OK: pool + text + tok sidecar khớp nhau")
PY
echo ">>> Soát mắt bảng trên (raw phải giữ dấu phẩy/cơ số; MV hợp lý). Ctrl-C trong 30s nếu sai."
sleep 30
rm -f "$P"/smoke_*

echo "#### S2.1 sinh lại 4 pool TEST kèm text + token (tham số y hệt A2) ####"
time $BP --model $Q15 --dataset math500 --n-samples 32 --max-tokens 1024 --out "$P/math500_q1.5b.jsonl.gz"
time $BP --model $Q7  --dataset math500 --n-samples 8  --max-tokens 1024 --out "$P/math500_q7b.jsonl.gz"
time $BP --model $Q15 --dataset gsm8k   --n-samples 16 --max-tokens 512  --out "$P/gsm8k_q1.5b.jsonl.gz"
time $BP --model $Q7  --dataset gsm8k   --n-samples 8  --max-tokens 512  --out "$P/gsm8k_q7b.jsonl.gz"

echo "#### S2.2 tái lập: pool mới vs pool A2 (greedy phải ~khớp) ####"
for n in math500_q1.5b math500_q7b gsm8k_q1.5b gsm8k_q7b; do
  python compare_pools.py "outputs/a2/$n.jsonl.gz" "$P/$n.jsonl.gz" || true
done

echo "#### S2.3 GO/NO-GO TIỀN ĐĂNG KÝ (CPU, vài phút) ####"
python gonogo.py --small "$P/math500_q1.5b.jsonl.gz" --big "$P/math500_q7b.jsonl.gz" --json "$OUT/gonogo_math500.json" || true
python gonogo.py --small "$P/gsm8k_q1.5b.jsonl.gz"   --big "$P/gsm8k_q7b.jsonl.gz"   --json "$OUT/gonogo_gsm8k.json"   || true
echo ">>> A3 vẫn chạy dù GO hay NO-GO (§8.4: NO-GO vẫn dùng split train/calib)."

echo "#### S2.4 A3 train/calib (1.5B 1+8, 7B 1+4) ####"
time $BP --model $Q15 --dataset gsm8k --split train --qids-file qids/a3_gsm8k_train.txt --n-samples 8 --max-tokens 512 --out "$P/a3_gsm8k_train_q1.5b.jsonl.gz"
time $BP --model $Q7  --dataset gsm8k --split train --qids-file qids/a3_gsm8k_train.txt --n-samples 4 --max-tokens 512 --out "$P/a3_gsm8k_train_q7b.jsonl.gz"
for D in math_train math_test_ex500; do
  S=train; [ $D = math_test_ex500 ] && S=test
  time $BP --model $Q15 --dataset $D --split $S --qids-file qids/a3_$D.txt --n-samples 8 --max-tokens 1024 --out "$P/a3_${D}_q1.5b.jsonl.gz"
  time $BP --model $Q7  --dataset $D --split $S --qids-file qids/a3_$D.txt --n-samples 4 --max-tokens 1024 --out "$P/a3_${D}_q7b.jsonl.gz"
done

echo "#### S2.5 A4 hidden state (1.5B, chỉ prompt) ####"
python extract_hidden.py --model $Q15 --dataset math500 --out "$H/math500_q1.5b.npz"
python extract_hidden.py --model $Q15 --dataset gsm8k   --out "$H/gsm8k_q1.5b.npz"
python extract_hidden.py --model $Q15 --dataset gsm8k --split train --qids-file qids/a3_gsm8k_train.txt --out "$H/a3_gsm8k_train_q1.5b.npz"
python extract_hidden.py --model $Q15 --dataset math_train --split train --qids-file qids/a3_math_train.txt --out "$H/a3_math_train_q1.5b.npz"
python extract_hidden.py --model $Q15 --dataset math_test_ex500 --split test --qids-file qids/a3_math_test_ex500.txt --out "$H/a3_math_test_ex500_q1.5b.npz"

echo "#### S2.6 thống kê grader (pool mới đã chấm MV khi sinh; đây là bảng bất đồng) ####"
for f in "$P"/*.jsonl.gz; do
  case "$f" in *.text.jsonl.gz|*.tok.jsonl.gz|*.regraded.jsonl.gz) continue;; esac
  python regrade_pool.py "$f" --show 25 --out /tmp/rg.jsonl.gz || true
done
rm -f /tmp/rg.jsonl.gz

echo "#### S2.7 gom kết quả ####"
cp "$P"/*.jsonl.gz.manifest.json "$OUT"/
for f in "$P"/*.jsonl.gz; do case "$f" in *.text.*|*.tok.*) ;; *) cp "$f" "$OUT"/;; esac; done
cp qids/*.txt qids/*.summary.json "$OUT"/ 2>/dev/null || true
cp /workspace/env_s2.txt "$OUT"/ 2>/dev/null || true
cp "$H"/*.manifest.json "$OUT"/ 2>/dev/null || true
tar -cf /workspace/s2_bulk.tar -C /workspace pools_v2 hidden   # sidecar + hidden: KHÔNG vào git
du -sh "$OUT" /workspace/s2_bulk.tar
echo "#### S2 DONE. Việc tay: cp /workspace/s2.log $OUT/ ; git add -f $OUT qids ; commit ; push ; tải s2_bulk.tar về ; DỪNG MÁY ####"
