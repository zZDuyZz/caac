#!/usr/bin/env bash
# =============================================================================
# PHIÊN GPU 2 — chạy tất cả (gọi từ session2_all.sh; có thể chạy riêng sau setup).
# - Smoke tự động (smoke_check.py) thay cho soát mắt: sai thì DỪNG trước khi tốn giờ.
# - Mỗi chặng có file mốc /workspace/.s2_done_<tên>: chạy lại script sẽ BỎ QUA chặng đã xong
#   (build_pool vốn đã resume theo câu). Xóa mốc để ép chạy lại: rm /workspace/.s2_done_*
# - Cuối cùng: s2_summary.md (GỬI FILE NÀY cho mình), gom outputs/s2, tạo tar.
# =============================================================================
set -uo pipefail
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:.
# Day 1 ran with the FlashInfer sampler OFF (the 'runtime' image has no nvcc to JIT it) -> keep it
# OFF for every pool so all pools share one configuration. Recorded in each manifest (env_flags).
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
# shellcheck disable=SC1091
set +u; source /workspace/venv/bin/activate; set -u     # dựng bởi session2_setup.sh (Python 3.12 + vLLM 0.32.0, V1); +u: script activate có thể đọc biến chưa đặt
cd /workspace/caac
# 2026-11-06: cặp chính mới (CAAC_MASTER §13.10). Tên file: qs = model nhỏ, qb = model lớn.
QS=Qwen/Qwen3.5-2B; QB=Qwen/Qwen3.5-9B
# Protocol tag + token caps. TAG=v2 = day-1 protocol (cap 1024 MATH / 512 GSM8K, max_model_len 2048),
# kept for the appendix ("tight token budget"). A new TAG gets its own pools_<TAG>/, outputs/s2_<TAG>/
# and marker files, so it never touches day-1 data. Caps for the new protocol come from the cap probe
# (STOP_AFTER=capprobe -> cap_curve.py prints "RECOMMEND: MT_M=.. MT_G=.. MML=.."):
#   TAG=v3 MT_M=<..> MT_G=<..> MML=<..> bash session2_all.sh
TAG=${TAG:-v2}; MT_M=${MT_M:-1024}; MT_G=${MT_G:-512}
case "${STOP_AFTER:-}" in ""|capprobe|test_pools) ;; *) echo "STOP_AFTER không hợp lệ: '$STOP_AFTER' (dùng capprobe hoặc test_pools)"; exit 1;; esac
# Every set the run generates, with its cap. preflight_prompts.py checks each prompt against
# max_model_len; for a new TAG it also chooses max_model_len, so the value is never guessed.
SETS="--set math500:test::$MT_M --set gsm8k:test::$MT_G \
 --set gsm8k:train:qids/a3_gsm8k_train.txt:$MT_G --set math_train:train:qids/a3_math_train.txt:$MT_M \
 --set math_test_ex500:test:qids/a3_math_test_ex500.txt:$MT_M"
if [ "$TAG" = v2 ]; then MML=${MML:-2048}; fi
P=/workspace/pools_$TAG; H=/workspace/hidden; M=/workspace
if [ "$TAG" = v2 ]; then OUT=outputs/s2; MK=".s2_done_"; BULK=$M/s2_bulk.tar; OUTTAR=$M/s2_outputs.tar.gz
else OUT=outputs/s2_$TAG; MK=".s2_done_${TAG}_"; BULK=$M/s2_bulk_$TAG.tar; OUTTAR=$M/s2_outputs_$TAG.tar.gz; fi
mkdir -p "$P" "$H" "$OUT"
if [ "${STOP_AFTER:-}" != capprobe ] && [ -z "${MML:-}" ]; then
  MML=$(python preflight_prompts.py --model $QS --model $QB $SETS) || { echo "preflight lỗi"; exit 1; }
fi
MML=${MML:-2048}
if [ "${STOP_AFTER:-}" = capprobe ]; then echo "PROTOCOL: TAG=$TAG cap probe (train sets, cap 4096, max_model_len 6144)  FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
else echo "PROTOCOL: TAG=$TAG  max_tokens MATH=$MT_M GSM8K=$MT_G  max_model_len=$MML  FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"; fi
T0=$(date +%s)
BP="python build_pool.py --max-model-len $MML"   # cấu hình MẶC ĐỊNH (batch 150, max_num_seqs mặc định của vLLM), thinking TẮT

tick() { local e=$(( $(date +%s) - T0 )); printf '[%02dh%02dm | ~%d VND] ' $((e/3600)) $(((e%3600)/60)) $((e*${RATE:-52516}/3600)); }
stage() {  # stage <tên> <hàm>
  if [ -f "$M/$MK$1" ]; then echo "$(tick)#### $1: đã xong trước đó — bỏ qua"; return 0; fi
  echo "$(tick)#### $1: BẮT ĐẦU"
  if "$2"; then touch "$M/$MK$1"; echo "$(tick)#### $1: XONG"; else echo "$(tick)#### $1: LỖI"; return 1; fi
}
die() { echo; echo "!!!!!! DỪNG: $* !!!!!!  (máy VẪN ĐANG TÍNH TIỀN — gửi mình log rồi tắt máy nếu chưa sửa được)"; exit 1; }

# ---------------------------------------------------------------- chặng
s0_capprobe() {
  # The cap is a protocol hyperparameter, so it is chosen on TRAIN questions (no test item is
  # touched) drawn stratified with seed 1 -- the loaders' first N follow file order, which is
  # not random. Greedy + 2 samples at a cap far above need; cap_curve.py reads lengths only.
  PR=$M/probe_$TAG; mkdir -p "$PR"
  [ -f qids/probe_math_train.txt ]  || python make_qids.py --dataset math_train --n 200 --seed 1 --out qids/probe_math_train.txt || return 1
  [ -f qids/probe_gsm8k_train.txt ] || python make_qids.py --dataset gsm8k --split train --n 200 --seed 1 --out qids/probe_gsm8k_train.txt || return 1
  PSETS="--set math_train:train:qids/probe_math_train.txt:4096 --set gsm8k:train:qids/probe_gsm8k_train.txt:4096"
  python preflight_prompts.py --model $QS --model $QB $PSETS --max-model-len 6144 || return 1
  PB="python build_pool.py --max-model-len 6144 --max-tokens 4096 --n-samples 2"
  for MD in "$QS qs" "$QB qb"; do
    set -- $MD
    $PB --model $1 --dataset math_train --split train --qids-file qids/probe_math_train.txt --out "$PR/math_train_$2.jsonl.gz" || return 1
    $PB --model $1 --dataset gsm8k --split train --qids-file qids/probe_gsm8k_train.txt --out "$PR/gsm8k_train_$2.jsonl.gz" || return 1
  done
  python cap_curve.py "$PR"/math_train_qs.jsonl.gz "$PR"/math_train_qb.jsonl.gz "$PR"/gsm8k_train_qs.jsonl.gz "$PR"/gsm8k_train_qb.jsonl.gz \
    | tee "$OUT/cap_curve.txt"
  cp qids/probe_* "$OUT"/
  tar -czf "$M/probe_$TAG.tar.gz" -C "$M" "probe_$TAG" "caac/$OUT" 2>/dev/null || true
  return 0
}
s0_smoke() {
  rm -f "$P"/smoke_*
  $BP --model $QS --dataset gsm8k   --n-questions 5 --n-samples 4 --max-tokens 512  --out "$P/smoke_gsm8k.jsonl.gz" || return 1
  $BP --model $QS --dataset math500 --n-questions 5 --n-samples 4 --max-tokens 1024 --out "$P/smoke_math.jsonl.gz" || return 1
  # model lớn cũng phải nạp + sinh được TRƯỚC khi tốn ~1 giờ cho pool model nhỏ
  $BP --model $QB --dataset gsm8k   --n-questions 3 --n-samples 2 --max-tokens 512  --out "$P/smoke_gsm8k_qb.jsonl.gz" || return 1
  python smoke_check.py "$P/smoke_gsm8k.jsonl.gz"    --min-correct 3 || return 1
  python smoke_check.py "$P/smoke_math.jsonl.gz"     --min-correct 1 || return 1
  python smoke_check.py "$P/smoke_gsm8k_qb.jsonl.gz" --min-correct 2 || return 1
  rm -f "$P"/smoke_*
}
s0_params() {
  # Exact parameter counts from the safetensors headers (vision/audio towers excluded). The FLOPs
  # axis uses the registry, so the registry is updated from this file before any analysis.
  python verify_params.py --weights $QS --weights $QB | tee "$OUT/params_verified.txt"
  return 0
}
# Big model: 16 samples on the TEST pools. It is the escalate target, so its pass@k ceiling is
# the ceiling of the whole controller.
s1_test_pools() {
  $BP --model $QS --dataset math500 --n-samples 32 --max-tokens $MT_M --out "$P/math500_qs.jsonl.gz" || return 1
  $BP --model $QB --dataset math500 --n-samples 16 --max-tokens $MT_M --out "$P/math500_qb.jsonl.gz" || return 1
  $BP --model $QS --dataset gsm8k   --n-samples 16 --max-tokens $MT_G  --out "$P/gsm8k_qs.jsonl.gz"   || return 1
  $BP --model $QB --dataset gsm8k   --n-samples 16 --max-tokens $MT_G  --out "$P/gsm8k_qb.jsonl.gz"   || return 1
}
s2_compare() {
  # Pool A2 là Qwen2.5 -> không còn đối chiếu tái lập với A2. Thay bằng KIỂM TÍNH XÁC ĐỊNH:
  # sinh lại 30 câu GSM8K đầu (cùng model, cùng seed, batch khác) rồi so với pool chính.
  # V1 không bất biến theo batch, nên có thể lệch nhẹ: CHỈ BÁO CÁO, không chặn.
  rm -f "$P"/det_gsm8k_qs*
  # (a) SAME batch composition as the main pool's first batch (150 questions, 16 samples,
  #     batch-size 150). Nothing about the generate() call differs, so a mismatch here means
  #     OUR pipeline is non-deterministic, not vLLM. Expect ~100%.
  $BP --model $QS --dataset gsm8k --n-questions 150 --n-samples 16 --batch-size 150 \
      --max-tokens $MT_G --out "$P/det_same_gsm8k_qs.jsonl.gz" || return 0
  python compare_pools.py "$P/gsm8k_qs.jsonl.gz" "$P/det_same_gsm8k_qs.jsonl.gz" \
      | tee "$OUT/determinism_same_batch.txt"
  # (b) DIFFERENT batch composition (30 questions, 4 samples). vLLM V1 is not batch-invariant,
  #     so this is expected to disagree; reporting both arms separates the two causes in the
  #     Limitations section instead of leaving one alarming number.
  $BP --model $QS --dataset gsm8k --n-questions 30 --n-samples 4 --max-tokens $MT_G --out "$P/det_gsm8k_qs.jsonl.gz" || return 0
  python compare_pools.py "$P/gsm8k_qs.jsonl.gz" "$P/det_gsm8k_qs.jsonl.gz" | tee "$OUT/determinism_gsm8k_qs.txt"
  return 0
}
# A3 = train + calibration. Calibration must run the SAME policies as test, so it needs as many
# draws as any policy may take: CAAC samples up to 16 from the small model, and PriceCheck's
# races draw up to 6 re-solves from the big one (8 leaves headroom).
s4_a3() {
  $BP --model $QS --dataset gsm8k --split train --qids-file qids/a3_gsm8k_train.txt --n-samples 16 --max-tokens $MT_G --out "$P/a3_gsm8k_train_qs.jsonl.gz" || return 1
  $BP --model $QB --dataset gsm8k --split train --qids-file qids/a3_gsm8k_train.txt --n-samples 8 --max-tokens $MT_G --out "$P/a3_gsm8k_train_qb.jsonl.gz" || return 1
  for D in math_train math_test_ex500; do
    S=train; [ $D = math_test_ex500 ] && S=test
    $BP --model $QS --dataset $D --split $S --qids-file qids/a3_$D.txt --n-samples 16 --max-tokens $MT_M --out "$P/a3_${D}_qs.jsonl.gz" || return 1
    $BP --model $QB --dataset $D --split $S --qids-file qids/a3_$D.txt --n-samples 8 --max-tokens $MT_M --out "$P/a3_${D}_qb.jsonl.gz" || return 1
  done
}
s5_hidden() {
  python extract_hidden.py --model $QS --dataset math500 --out "$H/math500_qs.npz" || return 1
  python extract_hidden.py --model $QS --dataset gsm8k   --out "$H/gsm8k_qs.npz" || return 1
  python extract_hidden.py --model $QS --dataset gsm8k --split train --qids-file qids/a3_gsm8k_train.txt --out "$H/a3_gsm8k_train_qs.npz" || return 1
  python extract_hidden.py --model $QS --dataset math_train --split train --qids-file qids/a3_math_train.txt --out "$H/a3_math_train_qs.npz" || return 1
  python extract_hidden.py --model $QS --dataset math_test_ex500 --split test --qids-file qids/a3_math_test_ex500.txt --out "$H/a3_math_test_ex500_qs.npz" || return 1
}
s6_collect() {
  cp "$P"/*.jsonl.gz.manifest.json "$OUT"/ 2>/dev/null
  for f in "$P"/*.jsonl.gz; do case "$f" in *.text.*|*.tok.*) ;; *) cp "$f" "$OUT"/;; esac; done
  cp qids/*.txt qids/*.summary.json "$OUT"/ 2>/dev/null
  cp "$M/env_s2.txt" "$OUT"/ 2>/dev/null
  ls -1 "$M"/"$MK"* 2>/dev/null | xargs -r -n1 basename > "$OUT/markers.txt"   # for session2_restore.sh
  { echo "tag=$TAG max_tokens_math=$MT_M max_tokens_gsm8k=$MT_G max_model_len=$MML"
    echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; hostname; } > "$OUT/protocol.txt"
  cp "$H"/*.manifest.json "$OUT"/ 2>/dev/null
  for f in "$P"/*.jsonl.gz; do   # bảng bất đồng grader (ghi vào log, không giữ file)
    case "$f" in *.text.*|*.tok.*|*.regraded.*|*/det_*) continue;; esac
    python regrade_pool.py "$f" --show 15 --out /tmp/rg.jsonl.gz
  done > "$OUT/grader_disagreement.log" 2>&1
  rm -f /tmp/rg.jsonl.gz
  python s2_summary.py --pools "$P" --out "$OUT/s2_summary.md" > /dev/null
  tar -cf "$BULK" -C "$M" "pools_$TAG" hidden                # sidecar + hidden: KHÔNG vào git
}

# ---------------------------------------------------------------- chạy
if [ "${STOP_AFTER:-}" = "capprobe" ]; then   # chỉ đo độ cắt rồi dừng (không chạy pool nào)
  stage capprobe s0_capprobe || die "cap probe lỗi"
  echo; echo "$(tick)>>> CAP PROBE XONG. Gửi mình toàn bộ phần 'cap_curve' ở trên (hoặc $OUT/cap_curve.txt)."
  exit 0
fi
python preflight_prompts.py --model $QS --model $QB $SETS --max-model-len $MML || die "prompt không vừa max_model_len=$MML"
stage s0_smoke s0_smoke      || die "smoke test lỗi (xem 'SMOKE FAIL' ở trên). Chưa tốn giờ GPU đáng kể."
stage s0_params s0_params
stage s1_test_pools s1_test_pools || die "sinh pool test lỗi"
stage s2_compare s2_compare
# STOP_AFTER=test_pools -> stop after the test pools + determinism check and collect for download (2a);
# the next machine restores them with session2_restore.sh and re-runs this command without STOP_AFTER (2b).
if [ "${STOP_AFTER:-}" = "test_pools" ]; then
  stage s6_part1 s6_collect  || die "gom kết quả lỗi"
  echo "$(tick)>>> PHẦN 2a XONG (đã dừng sau pool test theo STOP_AFTER). Chưa chạy A3/A4."
else
  stage s4_a3 s4_a3            || die "A3 lỗi"
  stage s5_hidden s5_hidden    || die "A4 hidden state lỗi"
  stage s6_collect s6_collect  || die "gom kết quả lỗi"
fi
cp "$M"/s2*.log "$OUT"/ 2>/dev/null || true

# Outputs always leave the machine as a tarball. No push from here: GH_TOKEN never goes on a rental.
tar -czf "$OUTTAR" -C . "$OUT" qids

echo; cat "$OUT/s2_summary.md"
echo
echo "============================================================================"
echo " $(tick)PHIÊN 2 HOÀN TẤT.  Tải về máy bạn (từ PowerShell trên máy bạn):"
echo "   scp -P <PORT> root@<IP>:$BULK .          # sidecar + hidden (bắt buộc)"
echo "   scp -P <PORT> root@<IP>:$OUTTAR .    # pool + summary"
[ -f "$M/probe_$TAG.tar.gz" ] && echo "   scp -P <PORT> root@<IP>:$M/probe_$TAG.tar.gz .   # cap probe"
echo " Gửi mình: $OUT/s2_summary.md  (trong tar) hoặc dán đoạn trên."
if [ "${STOP_AFTER:-}" = "test_pools" ]; then
  echo " >>> TẢI 2 FILE TRÊN VỀ, rồi xóa máy. Phần sau (A3 + hidden) chạy ở máy mới: scp 2 file lên /workspace, bash session2_restore.sh $TAG, chạy lại ĐÚNG lệnh này nhưng bỏ STOP_AFTER. <<<"
else
  echo " >>> SAU KHI TẢI XONG: DỪNG VÀ XÓA MÁY. <<<"
fi
echo "============================================================================"
