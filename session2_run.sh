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
# shellcheck disable=SC1091
set +u; source /workspace/venv/bin/activate; set -u     # dựng bởi session2_setup.sh (Python 3.12 + vLLM 0.32.0, V1); +u: script activate có thể đọc biến chưa đặt
cd /workspace/caac
# 2026-11-06: cặp chính mới (CAAC_MASTER §13.10). Tên file: qs = model nhỏ, qb = model lớn.
QS=Qwen/Qwen3.5-2B; QB=Qwen/Qwen3.5-9B
P=/workspace/pools_v2; H=/workspace/hidden; OUT=outputs/s2; M=/workspace
mkdir -p "$P" "$H" "$OUT"
T0=$(date +%s)
BP="python build_pool.py"   # cấu hình MẶC ĐỊNH (batch 150, max_num_seqs mặc định của vLLM), thinking TẮT

tick() { local e=$(( $(date +%s) - T0 )); printf '[%02dh%02dm | ~%d VND] ' $((e/3600)) $(((e%3600)/60)) $((e*${RATE:-52516}/3600)); }
stage() {  # stage <tên> <hàm>
  if [ -f "$M/.s2_done_$1" ]; then echo "$(tick)#### $1: đã xong trước đó — bỏ qua"; return 0; fi
  echo "$(tick)#### $1: BẮT ĐẦU"
  if "$2"; then touch "$M/.s2_done_$1"; echo "$(tick)#### $1: XONG"; else echo "$(tick)#### $1: LỖI"; return 1; fi
}
die() { echo; echo "!!!!!! DỪNG: $* !!!!!!  (máy VẪN ĐANG TÍNH TIỀN — gửi mình log rồi tắt máy nếu chưa sửa được)"; exit 1; }

# ---------------------------------------------------------------- chặng
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
  # Đếm tham số CHÍNH XÁC từ header safetensors (tháp ảnh/âm thanh bị loại). Registry hiện là số
  # từ model card (params_verified=False) -> go/no-go FLOPs ở s3 chỉ là TẠM cho tới khi mình cập
  # nhật registry bằng file này rồi chạy lại gonogo trên CPU (không cần GPU).
  python verify_params.py --weights $QS --weights $QB | tee "$OUT/params_verified.txt"
  return 0
}
s1_test_pools() {
  $BP --model $QS --dataset math500 --n-samples 32 --max-tokens 1024 --out "$P/math500_qs.jsonl.gz" || return 1
  $BP --model $QB --dataset math500 --n-samples 8  --max-tokens 1024 --out "$P/math500_qb.jsonl.gz" || return 1
  $BP --model $QS --dataset gsm8k   --n-samples 16 --max-tokens 512  --out "$P/gsm8k_qs.jsonl.gz"   || return 1
  $BP --model $QB --dataset gsm8k   --n-samples 8  --max-tokens 512  --out "$P/gsm8k_qb.jsonl.gz"   || return 1
}
s2_compare() {
  # Pool A2 là Qwen2.5 -> không còn đối chiếu tái lập với A2. Thay bằng KIỂM TÍNH XÁC ĐỊNH:
  # sinh lại 30 câu GSM8K đầu (cùng model, cùng seed, batch khác) rồi so với pool chính.
  # V1 không bất biến theo batch, nên có thể lệch nhẹ: CHỈ BÁO CÁO, không chặn.
  rm -f "$P"/det_gsm8k_qs*
  $BP --model $QS --dataset gsm8k --n-questions 30 --n-samples 4 --max-tokens 512 --out "$P/det_gsm8k_qs.jsonl.gz" || return 0
  python compare_pools.py "$P/gsm8k_qs.jsonl.gz" "$P/det_gsm8k_qs.jsonl.gz" | tee "$OUT/determinism_gsm8k_qs.txt"
  return 0
}
s3_gonogo() {
  python gonogo.py --small "$P/math500_qs.jsonl.gz" --big "$P/math500_qb.jsonl.gz" --json "$OUT/gonogo_math500.json"
  python gonogo.py --small "$P/gsm8k_qs.jsonl.gz"   --big "$P/gsm8k_qb.jsonl.gz"   --json "$OUT/gonogo_gsm8k.json"
  return 0   # §8.4: A3 vẫn chạy dù GO hay NO-GO
}
s4_a3() {
  $BP --model $QS --dataset gsm8k --split train --qids-file qids/a3_gsm8k_train.txt --n-samples 8 --max-tokens 512 --out "$P/a3_gsm8k_train_qs.jsonl.gz" || return 1
  $BP --model $QB --dataset gsm8k --split train --qids-file qids/a3_gsm8k_train.txt --n-samples 4 --max-tokens 512 --out "$P/a3_gsm8k_train_qb.jsonl.gz" || return 1
  for D in math_train math_test_ex500; do
    S=train; [ $D = math_test_ex500 ] && S=test
    $BP --model $QS --dataset $D --split $S --qids-file qids/a3_$D.txt --n-samples 8 --max-tokens 1024 --out "$P/a3_${D}_qs.jsonl.gz" || return 1
    $BP --model $QB --dataset $D --split $S --qids-file qids/a3_$D.txt --n-samples 4 --max-tokens 1024 --out "$P/a3_${D}_qb.jsonl.gz" || return 1
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
  cp "$H"/*.manifest.json "$OUT"/ 2>/dev/null
  for f in "$P"/*.jsonl.gz; do   # bảng bất đồng grader (ghi vào log, không giữ file)
    case "$f" in *.text.*|*.tok.*|*.regraded.*|*/det_*) continue;; esac
    python regrade_pool.py "$f" --show 15 --out /tmp/rg.jsonl.gz
  done > "$OUT/grader_disagreement.log" 2>&1
  rm -f /tmp/rg.jsonl.gz
  python s2_summary.py --pools "$P" --out "$OUT/s2_summary.md" > /dev/null
  tar -cf "$M/s2_bulk.tar" -C "$M" pools_v2 hidden          # sidecar + hidden: KHÔNG vào git
}

# ---------------------------------------------------------------- chạy
stage s0_smoke s0_smoke      || die "smoke test lỗi (xem 'SMOKE FAIL' ở trên). Chưa tốn giờ GPU đáng kể."
stage s0_params s0_params
stage s1_test_pools s1_test_pools || die "sinh pool test lỗi"
stage s2_compare s2_compare
stage s3_gonogo s3_gonogo
# STOP_AFTER=s3_gonogo -> NGÀY 1: dừng sau pool test + so sánh + go/no-go, rồi gom kết quả để tải về
# (không đặt biến -> chạy tiếp A3/A4/gom kết quả = ngày 2; các chặng đã xong tự bỏ qua nhờ file mốc).
if [ "${STOP_AFTER:-}" = "s3_gonogo" ]; then
  stage s6_day1 s6_collect   || die "gom kết quả lỗi"
  echo "$(tick)>>> NGÀY 1 XONG (đã dừng sau s3_gonogo theo STOP_AFTER). Chưa chạy A3/A4."
else
  stage s4_a3 s4_a3            || die "A3 lỗi"
  stage s5_hidden s5_hidden    || die "A4 hidden state lỗi"
  stage s6_collect s6_collect  || die "gom kết quả lỗi"
fi
cp "$M/s2.log" "$OUT/s2.log" 2>/dev/null || true

# ---------------------------------------------------------------- đẩy lên GitHub (tùy chọn)
if [ -n "${GH_TOKEN:-}" ]; then
  git add -f "$OUT" qids
  git -c user.name=zZDuyZz -c user.email=minhduytmd06@gmail.com commit -q -m "Session 2: regenerated test pools (text+token sidecars), A3 pools, go/no-go, summary" \
    && git push -q "https://x-access-token:${GH_TOKEN}@github.com/zZDuyZz/caac.git" HEAD:main \
    && echo "ĐÃ PUSH outputs/s2 lên GitHub" || echo "PUSH LỖI — dùng tar bên dưới"
else
  tar -czf "$M/s2_outputs.tar.gz" -C . "$OUT" qids
fi

echo; cat "$OUT/s2_summary.md"
echo
echo "============================================================================"
echo " $(tick)PHIÊN 2 HOÀN TẤT.  Tải về máy bạn (từ PowerShell trên máy bạn):"
echo "   scp -P <PORT> root@<IP>:/workspace/s2_bulk.tar .          # sidecar + hidden (bắt buộc)"
[ -z "${GH_TOKEN:-}" ] && echo "   scp -P <PORT> root@<IP>:/workspace/s2_outputs.tar.gz .    # pool + summary (thay cho git push)"
echo " Gửi mình: outputs/s2/s2_summary.md  (trong tar) hoặc dán đoạn trên."
if [ "${STOP_AFTER:-}" = "s3_gonogo" ]; then
  echo " >>> SAU KHI TẢI XONG: DỪNG MÁY (ĐỪNG XÓA nếu đĩa được giữ — ngày 2 cần pool + file mốc). <<<"
else
  echo " >>> SAU KHI TẢI XONG: DỪNG VÀ XÓA MÁY. <<<"
fi
echo "============================================================================"
