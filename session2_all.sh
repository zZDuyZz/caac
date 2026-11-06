#!/usr/bin/env bash
# =============================================================================
# MỘT LỆNH CHẠY TOÀN BỘ PHIÊN 2 (setup + smoke + pool test + go/no-go + A3 + A4 + gom kết quả)
#
#   cd /workspace && git clone https://github.com/zZDuyZz/caac.git && cd caac
#   tmux new -s caac 'bash session2_all.sh 2>&1 | tee /workspace/s2.log'
#
# (tùy chọn) tự push kết quả lên GitHub ở cuối:  export GH_TOKEN=<PAT>  trước lệnh tmux.
# Thoát tmux giữ job chạy: Ctrl+b rồi d.  Vào lại: tmux attach -t caac.
# Chạy lại lệnh y hệt sau khi bị ngắt -> tự bỏ qua chặng đã xong, resume pool dở.
# =============================================================================
set -uo pipefail
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU"
# (không còn kiểm tra loại GPU — người dùng tự chọn máy; mọi pool của một lần chạy phải cùng loại GPU)
cd /workspace/caac
if [ ! -f /workspace/.s2_setup_done ]; then
  # pipefail: trạng thái = của session2_setup.sh (set -e bên trong), không phải của tee
  if bash session2_setup.sh 2>&1 | tee /workspace/s2_setup.log; then
    touch /workspace/.s2_setup_done
  else
    echo "!!!!!! SETUP LỖI — xem /workspace/s2_setup.log. DỪNG (chưa tốn giờ sinh dữ liệu). !!!!!!"; exit 1
  fi
fi
exec bash session2_run.sh
