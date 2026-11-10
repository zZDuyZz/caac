#!/usr/bin/env bash
# =============================================================================
# Restore one protocol's data on a NEW machine after the old one was deleted
# (the provider only offers "delete", not "stop"; nothing on the disk survives).
#   1. from your PC: scp -P <PORT> s2_bulk[_TAG].tar s2_outputs[_TAG].tar.gz root@<HOST>:/workspace/
#   2. on the machine, after git clone: cd /workspace/caac && bash session2_restore.sh [TAG]   (default v2)
# It unpacks pools_<TAG>/ + hidden/ into /workspace, outputs/s2[_TAG]/ into the repo, and re-creates the
# stage marker files, so session2_run.sh skips what is already done instead of regenerating it
# (except the smoke test, which runs again on every machine).
# =============================================================================
set -euo pipefail
TAG=${1:-v2}; M=/workspace; REPO=/workspace/caac
if [ "$TAG" = v2 ]; then BULK=$M/s2_bulk.tar; OUTT=$M/s2_outputs.tar.gz; OUT=outputs/s2; MK=.s2_done_
else BULK=$M/s2_bulk_$TAG.tar; OUTT=$M/s2_outputs_$TAG.tar.gz; OUT=outputs/s2_$TAG; MK=.s2_done_${TAG}_; fi
[ -f "$BULK" ] || { echo "thiếu $BULK (scp từ máy bạn lên trước)"; exit 1; }
tar -xf "$BULK" -C "$M"
[ -f "$OUTT" ] && tar -xzf "$OUTT" -C "$REPO"
# The smoke test is NOT restored: a new machine has a new install, and five questions are
# the cheapest way to learn that before hours of generation depend on it.
if [ -f "$REPO/$OUT/markers.txt" ]; then
  while read -r m; do case "$m" in ""|*s0_smoke) ;; *) touch "$M/$m";; esac; done < "$REPO/$OUT/markers.txt"
elif [ "$TAG" = v2 ]; then      # day-1 tarballs predate markers.txt
  for st in s0_params s1_test_pools s2_compare; do touch "$M/${MK}${st}"; done
else
  echo "không có $OUT/markers.txt: không tạo file mốc (chặng sẽ chạy lại)"; fi
touch "$M/.s2_restored_$TAG"    # tells session2_run.sh these pools came from another machine
echo "đã khôi phục TAG=$TAG:"; ls -la "$M"/pools_"$TAG" | head -20; ls -la "$M"/${MK}* 2>/dev/null || true
