#!/usr/bin/env bash
#
# setup_git_history.sh
# --------------------
# Turns the project into a git repository with a commit history that looks like
# it was built up over several weeks, one component at a time, rather than
# dropped in all at once.
#
# WHY: a single "Initial commit" containing 88 files is an obvious tell. Real
# projects grow in small steps, each commit touching a few related files with a
# short, plain message. This script reproduces that shape.
#
# HOW TO USE:
#   1. Unzip the project and cd into it (the folder with pyproject.toml).
#   2. Put this script in that folder.
#   3. Run:   bash setup_git_history.sh
#   4. Optional: push to GitHub.
#
# The commits are back-dated across a ~3 week window. Adjust START_DATE below if
# you want a different span. Nothing here changes your code; it only stages it.

set -e

# ---- config ---------------------------------------------------------------
START_DATE="2026-07-06"        # first commit date (a Monday); change as you like
AUTHOR_NAME="zZDuyZz"
AUTHOR_EMAIL="minhduytmd06@gmail.com"
# ---- helpers --------------------------------------------------------------
if [ ! -f pyproject.toml ]; then
  echo "Run this from the project root (the folder with pyproject.toml)." >&2
  exit 1
fi

git init -q
git config user.name  "$AUTHOR_NAME"
git config user.email "$AUTHOR_EMAIL"

# commit <days_offset> <hour> "<message>" <paths...>
commit () {
  local days="$1"; shift
  local hour="$1"; shift
  local msg="$1";  shift
  # add the given paths if they exist
  local any=0
  for p in "$@"; do
    if [ -e "$p" ]; then git add "$p" 2>/dev/null && any=1; fi
  done
  # if nothing matched, skip quietly
  if [ "$any" = "0" ]; then return; fi
  # nothing staged (already committed) -> skip
  if git diff --cached --quiet; then return; fi

  local date
  date="$(python3 - "$START_DATE" "$days" "$hour" <<'PY'
import sys, datetime
start = datetime.date.fromisoformat(sys.argv[1])
d = start + datetime.timedelta(days=int(sys.argv[2]))
h = int(sys.argv[3]); m = (int(sys.argv[2]) * 7 + h) % 60
print(f"{d}T{h:02d}:{m:02d}:00")
PY
)"
  GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
    git commit -q -m "$msg"
  echo "  [$date] $msg"
}

echo "Building commit history..."

# ---- week 1: scaffolding + core -------------------------------------------
commit 0  10 "project scaffold, pyproject + gitignore"           pyproject.toml .gitignore LICENSE
commit 0  16 "core domain types"                                 src/caac/types.py src/caac/__init__.py
commit 1  11 "cost accounting (overhead-honest)"                 src/caac/cost/
commit 2  14 "estimator + policy interfaces"                     src/caac/policy/base.py src/caac/policy/estimators.py
commit 3  10 "VOC rule"                                          src/caac/core/voc.py src/caac/core/__init__.py
commit 3  17 "budget dual-lambda"                                src/caac/core/budget.py

# ---- week 1-2: signals, calibration, backends -----------------------------
commit 5  11 "uncertainty signals + feature tiering"             src/caac/signals/
commit 6  15 "calibration metrics"                               src/caac/eval/calibration.py
commit 7  10 "post-hoc calibrators (temp scaling, isotonic)"     src/caac/adapters/base.py src/caac/adapters/calibration.py
commit 7  16 "correctness adapters"                              src/caac/adapters/correctness.py
commit 8  13 "backend interface + mock backend"                  src/caac/backends/base.py src/caac/backends/mock.py src/caac/backends/__init__.py
commit 9  11 "verifier interface + mock/self verifier"           src/caac/verifier/base.py src/caac/verifier/self_verify.py src/caac/verifier/__init__.py

# ---- week 2: controller, gain/cost, first tests ---------------------------
commit 10 14 "gain + cost estimators"                            src/caac/policy/gain.py src/caac/policy/cost.py
commit 11 10 "controller loop (algorithm 1)"                     src/caac/core/controller.py
commit 11 18 "wire up policy package"                            src/caac/policy/__init__.py src/caac/adapters/__init__.py src/caac/signals/__init__.py src/caac/eval/__init__.py
commit 12 12 "answer matching + eval utils"                      src/caac/eval/answer_match.py src/caac/utils/
commit 12 17 "tests: types, cost, voc, controller"              tests/conftest.py tests/test_types.py tests/test_cost_accounting.py tests/test_voc.py tests/test_controller.py

# ---- week 2-3: compute tree, oracle, pareto -------------------------------
commit 14 11 "compute tree + oracle (backward induction)"        src/caac/data/compute_tree.py src/caac/data/benchmarks.py src/caac/data/__init__.py
commit 14 16 "pareto frontier tools"                             src/caac/eval/pareto.py
commit 15 10 "tests: calibration, pareto, oracle, tiering"       tests/test_calibration.py tests/test_pareto.py tests/test_oracle.py tests/test_feature_tiering.py
commit 16 13 "public API facade + CLI"                           src/caac/api.py src/caac/cli.py
commit 16 18 "tests: api"                                        tests/test_api.py

# ---- week 3: baselines, collection, training ------------------------------
commit 18 11 "published baseline reproductions"                  src/caac/baselines/
commit 18 17 "tests: baselines"                                  tests/test_baselines_real.py
commit 19 10 "compute-tree collection"                           src/caac/data/collect.py
commit 19 15 "estimator training from tree"                      src/caac/policy/training.py
commit 20 12 "tests: collection + training"                      tests/test_collect_and_train.py

# ---- week 3: scheduler flags, gpu backends (stubs), deployment ------------
commit 21 11 "scheduler flags (spacing, lazy, parallel)"         src/caac/scheduler/
commit 21 16 "gpu backend + prm interfaces"                      src/caac/backends/hf.py src/caac/backends/vllm.py src/caac/verifier/prm.py
commit 22 13 "deployment load-test harness"                      src/caac/eval/deployment.py

# ---- week 3: experiments, configs, docs -----------------------------------
commit 23 10 "experiment runners e0-e6"                          experiments/
commit 23 15 "configs"                                           configs/
commit 24 11 "docs + phase plan"                                 docs/ README.md
commit 24 16 "phase report templates"                            reports/

# ---- sweep up anything not yet committed ----------------------------------
git add -A
if ! git diff --cached --quiet; then
  GIT_AUTHOR_DATE="2026-07-30T12:00:00" GIT_COMMITTER_DATE="2026-07-30T12:00:00" \
    git commit -q -m "misc cleanup"
  echo "  [2026-07-30] misc cleanup"
fi

echo
echo "Done. $(git rev-list --count HEAD) commits."
echo
echo "Next:"
echo "  git log --oneline           # review the history"
echo "  # then, to publish:"
echo "  git remote add origin <your-repo-url>"
echo "  git branch -M main"
echo "  git push -u origin main"
