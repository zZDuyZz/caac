# Relative-claim test rounds v3-9, v3-10, v3-11 (lock amendments v3-9 to v3-11)

Each round has a design file hashed BEFORE anything was run (`DECISION_*_before_run.md` + `*_hash.txt`) and the
hashes of its code before the runs (`code_hashes_*.txt`). The scripts are kept byte-identical to the files that
were run, so those hashes still verify (`sha256sum -c code_hashes_before_dev_run.txt` in `v310/` and `v311/`,
`code_hashes_before_full_run.txt` in `v39/`).

| Round | Candidates | Result | Adopted |
|---|---|---|---|
| v39  | exact conditional quality test (R1), per-policy cost range (P), bet cap 0.9 (C), mixture bets (M) | R1+P+C chosen by its rule; lower synthetic power in 28/40 cells | no |
| v310 | quality by bet / R1 / combination, cap 0.9 / 0.99 | H1 (combination, cap 0.9) after deviation D1; confirmed on fresh draws | anytime mode |
| v311 | fixed-n bets: F3, F2, F1 | F2 (fixed-n bets on both GO legs); confirmed on fresh draws | fixed_n mode (default) |

Development: W1 seed tag `bl`, W2 seed tag `BL` on 8 pools; held out from every selection: `lrb_medqa_test`,
`lrb_winogrande_valid`, `lurume_mmlu_pro`. Confirmation: `bl2`/`BL2` (v310), `bl3`/`BL3` (v311). Results in each
`results/` (`*_w2_*.jsonl.gz` are the per-configuration outputs, gzip of the files the analysis read).

`x0_repo.py`: the integrated `audit3` reproduces, on the full data of all 2,807 configurations, the official map
(lock v3-8, `rel_test="bet"`), H1 (`rel_mode="anytime"`) and F2 (`rel_mode="fixed_n"`) of the experiments.
`x6_reports.py`: absolute-claim (v3) and design-v2 reports are byte-identical before and after the integration.

Rerunning: the scripts were run from a working folder where `../bl` was `protocols/v3_decision/baselines`. To rerun
from the repository, put that folder on the path and point `CAAC_REPO` at the repository root, e.g. (PowerShell)
`$env:PYTHONPATH="src;protocols\v3_decision\baselines;protocols\v3_decision\relative_rounds\v311"; $env:CAAC_REPO=(Get-Location)`
and run the scripts from inside their folder. `scipy` is needed for the experiments only (not for `caac`).
