# Evidence for the v3 amendment (recorded 26/11/2026, before the real-data run)

`DECISION_before_real.md` states the criteria and the chosen candidate; `DECISION_hashes.txt` gives the
SHA-256 of the decision file, the simulation code and its outputs, written at 2026-11-26T08:59:02+07:00,
before `real_v3.py` was run on the public pools. Check with `sha256sum -c DECISION_hashes.txt` (first 8 lines).

These scripts are the research prototypes, kept unchanged so that the hashes still match. They import the
v2 package from a sibling folder (`../pchk6/src`); to rerun them, point that path at `src/` of this repo
at tag `audit-lock-v2-final`. The official implementation of v3 is `src/caac/pool_v1/audit.py::audit3`
and `src/caac/pool_v1/gap.py`; running the cross-source map with lock v3 reproduces `real_v3.py`'s verdicts
on all 5,855 configurations (0 differences).
