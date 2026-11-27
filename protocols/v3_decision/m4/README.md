# M4: do the report's predictions come true on real data?

- `DECISION_m4_before_run.md` + `DECISION_m4_hash.txt`: design, follow-ups, readings and limits, hashed
  (sha256 d6b1eff6...) before any run on real data.
- `m4_run.py`: the harness. Every report is made by the official `make_report` with the official configuration of
  the v3-8 relative map; only the questions change. C0 checks, for every configuration, that the harness run on all
  questions reproduces the official map report byte for byte. The header records git commit, code, lock and data
  hashes.
- `m4_analyze.py`: the analysis (written before the real run); stops if any C0 mismatch.

Inputs (not in the repository): the 11 pools with n_common >= 1000 from LLMRouterBench and Lurume
(`pools_m4/`, about 4 MB) and the official v3-8 relative map (`map_rel38.jsonl`).

Run (from anywhere; paths are examples):

    python protocols/v3_decision/m4/m4_run.py --pools D:/caac_m4/pools_m4 --map D:/caac_m4/map_rel38.jsonl --out D:/caac_m4/out --workers 18
    python protocols/v3_decision/m4/m4_analyze.py D:/caac_m4/out/m4_runs.jsonl D:/caac_m4/out/m4_summary.json

Verification (Claude, after the run): re-run 30 random (configuration, n_sub, rep) jobs with `--verify` on
another machine and require identical report hashes; compare `m4_files.sha256`.
