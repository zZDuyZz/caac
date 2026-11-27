# U-c: action guidance (amendment v3-8) — evidence

- `DECISION_uc_before_run.md` + `DECISION_uc_hash.txt`: rule and pass criteria, hashed before any run on real data.
- `src/caac/pool_v1/action.py`: the rule (pure function of report fields; never changes the verdict).
- `verify_v38.py`: checks C1, C2, C4, C5 by comparing every report of the rerun maps with the v3-7 reports and by
  re-deriving the action independently (it does not import `action.py`).
  `verify_relative.txt`, `verify_absolute.txt`: outputs. `uc_summary_{relative,absolute}.json`: distribution used in the paper (Table "action").
- C3 (v2 byte-identical): v2 primary map rerun with the new code; 64/64 reports, map.jsonl and summary.json identical.

Clarification of C4 (stated here, not hidden): the independent re-derivation compares every structured field of
the action object; the free-text sentences (`text`) are produced only by `action.py`, so for them the check is that
`action_from_report` applied to the saved report gives exactly the action in the new report (test
`test_pure_function_and_verdict_untouched`), not a second implementation of the wording.

Note on `below_floor`: it uses R4's `minimum_questions_zero_errors` (exact harm-count / binomial floor at level
delta/|Pi|). The betting test used for the verdict needs at least as many questions, so `below_floor = true` is a
sufficient (conservative) condition for "GO impossible at this n"; some pools above this floor are still too small
for the betting test.
