# U-a: certified saving (supplementary; never changes a verdict)
- DECISION_ua_before_run.md: design, validity argument, criteria, written BEFORE any run (sha256 in DECISION_ua_hash.txt).
- sim_ua.py -> sim_ua_1000.{txt,json}: scenarios S1-S8 of ../relative/sim_rel.py, 1000 reps. Criterion (b) vectorised
  betting == audit._betting_p (max diff 1.1e-16). Criterion (a) P(statement false) <= .10: max 0.024 (S6, n=2400). PASS.
- certsave_map.py -> certsave.jsonl.gz: every configuration of LRB, Lurume, Monkey, both claims (12,062 rows), run ONCE
  with src/caac/pool_v1/certsave.py; audit3 recomputed first: verdict == official map verdict in 12,062/12,062.
- analyze_ua.py -> ua_summary.json.
Erratum (disclosed): the DECISION text gives the grid as -0.50..0.96 (copied from the docstring of S_GRID in audit.py);
S_GRID is actually -0.50..0.98 (range(-25, 50) x 0.02). Simulation and real run both used S_GRID itself, so they agree;
no real certified saving is above 0.68. The audit.py docstring is left as is (audit code unchanged).
Note: sim_ua.py imports ../relative/sim_rel.py; this copy differs from the one that ran ONLY in that import path line (the run used cpu1126/rel).

## v3-7: S* in every v3 report (approved by the user 26/11 20:10)
- report.py: R1_verdict.certified_saving {value, policy, saving_hat_of_policy, n_risk_pass, note} and one markdown line;
  v2 reports unchanged. Lock v3 (draft): amendment v3-7 and reporting.certified_saving.
- verify_v37.py -> verify_v37.txt: the official maps were rerun with the new code (relative and absolute on LRB +
  Lurume, both on Monkey). Every report equals the earlier one after removing R1.certified_saving and the lock hash
  (6,559 + 6,367 + 308 + 272 reports, 0 differences); S* in the reports equals certsave.jsonl in every configuration;
  map.jsonl lines are equal apart from report hashes. v2 primary: 64/64 reports and map.jsonl byte-identical.
