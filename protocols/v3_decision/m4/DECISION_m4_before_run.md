# M4: do the report's predictions come true on real data? — written BEFORE any run

Date: 2026-11-27. Nothing below has been run on real data when this file is hashed.

## Question

A v3 report on n questions makes four kinds of statement about what more data would show:
GO (the chosen policy meets the claim), NO-GO (no candidate does), INCONCLUSIVE-reachable ("with about Q
questions of the same kind, GO with probability about 0.8"), INCONCLUSIVE-unreachable ("more questions of
the same kind are not expected to give GO"). M4 checks how often each comes true, using large public pools
as stand-in ground truth: audit a random subsample, then look at the questions the subsample did not see.

## Data and scope (fixed now)

- Claim: relative (main claim), lock v3 draft with amendments up to v3-8 (canonical hash bd8a8e72...).
- Configurations: every configuration of the official v3-8 relative map (`map_rel38`) from LLMRouterBench and
  Lurume with n_common >= 1000 and verdict not NOT RUN: 2,807 configurations on 11 pools
  (primary 30, secondary 2,777). Monkey is excluded (127-140 questions).
- For each configuration the candidate set, cost measure and cost range are those of the official map, built
  from the whole pair (only the questions vary between runs). Every report is made by the official
  `make_report` with the official config; nothing in the audit is changed.
- Subsample sizes n_sub in {250, 500}; R = 10 subsamples per configuration and size, drawn without
  replacement with Python `random.Random(seed)`, seed = int(sha256("M4|" + config_id + "|" + n_sub + "|" + rep)[:16], 16).

## Follow-up for each subsample run (fixed now)

Let S be the subsample, F the configuration's questions, H = F \ S (held out, never seen by the subsample report).
- sub verdict GO, chosen policy pi: run the report on H; record pi's `delta_hat` and `saving` on H and its
  NO-GO p-values on H (`p_risk_above`, `p_saving_below`).
  * point violation: delta_hat_H > eps or saving_H < s;
  * clear violation: p_risk_above_H <= 0.05 or p_saving_below_H <= 0.05 (a valid one-sided 5% test on data the
    audit never saw).
- sub verdict INCONCLUSIVE, reachable with questions_needed Q: if Q <= |F|, draw E = S plus (Q - n_sub) more
  questions from H (seeded as above with tag "ext"), run the report on E, record the verdict.
  If Q > |F|: uncheckable (counted, not dropped).
- sub verdict INCONCLUSIVE, unreachable: compare with the official full-pool verdict (map).
- sub verdict NO-GO: compare with the official full-pool verdict (map).

## Harness check (must pass before any result is used)

- C0: for every configuration, the harness run on F reproduces the official map report byte for byte
  (sha16 equal). Any mismatch stops the analysis.
- C0b: run header records git commit, sha256 of `src/caac/pool_v1/*.py`, lock hash, sha256 of each pool's
  `meta.json` and `rows.jsonl.gz`.
- After the user's run: Claude re-runs 30 random (configuration, n_sub, rep) jobs in the container and requires
  identical sub and follow-up report hashes; file hashes and job counts must match the plan.

## What is reported (whatever it shows), relative claim, by n_sub and overall

- E1 reachable calibration: fraction of checkable reachable predictions whose extension run gives GO; planning
  target 0.8. Also by blocking leg, and the share uncheckable (Q > |F|).
  Prediction written now: the estimate is a plug-in power at the observed effect, selected for looking
  promising, so it is expected to be OPTIMISTIC (realised fraction below 0.8).
  Reading fixed now: >= 0.7 "usable as written"; 0.5-0.7 "optimistic, report with a correction";
  < 0.5 "not usable as a planning number" -> the report wording must change (deviation, user approval).
- E2 unreachable: fraction whose full-pool verdict is GO (a missed GO). Reading: <= 0.10 "the stop advice is sound".
- E3 GO on held-out data: point-violation rate and clear-violation rate among subsample GO runs.
  Reading: clear-violation rate <= delta = 0.10 is consistent with the guarantee; point violations are reported
  beside it (they include noise of H near the boundary).
- E4 NO-GO: fraction whose full-pool verdict is GO. Reading: <= 0.10 consistent.
- Every rate with a 95% interval from a cluster bootstrap over configurations (2,000 resamples, seed 0),
  because the R subsamples of one configuration are dependent; also counts.
- Also: share of subsample runs below the floor, verdict transition table subsample -> full pool.

## Limits stated now

Full-pool and held-out estimates are not the truth; they are larger samples of the same benchmark. Questions
are treated as independent (the cluster check is #171). Configurations share models and benchmarks, so rates
are descriptive of this map, not of all deployments.
