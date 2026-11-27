# U-c: action guidance in every v3 report (amendment v3-8) — written BEFORE any run

Date: 2026-11-26. Author of the rule: fixed here, before computing it on any real report.

## What is added

`R1_verdict.action` in every design-v3 report (v2 reports unchanged, byte for byte), and a
"What to do" section in the markdown report. It is a **pure function of fields already in the
report** (verdict, conflict, certified saving S*, R2 per-policy rows, R3, R4 evidence gap and
floor). No new test, no new p-value, no new data access. It never changes the verdict.

## Rule (fixed now)

Notation: lvl_ng = delta/2 (NO-GO level per leg); quality-ok estimate of a policy =
`delta_hat < eps` (relative) or `risk < alpha` with served > 0 (absolute);
saving-ok estimate = `saving > s`.

1. verdict NOT RUN -> kind `fix_data`.
2. verdict GO -> kind `deploy`: policy = chosen (cheapest passing); budget figure = S*
   (`certified_saving.value`), not the point estimate; flags: groups whose lower bound is above
   alpha (R2 by_group, absolute claim only — descriptive), `verdict_changes_with_measure` (R3).
3. verdict NO-GO -> kind `keep_big`: counts of candidates rejected by the quality leg only
   (`p_risk_above <= lvl_ng` and not saving), saving leg only, both legs.
4. verdict INCONCLUSIVE with conflict -> kind `check_data` (GO and NO-GO both fired).
5. verdict INCONCLUSIVE otherwise -> kind `collect_or_stop`, with
   - `type = reachable` iff `R4.certification.reachable` is true: `more_questions_needed`,
     `questions_needed`, `blocking` (risk / saving / coverage), copied from R4;
   - `type = unreachable` otherwise, with `blocking` decided from the point estimates:
     `risk` if no candidate is quality-ok; `saving` if some are quality-ok but none is
     saving-ok; `trade_off` if some are quality-ok and some saving-ok but none both;
     `power` if some candidate is both (it cannot reach power 0.8 within 200000 questions).
   - `below_floor` (bool): n_questions < `R4.minimum_questions_zero_errors`
     (even perfect data of this size cannot give GO).
6. For every verdict except GO: if S* > 0, add `smaller_saving_certified` = S* with its policy
   (the S* statement holds with the same guarantee; it is not GO at the locked s).

Wording of the guidance text is part of the code; changing the rule after seeing results is a
deviation and must be reported.

## Run plan (one run)

Apply the new code to every configuration of the four official v3-7 maps (relative and
absolute, LRB + Lurume, Monkey): 6,031 + 6,031 reports.

## Pass criteria (all must hold; any failure is reported, not hidden)

- C1 every verdict identical to the v3-7 map (12,062 rows).
- C2 every v3 report identical to the v3-7 report except the new `R1_verdict.action` and the
  lock hash fields.
- C3 v2 reports byte-identical (64/64, same check as v3-7).
- C4 an independent re-implementation of the rule, applied to the saved v3-7 reports, gives the
  same `action` object for every row.
- C5 consistency: GO <-> deploy, NO-GO <-> keep_big, INCONCLUSIVE <-> collect_or_stop/check_data;
  type reachable <-> R4 reachable; blocking of a reachable row equals R4 blocking.
- full pytest passes.

## What will be reported (whatever it shows)

For relative and absolute claims, primary and secondary tiers: counts of INCONCLUSIVE by type
(reachable / unreachable), by blocking leg, below_floor, and with a smaller certified saving;
NO-GO by rejecting leg; median more_questions_needed of reachable rows.
