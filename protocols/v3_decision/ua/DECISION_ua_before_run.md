# U-a: certified saving (cận dưới tiết kiệm được chứng nhận) — design and criteria, written BEFORE any run
Written 26/11/2026 ~19:55 (+07). Status: candidate, SUPPLEMENTARY output. It never changes a verdict.
Seen before writing: all verdicts of v1/v2/v3 (absolute and relative) on LRB, Lurume, Monkey. No
certified-saving number has been computed on real data with this definition.

## Definition
Input: the same AuditInput, candidate set Pi (|Pi| = m), delta, salt, claim (relative eps or absolute alpha)
as the main audit; lvl = delta / m (the GO level of each leg).
For each policy pi:
  risk_pass(pi): the GO risk leg of audit3 at lvl (relative: betting on eps - D; absolute: exact binomial
                 vs alpha), with the same gate as audit3 (mean D <= eps, resp. observed risk <= alpha).
  LCB(pi): fixed sequence over the fixed grid S_GRID = -0.50, -0.48, ..., 0.96 (the grid already in
           audit.py), ascending; at each s test "saving < s" by betting on d_s = (1-s)B - C in the
           structural range [-(a_max + max(s,0) b_max), (1-s) b_max] at level lvl; LCB = the last s
           rejected before the first non-rejection (None if the first fails).
Certified saving S* = max over {pi : risk_pass(pi), LCB(pi) not None} of LCB(pi); reported with its policy.
None when no policy passes.
Statement: "policy pi* has (relative) R_pi* - R_big <= eps [absolute: R_pi* <= alpha] AND saves >= S*
of the big-model-only cost", wrong with probability <= delta.

## Validity (proof sketch, to go to the appendix)
For each pi let E_pi = {risk_pass(pi) and (risk claim of pi false or LCB(pi) > true saving of pi)}.
 - risk claim false: P(E_pi) <= P(risk_pass) <= lvl (the leg is a valid level-lvl test).
 - risk claim true: E_pi needs LCB > true saving, i.e. the sequence rejected the first grid point g with
   g > true saving, whose null "saving < g" is TRUE: probability <= lvl (fixed-sequence: the first true
   null in a fixed order is rejected with probability <= its level; nested nulls).
The reported statement is false only if E_pi* occurs, so P <= sum_pi P(E_pi) <= m lvl = delta.

## Consistency with the main verdict (reported, not a criterion)
GO at s = 0.20 and S* < 0.20 can happen (betting p not exactly monotone in s; the sequence may stop
early), as can S* >= 0.20 without GO (different gate / order effects). Both counts are reported.

## Simulation criteria (sim_ua.py; scenarios S1-S8 of sim_rel.py, relative claim, eps .02, n in
## {150, 300, 600, 1200, 2400}, 1000 reps; delta .10)
 (a) VALIDITY: P(statement false) <= 0.10 in every cell (5000-rep rerun if above but within 2 MC SE).
 (b) vectorised betting with per-row lower bound equals audit._betting_p to 1e-12 on 200 random checks.
 Reported: mean S* when defined; share of reps with S* defined; mismatch with GO at s = .20.
## Real data (only after (a), (b)): run once on every configuration of LRB, Lurume, Monkey (primary +
## secondary) for BOTH claims, recomputing audit3 first and requiring its verdict == map.jsonl verdict.
