# Decision recorded BEFORE running the candidate on real data (26/11/2026, CPU day)

Status: candidate only. Not a lock. Lock v2 stays in force until the user approves a v3.
Revision 2: the simulation was corrected to the replay's semantics (on_unsure = escalate: every
question is served and the big model's errors on escalated questions count in the risk). The first
simulation (sim_u2_1000) treated escalated questions as unserved. No real-data result had been seen
when this was found (the first real run stopped at a range assertion before writing any verdict).

## Criteria written before looking at real data (MASTER 12.53)
(a) simulated P(wrong GO) <= delta and P(wrong NO-GO) <= delta in every scenario;
(b) power >= the v2 procedure (allowing small-sample noise);
(c) no unexplained change of conclusions.

## Simulation evidence (sim_v3sem_1000.txt: 1000 reps per cell, alpha = delta = .10, s = .20, 10 scenarios x 5 n)
- Largest wrong-verdict rate of any variant in any cell: 0.035 (v2, pi24_false, n=300). All <= 0.10.
  bonf+U2: largest wrong GO 0.009, largest wrong NO-GO 0.031.
- U1 (no split, Bonferroni over Pi at delta/|Pi|, both legs on all questions) vs v2:
  pi24_feasible n=600: 0.213 -> 0.991 (v2 picks the cheapest policy that passes half A, often a
  near-boundary one that then fails half B); feasible n=300: 0.550 -> 0.814.
  Only cell where U1 alone is lower: pi24 n=150 (0.025 vs 0.055).
- U2 (structural range of d) on top of U1: usd_feasible n=600: 0.255 -> 0.914; heavy_feasible n=600:
  0.182 -> 0.858; pi24 n=150: 0.025 -> 0.164. NO-GO power: -3 points (false_saving, light tail) to
  +7 points (heavy tail).
- bonf+U2 vs v2: GO power never lower in any cell.
- Planning (r4check_v3sem_150.txt): from a pilot of 150 or 400 questions, collecting the number v2's R4
  states and re-running v2 gives a GO in 4-28% of cases; collecting the two-leg evidence gap (target
  power .8) and running bonf+U2 gives a GO in 75-84% of cases.
- Certified frontier (frontier_v3sem_300.txt): P(any of the 2|Pi| bounds wrong) <= 0.073 (<= delta).
  At the main cell it is weaker than the direct test (usd n=600: 0.833 vs 0.914), as expected from
  delta/(2|Pi|). Decision: frontier = supplementary report; the verdict uses the direct test.

## Decision
Candidate = U1 + U2 ("bonf+U2"), with the two-leg evidence gap (power .8) as the planning output and the
certified frontier as a supplementary report.
alpha, delta, s, salt, Pi, pairs, measures, cost test (betting), NO-GO rule: UNCHANGED.
Run it ONCE on the real public pools (LRB + Lurume, primary + secondary), next to v2, and report every
configuration whose verdict changes, in both directions. Then the user decides (v3 amendment or keep v2).

## Known limits stated before the run
- The structural range needs: escalated cost = small part + the big-only cost of the same question. True by
  construction in replay.py (cost.append(used_cost + big_cost[qi])); real_v3.py asserts it on every config,
  using replay's serve_small flags.
- Bonferroni pays log|Pi|: at |Pi| = 30 the zero-error floor rises from 22 served (on half B, so about 44
  questions at full coverage) to 55 served (on all questions).
- The evidence gap is a plug-in estimate from the pilot; it is planning, never a verdict.
