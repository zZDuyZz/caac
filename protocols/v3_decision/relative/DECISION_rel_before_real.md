# Relative claim (R) and two-direction fixed-sequence (M1): design and criteria, written BEFORE any run
Written 26/11/2026 ~17:20 (+07). Status: candidate. Lock v3 is a draft (not tagged); nothing here changes
it until the user approves.

## What had been seen before this was written (disclosure)
- v1, v2 and v3 verdicts of the ABSOLUTE claim on the public pools (LRB, Lurume).
- One descriptive count (26/11 16:50): error rate of the big model alone (draw 0) per audited pair:
  primary 31/32 pairs > 0.10 (median 0.212); secondary 4,524/4,729 > 0.10 (median 0.354); small-only
  error <= big error + 0.02 in 14/32 primary pairs. This motivated the relative claim.
- NO test of the relative claim has been computed on real data. Monkey and the Qwen test split: unseen.

## 1. Relative claim (non-inferiority to big-only, paired)
Claim_rel(eps, s): there is pi in Pi with
  Delta_pi := R_pi - R_big <= eps     (R = P(served answer wrong); R_big = P(big draw 0 wrong))
  and saving_pi := 1 - E[C_pi]/E[B] >= s   (unchanged cost leg).
Per question i: D_i = e_pi,i - e_big,i in {-1, 0, 1}. With on_unsure = escalate the escalated question is
answered by the big model's draw 0, the same answer that defines big-only (checked in replay.py, R0),
so D_i = 0 exactly on escalated questions.
- GO risk leg: x_i = eps - D_i in [eps - 1, eps + 1]; betting test of H0: E[x] <= 0 (i.e. Delta >= eps),
  the same _betting_p as the cost leg, hash order. GO cost leg: unchanged (structural range).
- GO across Pi: Bonferroni, each leg at delta/|Pi| on all questions (as v3); cheapest passing policy.
- NO-GO: for EVERY pi, "Delta_pi <= eps" rejected (betting on D - eps, range [-1-eps, 1-eps]) OR
  "saving >= s" rejected, each at delta/2. INCONCLUSIVE otherwise (and on conflict).
- Evidence gap: both legs by bootstrap of the observed x and d (power 0.8, level delta/|Pi|);
  unreachable if mean(D) >= eps or mean(d) <= 0.
- Registered values: eps grid {0, 0.01, 0.02, 0.05}, MAIN eps = 0.02; s grid {0.10, 0.20, 0.30}, main 0.20;
  delta = 0.10. Pi, pairs, measures, salt: unchanged from lock v3.
- Reporting: the relative certificate is the paper's main answer to "is small-first worth deploying
  instead of big-only?"; the absolute certificate (alpha) stays, reported beside it as an SLA option.
  Each verdict controls its own error at delta; no joint guarantee is claimed across the two.
- Always reported (whatever is decided): NO-GO of the absolute claim split into "big-only alone fails
  alpha" vs the rest.

## 2. M1: two-direction fixed-sequence (power upgrade candidate)
Problem found while designing: ordering a tau-chain conservative -> liberal helps the risk leg but the
saving leg goes the other way (more escalation = less saving), so a fixed sequence on the combined
(intersection-union) null would stop at the first, conservative step.
Design: Pi is partitioned into C chains fixed in advance (in the real grid: one chain per (k, cut,
on_unsure), ordered by tau from 1.0 (conservative) to 0.5). In each chain, run
  - a fixed sequence on the RISK nulls from the conservative end, and
  - a fixed sequence on the SAVING nulls from the liberal end,
each at level delta/(2C), stopping at the first non-rejection. A policy passes if both its nulls are
rejected; GO if any passes; report the cheapest passing policy.
Validity: a wrong GO needs a false rejection of some true null; each fixed sequence controls its
family-wise error at its level for ANY fixed order; the 2C sequences sum to delta. NO-GO unchanged.
Caveat stated now: chains in the real grid have length <= 3 (tau has 3 values) and exist only for
k >= 2; LRB and Lurume are k = 1. The gain may be small; it can also be negative (2C vs |Pi|).

## 3. Simulation (sim_rel.py), criteria fixed now
Data model: big error ~ Bern(R_B = 0.25) per question; the small model answers a share q of questions;
on those, P(D = -1) = a (small right, big wrong), P(D = +1) = b (small wrong, big right); escalated
questions D = 0. Delta = q (b - a); saving = q - c E[a]/E[b]; costs as in sim_core (uniform; lognorm
for the heavy case). eps = 0.02, s = 0.20, delta = 0.10. n in {150, 300, 600, 1200, 2400}; 1000 reps.
Scenarios (c, q, a, b):
 S1 rel_feasible      target (1, .70, .04, .05) + decoys (1,.95,.02,.08) (1,.40,.03,.035) (2,.85,.03,.06) (2,.90,.02,.07)
 S2 rel_tight         target (1, .60, .03, .058) + same decoys
 S3 rel_bound_false   best saver Delta = .025 > eps: (1, .70, .03, .0657) + (1,.40,.03,.035) (1,.95,.02,.08) -> claim FALSE
 S4 rel_bound_true    Delta = .02 exactly: (1, .70, .03, .05857) + same two decoys -> claim TRUE (boundary)
 S5 rel_false_risk    savers Delta >= .04: (1,.80,.02,.07) (1,.95,.02,.08) (2,.90,.02,.07) + (1,.40,.03,.035)
 S6 rel_false_saving  low-Delta policies save < s: (1,.45,.03,.035) (1,.50,.04,.05) (2,.80,.03,.04)
 S7 rel_heavy         S1 with lognorm costs (cap 10x mean)
 S8 rel_discordant    S1 target with a = .15, b = .16 (Delta = .007, high variance)
 S9 rel_chains_true   8 chains x 3, order conservative -> liberal: q = (.45, .65, .85);
                      c = 1 in chains 1-4, c = 2 in chains 5-8 (chain j uses scale f_j = 1, 1.1, 1.2, 1.25
                      for j = 1..4 and again for 5..8); Delta = (.002, .016, .06) x f_j; a = .03 and
                      b = a + Delta / q. Savings (c = 1): .117, .317, .517; (c = 2): -.217, -.017, .183.
                      Only the middle c = 1 policies are feasible (chain 4: Delta = .02 = eps, boundary).
 S10 rel_chains_false as S9 with middle Delta = .03 (> eps) -> claim FALSE
 S11 rel_nonmono      as S9 but within each chain the order of the first two policies is swapped
                      (validity check of M1 when the declared order is wrong)
Variants: bonf (Section 1) in S1-S11; fs2 (Section 2) in S9-S11 only (S1-S8 have no chains).
Criteria:
 (a) VALIDITY: wrong-GO rate (claim false, verdict GO) and wrong-NO-GO rate (claim true, verdict NO-GO)
     <= delta = 0.10 in every cell, for every variant. If a cell is above 0.10 but within 2 Monte Carlo
     SE (0.019), rerun that cell with 5000 reps; the 5000-rep estimate decides.
 (b) M1 is ADOPTED (proposed to the user) only if (a) holds AND fs2 GO power >= bonf - 0.02 in every
     claim-true cell of S9 and S11 AND fs2 > bonf + 0.05 in at least 3 of the 5 n of S9. Otherwise it
     is reported as a negative result and not used.
 (c) The relative claim is PROPOSED for lock v3 if (a) holds for bonf in S1-S11. Power is reported,
     not a criterion (it is a different claim from the absolute one).
 (d) Vectorised betting must equal caac.pool_v1.audit._betting_p to 1e-12 on 200 random checks
     before any rate is computed.

## 4. Real data (only after 3 passes, with the official code)
Run ONCE on LRB + Lurume (primary + secondary), all registered eps and s; report every configuration,
the relative vs absolute verdicts side by side, and the absolute NO-GO decomposition. Monkey later, with
the same frozen code. Qwen: the relative claim enters lock v3 BEFORE the tag; the Qwen test split is
opened once after the tag.

## Known limits (stated now)
- The relative claim was added after the public data were seen (disclosed above); on public pools it is
  labelled post hoc; Qwen is the confirmatory test.
- eps = 0.02 is a convention (two points of accuracy), not derived; the grid is reported.
- D uses draw 0 of the big model; the claim is about that big-only baseline, not about the big model's
  best possible accuracy.

## Revision 2 (26/11 ~17:40), written while sim_rel.py (revision 1) was still running and BEFORE any of
## its output was read
Found while writing the report code: with zero discordance (D = 0 everywhere) the betting risk leg needs
n >= 563 questions to pass at eps = 0.02 and |Pi| = 30 (228 for one policy), because its bets are capped by
the range [eps - 1, eps + 1]. An exact alternative: Delta = P(D = +1) - P(D = -1) <= P(D = +1) =: h (the
"harm" rate: the policy wrong where big-only is right), so rejecting "h > eps" by the exact one-sided
binomial on the harm count (p = P(Bin(n, eps) <= #harm)) is a VALID test of "Delta > eps"; it needs
n >= 283 at zero harm (114 for one policy). It ignores the helps (D = -1), so it loses when harms and helps
are both frequent (S8). Variants of the GO risk leg added to the comparison (same levels, same everything
else; NO-GO risk leg unchanged: betting on D - eps):
  bet   (revision 1)   betting on eps - D
  harm                 exact binomial on the harm count vs eps
  mix                  p = min(1, 2 min(p_bet, p_harm))   (Bonferroni of the two; valid)
Choice rule (fixed now): among variants satisfying (a), take the highest mean GO power over all claim-true
cells (S1, S2, S4, S7, S8, S9, S11; all five n; Bonferroni procedure); a variant within 0.02 of the best is
preferred in the order harm, bet, mix (exact and simplest first). Implementation: sim_rel2.py (imports
sim_rel.py unchanged, so the revision-1 hash still holds). The revision-1 run is kept and reported.
