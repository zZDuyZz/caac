# Amendment v3-10 (candidate round 2): choose the relative-claim tests on a DEVELOPMENT set, confirm on fresh data — written BEFORE any run

Date: 2026-11-27. Follows v3-9 (DECISION_v39_before_run.md rev.1, d8ecdbc1...; its results are in MASTER 12.82). The user asked (13:49)
for the best way for the paper: iterating on the method is allowed; what is fixed here is how the choice is made and reported.
Nothing below has been run when this file is hashed.

## What v3-9 showed (development information, already seen)

V3 = exact quality test (R1) + per-policy cost range (P) + bet cap 0.9 (C) was the variant the v3-9 rule picked. Weaknesses seen:
in the synthetic world W1 it has LOWER power than the current test in 28 of 40 claim-true cells (mean .073 -> .058); on real
plasmode (W2) power rose by +.100/.076/.064 at n = 250/500/1000, mostly from P and C (P alone +.055/+.032/+.029; C alone
+.040/+.022/+.017; R1 alone +.027/+.030/+.028; M alone -.005/-.009/-.005). Mixture bets (M) are dropped.

## Candidates (fixed now; quality leg x cost-leg bet cap; all with the per-policy cost range P)

  B1 = bet quality + P + cap 0.9       B2 = bet quality + P + cap 0.99
  X1 = R1 (exact)   + P + cap 0.9      X2 = R1 (exact)   + P + cap 0.99      (X1 = V3 of v3-9)
  H1 = hybrid       + P + cap 0.9      H2 = hybrid       + P + cap 0.99
  V0 = lock v3-8 (reference).  Order of simplicity for ties: B1, B2, X1, X2, H1, H2.
hybrid quality p-value (GO and NO-GO form): min(1, 2 * min(p_bet, p_R1)), valid by a union bound (each p-value valid on its own).
R1 and P are exactly as in v3-9 (gamma = 1e-4; per-policy structural range lo_pi = -(a_pi + s big_max), tokens measure only).
cap 0.99: lambda <= 0.99/|lo| (1 + lambda x >= 0.01 > 0).
The absolute claim and design v2 are not touched.

## Data split (fixed now)

DEVELOPMENT: W1 (S1-S14, seed tag "bl", the draws already used) and W2 on the 8 pools NOT listed below (same seeds "BL").
HELD-OUT POOLS (3, fixed now): lrb_medqa_test, lrb_winogrande_valid, lurume_mmlu_pro. The selection uses none of their results.
Honest limit: earlier runs (#170, v3-9) already covered all pools for V0 and V3, so these pools are not blind for those two; they are
unseen for the other candidates.
CONFIRMATION: W1 with fresh seed tag "bl2" (1000 replicates per cell) and W2 with fresh seed tag "BL2" (200 replicates per
configuration and size) on all 11 pools, reported for all pools and for the 3 held-out pools separately.

## Selection on DEVELOPMENT (rule fixed now)

Eligible: (A) validity: W1 no cell fails (false GO > delta with exact binomial p < 0.01; wrong NO-GO likewise where the claim is
true); W2 (8 dev pools) per size pooled false GO <= delta, no configuration fails, pooled wrong NO-GO <= delta;
(C) on the full data of the dev-pool configurations: V0-GO lost <= 3% of the V0 GOs, no GO <-> NO-GO flip.
Score = 0.5 * gW2 + 0.5 * gW1, gW2 = mean over n in {250, 500, 1000} of (pooled W2 power - V0's), gW1 = mean over claim-true W1
cells of (power - V0's). Also required: W2 gain >= +0.05 at >= 2 of 3 sizes. Chosen = highest score; any variant earlier in the
simplicity order within 0.005 of the best score wins. If none is eligible: nothing is adopted, v3-8 stays.

## Confirmation criteria (on the CHOSEN variant vs V0, fresh seeds; fixed now)

A validity as above on all pools and cells. If it fails: STOP, find the cause, adopt nothing.
B power: W2 pooled gain >= +0.04 at >= 2 of 3 sizes and >= 0 at all three, all pools; on the 3 held-out pools gain > 0 at >= 2 of 3
  sizes. Cluster-bootstrap intervals (configurations, 1,000 resamples, seed 0) reported.
C W1 confirmation: mean gain over claim-true cells >= -0.01 (not materially worse than V0 in the synthetic world).
D Full-data compatibility as above (all pools).
If A holds but B, C or D fails: v3-8 stays and the result (including the dev table) goes to the paper as a negative result.
No variant outside the list above is tried after seeing confirmation results.

## Checks before any result is used (as v3-9)

X0 (V0 reproduces the official map on the full data of all 2,807 configurations; and equals bl_methods.caac on W1 draws);
X4 (exact enumeration of R1 and of the hybrid at the LEVELS used by the hybrid, i.e. also halved levels: GO delta/|Pi|/2, NO-GO
delta/4; single violation stops everything); X5 (plain-Python reimplementation of every candidate on 150 random draws, 0 mismatches);
X6 (absolute claim and v2 byte-identical after any repository change).

## Predictions written now

Hybrid recovers most of the W1 loss (gW1 closer to 0 than X1) and keeps most of W2's gain; cap 0.99 adds < 0.01 over 0.9; B1/B2
have gW1 about 0 or slightly positive and gW2 about +.06; the chosen variant is H1 or B1. These are guesses, not criteria.

## Limits stated now

The candidate list was formed after seeing the v3-9 development results (selection effect, stated in the paper: seven variants tried
over two rounds, plus the four of v3-9). Questions are treated as i.i.d. Pools are populations already seen in part. Power numbers of the
dev round are optimistic; the confirmation numbers are the ones to report.
