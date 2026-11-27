# Amendment v3-11 (candidate round 3): input profiler + fixed-n mode for the relative claim — written BEFORE any run

Date: 2026-11-27. Follows v3-10 (DECISION_v310_before_run.md e95a309e..., deviation D1 fe863afd...; adopted H1 after
confirmation, MASTER 12.83). The user approved (14:42) a profiler that adapts the test to information known before the data
are looked at. Nothing below has been run when this file is hashed, except the POST-HOC diagnostics named next.

## What is already known (development information, already seen)

Diagnostics on the 8 development pools only, seed tags DG..DG4 (never used for selection or confirmation; scripts
v311_diag*.py, results v311_diag*.json). At n = 500, power: H1 .436, CLT + Bonferroni .554; quality leg replaced by its point
gate .732; cost leg replaced by its gate .530; no Bonferroni (not valid) .549; cost leg with fixed-n bets .444; quality AND cost
legs with fixed-n bets (cap 0.9) .475; exact unconditional paired non-inferiority test (estimate) + fixed-n cost .473 (dropped:
no gain, much heavier); CLT quality (not finite-sample valid) + current cost .527. Reading: most of the gap is the quality leg;
a fixed-n bet recovers part of it; the remaining part is the price of a finite-sample guarantee.

## The profiler (fixed now)

Before any outcome is read, the audit reads only: the declared use (`fixed_n`: one audit of the n rows given, as in offline
replay and in every run of this paper; `anytime`: the user may look at the result while rows arrive and stop at any time),
n, |Pi|, the per-policy structural cost ranges (v3-9 P), eps, s, delta. Rule: `anytime` -> H1 (v3-10, anytime-valid);
`fixed_n` -> the candidate chosen below. Because the choice depends on no outcome, each branch keeps its own guarantee.

## Fixed-n bet (fixed now)

For a row x_1..x_n in [lo, hi], H0: E[x_t | past] <= 0. lambda_t = min(cap/|lo|, sqrt(2 log(1/alpha) / (n v_{t-1}))),
v_{t-1} = ((hi - lo)^2/4 + sum_{i<t} (x_i - xbar_{t-1})^2) / t (v_0 = (hi - lo)^2/4); K_t = prod (1 + lambda_i x_i);
p = min(1, 1 / max_{t <= n} K_t). lambda_t is predictable and >= 0, 1 + lambda x >= 1 - cap > 0, so K is a nonnegative
supermartingale under H0 and Ville gives P(p <= alpha) <= alpha (valid for all t, hence for t <= n; the tuning to n and alpha
is what makes it a fixed-n test in spirit). alpha = delta/|Pi| (the level each GO leg is tested at). Same order of rows as now.
Quality GO leg: x = eps - D, [lo, hi] = [eps - 1, eps + 1]. Cost GO leg: x = d, [lo_pi, hi] (P), gated by mean(D) <= eps as now.
NO-GO legs: exactly H1's (hybrid quality, v3-8 cost). Absolute claim and design v2 untouched.

## Candidates (fixed now)

  F3 = fixed-n quality (cap 0.9) + H1 cost leg (aGRAPA, P, cap 0.9)
  F2 = fixed-n quality (cap 0.9) + fixed-n cost (P, cap 0.9)
  F1 = fixed-n quality (cap 0.5) + fixed-n cost (P, cap 0.9)
  References: H1 (adopted in v3-10, not yet in the repository) and V0 (lock v3-8). Simplicity order for ties: F3, F2, F1.

## Data split (fixed now)

DEVELOPMENT: W1 seed tag "bl" and W2 seed tag "BL" on the 8 development pools (the v3-10 development draws).
HELD-OUT POOLS: lrb_medqa_test, lrb_winogrande_valid, lurume_mmlu_pro (not used for this choice; they were used in the v3-10
confirmation of H1 and V0 only, so they are unseen for the F candidates).
CONFIRMATION: W1 fresh seed tag "bl3" (1000 replicates per cell), W2 fresh seed tag "BL3" (200 replicates), all 11 pools, reported
for all pools and the 3 held-out pools separately.

## Selection on DEVELOPMENT (rule fixed now; includes the lesson of deviation D1)

Eligible: (A) validity exactly as in v3-10; (C) full-data compatibility vs V0 (V0-GO lost <= 3% of V0 GOs, no GO <-> NO-GO flip);
(Cw1) W1 mean gain vs V0 over claim-true cells >= -0.01; (gate) W2 gain vs H1 >= +0.02 at >= 2 of 3 sizes.
Score = 0.5 * gW2 + 0.5 * gW1, both vs H1. Chosen = highest score; a candidate earlier in the simplicity order within 0.005 wins.
None eligible: the fixed-n mode is not adopted; H1 serves both modes.

## Confirmation criteria (CHOSEN vs H1 and V0, fresh seeds; fixed now)

A validity as in v3-10 (all pools, all cells). Fails: STOP, find the cause, adopt nothing.
B power: W2 gain vs H1 >= +0.02 at >= 2 of 3 sizes and >= 0 at all three (all pools); held-out pools gain vs H1 > 0 at >= 2 of 3.
C W1 mean gain vs V0 >= -0.01.  D full-data compatibility vs V0 (all pools).
A holds but B, C or D fails: H1 serves both modes and the result (dev and confirmation tables) is reported as negative.
No candidate outside this list is tried after seeing the confirmation results.

## Checks (v311_check.py, before any result is used)

X0: V0 == bl_methods.caac and H1 == v3-10 H1 on 60 W1 draws. X5: plain-Python decision of every method on 150 draws, 0 mismatches;
X5b: fixed-n p-value vectorised vs plain Python on 200 random rows (|diff| < 1e-12). X7: size at the null boundary (two-point and
uniform, four ranges, alpha 0.1 and 0.02, 20,000 sequences): rate/alpha <= 1 + Monte Carlo slack. X0-map: V0 reproduces the official
map on every configuration (in v311_w2.py). X6 after any repository change.

## Predictions written now

Dev W2 gain vs H1 about +.03/+.04/+.04 for F2 (from the diagnostics); F3 smaller. W1: unknown; fixed-n bets may be weaker than
aGRAPA when the true margin is far from what sqrt(2 log(1/alpha)/(n v)) assumes. Probability that a candidate passes confirmation
about 0.6.

## Reporting

Paper: one sentence in the main text (profiler, modes), appendix table of all candidates at dev and confirmation, the diagnostics
named above as post-hoc, the number of rounds (v3-9, v3-10, v3-11) and deviation D1.
