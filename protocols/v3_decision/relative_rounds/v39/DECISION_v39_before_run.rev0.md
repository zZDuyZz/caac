# Amendment v3-9 (candidate): more powerful tests for the RELATIVE claim — written BEFORE any run

Date: 2026-11-27. User approved the direction at 11:45 ("Duyệt cách 1 + 2"): a more powerful quality leg and a more powerful cost leg,
as an amendment with a design hashed first and a deviations entry. Nothing below has been run when this file is hashed.
Motivation (measured, #170): CLT + Bonferroni finds a feasible policy more often than CAAC (W2 power at n = 250/500/1000:
.45/.57/.67 vs .25/.39/.52). Among draws where CLT + Bonferroni says GO and CAAC does not (n = 500), the CAAC quality leg is
the blocker in 52%, the cost leg in 33%, both in 15% (post-hoc diagnostic).
NO claim is lowered; if no variant meets the criteria below, lock v3-8 stays as is and the comparison is reported.

## What may change (the only changes considered)

Same claim, same candidate set Pi, same levels (GO: delta/|Pi| per leg; NO-GO: delta/2 per leg), same Bonferroni structure, same
verdict logic. Only the p-value of a leg may change. The ABSOLUTE claim and design v2 are not touched (checked byte-identical).

R1  Quality leg, exact conditional test (GO and NO-GO), replacing betting on eps - D.
    D_i in {-1, 0, 1}; H = #{D = +1}, Bn = #{D = -1}, m = H + Bn, theta = P(D != 0). With p_h - p_b = E[D]:
    given m, H ~ Bin(m, pi), pi = 1/2 + E[D]/(2 theta).
    GO (H0: E[D] >= eps, reject for small H): theta_U = one-sided Clopper-Pearson upper bound of theta from m ~ Bin(n, theta) at
      confidence 1 - gamma; pi_min = min(1, 1/2 + eps/(2 theta_U)) (if theta_U < eps, H0 is empty on the event theta <= theta_U:
      pi_min = 1); p_risk_go = P_{Bin(m, pi_min)}(H' <= H) + gamma.
    NO-GO (H0: E[D] <= eps, reject for large H): theta_L = lower Clopper-Pearson bound at 1 - gamma; pi_max = min(1, 1/2 + eps/(2 theta_L))
      (theta_L = 0 gives pi_max = 1); p_risk_above = P_{Bin(m, pi_max)}(H' >= H) + gamma.
    gamma = 1e-4 (fixed now). Validity: Berger-Boos (the nuisance parameter theta is confined to its confidence set; the
    probability that it is not is <= gamma). Exact for every n; no variance estimate; order-free.
P   Per-policy structural range of the GO cost leg (d = (1 - s) B - C): lo_pi = -(a_pi + s big_max) with a_pi = the structural
    bound of that policy's own small-model cost = k * min(cap_small, cut_pi or cap_small) * price_out_small + in_max_small *
    price_in_small (a function of the policy definition and the generation caps only, not of the data); hi unchanged. The audit
    already rejects any observation outside the range. Applies only where the cost bound comes from generation caps
    (measure tokens); with an explicit cost.c_max the global range stays. NO-GO cost leg unchanged.
C   Cap of the bets of the GO cost leg: lambda <= 0.9/|lo| instead of 0.5/|lo| (1 + lambda x >= 0.1 > 0 still holds).
M   Mixture of bets on the GO cost leg: wealth K_t = 1/2 K_t(plug-in, cap 0.9/|lo|) + 1/2 * mean_j K_t(lambda_j), fixed
    lambda_j = 0.9 / |lo| * 2^-j, j = 0..4; p = 1 / max_t K_t (an average of nonnegative supermartingales; Ville).

Variants (cumulative; V0 is lock v3-8): V0 current | V1 = R1 | V2 = R1 + P | V3 = R1 + P + C | V4 = R1 + P + C + M.
Component views (ablation, reported, not eligible): R1 only (= V1), P only, C only, M only (each on V0).
Planning (R4 evidence gap), certified saving S*, and the action rule must use the same tests as the chosen variant; they
never change a verdict, and they are updated only after a variant is chosen.

## Worlds, metrics, methods of evaluation (exactly as #170, DECISION_bl_before_run.md fa4c4228...)

W1: S1-S14 x n in {150, 300, 600, 1200, 2400} x 1000 replicates (same draws, seed tag "bl"); W2: the 2,807 M4 configurations,
n in {250, 500, 1000} x 200 replicates (same draws); W3: full data of the 2,807 configurations. The per-policy small bound a_pi
comes from the policy definition (replay), in W1 from c_pi * cap(kind, MEAN_A).
Metrics: false GO (GO naming a policy that violates the claim), power (GO naming a feasible policy), wrong NO-GO (NO-GO when
some policy is feasible); validity rule "fails" = false-GO (or wrong-NO-GO) rate > delta with exact one-sided binomial p < 0.01.

## Checks before any result is used

- X0: V0 inside the new harness reproduces `bl_methods.caac` verdicts and chosen policies on every W1 and W2 draw of 40 random
  cells (identical), and the official map on the full data (2,807 / 2,807).
- X4 (R1 validity by enumeration): for n in {30, 60, 120} and eps in {0.02, 0.10}, over a grid of (p_h, p_b) on and beyond the
  boundary of H0, the exact rejection probability (sum over all (H, Bn) of the multinomial pmf) of R1 at levels {0.1, 0.1/30,
  0.05} is <= level; same for the NO-GO form. A single violation stops everything.
- X5: independent plain-Python implementation of V4 (mixture, per-policy range) on 200 random draws; vectorised == plain.
- X6: absolute claim and design v2 byte-identical after any repository change (primary maps rerun, report hashes equal).

## Criteria fixed now (a variant is ELIGIBLE only if all of A hold; it is ADOPTED only if B holds too)

A (validity) W1: no cell fails validity (false GO; and wrong NO-GO where the claim is true). W2: for each n pooled false-GO
  <= delta; no configuration fails validity (p < 0.01); pooled wrong NO-GO <= delta. The heavy-tailed configuration lrb_arcc_test
  Llama-3.1-8B-Instruct -> Llama-3.1-8B-UltraMedical (tokens_lambda1) is reported separately.
B (gain) W2 pooled power gain over V0 >= +0.05 (absolute) at no fewer than 2 of the 3 sizes, and power not below V0 at any size.
C (compatibility, reported with every number): on the full data, V0-GO configurations that are not GO under the variant
  (expected few; > 10 of 333 stops adoption and is reported), no GO <-> NO-GO flips (any flip stops adoption), the list of all
  verdict transitions.
Selection: among eligible variants meeting B and C, the one with the highest W2 pooled power at n = 500; ties go to the one with
fewer components. If a stricter variant is eligible but a simpler one has power within 0.01, the simpler one is chosen.
If A fails for a variant: that variant is dropped and the cause reported; nothing else is changed to make it pass.
If no variant meets B: no change to the lock (negative result reported in the paper's comparison).

## Predictions written now

R1 raises power markedly where the quality leg blocks (Section motivation) and is valid (X4). P gives the largest cost-leg gain on
cut policies with token costs (range up to ~8x tighter in the example configuration). C adds little, M adds little (< 0.02).
V4 power in W2 stays below CLT + Bonferroni's at n = 250. V1-V4 false GO <= delta everywhere.

## Limits stated now

Validity rests on: R1 exact (X4 by enumeration, any n); P a data-independent structural bound (audited per observation); C, M
the same betting argument as the current test. Plasmode and synthetic worlds treat questions as i.i.d. (shift is #171). The choice
of variants and of the 0.05 threshold was fixed after seeing the #170 comparison, not before: this is a power repair motivated by
a measured gap, and its evaluation is on the same worlds in which the gap was measured; the paper states this.
