# #170 Baselines: does CAAC beat the usual ways to decide? — written BEFORE any run

Date: 2026-11-27. When this file is hashed, no baseline has been run on any data (synthetic or real).
Only a timing check of `replay` on 3 configurations was done (no method outcome computed).

## Question

A team that wants to know "is small-model-first worth deploying?" can decide in other ways than CAAC.
We compare CAAC with the usual rules on the SAME claim, the SAME candidate set Pi, the SAME data, and the
SAME levels, in worlds where the truth is known exactly. Two things are measured: how often a rule says
GO with a policy that in truth violates the claim (false GO), and how often it finds a policy that in truth
meets it (power).

## Claim (fixed; the main claim of the paper)

Relative claim, lock v3 draft with amendments up to v3-8 (canonical hash bd8a8e72...): for policy pi,
  R_pi - R_big = E[D] <= eps   (D_i = e_pi,i - e_big,i in {-1, 0, 1})   and   E[d] >= 0   (d_i = (1 - s) B_i - C_pi,i),
eps = 0.02, s = 0.20, delta = 0.10. d lies in the structural range [lo, hi] = [-(small_max + s big_max), (1 - s) big_max].
Policy pi is FEASIBLE if E[D] <= eps and E[d] >= 0 under the world's distribution, VIOLATING otherwise.

## Methods (all see the same n questions and the same Pi; |Pi| = number of policies)

Every method that says GO names the cheapest passing policy (smallest mean cost on its data), as CAAC does.
z_q = standard normal quantile. "Level l" means one-sided at level l.
- B1 naive point: pass if mean(D) <= eps and mean(d) >= 0 (the rule `naive_go_relative` of the map).
- B2 CLT interval, no correction: pass if mean(D) + z_{1-delta} sd(D)/sqrt(n) <= eps and
  mean(d) - z_{1-delta} sd(d)/sqrt(n) >= 0 (sample sd, n - 1). The interval a practitioner reports at 1 - delta.
- B3 CLT interval + Bonferroni: as B2 with z_{1-delta/|Pi|} (the same per-leg level as CAAC).
- B4 Hoeffding + Bonferroni: mean(D) + 2 sqrt(log(|Pi|/delta) / (2n)) <= eps and
  mean(d) - (hi - lo) sqrt(log(|Pi|/delta) / (2n)) >= 0.
- B5 LTT with Hoeffding-Bentkus p-values (Angelopoulos et al., Learn then Test), Bonferroni over Pi:
  each leg's p-value on the variable rescaled to [0, 1] (x = (D + 1)/2, null mean > (eps + 1)/2;
  y = (hi - d)/(hi - lo), null mean > hi/(hi - lo)), p = min(exp(-n h(min(mean, mu0), mu0)),
  e * P(Bin(n, mu0) <= ceil(n mean))), h(a, b) = a log(a/b) + (1 - a) log((1 - a)/(1 - b)); pass if both <= delta/|Pi|.
  CAAC is LTT with Bonferroni and betting p-values, so B5 isolates the choice of p-value.
- B6 conformal risk control (CRC, Angelopoulos et al. 2022) as commonly applied to a candidate set:
  pass if (n mean(D) + 1)/(n + 1) <= eps and (n mean(-d) - lo)/(n + 1) <= 0. Its guarantee is on the
  EXPECTED risk, not with probability 1 - delta, and it assumes a monotone one-parameter family; we apply it
  as a practitioner would. Stated in the paper.
- B7 hold-out: the first half of the questions (in the audit's fixed order) selects pi by B1; the second half
  tests that one pi with the CLT interval at level delta per leg (one hypothesis, no correction needed).
- CAAC: the official v3 relative audit (betting p-values, Bonferroni over Pi at delta/|Pi| per leg, the saving
  leg gated by mean(D) <= eps, NO-GO when every policy is rejected at delta/2 on one leg, GO and NO-GO both
  -> INCONCLUSIVE). Vectorised; must reproduce the official map (check X0 below).

Only CAAC has a NO-GO separate from "not GO". For the binary rules (B1-B7), "not GO" is what the team would act
on as "do not deploy"; we report it as such and do not invent a NO-GO for them.

## Worlds

W1 synthetic (truth exact). Scenarios S1-S11 of `relative/sim_rel.py` (fixed on 26/11 for another purpose,
not chosen for this comparison), plus three written now to stress multiplicity:
- S12_near_miss_grid (claim FALSE): 30 policies, uniform costs, c = 1, q = 0.60, 0.61, ..., 0.89 (saving
  q - 1/3 in [0.27, 0.56]), a = 0.03, b = a + 0.03/q (E[D] = 0.03 = eps + 0.01 for all).
- S13_heavy_saving_false (claim FALSE): lognormal costs, c = 1, q in {0.50, 0.505, 0.51, 0.515, 0.52}
  (saving 0.167-0.187 < s), a = 0.03, b = a + 0.005/q (E[D] = 0.005).
- S14_one_feasible_in_grid (claim TRUE): the 29 policies of S12 with q = 0.61..0.89, plus one feasible
  policy c = 1, q = 0.65, a = b = 0.03 (E[D] = 0, saving 0.317).
n in {150, 300, 600, 1200, 2400}; 1000 replicates per (scenario, n); seed and draw as in `sim_rel.one`
(seed tag "bl"), so every method sees the same draws.

W2 plasmode on real data (truth exact in that world). Populations: the 2,807 configurations used by M4
(official v3-8 relative map, LLMRouterBench and Lurume, n_common >= 1000, not NOT RUN; 11 pools). For each,
the replayed per-question (D, d) of every policy on all its questions is the population; truth per policy =
population mean of D and d. A study draws n in {250, 500, 1000} questions WITH replacement (i.i.d. from the
population); R = 200 replicates per (configuration, n); seed = int(sha256("BL|" + config_id + "|" + n + "|" + rep)[:16], 16)
with numpy default_rng. The betting order is the draw order (data-independent; with i.i.d. draws any fixed
order is equivalent to the salted order). Cost ranges (small_max, big_max) are the configuration's official ones.

W3 (descriptive, no truth): every method on the full data of the same 2,807 configurations; counts of GO
and overlap with CAAC's GO.

## Metrics (per world/cell; pooled over configurations in W2)

- M1 false GO = P(GO and the named policy is VIOLATING). Target for a valid method: <= delta.
- M2 power = P(GO and the named policy is FEASIBLE), reported where some policy is feasible.
- M3 CAAC wrong NO-GO = P(NO-GO) when some policy is feasible (<= delta); for B1-B7, P(not GO) when some policy
  is feasible, labelled "stop when it should not" under the practitioner reading.
- W2: rates pooled over configurations (ratio of counts) with 95% cluster-bootstrap intervals over
  configurations (2,000 resamples, seed 0); per configuration, the share of configurations where M1 > delta
  with exact binomial p < 0.01 (R = 200). Paired power differences CAAC - method on the same draws.

## Checks before any result is used

- X0: vectorised CAAC on the FULL data of each of the 2,807 configurations, with the official salted order,
  reproduces the official map verdict and chosen policy for every configuration. Any mismatch: stop.
- X1: every method has a second, independent plain-Python implementation (no numpy, written separately);
  on 300 random (world, n, replicate) draws both give the same pass/fail per policy and the same decision.
- X2: B5 p-values checked against scipy (binomial cdf) on random inputs; CLT quantiles against scipy.
- X3: W1 reruns of 20 random cells with a fresh process give identical counts.

## Predictions written now

- B1 naive: M1 well above delta in S3, S12, S14 and on many W2 configurations at n = 250.
- B2 CLT no correction: M1 above delta in S12 (30 near-miss policies).
- B3 CLT + Bonferroni: about valid; may exceed delta at small n / heavy tails (no firm prediction).
- B4, B5: M1 <= delta everywhere, power well below CAAC (D is mostly 0, so variance-blind bounds are loose).
- B6 CRC: M1 far above delta near the boundary (expected-risk guarantee only).
- B7 hold-out: M1 about <= delta; power below CAAC at small n.
- CAAC: M1 <= delta in every W1 cell and pooled in W2.

## Readings fixed now

- A method "fails validity" in a W1 cell if M1 > delta with exact binomial p < 0.01 (H0: rate <= delta).
- If CAAC fails validity in any W1 cell, or its pooled W2 M1 exceeds delta (lower end of the cluster interval
  above delta): STOP, find the cause; no claim is changed without the user's approval.
- If a valid baseline has higher power than CAAC in some world, it is reported as such (no rewording).
- Whatever the outcome, every cell and every method is reported (appendix table), not only favourable ones.

## Limits stated now

The plasmode world treats each pool as the population and draws questions i.i.d.; real deployment traffic can
differ (shift is #171). CRC is applied outside its monotone one-parameter setting, as practitioners do. B2 uses
nominal level delta per leg; other conventional levels (e.g. 95%) would change its numbers, not its lack of
multiplicity correction. Synthetic scenarios S12-S14 were written by us to stress multiplicity; S1-S11 were not.
