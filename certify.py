"""Learn-then-Test certification run: calibrate a policy, certify, evaluate.

This is the results pipeline for CAAC_MASTER §8.2, and it drives BOTH policies
through the identical protocol -- same grid size, same ordering, same binomial
p-value, same multiplicity correction, same held-out evaluation:

  --policy pricecheck   the baseline: label-free checks vote on the greedy
                        answer; the served answer is always the greedy one, so
                        coverage is capped by the Chow bound (§3.5).
  --policy caac         ours: the cheapest action whose CALIBRATED P(correct)
                        clears tau, else abstain. The answer-correcting actions
                        raise the cap the baseline is stuck under.

Running both from one driver is the point: the only difference the comparison
can be attributed to is the action set, not the statistics.

For each configuration lambda on a pre-specified grid:
  1. run the policy on the CALIBRATION pool -> (n_served, n_wrong, coverage, cost)
  2. certify P(selective risk <= alpha) >= 1-delta, two ways:
       * Bonferroni over the whole grid (what PriceCheck reports)
       * fixed-sequence, testing from the most conservative lambda downwards
         (spends no multiplicity, usually certifies more)
  3. pick the certified lambda with the most coverage
  4. run that lambda on the TEST pool and report what it actually achieves

It also reports the **price of the guarantee**: the coverage given up relative
to the best lambda that satisfies the risk constraint on test with hindsight.
That number is the honest cost of certification and, per §10, is one of the two
results nobody has published at realistic calibration sizes.

  python certify.py --calib-small A.jsonl.gz --calib-big B.jsonl.gz \
      --test-small C.jsonl.gz --test-big D.jsonl.gz --alpha 0.10 --json out.json

With only --calib-* given, the pool is split deterministically by a hash of the
query id. The TEST half is the same questions for both policies, so the two are
compared on identical data; for ``caac`` the other half is split again into
fitting and calibration questions, which leaves CAAC with HALF the calibration
set PriceCheck gets. That handicap runs against our own claim, so it is the
right way round for a preliminary number.
That is a PRELIMINARY mode for before the A3 split exists; with A3, pass
--train-qids plus disjoint --calib-*/--test-* pools and nothing is shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from caac.pool.ltt import (LambdaResult, certification_floor, certify_bonferroni,
                           certify_fixed_sequence, min_served_certifiable,
                           select_max_coverage)
from caac.pool.estimator import PerActionEstimator, SecondStageEstimator
from caac.pool.policies import caac_certified, caac_staged, pricecheck
from caac.pool.replay import make_orders, run_policy
from caac.pool.schema import load_pool

DELTA = 0.1


def split_half(pool: dict, seed: str = "caac-split-v1"):
    """Deterministic, model-independent half split by sha256 of the query id.

    Used ONLY in preliminary mode. The real calibration set is the A3 split,
    which is disjoint from the test questions by construction (§6.2)."""
    a, b = {}, {}
    for q, r in pool.items():
        h = hashlib.sha256(f"{seed}|{q}".encode()).hexdigest()
        (a if int(h[:8], 16) % 2 == 0 else b)[q] = r
    return a, b


def make_policy(small, big, check, k, conf_threshold):
    """lambda -> policy_fn(qid, order) suitable for replay.run_policy."""
    def factory(lam):
        def fn(qid, order):
            return pricecheck(small[qid], lam=lam, check=check, k=k, order=order,
                              conf_threshold=conf_threshold)
        return fn
    return factory


def make_staged_policy(small, big, est, est2, k, actions, cost_profile):
    """tau -> policy_fn for the three-stage controller."""
    def factory(tau):
        def fn(qid, order):
            return caac_staged(small[qid], (big or {}).get(qid), est, est2, tau=tau,
                               order=order, k=k, cost_profile=cost_profile,
                               actions=actions)
        return fn
    return factory


def make_caac_policy(small, big, est, k, actions, cost_profile):
    """tau -> policy_fn. Same signature as make_policy, so the sweep, the
    certification and the held-out evaluation below are shared code."""
    def factory(tau):
        def fn(qid, order):
            return caac_certified(small[qid], (big or {}).get(qid), est, tau=tau,
                                  order=order, k=k, cost_profile=cost_profile,
                                  actions=actions)
        return fn
    return factory


def sweep(qids, factory, grid, orders, cost_profile):
    """Run every lambda, returning LambdaResults in the GIVEN grid order."""
    out = []
    for lam in grid:
        m = run_policy(qids, factory(lam), orders)
        out.append((LambdaResult(lam=lam, n_served=m.served, n_wrong=m.wrong_served,
                                 coverage=m.coverage,
                                 mean_cost=m.mean_cost[cost_profile]), m))
    return out


def best_hindsight(pairs, alpha):
    """Most coverage among lambdas whose EMPIRICAL risk is already <= alpha.
    No guarantee attached -- this is the yardstick the guarantee is priced against."""
    ok = [(r, m) for r, m in pairs if m.served and m.selective_risk <= alpha]
    return max(ok, key=lambda rm: rm[0].coverage) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-small", required=True)
    ap.add_argument("--calib-big", default=None)
    ap.add_argument("--test-small", default=None)
    ap.add_argument("--test-big", default=None)
    ap.add_argument("--policy", default="pricecheck",
                    choices=["pricecheck", "caac", "caac_staged"])
    ap.add_argument("--estimator", default=None,
                    help="caac: a JSON estimator from train_estimator.py; "
                         "without it, one is fitted on the TRAIN part")
    ap.add_argument("--train-qids", default=None,
                    help="caac: fit the estimator on exactly these qids")
    ap.add_argument("--actions", default="answer_now,sample_more,escalate",
                    help="caac: action set (the ablation table varies this)")
    ap.add_argument("--check", default="resolve_vote",
                    choices=["confidence", "resolve_vote", "both"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--conf-threshold", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--grid", type=int, default=21, help="grid size")
    ap.add_argument("--grid-max", type=float, default=None,
                    help="most conservative grid value. Default: 1.0 for "
                         "pricecheck; 0.95 for caac, because tau=1.0 demands "
                         "calibrated CERTAINTY and so serves almost nothing -- a "
                         "degenerate point that cannot be certified at any "
                         "calibration size and, being first, would halt the "
                         "fixed-sequence test before it starts. Excluded by "
                         "design, not after seeing the numbers.")
    ap.add_argument("--split", default="auto", choices=["auto", "half", "three"],
                    help="preliminary splitting: 'three' also gives PRICECHECK "
                         "the smaller calibration set, for an exactly matched "
                         "comparison")
    ap.add_argument("--perms", type=int, default=20, help="common random permutations")
    ap.add_argument("--cost-profile", default="flops", choices=["flops", "tokens", "rounds"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cal_s = load_pool(a.calib_small)
    cal_b = load_pool(a.calib_big) if a.calib_big else None
    trn_s, trn_b = None, None
    if a.test_small:
        tst_s = load_pool(a.test_small)
        mode = "disjoint pools"
    elif a.split == "three" or (a.split == "auto"
                                and a.policy in ("caac", "caac_staged")
                                and not a.estimator):
        # three disjoint parts: fit | certify | report. The estimator must never
        # see a question that the guarantee or the reported number depends on.
        # test half FIRST, with the same seed pricecheck uses, so both policies
        # are scored on exactly the same held-out questions
        rest, tst_s = split_half(cal_s)
        trn_s, cal_s = split_half(rest, seed="caac-split-train")
        trn_b = cal_b
        mode = ("PRELIMINARY one pool: test half shared with pricecheck, "
                "remaining half split into fit/calib")
    else:
        cal_s, tst_s = split_half(cal_s)
        mode = "PRELIMINARY half-split of one pool"
    tst_b = load_pool(a.test_big) if a.test_big else cal_b

    actions = tuple(x.strip() for x in a.actions.split(",") if x.strip())
    est = est2 = None
    if a.policy in ("caac", "caac_staged"):
        if a.estimator:
            est = PerActionEstimator.from_json(Path(a.estimator).read_text())
            fit_note = f"loaded {a.estimator}"
        else:
            if a.train_qids:
                want = {l.strip() for l in Path(a.train_qids).read_text().splitlines()
                        if l.strip()}
                trn_s = {q: r for q, r in (trn_s or cal_s).items() if q in want}
                trn_b = trn_b or cal_b
            est = PerActionEstimator(k=a.k).fit(sorted(trn_s), trn_s, trn_b)
            fit_note = f"fitted on {len(trn_s)} held-apart questions"
        if a.policy == "caac_staged":
            # stage 2 is fitted on the SAME held-apart questions, never on the
            # calibration or test half
            src = trn_s if trn_s else cal_s
            est2 = SecondStageEstimator(k=a.k).fit(
                sorted(src), src,
                make_orders(min(r.n_samples for r in src.values()), 4, seed=11),
                stage1=lambda r: est(r))
            fit_note += f"; stage2 on {len(src)} (base {est2.base_rate:.3f})"

    def factory_for(small, big):
        if a.policy == "caac_staged":
            return make_staged_policy(small, big, est, est2, a.k, actions, a.cost_profile)
        if a.policy == "caac":
            return make_caac_policy(small, big, est, a.k, actions, a.cost_profile)
        return make_policy(small, big, a.check, a.k, a.conf_threshold)

    # grid ordered MOST CONSERVATIVE FIRST (high threshold -> abstains more).
    # The order is fixed here, before seeing any calibration number -- that is
    # what fixed-sequence testing requires.
    gmax = a.grid_max if a.grid_max is not None else (
        1.0 if a.policy == "pricecheck" else 0.95)
    grid = [round(gmax * (1.0 - i / (a.grid - 1)), 4) for i in range(a.grid)]
    n_samp = min(r.n_samples for r in cal_s.values())
    orders = make_orders(n_samp, a.perms, seed=0)

    cal_pairs = sweep(sorted(cal_s), factory_for(cal_s, cal_b),
                      grid, orders, a.cost_profile)
    bonf = certify_bonferroni([r for r, _ in cal_pairs], a.alpha, a.delta)
    pick_b = select_max_coverage(bonf)
    # fixed-sequence needs a FRESH list (certify_* mutate in place)
    fs_pairs = sweep(sorted(cal_s), factory_for(cal_s, cal_b),
                     grid, orders, a.cost_profile)
    fseq = certify_fixed_sequence([r for r, _ in fs_pairs], a.alpha, a.delta)
    pick_f = select_max_coverage(fseq)

    knob = "lambda" if a.policy == "pricecheck" else "tau"
    detail = (f"check={a.check}" if a.policy == "pricecheck"
              else f"actions={'+'.join(actions)}, estimator {fit_note}")
    print(f"\n=== LTT CERTIFICATION [{a.policy}] ({mode}; {detail}, k={a.k}, "
          f"alpha={a.alpha}, delta={a.delta}, |G|={len(grid)}) ===")
    n_min = min_served_certifiable(a.alpha, a.delta)
    n_min_b = min_served_certifiable(a.alpha, a.delta / len(grid))
    print(f"  calib n={len(cal_s)}   test n={len(tst_s)}   cost profile={a.cost_profile}"
          f"   grid max={gmax}")
    print(f"  certification floor: >={n_min} served (fixed-seq) / >={n_min_b} (Bonferroni)"
          f" => coverage >= {certification_floor(len(cal_s), a.alpha, a.delta):.3f}"
          f" / {certification_floor(len(cal_s), a.alpha, a.delta / len(grid)):.3f}")
    print(f"  {knob:>7} {'coverage':>9} {'risk':>7} {'p-value':>9}  {'Bonf':>5} {'FixSeq':>6}")
    for (r, m), rf in zip(cal_pairs, fseq):
        print(f"  {r.lam:7.3f} {r.coverage:9.3f} {m.selective_risk:7.3f} {r.pvalue:9.2e}"
              f"  {'YES' if r.certified else '  .':>5} {'YES' if rf.certified else '  .':>6}")

    out = {"policy": a.policy, "mode": mode, "check": a.check, "k": a.k,
           "actions": list(actions), "alpha": a.alpha, "delta": a.delta,
           "grid": grid, "grid_max": gmax, "n_calib": len(cal_s), "n_test": len(tst_s),
           "min_served_certifiable": n_min,
           "min_served_certifiable_bonferroni": n_min_b,
           "certification_floor_coverage": certification_floor(len(cal_s), a.alpha, a.delta),
           "cost_profile": a.cost_profile, "selected": {}}

    for name, pick in (("bonferroni", pick_b), ("fixed_sequence", pick_f)):
        if pick is None:
            print(f"  [{name}] nothing certified at alpha={a.alpha}")
            out["selected"][name] = None
            continue
        fn = factory_for(tst_s, tst_b)(pick.lam)
        m = run_policy(sorted(tst_s), fn, make_orders(
            min(r.n_samples for r in tst_s.values()), a.perms, seed=1))
        hs = best_hindsight(cal_pairs, a.alpha)
        price = (hs[0].coverage - pick.coverage) if hs else None
        print(f"  [{name}] {knob}={pick.lam:.3f}  calib coverage={pick.coverage:.3f}"
              f"  -> TEST coverage={m.coverage:.3f} risk={m.selective_risk:.3f}"
              f" acc={m.accuracy:.3f} cost={m.mean_cost[a.cost_profile]:.4g}"
              + (f"   price of guarantee = {price:+.3f} coverage" if price is not None else ""))
        out["selected"][name] = {
            "lambda": pick.lam, "calib_coverage": pick.coverage, "pvalue": pick.pvalue,
            "test_coverage": m.coverage, "test_selective_risk": m.selective_risk,
            "test_accuracy": m.accuracy, "test_cost": m.mean_cost[a.cost_profile],
            "price_of_guarantee_coverage": price,
            "test_risk_violated": m.selective_risk > a.alpha,
            "test_action_mix": dict(m.action_dist) if hasattr(m, "action_dist") else None,
        }
        if getattr(m, "action_dist", None):
            mix = ", ".join(f"{k2} {v}" for k2, v in sorted(m.action_dist.items()))
            print(f"           actions on test: {mix}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"  -> {a.json}")


if __name__ == "__main__":
    main()
