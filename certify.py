"""Learn-then-Test certification run: calibrate a policy, certify, evaluate.

This is the results pipeline for CAAC_MASTER §8.2. It currently drives the
PriceCheck baseline; the CAAC controller plugs into the same driver through
``POLICIES`` once its estimator is trained (§8.1).

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

With only --calib-* given, the pool is split in half deterministically (by a
hash of the query id) -- a PRELIMINARY mode for before the A3 split exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from caac.pool.ltt import (LambdaResult, certify_bonferroni, certify_fixed_sequence,
                           select_max_coverage)
from caac.pool.policies import pricecheck
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
    ap.add_argument("--check", default="resolve_vote",
                    choices=["confidence", "resolve_vote", "both"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--conf-threshold", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--grid", type=int, default=21, help="lambda values in [0,1]")
    ap.add_argument("--perms", type=int, default=20, help="common random permutations")
    ap.add_argument("--cost-profile", default="flops", choices=["flops", "tokens", "rounds"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cal_s = load_pool(a.calib_small)
    if a.test_small:
        tst_s = load_pool(a.test_small)
        mode = "disjoint pools"
    else:
        cal_s, tst_s = split_half(cal_s)
        mode = "PRELIMINARY half-split of one pool"
    cal_b = load_pool(a.calib_big) if a.calib_big else None
    tst_b = load_pool(a.test_big) if a.test_big else cal_b

    # grid ordered MOST CONSERVATIVE FIRST (high threshold -> abstains more).
    # The order is fixed here, before seeing any calibration number -- that is
    # what fixed-sequence testing requires.
    grid = [round(1.0 - i / (a.grid - 1), 4) for i in range(a.grid)]
    n_samp = min(r.n_samples for r in cal_s.values())
    orders = make_orders(n_samp, a.perms, seed=0)

    cal_pairs = sweep(sorted(cal_s), make_policy(cal_s, cal_b, a.check, a.k, a.conf_threshold),
                      grid, orders, a.cost_profile)
    bonf = certify_bonferroni([r for r, _ in cal_pairs], a.alpha, a.delta)
    pick_b = select_max_coverage(bonf)
    # fixed-sequence needs a FRESH list (certify_* mutate in place)
    fs_pairs = sweep(sorted(cal_s), make_policy(cal_s, cal_b, a.check, a.k, a.conf_threshold),
                     grid, orders, a.cost_profile)
    fseq = certify_fixed_sequence([r for r, _ in fs_pairs], a.alpha, a.delta)
    pick_f = select_max_coverage(fseq)

    print(f"\n=== LTT CERTIFICATION ({mode}; check={a.check}, k={a.k}, "
          f"alpha={a.alpha}, delta={a.delta}, |G|={len(grid)}) ===")
    print(f"  calib n={len(cal_s)}   test n={len(tst_s)}   cost profile={a.cost_profile}")
    print(f"  {'lambda':>7} {'coverage':>9} {'risk':>7} {'p-value':>9}  {'Bonf':>5} {'FixSeq':>6}")
    for (r, m), rf in zip(cal_pairs, fseq):
        print(f"  {r.lam:7.3f} {r.coverage:9.3f} {m.selective_risk:7.3f} {r.pvalue:9.2e}"
              f"  {'YES' if r.certified else '  .':>5} {'YES' if rf.certified else '  .':>6}")

    out = {"mode": mode, "check": a.check, "k": a.k, "alpha": a.alpha, "delta": a.delta,
           "grid": grid, "n_calib": len(cal_s), "n_test": len(tst_s),
           "cost_profile": a.cost_profile, "selected": {}}

    for name, pick in (("bonferroni", pick_b), ("fixed_sequence", pick_f)):
        if pick is None:
            print(f"  [{name}] nothing certified at alpha={a.alpha}")
            out["selected"][name] = None
            continue
        fn = make_policy(tst_s, tst_b, a.check, a.k, a.conf_threshold)(pick.lam)
        m = run_policy(sorted(tst_s), fn, make_orders(
            min(r.n_samples for r in tst_s.values()), a.perms, seed=1))
        hs = best_hindsight(cal_pairs, a.alpha)
        price = (hs[0].coverage - pick.coverage) if hs else None
        print(f"  [{name}] lambda={pick.lam:.3f}  calib coverage={pick.coverage:.3f}"
              f"  -> TEST coverage={m.coverage:.3f} risk={m.selective_risk:.3f}"
              f" acc={m.accuracy:.3f} cost={m.mean_cost[a.cost_profile]:.4g}"
              + (f"   price of guarantee = {price:+.3f} coverage" if price is not None else ""))
        out["selected"][name] = {
            "lambda": pick.lam, "calib_coverage": pick.coverage, "pvalue": pick.pvalue,
            "test_coverage": m.coverage, "test_selective_risk": m.selective_risk,
            "test_accuracy": m.accuracy, "test_cost": m.mean_cost[a.cost_profile],
            "price_of_guarantee_coverage": price,
            "test_risk_violated": m.selective_risk > a.alpha,
        }
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"  -> {a.json}")


if __name__ == "__main__":
    main()
