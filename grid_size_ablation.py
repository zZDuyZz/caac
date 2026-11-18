"""Does a bigger configuration grid help or hurt once it has to be certified?

    python grid_size_ablation.py --train-small T_s --train-big T_b --pool-small P_s --pool-big P_b \
        --sizes 3x8,6x16,11x32,21x64 --splits 500 --lam 5 --json out/grid_math.json

The staged CAAC policy (confidence gate, the "phat" arm) is run on (rho, tau)
grids of increasing size. For each size the grid is fixed before any data, its
testing order is fixed from a CROSS-FITTED training table (``prune``), and the
labelled POOL is re-split ``--splits`` times into calibration and test halves,
exactly as in resplit.py. Reported per size: configurations kept by prune, the
share of splits that certify something, the mean number certified, the test
violation rate (must stay near or below delta for every size), and the mean
coverage and cost of the pick.

What this can and cannot show. ``prune`` tests the survivors as ONE fixed
sequence ordered by their training p-value, at the full delta, so a bigger grid
does not split delta. Its cost, if any, is empirical: a longer sequence has more
places to stop early on calibration, and an order fitted on training noise
generalises worse. Whether that outweighs the extra flexibility is what the
numbers say -- in either direction -- and the paper reports whichever it is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from caac.pool import families as F
from caac.pool.protocol import build_table, concat_tables, prune
from caac.pool.schema import load_pool
from certify import fit_estimators, fit_fold_estimators
from resplit import run

FULL = ("answer_now", "sample_more", "escalate")


def grids(n_rho: int, n_tau: int) -> tuple[tuple, list]:
    """Rho grid with both special ends (0.0 buys freely, 1.01 buys nothing) and
    an evenly spaced interior; tau grid evenly spaced from 0.99 to 0.5. The
    default 6x16 returns the grids certify.py uses, so it is the paper's row."""
    if (n_rho, n_tau) == (len(F.RHO_GRID), len(F.TAU_GRID)):
        return tuple(F.RHO_GRID), list(F.TAU_GRID)
    if n_rho < 2 or n_tau < 2:
        raise ValueError("each grid needs at least two points")
    inner = np.linspace(0.3, 0.95, n_rho - 2) if n_rho > 2 else []
    rhos = tuple([0.0] + [round(float(x), 4) for x in inner] + [1.01])
    taus = [round(float(x), 4) for x in np.linspace(0.99, 0.5, n_tau)]
    return rhos, taus


def family(ests, lam, big_model, k_steps, rhos, taus):
    return F.caac_staged_family(ests.stage1, ests.stage2, lam, big_model, grid=taus, actions=FULL,
                                k_steps=k_steps, name=f"grid_{len(rhos)}x{len(taus)}", est3=ests.gate,
                                rhos=rhos, cost_model=ests.cost)


def evaluate_size(n_rho, n_tau, ests, fold_ests, train, pool, a, big_model, k_steps):
    rhos, taus = grids(n_rho, n_tau)
    tr_s, tr_b = train
    po_s, po_b = pool
    fam = family(ests, a.lam, big_model, k_steps, rhos, taus)
    tt = concat_tables([build_table(family(e, a.lam, big_model, k_steps, rhos, taus), held, tr_s, tr_b)
                        for held, e in fold_ests])
    qids = sorted(po_s)
    n_cal = int(a.calib_frac * len(qids))
    pruned = prune(fam, tt, n_cal, a.alpha, a.delta)
    out = {"n_rho": len(rhos), "n_tau": len(taus), "n_configs": len(fam.configs),
           "n_after_prune": len(pruned.configs)}
    if not pruned.configs:
        out["certify"] = out["plugin"] = None
        return out
    table = build_table(pruned, qids, po_s, po_b)
    out.update(run(table, len(qids), a.alpha, a.delta, a.splits, a.calib_frac, a.rule, a.lam,
                   a.min_coverage, a.seed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-small", required=True)
    ap.add_argument("--train-big", required=True)
    ap.add_argument("--pool-small", required=True)
    ap.add_argument("--pool-big", required=True)
    ap.add_argument("--sizes", default="3x8,6x16,11x32,21x64", help="RHOxTAU grid sizes")
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--rule", default="min_cost", choices=["max_coverage", "min_cost"])
    ap.add_argument("--min-coverage", type=float, default=0.6)
    ap.add_argument("--k-steps", default="2,4,8,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tr_s, tr_b = load_pool(a.train_small), load_pool(a.train_big)
    po_s, po_b = load_pool(a.pool_small), load_pool(a.pool_big)
    if set(tr_s) & set(po_s):
        raise SystemExit("train and pool must not share a question")
    k_steps = tuple(int(x) for x in a.k_steps.split(","))
    sizes = [tuple(int(v) for v in s.lower().split("x")) for s in a.sizes.split(",") if s.strip()]
    big_model = next(iter(po_b.values())).model
    ests = fit_estimators(tr_s, tr_b, k_steps)
    fold_ests = fit_fold_estimators(tr_s, tr_b, k_steps)

    res = {"alpha": a.alpha, "delta": a.delta, "lam": a.lam, "rule": a.rule, "splits": a.splits,
           "min_coverage": a.min_coverage, "n_pool": len(po_s), "sizes": []}
    print(f"{'grid':>7s} {'configs':>7s} {'kept':>5s} {'cert>0':>7s} {'#cert':>6s} {'viol':>6s} "
          f"{'cov':>6s} {'cost':>8s}   (LTT; delta={a.delta})")
    for n_rho, n_tau in sizes:
        r = evaluate_size(n_rho, n_tau, ests, fold_ests, (tr_s, tr_b), (po_s, po_b), a, big_model, k_steps)
        res["sizes"].append(r)
        c = r.get("certify")
        if c is None:
            print(f"{n_rho}x{n_tau:<4d} {r['n_configs']:7d} {r['n_after_prune']:5d}   nothing survives prune")
            continue
        share = 1 - c["nothing_certified"] / c["splits"]
        print(f"{n_rho:>3d}x{n_tau:<3d} {r['n_configs']:7d} {r['n_after_prune']:5d} {share:7.3f} "
              f"{c['mean_n_certified']:6.1f} {c['violation_rate']:6.3f} {c['mean_coverage']:6.3f} "
              f"{c['mean_cost']:8.0f}", flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
