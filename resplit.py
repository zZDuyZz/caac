"""How often does each selection rule break its promise? Re-split and count.

    python resplit.py --train-small T_s --train-big T_b --pool-small P_s --pool-big P_b \
        --splits 500 --alpha 0.10 --json out/resplit_math.json

The estimators are fitted once on TRAIN. The labelled POOL is split at random
into calibration and test halves ``--splits`` times; on each split every family
is certified on the calibration half and its pick is scored on the test half.
A violation is a test selective risk above alpha. With delta = 0.10, a valid
certificate should violate on at most about 10% of splits; the plug-in rule,
which takes calibration risk at face value, shows what happens without one.
Outcome tables are built once, so each split is just a sum over rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from caac.pool.protocol import (build_table, certify, pick_max_coverage, pick_min_cost,
                                plugin_select, prune, summarize)
from caac.pool.schema import load_pool
from certify import CAAC, baseline_family, caac_family, fit_estimators


def run(table, n, alpha, delta, splits, frac, rule, lam, min_cov, seed):
    rng = np.random.default_rng(seed)
    stats = {"certify": [], "plugin": []}
    for _ in range(splits):
        perm = rng.permutation(n)
        cal, tst = perm[:int(frac * n)], perm[int(frac * n):]
        for name, ok in (("certify", certify(table, cal, alpha, delta)[0]),
                         ("plugin", plugin_select(table, cal, alpha))):
            j = (pick_max_coverage(table, cal, ok) if rule == "max_coverage"
                 else pick_min_cost(table, cal, ok, lam, min_cov))
            if j is None:
                stats[name].append(None)
                continue
            m = summarize(table, tst, j, lam)
            stats[name].append((m["selective_risk"] > alpha, m["coverage"], m["cost"]))
    out = {}
    for name, xs in stats.items():
        picked = [x for x in xs if x is not None]
        out[name] = {
            "splits": len(xs),
            "nothing_certified": len(xs) - len(picked),
            # a split that certifies nothing serves nothing and cannot violate
            "violation_rate": sum(v for v, _, _ in picked) / len(xs) if xs else 0.0,
            "mean_coverage": float(np.mean([c for _, c, _ in picked])) if picked else 0.0,
            "mean_cost": float(np.mean([c for _, _, c in picked])) if picked else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-small", required=True)
    ap.add_argument("--train-big", default=None)
    ap.add_argument("--pool-small", required=True)
    ap.add_argument("--pool-big", default=None)
    ap.add_argument("--families", default="pricecheck,cascade_abstain,sc_abstain,caac_staged")
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--rule", default="max_coverage", choices=["max_coverage", "min_cost"])
    ap.add_argument("--min-coverage", type=float, default=0.6)
    ap.add_argument("--k-steps", default="4,8,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tr_s = load_pool(a.train_small)
    tr_b = load_pool(a.train_big) if a.train_big else None
    po_s = load_pool(a.pool_small)
    po_b = load_pool(a.pool_big) if a.pool_big else None
    if set(tr_s) & set(po_s):
        raise SystemExit("train and pool must not share a question")
    k_steps = tuple(int(x) for x in a.k_steps.split(","))
    names = [x.strip() for x in a.families.split(",") if x.strip()]
    big_model = next(iter(po_b.values())).model if po_b else None
    n_big = min((r.n_samples for r in (po_b or {}).values()), default=0)

    ests = fit_estimators(tr_s, tr_b, k_steps) if any(n in CAAC for n in names) else None

    qids = sorted(po_s)
    n_cal = int(a.calib_frac * len(qids))
    res = {"alpha": a.alpha, "delta": a.delta, "splits": a.splits, "n_pool": len(qids),
           "n_calib": n_cal, "rule": a.rule, "lam": a.lam, "families": {}}
    for nm in names:
        fam = (caac_family(nm, ests, a.lam, big_model, k_steps) if nm in CAAC
               else baseline_family(nm, n_big))
        fam = prune(fam, build_table(fam, sorted(tr_s), tr_s, tr_b), n_cal, a.alpha, a.delta)
        table = build_table(fam, qids, po_s, po_b)
        res["families"][nm] = run(table, len(qids), a.alpha, a.delta, a.splits, a.calib_frac,
                                  a.rule, a.lam, a.min_coverage, a.seed)
        c, p = res["families"][nm]["certify"], res["families"][nm]["plugin"]
        print(f"  {nm:30s} LTT: violation {c['violation_rate']:.3f}  coverage {c['mean_coverage']:.3f}"
              f"   |  plug-in: violation {p['violation_rate']:.3f}  coverage {p['mean_coverage']:.3f}",
              flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
