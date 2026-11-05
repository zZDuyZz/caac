"""Fit + evaluate the calibrated per-action estimator (CAAC_MASTER §8.1).

    python train_estimator.py --small pools/math_1.5b.jsonl.gz \
        --big pools/math_7b.jsonl.gz --out models/est_math.json

What it reports, per action and on HELD-OUT questions only:
  * ECE / Brier      -- is the probability the controller gates on honest?
  * AUROC            -- does it rank questions at all? (ECE can be good while
                        the score is uninformative, so both are needed)
  * base rate        -- the "always predict the mean" reference
and the ABLATION over feature groups:
  * ``qonly``        -- question-only information (prompt length, and the A4
                        hidden-state probe once it exists)
  * ``greedy``       -- the greedy turn the policy has already paid for
  * both
That ablation is the second of the two results CAAC_MASTER §10 lists as unclaimed:
how much the greedy turn adds over a question-only probe on a short-CoT instruct
model -- the gap Bulut (2609.03436) leaves open. If ``greedy`` adds nothing, the
paper says so and the controller still works on ``qonly``.

Evaluation uses REALISED outcomes (one concrete draw order for sample_more),
never the soft maj@k target used for fitting, so every number here is one a
deployed policy could actually hit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from caac.eval.calibration import brier_score, expected_calibration_error
from caac.pool.estimator import (SERVING_ACTIONS, PerActionEstimator, feature_names,
                                 realised_targets)
from caac.pool.replay import make_orders
from caac.pool.schema import load_pool


def split_by_hash(qids, fracs=(0.5, 0.5), seed="caac-est-v1"):
    """Deterministic, model-independent split into len(fracs) parts."""
    cuts, acc = [], 0.0
    for f in fracs[:-1]:
        acc += f
        cuts.append(acc)
    parts = [[] for _ in fracs]
    for q in qids:
        u = int(hashlib.sha256(f"{seed}|{q}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        i = sum(1 for c in cuts if u >= c)
        parts[i].append(q)
    return [sorted(p) for p in parts]


def auroc(scores, labels) -> float:
    """Rank-based AUROC with tie correction; nan when one class is absent."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    pos, neg = y > 0.5, y <= 0.5
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks within ties so a constant predictor scores exactly 0.5
    i = 0
    ss = s[order]
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(est: PerActionEstimator, qids, small, big, orders, k) -> dict:
    rows = {a: {"p": [], "y": []} for a in SERVING_ACTIONS}
    for q in qids:
        p = est(small[q], (big or {}).get(q))
        t = realised_targets(small[q], (big or {}).get(q), orders[0], k)
        for a in SERVING_ACTIONS:
            if a in t and a in est.heads:
                rows[a]["p"].append(p[a])
                rows[a]["y"].append(t[a])
    out = {}
    for a, d in rows.items():
        if len(d["y"]) < 10:
            continue
        pr, y = d["p"], [int(v) for v in d["y"]]
        out[a] = {"n": len(y), "base_rate": float(np.mean(y)),
                  "mean_pred": float(np.mean(pr)),
                  "ece": expected_calibration_error(pr, y),
                  "brier": brier_score(pr, y),
                  "auroc": auroc(pr, y)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--big", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--train-qids", default=None,
                    help="file of qids to FIT on; the rest is held out "
                         "(use the A3 split; hash split is the fallback)")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--out", default=None, help="write the fitted estimator as JSON")
    ap.add_argument("--json", default=None, help="write the report as JSON")
    a = ap.parse_args()

    small = load_pool(a.small)
    big = load_pool(a.big) if a.big else None
    qids = sorted(small if big is None else set(small) & set(big))
    if a.train_qids:
        want = {l.strip() for l in Path(a.train_qids).read_text().splitlines() if l.strip()}
        tr = [q for q in qids if q in want]
        te = [q for q in qids if q not in want]
        mode = f"qids file {a.train_qids}"
    else:
        tr, te = split_by_hash(qids, (a.train_frac, 1 - a.train_frac))
        mode = f"PRELIMINARY hash split {a.train_frac:.2f}/{1 - a.train_frac:.2f}"
    orders = make_orders(min(r.n_samples for r in small.values()), 1, seed=7)

    print(f"\n=== PER-ACTION ESTIMATOR ({mode}) ===")
    print(f"  pool {a.small}  n={len(qids)}   fit n={len(tr)}   held-out n={len(te)}")
    print(f"  k={a.k}  l2={a.l2}  folds={a.folds}  calibrate={not a.no_calibrate}")

    report = {"small": a.small, "big": a.big, "mode": mode, "n_fit": len(tr),
              "n_held_out": len(te), "k": a.k, "l2": a.l2, "ablation": {}}
    best = None
    for groups in (("qonly",), ("greedy",), ("qonly", "greedy")):
        est = PerActionEstimator(groups=groups, k=a.k, l2=a.l2, n_folds=a.folds,
                                 calibrate=not a.no_calibrate).fit(tr, small, big)
        ev = evaluate(est, te, small, big, orders, a.k)
        report["ablation"]["+".join(groups)] = {
            "features": feature_names(groups), "held_out": ev}
        print(f"\n  features = {'+'.join(groups):15} ({len(feature_names(groups))} cols)")
        print(f"    {'action':12} {'n':>5} {'base':>6} {'pred':>6} {'ECE':>7} {'Brier':>7} {'AUROC':>7}")
        for act, m in ev.items():
            print(f"    {act:12} {m['n']:5d} {m['base_rate']:6.3f} {m['mean_pred']:6.3f} "
                  f"{m['ece']:7.3f} {m['brier']:7.3f} {m['auroc']:7.3f}")
        if groups == ("qonly", "greedy"):
            best = est

    # the headline of the ablation, stated once so it cannot be read backwards
    try:
        q = report["ablation"]["qonly"]["held_out"]
        f = report["ablation"]["qonly+greedy"]["held_out"]
        print("\n  greedy turn adds (AUROC, held out):")
        for act in SERVING_ACTIONS:
            if act in q and act in f:
                print(f"    {act:12} {q[act]['auroc']:.3f} -> {f[act]['auroc']:.3f}"
                      f"   ({f[act]['auroc'] - q[act]['auroc']:+.3f})")
    except KeyError:
        pass

    if a.out and best is not None:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(best.to_json())
        print(f"\n  -> {a.out}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=2))
        print(f"  -> {a.json}")


if __name__ == "__main__":
    main()
