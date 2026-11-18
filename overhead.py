"""How heavy is the controller itself? (MASTER 12.7m C, second tier)

    python overhead.py --train-small T_s --train-big T_b --pool-small P_s --pool-big P_b \
        --family caac_staged --lam 5 --rho 0.8 --tau 0.9 --json out/overhead.json

Measured on CPU, one thread of this process:
  1. fit time: everything CAAC learns on TRAIN, once (amortised over all queries);
  2. bundle size: the fitted estimators serialised to JSON, per component;
  3. decision time per question for ONE configuration, every estimator cache
     cleared before each question so every call is cold. The replay also builds
     the cost ledger from the pool, which a deployed controller does not do, so
     this is an UPPER bound on the decision overhead;
  4. the reference it is compared with: the GPU time the same question costs,
     amortised, from the pool manifests' measured throughput (decode tokens per
     second of each pool's whole run). Without a manifest it is left out;
  5. which inputs are free (already produced by a paid step) and which cost
     extra: the question-only hidden-state probe is one more prefill of the
     question on the small model, priced here in small-model tokens.
Nothing here enters the certificate; it is the "is the controller cheap" table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from caac.pool.estimator import FEATURE_GROUPS, POST_FEATURES, ESC_FEATURES
from caac.pool.protocol import build_table, summarize
from caac.pool.schema import load_pool
from certify import CAAC, caac_family, fit_estimators

FEATURE_COST = {
    "question (prompt length)": ("free", "known before anything runs"),
    "greedy turn (" + ", ".join(FEATURE_GROUPS["greedy"]) + ")":
        ("free", "read from the greedy answer, which every policy pays for first"),
    "after sampling (" + ", ".join(POST_FEATURES) + ")":
        ("free", "read from the samples the step just paid for"),
    "after escalating (" + ", ".join(ESC_FEATURES) + ")":
        ("free", "read from the big model's answer the step just paid for"),
    "question-only hidden state (router, A4 probe)":
        ("extra", "one more prefill of the question on the small model"),
}


def _json_size(obj) -> int | None:
    """Bytes of the JSON a fitted object serialises to; None if it has no serialiser."""
    inner = getattr(obj, "fn", obj)              # Memo wraps the estimator
    if hasattr(inner, "to_json"):
        return len(inner.to_json().encode())
    if hasattr(inner, "to_dict"):
        return len(json.dumps(inner.to_dict()).encode())
    if getattr(inner, "head", None) is not None:  # EscalateGate: head + calibrator
        d = {"head": inner.head.to_dict(), "cal": inner.cal.to_dict() if inner.cal else None}
        return len(json.dumps(d).encode())
    return None


def _clear(ests) -> None:
    for name in ("stage1", "stage2", "gate", "recovery", "state", "cost"):
        obj = getattr(ests, name, None)
        for o in (obj, getattr(obj, "fn", None)):
            for attr in ("cache", "_cache"):
                c = getattr(o, attr, None)
                if isinstance(c, dict):
                    c.clear()


def _throughput(pool_path: str) -> float | None:
    m = Path(str(pool_path) + ".manifest.json")
    if not m.exists():
        return None
    thr = json.loads(m.read_text()).get("throughput") or {}
    tps = thr.get("tok_per_s")
    return float(tps) if tps else None


def pick_config(fam, rho, tau) -> int:
    """Index of (rho, tau) in the family; nearest point when it is not on the grid."""
    best, bi = None, 0
    for i, cfg in enumerate(fam.configs):
        r, t = cfg if isinstance(cfg, tuple) else (None, cfg)
        d = abs((r if r is not None else rho) - rho) + abs(t - tau)
        if best is None or d < best:
            best, bi = d, i
    return bi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-small", required=True)
    ap.add_argument("--train-big", required=True)
    ap.add_argument("--pool-small", required=True)
    ap.add_argument("--pool-big", required=True)
    ap.add_argument("--family", default="caac_staged", choices=[c for c in CAAC if c != "caac"])
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=1000, help="questions timed (first N of the pool)")
    ap.add_argument("--k-steps", default="2,4,8,16")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tr_s, tr_b = load_pool(a.train_small), load_pool(a.train_big)
    po_s, po_b = load_pool(a.pool_small), load_pool(a.pool_big)
    k_steps = tuple(int(x) for x in a.k_steps.split(","))
    big_model = next(iter(po_b.values())).model

    t0 = time.perf_counter()
    ests = fit_estimators(tr_s, tr_b, k_steps)
    fit_s = time.perf_counter() - t0
    sizes = {nm: _json_size(getattr(ests, nm)) for nm in ("stage1", "stage2", "gate", "recovery",
                                                          "state", "cost")}

    fam = caac_family(a.family, ests, a.lam, big_model, k_steps)
    j = pick_config(fam, a.rho, a.tau)
    fn = fam.make(fam.configs[j])
    qids = sorted(po_s)[: a.n]
    per_q = []
    for q in qids:
        _clear(ests)
        s, b = po_s[q], po_b.get(q)
        order = list(range(s.n_samples))
        t1 = time.perf_counter()
        fn(s, b, order)
        per_q.append(time.perf_counter() - t1)
    ms = np.asarray(per_q) * 1000.0

    # amortised GPU time of the same questions under the same configuration
    _clear(ests)
    one = type(fam)(fam.name, [fam.configs[j]], fam.make, None, fam.needs_big, fam.uses_order)
    tab = build_table(one, qids, po_s, po_b)
    summ = summarize(tab, None, 0, a.lam)
    tps_s, tps_b = _throughput(a.pool_small), _throughput(a.pool_big)
    gpu_ms = None
    if tps_s and tps_b:
        gpu_ms = 1000.0 * (summ["tok_small"] / tps_s + summ["tok_big"] / tps_b)
    probe_tokens = float(np.mean([po_s[q].prompt_tokens for q in qids]))

    res = {
        "family": a.family, "config": list(fam.configs[j]) if isinstance(fam.configs[j], tuple)
        else fam.configs[j], "lam": a.lam, "n_questions": len(qids),
        "fit_seconds": fit_s, "bundle_bytes": sizes,
        "bundle_bytes_total": sum(v for v in sizes.values() if v),
        "decision_ms": {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
                        "mean": float(ms.mean()), "upper_bound": True},
        "cost_tokens_per_question": {"small": summ["tok_small"], "big": summ["tok_big"],
                                     "lam_cost": summ["cost"]},
        "gpu_ms_per_question_amortised": gpu_ms,
        "throughput_tok_per_s": {"small": tps_s, "big": tps_b},
        "overhead_share_of_gpu_time": (float(np.median(ms)) / gpu_ms) if gpu_ms else None,
        "probe_prefill_tokens_per_question": probe_tokens,
        "probe_share_of_cost": probe_tokens / summ["cost"] if summ["cost"] else None,
        "feature_cost": {k: {"cost": v[0], "why": v[1]} for k, v in FEATURE_COST.items()},
    }
    print(f"controller overhead: {a.family} config {res['config']} at lam={a.lam:g}, "
          f"{len(qids)} questions")
    print(f"  fit once on train      : {fit_s:.1f} s")
    print(f"  bundle (JSON)          : {res['bundle_bytes_total'] / 1024:.1f} KB  "
          + ", ".join(f"{k} {v / 1024:.1f}" for k, v in sizes.items() if v))
    print(f"  decision per question  : median {res['decision_ms']['median']:.2f} ms, "
          f"p95 {res['decision_ms']['p95']:.2f} ms (upper bound: includes replay bookkeeping)")
    if gpu_ms:
        print(f"  GPU per question       : {gpu_ms:.0f} ms amortised -> controller = "
              f"{100 * res['overhead_share_of_gpu_time']:.3f}% of it")
    else:
        print("  GPU per question       : no pool manifest throughput -> not compared")
    print(f"  hidden-state probe     : {probe_tokens:.0f} prefill tokens/question = "
          f"{100 * (res['probe_share_of_cost'] or 0):.1f}% of this config's cost (only if the probe is used)")
    for k, v in FEATURE_COST.items():
        print(f"    {v[0]:5s}  {k}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
