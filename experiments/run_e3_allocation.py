"""E3 -- allocation structure (RQ3 / H2).

Records every decision the controller makes and reports how the actions are
distributed across belief bins. The headline output is the curve of
VOC(Verify) against belief, which tests H2:

    VOC(Verify) is non-monotonic in belief -- near zero when the belief is very
    confident (verification cannot change the decision) and also low when the
    belief is very uncertain, peaking in between.

This is the deliverable that threshold-based methods cannot produce: a fixed
threshold yields no structured prediction about when an action is worth taking.
If the curve turns out monotonic, H2 is falsified and that is reported as a
limit of the EVOI argument at small scale.
"""

from __future__ import annotations

import argparse

import numpy as np

from caac.core.controller import CAACController, ControllerConfig
from caac.core.voc import VOCPolicy
from caac.cost.accounting import DEPLOYMENT_PROFILES
from caac.eval.answer_match import answers_match
from caac.policy.cost import AnalyticCost
from caac.policy.gain import HeuristicGain
from caac.signals.features import FeatureExtractor
from caac.types import Action
from caac.utils.io import write_json
from caac.utils.logging import setup_logging


def collect_decisions(backend, verifier, adapter, gain, problems, lam, weights, budget):
    """Run the controller and keep every decision, with its VOC map."""
    policy = VOCPolicy(gain=gain, cost=AnalyticCost(), weights=weights, lam=lam,
                       max_branches=2, max_verify=2)
    ctrl = CAACController(
        backend=backend, policy=policy, adapter=adapter, features=FeatureExtractor(),
        weights=weights, verifier=verifier, config=ControllerConfig(initial_budget=budget),
    )
    records = []
    for prob in problems:
        state, meter, decisions = ctrl.run(prob["id"], prob["question"])
        res = ctrl.finalise(state, meter, decisions, prob["answer"], answers_match)
        for step, d in enumerate(decisions):
            records.append({
                "query_id": prob["id"],
                "step": step,
                "belief": d.belief,
                "action": d.action.value,
                "voc": {a.value: (None if not np.isfinite(v) else float(v))
                        for a, v in d.voc.items()},
                "correct": res.correct,
            })
    return records


def bin_by_belief(records, n_bins=10):
    """Action frequency and mean VOC per action, per belief bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = [r for r in records if lo <= r["belief"] < hi or (b == n_bins - 1 and r["belief"] == 1.0)]
        if not sel:
            continue
        row = {"bin_lo": float(lo), "bin_hi": float(hi), "n": len(sel)}
        for action in Action:
            row[f"freq_{action.value}"] = sum(
                1 for r in sel if r["action"] == action.value
            ) / len(sel)
        for action in Action.spending():
            vals = [r["voc"][action.value] for r in sel if r["voc"].get(action.value) is not None]
            row[f"voc_{action.value}"] = float(np.mean(vals)) if vals else None
        out.append(row)
    return out


def check_h2(bins) -> dict:
    """Is VOC(Verify) non-monotonic with an interior maximum?"""
    xs, ys = [], []
    for row in bins:
        v = row.get("voc_verify")
        if v is not None:
            xs.append((row["bin_lo"] + row["bin_hi"]) / 2)
            ys.append(v)
    if len(ys) < 3:
        return {"testable": False, "reason": "too few populated belief bins"}
    peak = int(np.argmax(ys))
    interior = 0 < peak < len(ys) - 1
    return {
        "testable": True,
        "interior_peak": bool(interior),
        "peak_belief": float(xs[peak]),
        "peak_voc": float(ys[peak]),
        "h2_supported": bool(interior),
    }


def main():
    p = argparse.ArgumentParser(description="E3 allocation structure")
    p.add_argument("--backend", default="mock", choices=["mock", "vllm", "hf"])
    p.add_argument("--model", default="mock-model")
    p.add_argument("--n-problems", type=int, default=30)
    p.add_argument("--lam", type=float, default=5e-4)
    p.add_argument("--profile", default="token_only")
    p.add_argument("--budget", type=float, default=4096)
    p.add_argument("--out", default="outputs/e3_allocation.json")
    args = p.parse_args()
    setup_logging()

    from caac.adapters.correctness import SignalAdapter
    from caac.backends import get_backend
    from caac.backends.mock import MockVerifier
    from caac.data.benchmarks import synthetic_benchmark

    weights = DEPLOYMENT_PROFILES[args.profile]
    backend = get_backend(args.backend,
                          **({} if args.backend == "mock" else {"model_name": args.model}))
    verifier = MockVerifier() if args.backend == "mock" else None
    problems = synthetic_benchmark(n=args.n_problems)
    for prob in problems:
        prob["answer"] = "42"

    records = collect_decisions(backend, verifier, SignalAdapter(), HeuristicGain(),
                                problems, args.lam, weights, args.budget)
    bins = bin_by_belief(records)
    h2 = check_h2(bins)

    print(f"\n{'belief bin':>14s} {'n':>4s} {'stop':>6s} {'cont':>6s} {'verf':>6s} {'brch':>6s} {'VOC(verify)':>12s}")
    for row in bins:
        v = row.get("voc_verify")
        print(f"{row['bin_lo']:.1f}-{row['bin_hi']:.1f}".rjust(14)
              + f" {row['n']:4d} {row['freq_stop']:6.2f} {row['freq_continue']:6.2f}"
              + f" {row['freq_verify']:6.2f} {row['freq_branch']:6.2f}"
              + (f" {v:12.5f}" if v is not None else f" {'-':>12s}"))

    print(f"\nH2 (interior peak in VOC(Verify)): "
          f"{'SUPPORTED' if h2.get('h2_supported') else 'NOT SUPPORTED'}")
    if h2.get("testable"):
        print(f"  peak at belief={h2['peak_belief']:.2f}, VOC={h2['peak_voc']:.5f}")

    write_json(args.out, {"bins": bins, "h2": h2, "n_decisions": len(records)})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
