"""E2 -- accuracy/cost Pareto sweep (the main result).

Sweeps the exchange rate lambda for CAAC and the tuning knob of every baseline,
runs all of them through the same backend and the same CostMeter, and reports
the frontier comparison: cost@iso-accuracy and accuracy@iso-cost.

The accounting rule is the point. Every method is charged for everything it
uses, CAAC included: its controller and any verifier calls are counted in the
same budget as reasoning tokens. A method that looks cheap only because its
overhead was omitted cannot hide here.
"""

from __future__ import annotations

import argparse

from caac.baselines import BASELINE_REGISTRY, build_baseline
from caac.core.budget import lambda_grid
from caac.core.controller import CAACController, ControllerConfig
from caac.core.voc import VOCPolicy
from caac.cost.accounting import CostMeter, DEPLOYMENT_PROFILES
from caac.eval.answer_match import answers_match
from caac.eval.pareto import OperatingPoint, compare_frontiers, pareto_frontier
from caac.policy.cost import AnalyticCost
from caac.policy.gain import HeuristicGain
from caac.signals.features import FeatureExtractor
from caac.utils.io import write_json
from caac.utils.logging import get_logger, setup_logging

log = get_logger("e2")


def run_caac(backend, verifier, adapter, gain, problems, lam, weights, budget):
    """One CAAC operating point at a given lambda."""
    policy = VOCPolicy(
        gain=gain, cost=AnalyticCost(), weights=weights, lam=lam,
        max_branches=2, max_verify=2,
    )
    ctrl = CAACController(
        backend=backend, policy=policy, adapter=adapter,
        features=FeatureExtractor(), weights=weights, verifier=verifier,
        config=ControllerConfig(initial_budget=budget),
    )
    correct = 0
    total = CostMeter()
    for prob in problems:
        state, meter, decisions = ctrl.run(prob["id"], prob["question"])
        res = ctrl.finalise(state, meter, decisions, prob["answer"], answers_match)
        correct += int(res.correct)
        total.merge(meter)
    n = len(problems)
    return OperatingPoint(
        cost=total.total().scalar(weights) / n,
        accuracy=correct / n,
        label=f"caac(lam={lam:.4g})",
    )


def run_baseline(backend, name, problems, weights, budget, **kw):
    """One baseline operating point at a given setting."""
    method = build_baseline(name, **kw)
    correct = 0
    total = CostMeter()
    for prob in problems:
        meter = CostMeter()
        res = method.run(backend, prob["question"], budget, meter)
        correct += int(answers_match(res.answer, prob["answer"]))
        total.merge(meter)
    n = len(problems)
    setting = ",".join(f"{k}={v}" for k, v in kw.items()) or "default"
    return OperatingPoint(
        cost=total.total().scalar(weights) / n,
        accuracy=correct / n,
        label=f"{name}({setting})",
    )


# Sweep knobs: each baseline needs its own curve, not a single point.
BASELINE_SWEEPS = {
    "greedy_cot": [{"max_segments": n} for n in (2, 4, 6, 8)],
    "self_consistency": [{"k": k} for k in (2, 4, 8)],
    "budget_forcing": [{"target_segments": n} for n in (2, 4, 6, 8)],
    "deer": [{"threshold": t} for t in (0.5, 0.7, 0.85, 0.95)],
    "deepconf": [{"k": k, "conf_threshold": 0.45} for k in (2, 4, 8)],
}


def main():
    p = argparse.ArgumentParser(description="E2 Pareto sweep")
    p.add_argument("--backend", default="mock", choices=["mock", "vllm", "hf"])
    p.add_argument("--model", default="mock-model")
    p.add_argument("--n-problems", type=int, default=20)
    p.add_argument("--profile", default="token_only", choices=sorted(DEPLOYMENT_PROFILES))
    p.add_argument("--budget", type=float, default=4096)
    p.add_argument("--n-lambda", type=int, default=8)
    p.add_argument("--out", default="outputs/e2_pareto.json")
    args = p.parse_args()
    setup_logging()

    from caac.adapters.correctness import SignalAdapter
    from caac.backends import get_backend
    from caac.backends.mock import MockVerifier
    from caac.data.benchmarks import synthetic_benchmark

    weights = DEPLOYMENT_PROFILES[args.profile]
    backend = get_backend(
        args.backend, **({} if args.backend == "mock" else {"model_name": args.model})
    )
    verifier = MockVerifier() if args.backend == "mock" else None
    problems = synthetic_benchmark(n=args.n_problems)
    for prob in problems:
        prob["answer"] = "42"

    # CAAC frontier
    caac_pts = []
    for lam in lambda_grid(1e-5, 1e-2, args.n_lambda):
        pt = run_caac(
            backend, verifier, SignalAdapter(), HeuristicGain(),
            problems, lam, weights, args.budget,
        )
        caac_pts.append(pt)
        log.info("caac lam=%.5g  acc=%.3f  cost=%.0f", lam, pt.accuracy, pt.cost)

    # Baseline frontiers
    baseline_pts = {}
    for name, settings in BASELINE_SWEEPS.items():
        pts = []
        for kw in settings:
            pt = run_baseline(backend, name, problems, weights, args.budget, **kw)
            pts.append(pt)
            log.info("%s %s  acc=%.3f  cost=%.0f", name, kw, pt.accuracy, pt.cost)
        baseline_pts[name] = pts

    # Report
    print(f"\n=== CAAC frontier (profile={args.profile}) ===")
    for pt in pareto_frontier(caac_pts):
        print(f"  cost={pt.cost:8.1f}  acc={pt.accuracy:.3f}  {pt.label}")

    summary = {}
    for name, pts in baseline_pts.items():
        cmp = compare_frontiers(caac_pts, pts)
        summary[name] = cmp
        ratio = cmp["mean_cost_ratio"]
        verdict = "CAAC cheaper" if ratio < 1 else "baseline cheaper"
        print(f"\nvs {name}: mean cost ratio = {ratio:.3f}  ({verdict})")
        for row in cmp["per_target"]:
            print(f"    @acc>={row['accuracy_target']:.2f}: "
                  f"caac={row['method_cost']:.0f} vs {name}={row['baseline_cost']:.0f} "
                  f"(saving {row['saving']:.1%})")

    write_json(args.out, {
        "profile": args.profile,
        "caac": [pt.__dict__ for pt in caac_pts],
        "baselines": {k: [pt.__dict__ for pt in v] for k, v in baseline_pts.items()},
        "comparison": summary,
    })
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
