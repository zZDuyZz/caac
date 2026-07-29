"""E6 -- deployment-profile sweep.

Keeps every estimator fixed and only changes the cost weights, then checks that
the operating point moves in the predicted direction:

  * latency-critical    -> lower P95, fewer BRANCH actions (branching costs
                           memory and hurts tail latency)
  * throughput-critical -> more willing to branch when memory allows

The experiment is cheap, since nothing is retrained, but it tests a practical
claim of the framework directly: one controller serves several deployment
targets by configuration alone.
"""

from __future__ import annotations

import argparse
from collections import Counter

from caac.adapters.correctness import SignalAdapter
from caac.core.controller import CAACController, ControllerConfig
from caac.core.voc import VOCPolicy
from caac.cost.accounting import CostMeter, DEPLOYMENT_PROFILES
from caac.eval.answer_match import answers_match
from caac.policy.cost import AnalyticCost
from caac.policy.gain import HeuristicGain
from caac.signals.features import FeatureExtractor
from caac.utils.io import write_json
from caac.utils.logging import setup_logging


def run_profile(backend, verifier, problems, profile, lam, budget):
    """Run the same controller under one cost profile."""
    weights = DEPLOYMENT_PROFILES[profile]
    policy = VOCPolicy(gain=HeuristicGain(), cost=AnalyticCost(), weights=weights,
                       lam=lam, max_branches=2, max_verify=2)
    ctrl = CAACController(
        backend=backend, policy=policy, adapter=SignalAdapter(),
        features=FeatureExtractor(), weights=weights, verifier=verifier,
        config=ControllerConfig(initial_budget=budget),
    )
    correct, total, actions = 0, CostMeter(), Counter()
    for prob in problems:
        state, meter, decisions = ctrl.run(prob["id"], prob["question"])
        res = ctrl.finalise(state, meter, decisions, prob["answer"], answers_match)
        correct += int(res.correct)
        total.merge(meter)
        actions.update(d.action.value for d in decisions)
    n = len(problems)
    n_dec = sum(actions.values()) or 1
    return {
        "profile": profile,
        "accuracy": correct / n,
        "cost": total.total().scalar(weights) / n,
        "tokens": total.total().tokens / n,
        "latency_s": total.total().latency_s / n,
        "branch_rate": actions["branch"] / n_dec,
        "verify_rate": actions["verify"] / n_dec,
        "actions": dict(actions),
    }


def main():
    p = argparse.ArgumentParser(description="E6 deployment-profile sweep")
    p.add_argument("--backend", default="mock", choices=["mock", "vllm", "hf"])
    p.add_argument("--model", default="mock-model")
    p.add_argument("--n-problems", type=int, default=20)
    p.add_argument("--lam", type=float, default=5e-4)
    p.add_argument("--budget", type=float, default=4096)
    p.add_argument("--out", default="outputs/e6_profile.json")
    args = p.parse_args()
    setup_logging()

    from caac.backends import get_backend
    from caac.backends.mock import MockVerifier
    from caac.data.benchmarks import synthetic_benchmark

    backend = get_backend(args.backend,
                          **({} if args.backend == "mock" else {"model_name": args.model}))
    verifier = MockVerifier() if args.backend == "mock" else None
    problems = synthetic_benchmark(n=args.n_problems)
    for prob in problems:
        prob["answer"] = "42"

    rows = [run_profile(backend, verifier, problems, prof, args.lam, args.budget)
            for prof in ("token_only", "throughput", "latency")]

    print(f"\n{'profile':>14s} {'acc':>6s} {'tokens':>8s} {'latency':>9s} {'branch%':>8s} {'verify%':>8s}")
    for r in rows:
        print(f"{r['profile']:>14s} {r['accuracy']:6.3f} {r['tokens']:8.0f} "
              f"{r['latency_s']:9.4f} {r['branch_rate']:8.1%} {r['verify_rate']:8.1%}")

    print("\nNote: no estimator was retrained; only the cost weights changed.")
    write_json(args.out, {"rows": rows})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
