"""E4 -- deployment under load (phase P7).

The first-class systems experiment: does the compute saved at the reasoning
level turn into a system-level benefit when many requests run together?

Needs a dedicated GPU for the real result. The scheduling and statistics run on
CPU with the mock backend so the harness can be validated first.
"""

from __future__ import annotations

import argparse
import time

from caac.adapters.correctness import SignalAdapter
from caac.baselines import build_baseline
from caac.core.controller import CAACController, ControllerConfig
from caac.core.voc import VOCPolicy
from caac.cost.accounting import CostMeter, DEPLOYMENT_PROFILES
from caac.eval.answer_match import answers_match
from caac.eval.deployment import run_load_test
from caac.policy.cost import AnalyticCost
from caac.policy.gain import HeuristicGain
from caac.signals.features import FeatureExtractor
from caac.utils.io import write_json
from caac.utils.logging import setup_logging


def caac_runner(backend, verifier, weights, lam, budget):
    def run_one(prob):
        policy = VOCPolicy(gain=HeuristicGain(), cost=AnalyticCost(), weights=weights,
                           lam=lam, max_branches=2, max_verify=2)
        ctrl = CAACController(
            backend=backend, policy=policy, adapter=SignalAdapter(),
            features=FeatureExtractor(), weights=weights, verifier=verifier,
            config=ControllerConfig(initial_budget=budget),
        )
        t0 = time.perf_counter()
        state, meter, decisions = ctrl.run(prob["id"], prob["question"])
        res = ctrl.finalise(state, meter, decisions, prob["answer"], answers_match)
        return res.correct, meter, time.perf_counter() - t0
    return run_one


def baseline_runner(backend, name, budget, **kw):
    method = build_baseline(name, **kw)
    def run_one(prob):
        meter = CostMeter()
        t0 = time.perf_counter()
        res = method.run(backend, prob["question"], budget, meter)
        return answers_match(res.answer, prob["answer"]), meter, time.perf_counter() - t0
    return run_one


def main():
    p = argparse.ArgumentParser(description="E4 deployment under load")
    p.add_argument("--backend", default="mock", choices=["mock", "vllm", "hf"])
    p.add_argument("--model", default="mock-model")
    p.add_argument("--n-problems", type=int, default=20)
    p.add_argument("--qps", default="1,2,4,8")
    p.add_argument("--lam", type=float, default=5e-4)
    p.add_argument("--profile", default="latency")
    p.add_argument("--budget", type=float, default=4096)
    p.add_argument("--compare", default="deer", help="baseline to compare against")
    p.add_argument("--out", default="outputs/e4_deployment.json")
    args = p.parse_args()
    setup_logging()

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
    qps_levels = tuple(float(x) for x in args.qps.split(","))

    runs = {
        "caac": run_load_test(caac_runner(backend, verifier, weights, args.lam, args.budget),
                              problems, qps_levels=qps_levels, weights=weights),
        args.compare: run_load_test(baseline_runner(backend, args.compare, args.budget),
                                    problems, qps_levels=qps_levels, weights=weights),
    }

    print(f"\n{'method':>10s} {'qps':>5s} {'p50(s)':>8s} {'p95(s)':>8s} "
          f"{'thr/s':>7s} {'cost':>7s} {'ovh%':>6s} {'ctrl%':>6s} {'acc':>6s}")
    for name, rows in runs.items():
        for r in rows:
            print(f"{name:>10s} {r.qps:5.1f} {r.latency_p50:8.3f} {r.latency_p95:8.3f} "
                  f"{r.throughput:7.2f} {r.mean_cost:7.0f} {r.overhead_ratio:6.1%} "
                  f"{r.controller_ratio:6.1%} {r.accuracy:6.2f}")

    write_json(args.out, {k: [r.__dict__ for r in v] for k, v in runs.items()})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
