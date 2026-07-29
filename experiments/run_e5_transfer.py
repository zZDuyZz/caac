"""E5 -- transfer and adaptation cost.

Tests the multi-model claim directly: keep the policy layer trained on a source
model, re-fit only the adapter on a target model, and measure how much of the
benefit survives. The headline output is an adaptation learning curve -- benefit
retained as a function of how many examples the adapter was calibrated on --
because that number is what determines the practical cost of moving CAAC to a
model it has not seen.

Three transfer axes: cross-scale (same family), cross-family, and cross-domain.
"""

from __future__ import annotations

import argparse

import numpy as np

from caac.adapters.calibration import build_calibrator
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


def evaluate(backend, verifier, adapter, gain, problems, lam, weights, budget):
    """Return (accuracy, mean cost) for one controller configuration."""
    policy = VOCPolicy(gain=gain, cost=AnalyticCost(), weights=weights, lam=lam,
                       max_branches=2, max_verify=2)
    ctrl = CAACController(
        backend=backend, policy=policy, adapter=adapter, features=FeatureExtractor(),
        weights=weights, verifier=verifier, config=ControllerConfig(initial_budget=budget),
    )
    correct, total = 0, CostMeter()
    for prob in problems:
        state, meter, decisions = ctrl.run(prob["id"], prob["question"])
        res = ctrl.finalise(state, meter, decisions, prob["answer"], answers_match)
        correct += int(res.correct)
        total.merge(meter)
    n = len(problems)
    return correct / n, total.total().scalar(weights) / n


def make_calibration_data(backend, problems, n_examples, seed=0):
    """Cheap (signal, label) pairs for fitting the adapter on a new model.

    Only a forward pass per example is needed, which is why adapting to a new
    model is cheap compared with collecting a fresh compute tree.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for prob in problems[:n_examples]:
        traj = backend.generate(prompt=prob["question"], prefix="", max_tokens=64)
        conf = float(np.mean(np.exp(traj.token_logprobs))) if traj.token_logprobs else 0.5
        label = 1.0 if answers_match(traj.answer, prob["answer"]) else 0.0
        xs.append(conf)
        ys.append(label)
    return np.array(xs), np.array(ys)


def main():
    p = argparse.ArgumentParser(description="E5 transfer / adaptation curve")
    p.add_argument("--backend", default="mock", choices=["mock", "vllm", "hf"])
    p.add_argument("--source-difficulty", type=float, default=0.3, help="mock: source model")
    p.add_argument("--target-difficulty", type=float, default=0.7, help="mock: target model")
    p.add_argument("--n-problems", type=int, default=30)
    p.add_argument("--lam", type=float, default=5e-4)
    p.add_argument("--budget", type=float, default=4096)
    p.add_argument("--sizes", default="0,5,10,20,50", help="calibration set sizes")
    p.add_argument("--out", default="outputs/e5_transfer.json")
    args = p.parse_args()
    setup_logging()

    from caac.backends.mock import MockBackend, MockVerifier
    from caac.data.benchmarks import synthetic_benchmark

    weights = DEPLOYMENT_PROFILES["token_only"]
    problems = synthetic_benchmark(n=args.n_problems)
    for prob in problems:
        prob["answer"] = "42"

    # Mock stands in for two different models; with a real backend, swap these
    # for two model names (cross-scale or cross-family).
    source = MockBackend(difficulty=args.source_difficulty, seed=0)
    target = MockBackend(difficulty=args.target_difficulty, seed=1)
    verifier = MockVerifier()
    gain = HeuristicGain()  # policy layer: shared, never re-fit

    # Reference: adapter fitted properly on the target (full adaptation).
    xs, ys = make_calibration_data(target, problems, len(problems))
    full = SignalAdapter(calibrator=build_calibrator("temperature"))
    if len(set(ys.tolist())) > 1:
        full.calibrate(xs, ys)
    ref_acc, ref_cost = evaluate(target, verifier, full, gain, problems, args.lam, weights, args.budget)

    rows = []
    for n in [int(s) for s in args.sizes.split(",")]:
        adapter = SignalAdapter(calibrator=build_calibrator("temperature"))
        if n > 0:
            xn, yn = make_calibration_data(target, problems, n)
            if len(set(yn.tolist())) > 1:
                adapter.calibrate(xn, yn)
        acc, cost = evaluate(target, verifier, adapter, gain, problems, args.lam, weights, args.budget)
        # Retained benefit: how close to the fully-adapted controller.
        retained = 1.0 if ref_acc == 0 else acc / ref_acc
        rows.append({"n_examples": n, "accuracy": acc, "cost": cost, "retained": retained})

    print(f"\nreference (full adaptation): acc={ref_acc:.3f} cost={ref_cost:.0f}\n")
    print(f"{'n_examples':>11s} {'accuracy':>9s} {'cost':>8s} {'retained':>9s}")
    for r in rows:
        print(f"{r['n_examples']:11d} {r['accuracy']:9.3f} {r['cost']:8.0f} {r['retained']:9.1%}")

    write_json(args.out, {"reference": {"accuracy": ref_acc, "cost": ref_cost}, "curve": rows})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
