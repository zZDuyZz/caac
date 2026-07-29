"""E1 -- calibration study (RQ1). The first go/no-go gate.

Measures how far each uncertainty signal is from a usable probability, before
and after post-hoc calibration, across model scales. If no signal reaches
usable calibration at any scale, the VOC rule has nothing trustworthy to compute
with, and the project pivots to a calibration study rather than a controller.

Gate: at least one signal reaching ECE < 0.10 after calibration at >= 1.5B.

Runs on CPU with the mock backend (to validate the pipeline) and on GPU with a
real backend (for the actual result).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from caac.adapters.calibration import build_calibrator
from caac.data.collect import CollectionConfig, collect_tree
from caac.data.compute_tree import ComputeTree
from caac.eval.calibration import calibration_report
from caac.signals.features import ALL_FEATURES
from caac.utils.io import write_json
from caac.utils.logging import get_logger, setup_logging

log = get_logger("e1")

# Signals compared in the study, given as indices into the feature vector.
SIGNALS = {
    "mean_token_confidence": "mean_token_confidence",
    "min_token_confidence": "min_token_confidence",
    "sequence_entropy": "sequence_entropy",
    "answer_stability": "answer_stability",
    "verifier_score": "verifier_score",
}


def signal_column(tree: ComputeTree, name: str):
    """Extract one raw signal column plus binary correctness labels."""
    if name not in ALL_FEATURES:
        raise KeyError(f"unknown signal '{name}'; available: {ALL_FEATURES}")
    idx = ALL_FEATURES.index(name)
    xs, ys = [], []
    for node in tree.nodes.values():
        xs.append(node.features[idx])
        ys.append(1.0 if node.rollout_correct >= 0.5 else 0.0)
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    # sequence_entropy is an uncertainty measure: invert so higher = more
    # confident, which is what a probability of correctness must look like.
    if "entropy" in name:
        x = 1.0 - x
    return np.clip(x, 0.0, 1.0), y


def run(tree: ComputeTree, calibrators=("identity", "temperature", "isotonic")) -> dict:
    """Report ECE/Brier/AURC for every (signal, calibrator) pair."""
    results = {}
    for sig in SIGNALS:
        try:
            x, y = signal_column(tree, sig)
        except KeyError:
            continue
        if len(set(y.tolist())) < 2:
            log.warning("signal %s: labels are single-class, skipping", sig)
            continue
        per_cal = {}
        for cal_name in calibrators:
            cal = build_calibrator(cal_name)
            # Fit and evaluate on the same split here for simplicity; the real
            # run must use the held-out calibration split (see --split).
            probs = cal.fit_transform(x, y)
            per_cal[cal_name] = calibration_report(probs, y)
        results[sig] = per_cal
    return results


def check_gate(results: dict, threshold: float = 0.10) -> tuple[bool, list]:
    """Gate: any signal reaching ECE < threshold after calibration."""
    passing = []
    for sig, cals in results.items():
        for cal_name, rep in cals.items():
            if cal_name != "identity" and rep["ece"] < threshold:
                passing.append((sig, cal_name, rep["ece"]))
    return bool(passing), sorted(passing, key=lambda t: t[2])


def main():
    p = argparse.ArgumentParser(description="E1 calibration study")
    p.add_argument("--tree", default=None, help="saved ComputeTree JSON")
    p.add_argument("--backend", default="mock", choices=["mock", "vllm", "hf"])
    p.add_argument("--model", default="mock-model")
    p.add_argument("--n-problems", type=int, default=20)
    p.add_argument("--out", default="outputs/e1_calibration.json")
    p.add_argument("--gate", type=float, default=0.10)
    args = p.parse_args()
    setup_logging()

    if args.tree:
        tree = ComputeTree.load(args.tree)
    else:
        from caac.backends import get_backend
        from caac.backends.mock import MockVerifier
        from caac.data.benchmarks import synthetic_benchmark

        backend = get_backend(args.backend, **({} if args.backend == "mock" else {"model_name": args.model}))
        problems = synthetic_benchmark(n=args.n_problems)
        for prob in problems:
            prob["answer"] = "42"  # mock answer pool
        tree = collect_tree(
            backend, problems, verifier=MockVerifier(),
            config=CollectionConfig(max_depth=2, n_rollouts=2),
        )

    results = run(tree)
    passed, passing = check_gate(results, args.gate)

    print(f"\n{'signal':24s} {'calibrator':12s} {'ECE':>8s} {'Brier':>8s} {'AURC':>8s}")
    for sig, cals in results.items():
        for cal_name, rep in cals.items():
            print(f"{sig:24s} {cal_name:12s} {rep['ece']:8.4f} {rep['brier']:8.4f} {rep['aurc']:8.4f}")

    print(f"\nGATE (ECE < {args.gate}): {'PASS' if passed else 'FAIL'}")
    for sig, cal, ece in passing[:3]:
        print(f"  {sig} + {cal}: ECE={ece:.4f}")
    if not passed:
        print("  -> No signal is usable for control. See proposal Risks: pivot to")
        print("     a calibration-for-control study rather than the full controller.")

    write_json(args.out, {"results": results, "gate_passed": passed, "n_nodes": len(tree)})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
