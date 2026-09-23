"""P3 main step -- collect the real compute tree, then check the Gate 2 (E0 oracle).

This is the expensive step (docs/PHASES.md). Run p3_pilot.py first to sanity
check throughput before committing to the full size here.

Usage:
    python p3_collect.py --model Qwen/Qwen2.5-1.5B-Instruct --n-problems 40

Output:
    outputs/tree.json          -- the saved ComputeTree (reusable by P4+)
    Then prints the same oracle-frontier table as
    `python experiments/run_e0_oracle.py --tree outputs/tree.json`,
    plus an explicit PASS/FAIL against the pre-registered Gate 2
    (oracle beats the best P1 baseline by >= 20% cost at iso-accuracy).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-problems", type=int, default=40)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--n-rollouts", type=int, default=4)
    p.add_argument("--max-rollout-segments", type=int, default=6)
    p.add_argument("--segment-tokens", type=int, default=64)
    p.add_argument("--verifier", choices=["none", "self"], default="none",
                    help="none = skip VERIFY branch entirely (recommended while "
                         "the real PRM is still P4 debt, see reports/P1_baselines.md); "
                         "self = SelfVerifier (real cost charged, but score is a "
                         "known placeholder -- do not trust VOC(Verify) numbers yet).")
    p.add_argument("--out", default="outputs/tree.json")
    # Gate 2 threshold, mirrors configs/experiments/e0_oracle.yaml.
    p.add_argument("--min-oracle-gap", type=float, default=0.20)
    # Best baseline cost @ iso-accuracy from P1 (reports/P1_baselines.md, n=5 --
    # PRELIMINARY. Re-measure with a proper n before trusting the gate decision.)
    p.add_argument("--baseline-cost", type=float, default=None,
                    help="best baseline cost at the accuracy level you gate on. "
                         "If omitted, the gate check is skipped and only the "
                         "oracle frontier is printed.")
    args = p.parse_args()

    from caac.backends.vllm import VLLMBackend
    from caac.core.budget import lambda_grid
    from caac.data.benchmarks import load_gsm8k
    from caac.data.compute_tree import oracle_frontier
    from caac.data.collect import CollectionConfig, collect_tree
    from caac.types import CostWeights
    from caac.utils.logging import setup_logging

    setup_logging("INFO")

    print(f"Loading backend: {args.model} ...")
    backend = VLLMBackend(model_name=args.model)

    verifier = None
    if args.verifier == "self":
        from caac.verifier.self_verify import SelfVerifier
        verifier = SelfVerifier(backend=backend)
        print("WARNING: SelfVerifier score is a placeholder (constant 0.5) -- "
              "VOC(Verify) values from this tree are not meaningful yet.")

    problems = load_gsm8k(n=args.n_problems)
    print(f"Loaded {len(problems)} GSM8K problems.")

    cfg = CollectionConfig(
        segment_tokens=args.segment_tokens,
        max_depth=args.max_depth,
        n_rollouts=args.n_rollouts,
        max_rollout_segments=args.max_rollout_segments,
        verify_enabled=(args.verifier != "none"),
    )

    print(
        f"Collecting tree: n_problems={args.n_problems} max_depth={args.max_depth} "
        f"n_rollouts={args.n_rollouts} verifier={args.verifier}"
    )
    t0 = time.time()
    tree = collect_tree(backend, problems, verifier=verifier, config=cfg)
    elapsed = time.time() - t0

    out_path = Path(args.out)
    tree.save(out_path)
    print(f"\nCollected {len(tree)} nodes in {elapsed/60:.1f} min "
          f"(~{elapsed/3600:.2f} GPU-h). Saved to {out_path}")

    # -- E0 oracle frontier, same code path as experiments/run_e0_oracle.py ----
    weights = CostWeights(tokens=1.0)
    frontier = oracle_frontier(tree, lambda_grid(1e-3, 10.0, 12), weights)

    print(f"\n{'lambda':>10} {'accuracy':>10} {'cost':>10}")
    for row in frontier:
        print(f"{row['lambda']:>10.4f} {row['accuracy']:>10.4f} {row['cost']:>10.2f}")

    result = {
        "n_problems": args.n_problems,
        "n_nodes": len(tree),
        "elapsed_s": elapsed,
        "frontier": frontier,
    }

    # -- Gate 2 check -----------------------------------------------------
    if args.baseline_cost is not None:
        # Find the oracle point with accuracy >= the baseline's accuracy band,
        # take the cheapest such point, and compare cost.
        best_baseline_cost = args.baseline_cost
        candidates = [r for r in frontier if r["cost"] <= best_baseline_cost]
        if candidates:
            best = max(candidates, key=lambda r: r["accuracy"])
            gap = 1.0 - (best["cost"] / best_baseline_cost) if best_baseline_cost else 0.0
            passed = gap >= args.min_oracle_gap
            print(f"\nGate 2 (oracle gap >= {args.min_oracle_gap:.0%}):")
            print(f"  oracle cost at acc>={best['accuracy']:.3f}: {best['cost']:.2f}")
            print(f"  baseline cost:                            {best_baseline_cost:.2f}")
            print(f"  oracle gap: {gap:.1%}  ->  {'PASS' if passed else 'FAIL'}")
            result["gate2_gap"] = gap
            result["gate2_pass"] = passed
        else:
            print("\nGate 2: no oracle point found at or below baseline cost -- "
                  "check tree size / baseline_cost value.")
            result["gate2_pass"] = False
    else:
        print("\nGate 2: skipped (pass --baseline-cost to check against P1's "
              "strongest baseline, see reports/P1_baselines.md).")

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/p3_collect_result.json").write_text(json.dumps(result, indent=2, default=str))
    print("Summary written to outputs/p3_collect_result.json")


if __name__ == "__main__":
    main()
