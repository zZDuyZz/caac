"""P3.1 -- pilot run.

Collects a *tiny* compute tree (few problems, shallow depth, few rollouts) to
measure real wall-clock throughput on this GPU, then extrapolates the cost of
the full P3 collection run so you can decide the tree size before committing
GPU-hours to it.

Usage:
    python p3_pilot.py --model Qwen/Qwen2.5-1.5B-Instruct --n-problems 3

This does NOT save a usable tree -- it's a timing probe. Run p3_collect.py for
the real thing once the extrapolated cost looks acceptable.
"""

from __future__ import annotations

import argparse
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-problems", type=int, default=3)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--n-rollouts", type=int, default=2)
    p.add_argument("--max-rollout-segments", type=int, default=3)
    p.add_argument("--segment-tokens", type=int, default=64)
    # What you actually intend to run in p3_collect.py -- used only to
    # extrapolate the estimate, not to run anything bigger here.
    p.add_argument("--target-n-problems", type=int, default=40)
    p.add_argument("--target-max-depth", type=int, default=3)
    p.add_argument("--target-n-rollouts", type=int, default=4)
    args = p.parse_args()

    from caac.backends.vllm import VLLMBackend
    from caac.data.benchmarks import load_gsm8k
    from caac.data.collect import CollectionConfig, collect_tree

    print(f"Loading backend: {args.model} ...")
    t_load0 = time.time()
    backend = VLLMBackend(model_name=args.model)
    t_load1 = time.time()
    print(f"  backend ready in {t_load1 - t_load0:.1f}s (one-time cost, not scaled)")

    problems = load_gsm8k(n=args.n_problems)
    print(f"Loaded {len(problems)} GSM8K problems.")

    cfg = CollectionConfig(
        segment_tokens=args.segment_tokens,
        max_depth=args.max_depth,
        n_rollouts=args.n_rollouts,
        max_rollout_segments=args.max_rollout_segments,
        verify_enabled=False,  # verifier is still MockVerifier-grade; see P4 debt
    )

    print(
        f"Pilot config: n_problems={args.n_problems} max_depth={args.max_depth} "
        f"n_rollouts={args.n_rollouts} (verify disabled)"
    )
    t0 = time.time()
    tree = collect_tree(backend, problems, verifier=None, config=cfg)
    elapsed = time.time() - t0

    n_nodes = len(tree)
    per_problem = elapsed / max(1, args.n_problems)
    print(f"\nPilot done: {n_nodes} nodes, {elapsed:.1f}s total, {per_problem:.1f}s/problem")

    # Rough extrapolation: cost scales ~ n_problems * max_depth * n_rollouts
    # (see CollectionConfig docstring in collect.py).
    def work_units(n_problems, max_depth, n_rollouts):
        return n_problems * max_depth * n_rollouts

    pilot_units = work_units(args.n_problems, args.max_depth, args.n_rollouts)
    target_units = work_units(
        args.target_n_problems, args.target_max_depth, args.target_n_rollouts
    )
    scale = target_units / max(1, pilot_units)
    est_seconds = elapsed * scale
    est_hours = est_seconds / 3600

    print(
        f"\nExtrapolated full run (n_problems={args.target_n_problems}, "
        f"max_depth={args.target_max_depth}, n_rollouts={args.target_n_rollouts}):"
    )
    print(f"  estimated wall-clock: {est_seconds/60:.1f} min (~{est_hours:.2f} GPU-h)")
    print(
        "  NOTE: this scales rollout *count*, not tree branching factor exactly -- "
        "treat as an order-of-magnitude check, not a firm budget."
    )
    if est_hours > 3:
        print(
            "  -> This exceeds a comfortable single sitting on a rented GPU. "
            "Consider lowering --target-n-problems or --target-n-rollouts."
        )


if __name__ == "__main__":
    main()
