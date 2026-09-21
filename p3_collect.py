#!/usr/bin/env python3
"""P3 Step 2 / Step 4 - Collect a compute tree on REAL GSM8K.

    # gate tree (small, to read the oracle gate):
    python p3_collect.py --n 40  --depth 3 --rollouts 2 --out outputs/tree_gate.json

    # full tree for P4 (only if GATE 1 passes):
    python p3_collect.py --n 120 --depth 3 --rollouts 3 --out outputs/tree_full.json
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
from caac.data.collect import collect_tree, CollectionConfig
from caac.data.benchmarks import load_benchmark
from caac.backends.vllm import VLLMBackend
from caac.backends.mock import MockVerifier

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--out", default="outputs/tree_gate.json")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    backend = VLLMBackend(model_name=args.model)
    verifier = MockVerifier()
    problems = load_benchmark("gsm8k", n=args.n)      # REAL data, explicit

    tree = collect_tree(
        backend, problems, verifier=verifier,
        config=CollectionConfig(max_depth=args.depth, n_rollouts=args.rollouts),
    )
    tree.save(args.out)
    print(f"\n=== SAVED {len(tree)} nodes -> {args.out} ===")


if __name__ == "__main__":
    main()
