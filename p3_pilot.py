#!/usr/bin/env python3
"""P3 Step 1 - Pilot: measure REAL collection speed on today's GPU.

    python p3_pilot.py

Uses real GSM8K (never the synthetic '42' set) and MockVerifier (real PRM is P4).
The s/problem it prints decides the size of the gate tree in Step 2.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # fix UUID-device ValueError

import time
from caac.data.collect import collect_tree, CollectionConfig
from caac.data.benchmarks import load_benchmark
from caac.backends.vllm import VLLMBackend
from caac.backends.mock import MockVerifier

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    backend = VLLMBackend(model_name=MODEL)
    verifier = MockVerifier()
    problems = load_benchmark("gsm8k", n=10)          # REAL data

    t0 = time.time()
    tree = collect_tree(
        backend, problems, verifier=verifier,
        config=CollectionConfig(max_depth=2, n_rollouts=2),
    )
    dt = time.time() - t0

    print("\n=== PILOT RESULT ===")
    print(f"10 problems, depth=2, rollout=2 -> {dt:.0f}s total, "
          f"{dt/10:.1f}s/problem, {len(tree)} nodes")
    print("Use s/problem to size Step 2 (aim ~2-3h of GPU for the gate tree).")
    tree.save("outputs/tree_pilot.json")
    print("saved outputs/tree_pilot.json")


if __name__ == "__main__":
    main()
