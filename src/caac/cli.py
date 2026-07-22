"""Command-line interface: caac solve / calibrate / evaluate.

A thin wrapper over the public API so non-coders can run the framework from a
terminal. Defaults to the CPU mock backend; pass --backend vllm for GPU (P1).
"""

from __future__ import annotations

import argparse
import sys

from caac.utils.logging import setup_logging


def _cmd_solve(args) -> int:
    from caac.api import CAAC

    controller = CAAC.from_pretrained(
        args.model, backend=args.backend, profile=args.profile
    )
    answer = controller.solve(args.query, budget=args.budget)
    print(f"answer: {answer}")
    if hasattr(controller, "last_meter"):
        s = controller.last_meter.summary(controller.weights)
        print(f"cost: {s['total_scalar']:.1f}  overhead: {s['overhead_ratio']:.1%}"
              f"  controller: {s['controller_ratio']:.1%}")
    if hasattr(controller, "last_decisions"):
        from collections import Counter
        counts = Counter(d.action.value for d in controller.last_decisions)
        print("actions: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


def _cmd_evaluate(args) -> int:
    from caac.api import CAAC
    from caac.data.benchmarks import synthetic_benchmark
    from caac.eval.answer_match import answers_match

    controller = CAAC.from_pretrained(args.model, backend=args.backend)
    items = synthetic_benchmark(n=args.n)
    correct = 0
    for it in items:
        ans = controller.solve(it["question"], budget=args.budget, query_id=it["id"])
        correct += int(answers_match(ans, it["answer"]))
    print(f"accuracy: {correct}/{len(items)} = {correct / len(items):.1%}")
    return 0


def _cmd_calibrate(args) -> int:
    print("calibrate: wire an adapter to a labelled set; see docs/quickstart.md")
    return 0


def main(argv=None) -> int:
    setup_logging()
    p = argparse.ArgumentParser(prog="caac", description="Cost-Aware Adaptive Computation")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("solve", help="solve one query")
    s.add_argument("query")
    s.add_argument("--model", default="mock-model")
    s.add_argument("--backend", default="mock")
    s.add_argument("--profile", default="token_only")
    s.add_argument("--budget", type=float, default=2048.0)
    s.set_defaults(func=_cmd_solve)

    e = sub.add_parser("evaluate", help="evaluate on a benchmark")
    e.add_argument("--model", default="mock-model")
    e.add_argument("--backend", default="mock")
    e.add_argument("--n", type=int, default=20)
    e.add_argument("--budget", type=float, default=2048.0)
    e.set_defaults(func=_cmd_evaluate)

    c = sub.add_parser("calibrate", help="calibrate the adapter for a model")
    c.add_argument("--model", default="mock-model")
    c.set_defaults(func=_cmd_calibrate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
