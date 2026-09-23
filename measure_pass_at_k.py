"""P3 supplementary check: does a *perfect* verifier have real headroom over
free majority-vote self-consistency, at the SAME generation cost?

Why this exists: the P3 compute-tree oracle (Gate 2) could not measure
VERIFY's value at all -- an oracle already has ground truth, so "verifying"
never helps it, by construction (see PROGRESS.md, 2026-09-23 entry). This
script asks a narrower, answerable question instead: generate k independent
samples per problem (exactly like the self_consistency baseline already
measured), and compare two ways of picking among them --
  (a) self_consistency's free majority vote (already measured tonight), vs
  (b) a PERFECT verifier that always picks a correct sample if one exists
      among the k (the "pass@k" ceiling).
The gap between them is the maximum possible value a real verifier could
ever add, at no extra generation cost (a verifier only adds a small scoring
cost on top of samples you'd generate anyway).

Uses the unbiased pass@k estimator (Chen et al. 2021, the Codex/HumanEval
paper) so ONE run of n=8 samples/problem yields pass@1, pass@2, pass@4 and
pass@8 together, instead of separate runs per k:

    pass@k = E_problems[ 1 - C(n-c, k) / C(n, k) ]

where n = samples drawn per problem, c = how many of those were correct.

Usage:
    python measure_pass_at_k.py --n-problems 40 --n-samples 8

Prints, per k in {1,2,4,8}: pass@k (ceiling), the actual self-consistency
accuracy at that k, the gap between them ("verifier headroom"), and the
generation cost at that k. Writes the same to outputs/pass_at_k.json.

PRE-REGISTERED DECISION RULE (set before running, see PROGRESS.md):
headroom >= 15 percentage points at any k is treated as real (roughly 2x the
sampling noise at n=40 problems, std err ~= sqrt(0.5*0.5/40) ~= 7.9 pts) and
worth pursuing with a real verifier. Below that, treat it as noise / not
worth building a verifier for -- accept the P3 Gate 2 FAIL as the honest
result for this model/dataset/action-set configuration and rescope instead
of re-running more variations hoping for a different answer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. 2021 unbiased pass@k estimator from n samples, c correct."""
    if n - c < k:
        return 1.0
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - prob_all_wrong


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-problems", type=int, default=40)
    p.add_argument("--n-samples", type=int, default=8,
                    help="samples drawn per problem (n in pass@k); pass@1/2/4/8 "
                         "are all derived from this one set of samples")
    p.add_argument("--segment-tokens", type=int, default=64)
    p.add_argument("--max-segments", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8,
                    help="matches SelfConsistency's default so this is a fair "
                         "apples-to-apples comparison against tonight's numbers")
    p.add_argument("--headroom-threshold", type=float, default=0.15,
                    help="pre-registered bar for 'a verifier is worth building'")
    p.add_argument("--out", default="outputs/pass_at_k.json")
    args = p.parse_args()

    from caac.backends.vllm import VLLMBackend
    from caac.baselines.methods import _run_to_completion
    from caac.cost.accounting import CostMeter, DEPLOYMENT_PROFILES
    from caac.data.benchmarks import load_benchmark
    from caac.eval.answer_match import answers_match, normalize_answer
    from caac.utils.logging import setup_logging

    setup_logging("INFO")
    weights = DEPLOYMENT_PROFILES["token_only"]

    print(f"Loading backend: {args.model} ...")
    backend = VLLMBackend(model_name=args.model)
    problems = load_benchmark("gsm8k", n=args.n_problems)
    print(f"Loaded {len(problems)} GSM8K problems. Sampling n={args.n_samples} each...")

    n = args.n_samples
    per_problem_correct: list[int] = []
    per_problem_answers: list[list[str | None]] = []
    total_cost_tokens = 0.0

    for i, prob in enumerate(problems):
        print(f"sampling {i + 1}/{len(problems)} id={prob['id']}", flush=True)
        meter = CostMeter()
        answers: list[str | None] = []
        n_correct = 0
        for _ in range(n):
            traj = _run_to_completion(
                backend, prob["question"], meter, args.segment_tokens, args.max_segments,
                temperature=args.temperature,
            )
            norm = normalize_answer(traj.answer) if traj.answer is not None else None
            answers.append(norm)
            if traj.answer is not None and answers_match(traj.answer, prob["answer"]):
                n_correct += 1
        per_problem_correct.append(n_correct)
        per_problem_answers.append(answers)
        total_cost_tokens += meter.total().scalar(weights)

    avg_cost_per_sample = total_cost_tokens / (len(problems) * n)

    print(f"\n{'k':>3} {'pass@k':>8} {'self_cons':>10} {'headroom':>9} {'cost':>10}")
    results = {}
    for k in [kk for kk in (1, 2, 4, 8) if kk <= n]:
        pass_at_k = sum(unbiased_pass_at_k(n, c, k) for c in per_problem_correct) / len(problems)

        mv_correct = 0
        for prob, answers in zip(problems, per_problem_answers):
            subset = [a for a in answers[:k] if a is not None]
            if not subset:
                continue
            voted, _ = Counter(subset).most_common(1)[0]
            if answers_match(voted, prob["answer"]):
                mv_correct += 1
        mv_acc = mv_correct / len(problems)
        headroom = pass_at_k - mv_acc
        cost_k = avg_cost_per_sample * k

        results[f"k={k}"] = {
            "pass_at_k": pass_at_k,
            "self_consistency_acc": mv_acc,
            "verifier_headroom": headroom,
            "cost": cost_k,
        }
        print(f"{k:>3} {pass_at_k:>8.3f} {mv_acc:>10.3f} {headroom:>+9.3f} {cost_k:>10.1f}")

    Path("outputs").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {args.out}")

    best_headroom = max(r["verifier_headroom"] for r in results.values())
    best_k = max(results, key=lambda kk: results[kk]["verifier_headroom"])
    print(f"\nBest verifier headroom: {best_headroom:+.3f} (at {best_k})")
    if best_headroom >= args.headroom_threshold:
        print(f"-> MEANINGFUL headroom found (>= {args.headroom_threshold:.0%}). "
              f"A real verifier is worth building -- proceed to design one for P4/P5.")
    else:
        print(f"-> Headroom below the pre-registered bar ({args.headroom_threshold:.0%}). "
              f"A verifier is unlikely to be worth building for this model/dataset/action-set "
              f"combination. Treat the P3 Gate 2 FAIL as the honest result here and rescope "
              f"(bigger model, harder dataset, or a different mechanism) rather than probing "
              f"further variations of this same setup.")


if __name__ == "__main__":
    main()
