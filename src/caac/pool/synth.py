"""Synthetic sample pool generator -- lets the WHOLE replay/LTT/Chow pipeline
be developed and tested on CPU before renting a GPU (PLAN_V2 rule 1).

It fabricates per-question difficulty, then draws greedy + sampled answers for
a small and a big model whose accuracies and confidence calibration roughly
match the planning numbers (small ~44% on hard set, big ~75%), with confidence
correlated to correctness so that confidence-threshold policies actually work.
The numbers are NOT meant to be realistic physics -- only to exercise every
code path and let the tests assert ordering properties that must hold on real
data too (e.g. escalate raises pi_pop; maj@k >= greedy accuracy).
"""

from __future__ import annotations

import random

from caac.pool.schema import Completion, PoolRecord


def _draw_answer(gold: str, p_correct: float, rng: random.Random, n_distractors: int = 3):
    if rng.random() < p_correct:
        return gold
    # a wrong answer from a small set of plausible distractors (clustered, as in math)
    return f"W{rng.randint(1, n_distractors)}"


def make_pair(
    query_id: str,
    dataset: str,
    small_model: str,
    big_model: str,
    n_small: int = 32,
    n_big: int = 8,
    seed: int = 0,
) -> tuple[PoolRecord, PoolRecord]:
    """One question's pool records for (small, big)."""
    rng = random.Random(hash((query_id, seed)) & 0xFFFFFFFF)
    gold = "G"
    difficulty = rng.random()  # 0 easy .. 1 hard

    # per-sample correctness prob: small model struggles on hard, big less so
    p_small = max(0.02, 0.95 - 0.9 * difficulty)
    p_big = max(0.05, 0.98 - 0.5 * difficulty)

    def completion(p: float, temp0: bool) -> Completion:
        # greedy (temp0) picks the modal answer: single draw at a mildly higher
        # prob than a temperature sample (argmax favors the likeliest answer).
        pc = min(0.99, p + 0.05) if temp0 else p
        ans = _draw_answer(gold, pc, rng)
        correct = ans == gold
        # confidence correlated with correctness (+ noise), in [0,1]
        base = 0.75 if correct else 0.45
        conf = min(1.0, max(0.0, rng.gauss(base, 0.15)))
        dec = rng.randint(120, 700)
        fr = "length" if dec > 680 else "stop"
        return Completion(answer=ans, correct=correct, decode_tokens=dec,
                          finish_reason=fr, confidence=conf,
                          seed=rng.randint(0, 2**31))

    small = PoolRecord(
        query_id=query_id, dataset=dataset, model=small_model, gold=gold,
        prompt_tokens=rng.randint(60, 160),
        greedy=completion(p_small, temp0=True),
        samples=[completion(p_small, temp0=False) for _ in range(n_small)],
    )
    big = PoolRecord(
        query_id=query_id, dataset=dataset, model=big_model, gold=gold,
        prompt_tokens=small.prompt_tokens,
        greedy=completion(p_big, temp0=True),
        samples=[completion(p_big, temp0=False) for _ in range(n_big)],
    )
    return small, big


def make_pools(n_questions: int = 200, dataset: str = "synth",
               small_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
               big_model: str = "Qwen/Qwen2.5-7B-Instruct",
               n_small: int = 32, n_big: int = 8, seed: int = 0):
    small_pool, big_pool = {}, {}
    for i in range(n_questions):
        qid = f"{dataset}_{i}"
        s, b = make_pair(qid, dataset, small_model, big_model, n_small, n_big, seed)
        small_pool[qid] = s
        big_pool[qid] = b
    return small_pool, big_pool
