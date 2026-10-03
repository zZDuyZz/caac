"""Finite-pool estimators: pass@k, EXACT maj@k, and the Chow coverage bound.

These are the three numbers the whole paper leans on, so they are implemented
from first principles and unit-tested on a synthetic pool (test_pool_replay.py).

maj@k note (PLAN_V2 section 6): the exact estimator averages over EVERY size-k
subset drawn WITHOUT replacement, via the multivariate hypergeometric over the
answer-count vector. This is a U-statistic of degree k, hence unbiased for the
true maj@k and free of Monte-Carlo noise. There is no published reference for
this exact form that the research turned up -- it is our derivation, so it is
cross-checked against a Monte-Carlo estimate in the tests.
"""

from __future__ import annotations

from collections import Counter
from math import comb, isclose

__all__ = ["pass_at_k", "maj_at_k_exact", "chow_coverage_bound"]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): 1 - C(n-c,k)/C(n,k).

    n = samples drawn, c = how many correct, k = budget. If k > n, treat as
    k = n (can't draw more than the pool holds).
    """
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _plurality_credit(counts: tuple[int, ...], gold_idx: int, tie: str,
                      none_idx: int | None) -> float:
    """Credit in [0,1] that the plurality of this drawn count vector is gold.

    Unparsable answers (the None bucket at ``none_idx``) consume draw slots but
    DO NOT vote: they are excluded from the max/winner comparison, matching
    self-consistency's "drop unparsable, then vote" behavior. If every drawn
    answer is None, no one wins -> credit 0.

    tie:
      "fair"       -- split credit uniformly among the tied winners (default;
                      the unbiased expected accuracy of random tie-break)
      "correct"    -- gold wins any tie it is part of (optimistic upper bound)
      "incorrect"  -- gold loses any tie (pessimistic lower bound)
    """
    voting = [v if i != none_idx else -1 for i, v in enumerate(counts)]
    mx = max(voting)
    if mx <= 0:
        return 0.0  # all drawn answers were None (or empty)
    if voting[gold_idx] != mx:
        return 0.0
    winners = [i for i, v in enumerate(voting) if v == mx]
    if len(winners) == 1:
        return 1.0
    if tie == "correct":
        return 1.0
    if tie == "incorrect":
        return 0.0
    return 1.0 / len(winners)  # "fair"


def maj_at_k_exact(
    answers: list,
    gold,
    k: int,
    tie: str = "fair",
    max_states: int = 2_000_000,
) -> float:
    """Exact maj@k over a finite pool of answers (without replacement).

    ``answers`` is the list of extracted answers for this question (None for
    unparsable; counted as its own "answer" that can never equal gold).
    ``gold`` is the normalized correct answer.

    Returns the probability that a uniformly random size-k subset's plurality
    vote (tie-broken per ``tie``) equals gold. Exact when the enumeration fits
    in ``max_states`` composition states; raises if not (caller may fall back
    to Monte-Carlo, but for math with few distinct answers this never trips).
    """
    n = len(answers)
    if k >= n:
        k = n
    if n == 0:
        return 0.0

    # Distinct-answer count vector. None -> a sentinel distinct bucket.
    norm = [a if a is not None else ("__none__", i) for i, a in enumerate(answers)]
    # collapse Nones into ONE bucket (they're all "unparsable", never gold)
    norm = ["__NONE__" if (isinstance(a, tuple) and a[0] == "__none__") else a for a in norm]
    counts_by_ans = Counter(norm)
    labels = list(counts_by_ans.keys())
    counts = [counts_by_ans[l] for l in labels]
    m = len(labels)
    try:
        gold_idx = labels.index(gold)
    except ValueError:
        return 0.0  # gold never appears -> maj@k can't be correct
    none_idx = labels.index("__NONE__") if "__NONE__" in labels else None

    total_subsets = comb(n, k)

    # Enumerate count vectors (k_0..k_{m-1}) with 0<=k_i<=counts[i], sum=k.
    # Prob of each = prod C(counts[i], k_i) / C(n, k).
    acc = 0.0
    state_budget = [0]

    def rec(idx: int, remaining: int, chosen: list[int], ways: int):
        if state_budget[0] > max_states:
            raise RuntimeError(
                f"maj_at_k_exact: too many states (>{max_states}); "
                f"m={m} distinct answers, k={k}. Fall back to Monte-Carlo."
            )
        if idx == m - 1:
            if remaining <= counts[idx]:
                state_budget[0] += 1
                w = ways * comb(counts[idx], remaining)
                full = tuple(chosen + [remaining])
                credit = _plurality_credit(full, gold_idx, tie, none_idx)
                if credit > 0.0:
                    nonlocal acc
                    acc += credit * w
            return
        lo = max(0, remaining - sum(counts[idx + 1:]))
        hi = min(counts[idx], remaining)
        for ki in range(lo, hi + 1):
            rec(idx + 1, remaining - ki, chosen + [ki], ways * comb(counts[idx], ki))

    rec(0, k, [], 1)
    return acc / total_subsets


def chow_coverage_bound(pi_pop: float, alpha: float) -> float:
    """Max coverage any serve/abstain rule can reach at selective-risk alpha.

    Chow (1970), as cited in PriceCheck: coverage <= min(1, pi_pop/(1-alpha)),
    where pi_pop is the fraction of candidate answers that are correct.

    This is THE bound behind the paper's central argument (PLAN_V2 section 8):
    in the small-model regime pi_pop is low, so answer-correcting actions
    (sample-more, escalate) are a precondition for usable certified coverage,
    not an optimization.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    return min(1.0, pi_pop / (1.0 - alpha))
