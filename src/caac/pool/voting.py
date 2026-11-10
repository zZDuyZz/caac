"""The vote CAAC serves when it samples more.

A plain majority of k fresh samples ignores an answer the small model already
paid for: at k=2 two disagreeing samples can overturn a correct greedy answer
and the vote is then worse than not sampling. Counting the greedy answer as one
more vote, ties going to greedy, fixes that without any label or extra cost.
Truncated samples never vote (see ``PoolRecord.canonical_sample_answers``).

Baselines keep the plain majority, as in their papers; ``sc_abstain_greedy``
gives self-consistency the same selector so the comparison is fair.
"""

from __future__ import annotations

import zlib
from itertools import combinations
from math import comb
import random

from caac.pool.schema import PoolRecord

__all__ = ["serve_with_greedy", "greedy_vote_correct", "greedy_vote_prob"]

_MAX_SUBSETS = 512


def serve_with_greedy(rec: PoolRecord, idxs: list[int]) -> str | None:
    """Answer class served by the sample votes plus one vote for greedy."""
    canon = rec.canonical_sample_answers()
    g = rec.greedy_label()
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for pos, i in enumerate(idxs):
        a = canon[i]
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
            first.setdefault(a, pos)
    if g is not None:
        counts[g] = counts.get(g, 0) + 1
        first.setdefault(g, -1)
    if not counts:
        return None
    best = max(counts.values())
    top = [a for a, c in counts.items() if c == best]
    if g in top:
        return g
    return min(top, key=lambda a: first[a])


def greedy_vote_correct(rec: PoolRecord, idxs: list[int]) -> float:
    return float(rec.label_correct(serve_with_greedy(rec, idxs)))


def greedy_vote_prob(rec: PoolRecord, k: int) -> float:
    """P(the served vote is correct) over draws of k samples, fixed per question.

    Enumerated when the subsets are few, otherwise a seeded sample of them, so
    the target is deterministic and has no lucky permutation.
    """
    n = rec.n_samples
    k = min(k, n)
    if n == 0 or k == 0:
        return float(rec.greedy.correct)
    if comb(n, k) <= _MAX_SUBSETS:
        subsets = combinations(range(n), k)
    else:
        rng = random.Random(zlib.crc32(f"{rec.query_id}:{k}".encode()))
        subsets = (rng.sample(range(n), k) for _ in range(_MAX_SUBSETS))
    vals = [greedy_vote_correct(rec, list(s)) for s in subsets]
    return sum(vals) / len(vals)
