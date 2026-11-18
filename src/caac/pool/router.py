"""A question-only router: the RouteLLM-style baseline (2406.18665).

RouteLLM decides, from the query ALONE and before any model runs, whether the
weak or the strong model answers; its routers (matrix factorisation, a BERT
classifier, a causal-LM classifier, similarity-weighted ranking) differ in the
predictor, not in that decision. Here the predictor is a logistic head on what
the pools hold before generation: the prompt length and, when available, the
small model's question-only hidden state (extract_hidden.py, projected by
``caac.pool.hidden``). It estimates

    p_small = P(small greedy correct | question),  p_big = P(big greedy correct | question)

and the family (``families.router``) routes on p_small. No probability needs to
be calibrated for validity: thresholds are certified by the same Learn-then-Test
protocol as every other method. This is our reimplementation, not RouteLLM's
code or trained routers, and the paper must say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from caac.pool.estimator import LogisticHead, features
from caac.pool.schema import PoolRecord

__all__ = ["QuestionRouter"]


@dataclass
class QuestionRouter:
    l2: float = 1.0
    extras: dict = field(default_factory=dict)   # qid -> projected hidden state (may be empty)
    head_small: LogisticHead | None = None
    head_big: LogisticHead | None = None
    extra_dim: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    def _row(self, rec: PoolRecord) -> list[float]:
        extra = self.extras.get(rec.query_id)
        if self.extra_dim and extra is None:
            raise KeyError(f"no hidden state for {rec.query_id}")
        return features(rec, ("qonly",), extra if self.extra_dim else None)

    def fit(self, qids, small: dict, big: dict) -> "QuestionRouter":
        qids = list(qids)
        self.extra_dim = len(self.extras.get(qids[0], [])) if self.extras else 0
        X = np.asarray([self._row(small[q]) for q in qids], dtype=float)
        ys = np.asarray([float(small[q].greedy.correct) for q in qids])
        yb = np.asarray([float(big[q].greedy.correct) for q in qids])
        self.head_small = LogisticHead(l2=self.l2).fit(X, ys)
        self.head_big = LogisticHead(l2=self.l2).fit(X, yb)
        self._cache = {}
        return self

    def __call__(self, rec: PoolRecord) -> tuple[float, float]:
        key = rec.query_id
        if key not in self._cache:
            x = np.asarray([self._row(rec)], dtype=float)
            self._cache[key] = (float(self.head_small.predict(x)[0]), float(self.head_big.predict(x)[0]))
        return self._cache[key]

    def to_json(self) -> str:
        """The fitted heads only (the hidden-state projection is saved by its owner)."""
        return json.dumps({"kind": "QuestionRouter", "l2": self.l2, "extra_dim": self.extra_dim,
                           "head_small": self.head_small.to_dict(), "head_big": self.head_big.to_dict()})
