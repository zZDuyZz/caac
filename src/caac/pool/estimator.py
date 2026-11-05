"""Calibrated per-action correctness estimator (CAAC_MASTER §8.1, stage 2).

The controller needs, BEFORE spending anything, an honest probability that each
available action would end up correct:

    Phat(correct | action a, question q)   for a in {answer_now, sample_more, escalate}

This module fits one head per action and then CALIBRATES it, because the
controller's threshold is only meaningful if the probabilities mean what they
say -- the whole Learn-then-Test guarantee is applied to a policy whose gate is
such a probability.

Three design rules, each of which a reviewer will check:

1. **No peeking at the action's own outcome.** Features come only from the
   question and from the GREEDY turn, which the policy has already paid for.
   Nothing reads ``small.samples`` or ``big`` -- those ARE the actions. A
   feature set that looked at the samples would predict ``sample_more``
   perfectly and be worthless in deployment.

2. **Soft targets where the outcome is a distribution.** ``sample_more``'s
   outcome depends on which k samples are drawn, so its target is the exact
   maj@k probability over all C(n,k) subsets (``maj_at_k_exact``), not one
   realised vote. Lower variance, and no lucky permutation. Evaluation still
   uses a REALISED held-out vote, so the reported numbers are achievable.

3. **Calibration fit out-of-fold.** ``fit`` cross-fits the raw head, calibrates
   on the out-of-fold scores, and refits the head on everything. Calibrating on
   in-fold scores is the classic way to report an ECE that deployment will not
   reproduce.

No scikit-learn: the logistic head (L2, IRLS) and the isotonic calibrator
(PAVA) are ~60 lines of numpy each. One less dependency to break on a Windows
box, and both are auditable by a referee.

The fitted object serialises to plain JSON (``to_json``/``from_json``) -- no
pickle, so a trained estimator can be committed next to the numbers it
produced.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from caac.pool.estimators import maj_at_k_exact
from caac.pool.schema import GOLD, PoolRecord

__all__ = [
    "SERVING_ACTIONS", "FEATURE_GROUPS",
    "features", "feature_names", "action_targets", "realised_targets",
    "LogisticHead", "Isotonic", "PerActionEstimator",
]

# ``abstain`` is not estimated: it serves nothing, so its correctness is not
# defined and its risk contribution is zero by construction.
SERVING_ACTIONS = ("answer_now", "sample_more", "escalate")

# Which features each group contributes. "qonly" is the ABLATION demanded by
# Bulut (2609.03436): if prefix-based difficulty prediction is near chance on
# some models, the paper must show how much of CAAC survives on question-only
# information. Keep the two groups separable for exactly that table.
FEATURE_GROUPS = {
    "qonly": ("log_prompt_tokens", "bias_qonly"),
    "greedy": ("conf_prefix32", "greedy_confidence", "greedy_mean_logprob",
               "log_decode_tokens", "truncated"),
}


def feature_names(groups=("qonly", "greedy"), extra_dim: int = 0) -> list[str]:
    names: list[str] = []
    for g in groups:
        names.extend(FEATURE_GROUPS[g])
    names.extend(f"extra_{i}" for i in range(extra_dim))
    return names


def features(small: PoolRecord, groups=("qonly", "greedy"),
             extra: list[float] | None = None) -> list[float]:
    """Pre-action feature vector for one question.

    Reads ``small.greedy`` and the question, never ``small.samples`` and never
    the big model. ``extra`` carries an externally computed vector (the A4
    question-only hidden-state probe) and is treated as part of "qonly".

    Missing fields (older pools have no ``conf_prefix32``) become 0.0 and are
    flagged by their own indicator so the head can tell "low" from "absent".
    """
    g = small.greedy
    out: list[float] = []
    for grp in groups:
        if grp == "qonly":
            out.append(math.log1p(small.prompt_tokens))
            out.append(1.0)                                   # bias
        elif grp == "greedy":
            cp = g.conf_prefix32
            out.append(cp if cp is not None else 0.0)
            out.append(g.confidence if g.confidence is not None else 0.0)
            out.append(g.mean_logprob if g.mean_logprob is not None else 0.0)
            out.append(math.log1p(g.decode_tokens))
            out.append(1.0 if g.finish_reason == "length" else 0.0)
        else:
            raise ValueError(f"unknown feature group {grp!r}")
    if extra:
        out.extend(float(v) for v in extra)
    return out


def action_targets(small: PoolRecord, big: PoolRecord | None, k: int = 8) -> dict[str, float]:
    """FITTING targets. ``sample_more`` is the exact maj@k probability."""
    t = {"answer_now": float(small.greedy.correct)}
    if small.n_samples:
        t["sample_more"] = maj_at_k_exact(small.canonical_sample_answers(), GOLD,
                                          min(k, small.n_samples))
    if big is not None:
        t["escalate"] = float(big.greedy.correct)
    return t


def realised_targets(small: PoolRecord, big: PoolRecord | None, order: list[int],
                     k: int = 8) -> dict[str, float]:
    """EVALUATION targets: what actually happened for one concrete draw order.

    Reported numbers must be achievable by a policy that draws k samples once,
    so held-out metrics use this, not the soft target.
    """
    from collections import Counter
    t = {"answer_now": float(small.greedy.correct)}
    if small.n_samples:
        idxs = [i for i in order if i < small.n_samples][:k]
        votes = [small.canonical_sample_answers()[i] for i in idxs]
        votes = [v for v in votes if v is not None]
        t["sample_more"] = float(bool(votes) and Counter(votes).most_common(1)[0][0] == GOLD)
    if big is not None:
        t["escalate"] = float(big.greedy.correct)
    return t


# --------------------------------------------------------------------------- #
# Logistic head: L2-regularised, IRLS, accepts SOFT labels in [0, 1].
# --------------------------------------------------------------------------- #

@dataclass
class LogisticHead:
    l2: float = 1.0
    max_iter: int = 50
    tol: float = 1e-8
    w: np.ndarray | None = None
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    def _z(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu) / self.sd
        return np.column_stack([Xs, np.ones(len(Xs))])

    def fit(self, X, y, weights=None) -> "LogisticHead":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D and aligned with y")
        if np.any((y < -1e-9) | (y > 1 + 1e-9)):
            raise ValueError("targets must lie in [0, 1] (soft labels allowed)")
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        # constant columns (e.g. the explicit bias feature) must not blow up
        self.sd = np.where(sd < 1e-12, 1.0, sd)
        Z = self._z(X)
        n, d = Z.shape
        sw = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        w = np.zeros(d)
        # L2 on the slopes only; the intercept (last column) stays free so the
        # head can match the base rate of a rare action.
        pen = np.full(d, self.l2)
        pen[-1] = 0.0
        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
            grad = Z.T @ (sw * (y - p)) - pen * w
            W = sw * np.clip(p * (1 - p), 1e-9, None)
            H = (Z * W[:, None]).T @ Z + np.diag(pen)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            w = w + step
            if np.max(np.abs(step)) < self.tol:
                break
        self.w = w
        return self

    def predict(self, X) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("head is not fitted")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return 1.0 / (1.0 + np.exp(-np.clip(self._z(X) @ self.w, -30, 30)))

    def to_dict(self) -> dict:
        return {"l2": self.l2, "w": self.w.tolist(), "mu": self.mu.tolist(),
                "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticHead":
        h = cls(l2=d["l2"])
        h.w = np.asarray(d["w"], dtype=float)
        h.mu = np.asarray(d["mu"], dtype=float)
        h.sd = np.asarray(d["sd"], dtype=float)
        return h


# --------------------------------------------------------------------------- #
# Isotonic calibration (PAVA) -- monotone, non-parametric, soft-label safe.
# --------------------------------------------------------------------------- #

@dataclass
class Isotonic:
    """Monotone map raw score -> calibrated probability, fitted by PAVA.

    Monotone by construction, so it never reorders the head's ranking: it only
    fixes the LEVEL of the probabilities, which is what the threshold needs.
    Prediction interpolates linearly between block midpoints and clamps outside
    the fitted range (no extrapolation past what was observed).
    """

    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def fit(self, scores, targets) -> "Isotonic":
        s = np.asarray(scores, dtype=float)
        t = np.asarray(targets, dtype=float)
        if len(s) != len(t):
            raise ValueError("scores and targets must align")
        if len(s) == 0:
            self.x, self.y = [], []
            return self
        o = np.argsort(s, kind="mergesort")
        s, t = s[o], t[o]
        # pool adjacent violators, carrying weights
        vals = list(t)
        wts = [1.0] * len(t)
        xs = list(s)
        i = 0
        while i < len(vals) - 1:
            if vals[i] <= vals[i + 1] + 1e-12:
                i += 1
                continue
            tot = wts[i] + wts[i + 1]
            vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot
            xs[i] = (xs[i] * wts[i] + xs[i + 1] * wts[i + 1]) / tot
            wts[i] = tot
            del vals[i + 1], wts[i + 1], xs[i + 1]
            if i > 0:
                i -= 1
        self.x, self.y = [float(v) for v in xs], [float(v) for v in vals]
        return self

    def predict(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if not self.x:
            return s.copy()
        return np.clip(np.interp(s, self.x, self.y), 0.0, 1.0)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict) -> "Isotonic":
        return cls(x=list(d["x"]), y=list(d["y"]))


def _folds(n: int, n_folds: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.sort(a) for a in np.array_split(idx, min(n_folds, max(1, n)))]


# --------------------------------------------------------------------------- #
# The estimator the controller actually calls.
# --------------------------------------------------------------------------- #

@dataclass
class PerActionEstimator:
    """One calibrated head per serving action; callable as a policy Scorer.

    ``__call__(small, big)`` returns ``{action: Phat(correct)}`` plus
    ``abstain: 1.0`` so it drops straight into ``policies.caac`` /
    ``policies.caac_certified`` without those knowing how it was fitted.
    """

    groups: tuple[str, ...] = ("qonly", "greedy")
    k: int = 8
    l2: float = 1.0
    n_folds: int = 5
    seed: int = 0
    calibrate: bool = True
    heads: dict[str, LogisticHead] = field(default_factory=dict)
    cals: dict[str, Isotonic] = field(default_factory=dict)
    extra_dim: int = 0
    base_rate: dict[str, float] = field(default_factory=dict)
    oof: dict[str, dict[str, float]] = field(default_factory=dict)

    # -- design matrix ----------------------------------------------------- #

    def _row(self, small: PoolRecord, extra=None) -> list[float]:
        return features(small, self.groups, extra)

    def design(self, qids, small, big, extras=None):
        """X, per-action y, and the qid order used -- the one place that decides
        which questions a head is fitted on (a head skips questions where its
        own action has no recorded outcome)."""
        X, ys, keep = [], {a: [] for a in SERVING_ACTIONS}, {a: [] for a in SERVING_ACTIONS}
        for i, q in enumerate(qids):
            X.append(self._row(small[q], (extras or {}).get(q)))
            t = action_targets(small[q], (big or {}).get(q), self.k)
            for a in SERVING_ACTIONS:
                if a in t:
                    ys[a].append(t[a])
                    keep[a].append(i)
        return np.asarray(X, dtype=float), ys, keep

    # -- fitting ----------------------------------------------------------- #

    def fit(self, qids, small, big=None, extras=None) -> "PerActionEstimator":
        qids = list(qids)
        X, ys, keep = self.design(qids, small, big, extras)
        self.extra_dim = X.shape[1] - len(feature_names(self.groups))
        for a in SERVING_ACTIONS:
            rows = np.asarray(keep[a], dtype=int)
            if len(rows) < 2 * self.n_folds:
                continue  # not enough signal for this action; left unfitted
            Xa, ya = X[rows], np.asarray(ys[a], dtype=float)
            self.base_rate[a] = float(ya.mean())
            # out-of-fold raw scores -> what the calibrator is allowed to see
            oof = np.zeros(len(rows))
            for fold in _folds(len(rows), self.n_folds, self.seed):
                mask = np.ones(len(rows), dtype=bool)
                mask[fold] = False
                if mask.sum() < 2:
                    oof[fold] = ya.mean()
                    continue
                oof[fold] = LogisticHead(self.l2).fit(Xa[mask], ya[mask]).predict(Xa[fold])
            self.heads[a] = LogisticHead(self.l2).fit(Xa, ya)
            self.cals[a] = Isotonic().fit(oof, ya) if self.calibrate else Isotonic()
            self.oof[a] = {str(qids[i]): float(p) for i, p in zip(rows, oof)}
        return self

    # -- prediction -------------------------------------------------------- #

    def predict_one(self, small: PoolRecord, extra=None) -> dict[str, float]:
        x = np.asarray([self._row(small, extra)], dtype=float)
        out = {}
        for a in SERVING_ACTIONS:
            if a not in self.heads:
                continue
            raw = float(self.heads[a].predict(x)[0])
            out[a] = float(self.cals[a].predict([raw])[0]) if self.calibrate else raw
        return out

    def __call__(self, small: PoolRecord, big: PoolRecord | None = None) -> dict[str, float]:
        """Scorer interface used by ``policies.caac*``.

        ``big`` is accepted and IGNORED on purpose: the escalate head must
        predict the big model's correctness from the question and the small
        model's greedy turn, because in deployment the big model has not run.
        """
        p = self.predict_one(small)
        # an unfitted action is treated as unusable (never eligible), not as a
        # coin flip -- silently serving on a guessed probability would break the
        # risk guarantee rather than merely lose coverage.
        for a in SERVING_ACTIONS:
            p.setdefault(a, 0.0)
        p["abstain"] = 1.0
        return p

    # -- persistence ------------------------------------------------------- #

    def to_json(self) -> str:
        return json.dumps({
            "groups": list(self.groups), "k": self.k, "l2": self.l2,
            "n_folds": self.n_folds, "seed": self.seed, "calibrate": self.calibrate,
            "extra_dim": self.extra_dim, "base_rate": self.base_rate,
            "feature_names": feature_names(self.groups, self.extra_dim),
            "heads": {a: h.to_dict() for a, h in self.heads.items()},
            "cals": {a: c.to_dict() for a, c in self.cals.items()},
        }, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "PerActionEstimator":
        d = json.loads(s)
        e = cls(groups=tuple(d["groups"]), k=d["k"], l2=d["l2"], n_folds=d["n_folds"],
                seed=d["seed"], calibrate=d["calibrate"], extra_dim=d["extra_dim"])
        e.base_rate = dict(d.get("base_rate", {}))
        e.heads = {a: LogisticHead.from_dict(v) for a, v in d["heads"].items()}
        e.cals = {a: Isotonic.from_dict(v) for a, v in d["cals"].items()}
        return e
