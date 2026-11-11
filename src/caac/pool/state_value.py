"""Is the next step worth its price, given everything already paid for?

The recovery estimator of ``estimator.py`` reads the greedy turn and nothing
else, so it is asked once and cannot see that two samples disagreed, or that
the big model's answer was just rejected. This one is asked at every state of
the staged controller -- after the greedy turn, after each batch of samples,
after an escalation -- and scores the next step from the evidence that has been
bought so far:

    Delta_a(state) = P(gain) - P(harm),  gain = now wrong AND after a right,
                                         harm = now right AND after a wrong,

where "now" is the answer the controller would serve if it stopped here (greedy
at the start, the vote with greedy as one voter once samples are drawn) and
"after a" is what the step would put on the table. Sampling is scored per
horizon -- sample up to h in total -- because two more draws rarely rescue a
question that sixteen would, and a one-step lookahead would never start.

Same machinery as everywhere else: a logistic head per target, isotonic
calibration out of fold, folds by question. Nothing reads the gold label except
the targets, and the targets are built on TRAIN only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from caac.pool.estimator import (POST_FEATURES, ESC_FEATURES, Isotonic, LogisticHead, _fit_calibrated,
                                 features, post_escalate_features, post_sample_features)
from caac.pool.voting import greedy_vote_correct

__all__ = ["STATE_ACTIONS", "state_features", "state_feature_names", "StateValueEstimator"]

STATE_ACTIONS = ("sample_more", "escalate")
_PARTS = ("gain", "harm")
_N_GREEDY = 7          # len(features()) for the default groups


def state_feature_names() -> list[str]:
    return (["g_log_prompt", "g_bias", "g_conf_prefix32", "g_confidence", "g_mean_logprob",
             "g_log_decode", "g_truncated"]
            + ["s_" + n for n in POST_FEATURES]
            + ["s_has_samples", "s_log_drawn"]
            + ["e_" + n for n in ESC_FEATURES] + ["e_escalated"]
            + ["horizon_log", "horizon_gap"])


def state_features(small, big, drawn: list[int], escalated: bool, horizon: int,
                   stage1: dict | None = None) -> list[float]:
    """Everything visible at this state, plus the horizon being asked about."""
    g = features(small)
    if drawn:
        s = post_sample_features(small, drawn, stage1) + [1.0, np.log2(len(drawn) + 1) / 4.0]
    else:
        s = [0.0] * (len(POST_FEATURES) + 2)
    if escalated and big is not None:
        e = post_escalate_features(small, big, drawn, stage1) + [1.0]
    else:
        e = [0.0] * (len(ESC_FEATURES) + 1)
    h = [np.log2(horizon) / 4.0 if horizon else 0.0, (horizon - len(drawn)) / 16.0 if horizon else 0.0]
    return g + s + e + h


def _answer_now(small, order: list[int], n: int) -> float:
    return float(small.greedy.correct) if n == 0 else greedy_vote_correct(small, order[:n])


@dataclass
class StateValueEstimator:
    steps: tuple = (2, 4, 8, 16)
    l2: float = 1.0
    n_folds: int = 5
    seed: int = 0
    calibrate: bool = True
    heads: dict[str, LogisticHead] = field(default_factory=dict)
    cals: dict[str, Isotonic] = field(default_factory=dict)
    base_rate: dict[str, float] = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def _key(action: str, part: str) -> str:
        return f"{action}.{part}"

    def _states(self, n_avail: int):
        """The (drawn, escalated) states the controller can be in."""
        ns = [0] + [h for h in self.steps if h < min(max(self.steps), n_avail)]
        return [(n, e) for n in ns for e in (False, True)]

    def _horizons(self, n: int, n_avail: int):
        return sorted({min(h, n_avail) for h in self.steps if min(h, n_avail) > n})

    def fit(self, qids, small, big, orders, stage1=None) -> "StateValueEstimator":
        rows = {a: [] for a in STATE_ACTIONS}
        ys = {self._key(a, p): [] for a in STATE_ACTIONS for p in _PARTS}
        groups = {a: [] for a in STATE_ACTIONS}
        for gi, q in enumerate(qids):
            s, b = small[q], (big or {}).get(q)
            p1 = stage1(s) if stage1 is not None else None
            for o in orders:
                avail = [i for i in o if i < s.n_samples]
                for n, esc in self._states(len(avail)):
                    if esc and b is None:
                        continue
                    drawn = avail[:n]
                    now = _answer_now(s, avail, n)
                    for a in STATE_ACTIONS:
                        if a == "escalate":
                            if esc or b is None:
                                continue
                            targets = [(0, float(b.greedy.correct))]
                        else:
                            targets = [(h, greedy_vote_correct(s, avail[:h]))
                                       for h in self._horizons(n, len(avail))]
                        for h, after in targets:
                            rows[a].append(state_features(s, b, drawn, esc, h, p1))
                            ys[self._key(a, "gain")].append((1.0 - now) * after)
                            ys[self._key(a, "harm")].append(now * (1.0 - after))
                            groups[a].append(gi)
        for a in STATE_ACTIONS:
            if len(set(groups[a])) < 2 * self.n_folds:
                continue
            X = np.asarray(rows[a], dtype=float)
            for p in _PARTS:
                key = self._key(a, p)
                head, cal, base = _fit_calibrated(X, ys[key], groups[a], self.l2, self.n_folds,
                                                  self.seed, self.calibrate)
                self.heads[key], self.cals[key], self.base_rate[key] = head, cal, base
        self._cache.clear()
        return self

    def _p(self, key: str, x) -> float:
        if key not in self.heads:
            return 0.0
        raw = float(self.heads[key].predict(x)[0])
        return float(self.cals[key].predict([raw])[0]) if self.calibrate else raw

    def delta(self, small, big, drawn: list[int], escalated: bool, action: str,
              horizon: int = 0, stage1: dict | None = None) -> float:
        """Estimated net gain of ``action`` from this state; 0 when unfitted, which
        buys nothing at a positive threshold -- the safe direction."""
        key = (small.model, small.query_id, tuple(drawn), escalated, action, horizon)
        if key not in self._cache:
            x = np.asarray([state_features(small, big, drawn, escalated, horizon, stage1)], dtype=float)
            self._cache[key] = (self._p(self._key(action, "gain"), x)
                                - self._p(self._key(action, "harm"), x))
        return self._cache[key]

    def to_json(self) -> str:
        return json.dumps({"steps": list(self.steps), "l2": self.l2, "n_folds": self.n_folds,
                           "seed": self.seed, "calibrate": self.calibrate,
                           "base_rate": self.base_rate, "feature_names": state_feature_names(),
                           "heads": {k: h.to_dict() for k, h in self.heads.items()},
                           "cals": {k: c.to_dict() for k, c in self.cals.items()}}, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "StateValueEstimator":
        d = json.loads(s)
        e = cls(steps=tuple(d["steps"]), l2=d["l2"], n_folds=d["n_folds"], seed=d["seed"],
                calibrate=d["calibrate"])
        e.base_rate = dict(d.get("base_rate", {}))
        e.heads = {k: LogisticHead.from_dict(v) for k, v in d["heads"].items()}
        e.cals = {k: Isotonic.from_dict(v) for k, v in d["cals"].items()}
        return e
