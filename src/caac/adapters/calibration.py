"""Post-hoc calibration of raw uncertainty signals.

Raw signals (mean token confidence, negative entropy, PRM score) are ordinal
at best. These map them onto a probability scale so the belief entering the VOC
rule means what it claims -- a precondition for the martingale argument (H2).

  * TemperatureScaler  -- one parameter + bias, robust on small sets.
  * IsotonicCalibrator -- non-parametric, stronger but needs more data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

__all__ = [
    "Calibrator",
    "IdentityCalibrator",
    "TemperatureScaler",
    "IsotonicCalibrator",
    "build_calibrator",
]

_EPS = 1e-6


class Calibrator:
    """Maps a raw score to P(correct)."""

    def fit(self, scores, labels):
        raise NotImplementedError

    def transform(self, scores):
        raise NotImplementedError

    def fit_transform(self, scores, labels):
        return self.fit(scores, labels).transform(scores)


class IdentityCalibrator(Calibrator):
    """No-op baseline: use when reporting *uncalibrated* numbers."""

    def fit(self, scores, labels):
        return self

    def transform(self, scores):
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


@dataclass
class TemperatureScaler(Calibrator):
    """Single-parameter logistic recalibration with a bias term.

    The bias is included because small models are often not merely
    over-confident but systematically shifted.
    """

    temperature: float = 1.0
    bias: float = 0.0
    fitted: bool = False

    def _logit(self, scores):
        s = np.clip(np.asarray(scores, dtype=np.float64), _EPS, 1 - _EPS)
        return np.log(s / (1 - s))

    def fit(self, scores, labels):
        logits = self._logit(scores)
        labels = np.asarray(labels, dtype=np.float64)
        base = float(np.clip(labels.mean(), _EPS, 1 - _EPS))
        self.bias = float(np.log(base / (1 - base)) - np.mean(logits))

        def nll(log_t):
            t = float(np.exp(log_t))
            p = 1.0 / (1.0 + np.exp(-(logits + self.bias) / t))
            p = np.clip(p, _EPS, 1 - _EPS)
            return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

        res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
        self.temperature = float(np.exp(res.x))
        self.fitted = True
        return self

    def transform(self, scores):
        logits = self._logit(scores)
        return 1.0 / (1.0 + np.exp(-(logits + self.bias) / self.temperature))


@dataclass
class IsotonicCalibrator(Calibrator):
    """Monotone non-parametric calibration.

    More expressive than temperature scaling but prone to overfitting on small
    calibration sets; prefer it only when the calibration split is large.
    """

    _model: IsotonicRegression = field(
        default_factory=lambda: IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    )
    fitted: bool = False

    def fit(self, scores, labels):
        self._model.fit(
            np.asarray(scores, dtype=np.float64).ravel(),
            np.asarray(labels, dtype=np.float64).ravel(),
        )
        self.fitted = True
        return self

    def transform(self, scores):
        return np.clip(
            self._model.predict(np.asarray(scores, dtype=np.float64).ravel()), 0.0, 1.0
        )


def build_calibrator(name: str) -> Calibrator:
    table = {
        "identity": IdentityCalibrator,
        "temperature": TemperatureScaler,
        "isotonic": IsotonicCalibrator,
    }
    if name not in table:
        raise KeyError(f"unknown calibrator '{name}'; choose {sorted(table)}")
    return table[name]()
