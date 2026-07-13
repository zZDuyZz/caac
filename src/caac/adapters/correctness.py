"""Correctness adapters: raw model signals -> calibrated belief.

This is the per-model layer. Two variants:
  * SignalAdapter  -- belief from one signal + calibration; used in the E1
    study where the question is how far a raw signal is from a probability.
  * FittedAdapter  -- logistic model over the full feature vector + calibration.

Both live in ``adapters`` because they are the ONLY model-dependent component;
moving to a new model re-fits one of these, not the shared policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from caac.adapters.base import Adapter
from caac.adapters.calibration import Calibrator, IdentityCalibrator
from caac.types import ReasoningState

__all__ = ["SignalAdapter", "FittedAdapter"]


@dataclass
class SignalAdapter(Adapter):
    """Belief taken directly from one uncertainty signal, then calibrated."""

    signal_key: str = "mean_token_confidence"
    calibrator: Calibrator = field(default_factory=IdentityCalibrator)
    name: str = "signal_adapter"

    def belief(self, state: ReasoningState) -> float:
        raw = float(state.meta.get(self.signal_key, 0.5))
        return float(self.calibrator.transform(np.array([raw]))[0])

    def calibrate(self, features, labels):
        self.calibrator.fit(np.asarray(features).ravel(), labels)
        return self


@dataclass
class FittedAdapter(Adapter):
    """Logistic model over the full feature vector, then calibrated.

    Logistic regression is chosen over a deeper model on purpose: the feature
    vector is small, training is one compute tree, and a convex model keeps the
    belief well-behaved at the tails where STOP decisions are made.
    """

    calibrator: Calibrator = field(default_factory=IdentityCalibrator)
    C: float = 1.0
    _model: LogisticRegression | None = None
    name: str = "fitted_adapter"

    def calibrate(self, features, labels):
        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).ravel()
        self._model = LogisticRegression(C=self.C, max_iter=1000)
        self._model.fit(X, y)
        raw = self._model.predict_proba(X)[:, 1]
        self.calibrator.fit(raw, y)
        return self

    def belief(self, state: ReasoningState) -> float:
        if self._model is None:
            raise RuntimeError("FittedAdapter used before calibrate()")
        if state.features is None:
            raise ValueError("state.features is None; run the extractor first")
        raw = self._model.predict_proba(state.features.reshape(1, -1))[0, 1]
        return float(self.calibrator.transform(np.array([raw]))[0])
