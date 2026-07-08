"""Estimator interfaces feeding the VOC rule.

Three quantities, each estimated separately:

    p_theta(s)      -- P(current answer correct | s)   [correctness -> adapter]
    Delta_psi(a, s) -- E[V(s')] - U_stop(s)            [gain        -> policy]
    c_phi(a, s)     -- multi-dimensional cost of a     [cost        -> policy]

Deliberately NOT fused into one end-to-end reward: separation is what lets a
failing component be diagnosed. A single scalar reward would hide that.
"""

from __future__ import annotations

import numpy as np

from caac.types import Action, CostVector, ReasoningState

__all__ = ["CorrectnessEstimator", "GainEstimator", "CostEstimator"]


class CorrectnessEstimator:
    """Calibrated belief that the current answer is correct."""

    def predict(self, state: ReasoningState) -> float:
        raise NotImplementedError

    def fit(self, features: np.ndarray, labels: np.ndarray):
        raise NotImplementedError

    def reset(self) -> None:
        return None


class GainEstimator:
    """Expected improvement in value from taking an action.

    Returns E[V(s')] - U_stop(s): the *gross* gain before cost is subtracted.
    Positive means the action is expected to help.
    """

    def predict(self, state: ReasoningState, action: Action) -> float:
        raise NotImplementedError

    def fit(self, features: np.ndarray, actions: np.ndarray, values: np.ndarray):
        raise NotImplementedError

    def reset(self) -> None:
        return None


class CostEstimator:
    """Predicted cost of an action, before it is executed.

    Prediction is required because the VOC rule compares actions *ahead of
    time*. Measured cost is recorded separately by the CostMeter; the gap
    between predicted and measured is itself a diagnostic.
    """

    def predict(self, state: ReasoningState, action: Action) -> CostVector:
        raise NotImplementedError

    def reset(self) -> None:
        return None
