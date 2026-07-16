"""Gain estimators: Delta_psi(a, s) = E[V(s')] - U_stop(s).

The heuristic variant encodes the qualitative predictions of the proposal so
the pipeline runs before any tree is collected, and gives H2 a concrete shape
to be tested against. The fitted variant trains one regressor per action on
compute-tree value labels.

These read only tier-0/1 features (via the extractor's policy_features), which
is what keeps the gain model transferable across models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caac.policy.estimators import GainEstimator
from caac.types import Action, ReasoningState

__all__ = ["HeuristicGain", "FittedGain"]


@dataclass
class HeuristicGain(GainEstimator):
    """Closed-form gain to smoke-test the pipeline. A placeholder, not a result.

    Encodes: CONTINUE gains most when belief is low; VERIFY follows an
    inverted-U in belief (the EVOI prediction, H2); BRANCH pays off only when
    belief is genuinely poor. Real numbers come from the fitted estimator.
    """

    continue_scale: float = 0.08
    verify_scale: float = 0.05
    branch_scale: float = 0.06
    verify_peak: float = 0.6
    verify_width: float = 0.18

    def predict(self, state: ReasoningState, action: Action) -> float:
        b = float(np.clip(state.belief, 0.0, 1.0))
        if action is Action.CONTINUE:
            return self.continue_scale * (1.0 - b)
        if action is Action.VERIFY:
            return self.verify_scale * float(
                np.exp(-((b - self.verify_peak) ** 2) / (2 * self.verify_width**2))
            )
        if action is Action.BRANCH:
            return self.branch_scale * max(0.0, 1.0 - 2.0 * b)
        return 0.0


@dataclass
class FittedGain(GainEstimator):
    """One regressor per action, trained on compute-tree value labels.

    Fitting per action (rather than one model with action as a feature) keeps
    the comparison between actions from being smoothed together -- the whole
    point is to tell them apart.
    """

    max_depth: int = 4
    max_iter: int = 200
    _models: dict = field(default_factory=dict)

    def fit(self, features, actions, values):
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = np.asarray(features, dtype=np.float64)
        a = np.asarray(actions)
        y = np.asarray(values, dtype=np.float64).ravel()
        for action in Action.spending():
            m = a == action.value
            if m.sum() < 10:
                continue
            model = HistGradientBoostingRegressor(max_depth=self.max_depth, max_iter=self.max_iter)
            model.fit(X[m], y[m])
            self._models[action] = model
        return self

    def predict(self, state: ReasoningState, action: Action) -> float:
        model = self._models.get(action)
        if model is None:
            return 0.0
        if state.features is None:
            raise ValueError("state.features is None; run the extractor first")
        return float(model.predict(state.features.reshape(1, -1))[0])

    @property
    def fitted_actions(self):
        return tuple(self._models)
