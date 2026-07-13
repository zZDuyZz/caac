"""Adapter interface: the per-model layer.

An adapter maps a specific model's raw signals to a calibrated belief. It is
the ONLY model-dependent part of the controller; the policy layer works on the
calibrated belief and is shared across models. Moving to a new model means
re-fitting an adapter, not retraining the policy.
"""

from __future__ import annotations

import numpy as np

from caac.types import ReasoningState

__all__ = ["Adapter"]


class Adapter:
    """Raw model signals -> calibrated belief P(correct)."""

    name: str = "adapter"

    def belief(self, state: ReasoningState) -> float:
        raise NotImplementedError

    def calibrate(self, features: np.ndarray, labels: np.ndarray) -> "Adapter":
        raise NotImplementedError
