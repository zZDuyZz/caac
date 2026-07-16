"""Cost estimator: c_phi(a, s) as a multi-dimensional CostVector.

Token counts and forward passes per action are largely determined by the
decoding configuration, so a closed-form estimate is accurate enough to start.
The measured-vs-predicted gap is tracked separately by the CostMeter; if it
proves large, this class is swapped for a fitted regressor behind the same
interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from caac.policy.estimators import CostEstimator
from caac.types import Action, CostVector, ReasoningState

__all__ = ["AnalyticCost"]


@dataclass
class AnalyticCost(CostEstimator):
    """Cost predicted from configuration rather than learned."""

    segment_tokens: int = 64
    verify_tokens: int = 32
    verify_forward_passes: float = 1.0
    branch_tokens: int = 64
    bytes_per_token_kv: float = 128.0
    seconds_per_token: float = 0.01

    def predict(self, state: ReasoningState, action: Action) -> CostVector:
        if action is Action.STOP:
            return CostVector.zero()
        if action is Action.CONTINUE:
            tokens, passes = float(self.segment_tokens), float(self.segment_tokens)
        elif action is Action.VERIFY:
            tokens = float(self.verify_tokens)
            passes = float(self.verify_forward_passes * self.verify_tokens)
        elif action is Action.BRANCH:
            tokens, passes = float(self.branch_tokens), float(self.branch_tokens)
        else:  # pragma: no cover
            raise ValueError(f"unhandled action {action}")
        return CostVector(
            tokens=tokens,
            forward_passes=passes,
            kv_bytes=tokens * self.bytes_per_token_kv,
            latency_s=tokens * self.seconds_per_token,
        )
