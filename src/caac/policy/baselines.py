"""Baseline allocation policies.

Each embodies a different published answer to 'when and where to spend compute'.
They run through the same controller as VOC so any difference on the Pareto
frontier comes from the decision rule alone.

Scope note: these reimplement the decision *rule*, not the full published
systems. That is the correct comparison for the allocation question;
reproducing each paper's headline numbers is a separate task (phase P1).
"""

from __future__ import annotations

from dataclasses import dataclass

from caac.policy.base import Policy
from caac.types import Action, Decision, ReasoningState

__all__ = [
    "FixedBudgetPolicy",
    "ConfidenceThresholdPolicy",
    "AlwaysVerifyPolicy",
    "RandomPolicy",
]


@dataclass
class FixedBudgetPolicy(Policy):
    """Spend a fixed number of segments, then stop. The non-adaptive reference."""

    n_segments: int = 8
    name: str = "fixed_budget"

    def decide(self, state: ReasoningState) -> Decision:
        action = Action.CONTINUE if state.step < self.n_segments else Action.STOP
        return Decision(action, {}, state.belief, f"step {state.step}/{self.n_segments}")


@dataclass
class ConfidenceThresholdPolicy(Policy):
    """Stop as soon as belief clears a fixed threshold.

    The shape shared by confidence-gated early-exit methods (DEER/DeepConf-style).
    It answers *when* but cannot express *where* -- the limitation the VOC
    formulation removes, which makes it the most informative single baseline.
    """

    threshold: float = 0.9
    min_steps: int = 1
    name: str = "confidence_threshold"

    def decide(self, state: ReasoningState) -> Decision:
        if state.step < self.min_steps:
            return Decision(Action.CONTINUE, {}, state.belief, "warmup")
        action = Action.STOP if state.belief >= self.threshold else Action.CONTINUE
        return Decision(action, {}, state.belief, f"belief {state.belief:.3f} vs {self.threshold}")


@dataclass
class AlwaysVerifyPolicy(Policy):
    """Verify after every segment, then stop on high belief.

    Tests H2 directly: if unconditional verification is no worse than selective
    at matched cost, the inverted-U prediction is falsified.
    """

    threshold: float = 0.9
    name: str = "always_verify"

    def decide(self, state: ReasoningState) -> Decision:
        if state.belief >= self.threshold:
            return Decision(Action.STOP, {}, state.belief, "confident")
        action = Action.VERIFY if state.step % 2 == 0 else Action.CONTINUE
        return Decision(action, {}, state.belief, "alternating")


@dataclass
class RandomPolicy(Policy):
    """Uniformly random spending action. Sanity floor, not a real baseline."""

    stop_prob: float = 0.2
    seed: int = 0
    name: str = "random"

    def __post_init__(self):
        import numpy as np

        self._rng = np.random.default_rng(self.seed)

    def decide(self, state: ReasoningState) -> Decision:
        if self._rng.random() < self.stop_prob:
            return Decision(Action.STOP, {}, state.belief, "random stop")
        choice = self._rng.choice([a.value for a in Action.spending()])
        return Decision(Action(choice), {}, state.belief, "random")

    def reset(self):
        import numpy as np

        self._rng = np.random.default_rng(self.seed)
