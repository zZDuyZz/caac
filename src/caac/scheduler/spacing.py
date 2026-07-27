"""Adaptive decision spacing -- a scheduler flag, not decision logic.

Widen the gap between decision points when the belief is stable, hold it tight
when the belief moves. Reduces the number of controller calls (n_d) without
losing control resolution where it matters.

SAFETY: only enable when the correctness estimator has passed the E1
calibration gate. If the belief is untrustworthy, 'stable belief -> ask less
often' can be an illusion that misses the moment intervention is needed. This
is why the flag is calibration-gated, tying it back to RQ1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AdaptiveSpacing"]


@dataclass
class AdaptiveSpacing:
    """Decides whether to run the controller at the current step."""

    base_k: int = 64
    max_k: int = 256
    stable_eps: float = 0.02       # belief change below this counts as 'stable'
    enabled: bool = False          # off unless calibration gate passed
    _last_belief: float | None = field(default=None, repr=False)
    _current_k: int = field(default=0, repr=False)

    def should_decide(self, belief: float, tokens_since: int) -> bool:
        if not self.enabled:
            return tokens_since >= self.base_k
        if self._last_belief is None:
            self._last_belief = belief
            self._current_k = self.base_k
            return tokens_since >= self.base_k

        moved = abs(belief - self._last_belief) > self.stable_eps
        self._current_k = self.base_k if moved else min(self.max_k, self._current_k * 2)
        self._last_belief = belief
        return tokens_since >= self._current_k

    def reset(self) -> None:
        self._last_belief = None
        self._current_k = self.base_k
