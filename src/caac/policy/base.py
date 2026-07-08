"""Policy interface.

Every allocation strategy -- VOC, fixed budget, confidence threshold, oracle --
implements this one method, so the evaluation harness compares them without
special-casing.
"""

from __future__ import annotations

from caac.types import Decision, ReasoningState

__all__ = ["Policy"]


class Policy:
    """Maps a reasoning state to an action."""

    name: str = "policy"

    def decide(self, state: ReasoningState) -> Decision:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any per-query internal state. Default: nothing."""
        return None
