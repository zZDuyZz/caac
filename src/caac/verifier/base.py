"""Verifier interface.

Cost is returned alongside the score so the caller cannot forget to charge for
it -- the accounting discipline is enforced by the type, not by convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = ["VerifierResult", "Verifier"]


@dataclass
class VerifierResult:
    """Outcome of one VERIFY action."""

    score: float           # in [0, 1]; higher means "looks correct"
    n_tokens: float = 0.0
    n_forward_passes: float = 0.0


class Verifier(Protocol):
    def verify(self, prompt: str, trace: str) -> VerifierResult:
        ...
