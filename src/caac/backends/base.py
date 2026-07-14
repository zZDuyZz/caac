"""Generation backend interface.

The controller talks to models only through these protocols. Keeping the
surface this small lets the whole decision layer be tested with a mock backend
on CPU, and swapped to vLLM for real runs without touching policy code.
"""

from __future__ import annotations

from typing import Protocol

from caac.types import Trajectory

__all__ = ["GenerationBackend"]


class GenerationBackend(Protocol):
    """Produces one reasoning segment at a time.

    Segment-wise generation (not one long call) is required because decision
    points only exist if generation can be interrupted.
    """

    def generate(
        self, prompt: str, prefix: str, max_tokens: int, *, temperature: float | None = None
    ) -> Trajectory:
        """Continue ``prefix`` for at most ``max_tokens`` tokens.

        The returned Trajectory must carry per-token logprobs and entropies:
        raw material for every uncertainty signal, free at decode time,
        impossible to recover afterwards.
        """
        ...
