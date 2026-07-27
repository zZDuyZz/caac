"""CPU-parallel controller execution.

Runs the (tiny) controller on CPU while the GPU generates the next segment, so
most controller cost is hidden behind generation at the wall-clock level. This
only produces a measurable benefit under a real serving engine and load, so the
implementation is deferred to phase P7; the interface is fixed here.
"""

from __future__ import annotations

__all__ = ["ParallelController"]


class ParallelController:
    """Deferred to P7 (needs a real serving engine to measure)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CPU-parallel controller requires a real serving backend (phase P7). "
            "Use the synchronous path with the mock/HF/vLLM backend until then."
        )
