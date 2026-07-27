"""HuggingFace transformers backend (phase P1+).

Interface is fixed; the body requires GPU + model weights and is filled in when
GPU access is available. It must return per-token logprobs and entropies -- the
raw material every uncertainty signal depends on.
"""

from __future__ import annotations

from caac.types import Trajectory

__all__ = ["HFBackend"]


class HFBackend:
    """Segment-wise generation via transformers. Deferred to P1."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        raise NotImplementedError(
            "HFBackend requires transformers + GPU (phase P1). Use MockBackend on CPU. "
            "Implementation must expose per-token logprobs and entropies."
        )

    def generate(self, prompt: str, prefix: str, max_tokens: int) -> Trajectory:  # pragma: no cover
        raise NotImplementedError
