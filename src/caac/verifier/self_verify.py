"""Self-verification using the generation model itself.

The cheapest VERIFY tier: prompt the model to check its own trace. On the mock
backend this is simulated; with a real backend it issues a short verification
request whose cost is charged like any other forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from caac.verifier.base import VerifierResult

__all__ = ["SelfVerifier"]


@dataclass
class SelfVerifier:
    """Wraps a backend to score a trace via a short check prompt.

    On the mock backend, ``backend`` may expose a ``verify`` shortcut; a real
    backend generates a short yes/no continuation and maps it to a score.
    """

    backend: object
    check_tokens: int = 32

    def verify(self, prompt: str, trace: str) -> VerifierResult:
        if hasattr(self.backend, "verify"):
            return self.backend.verify(prompt, trace)
        seg = self.backend.generate(
            prompt=prompt + "\n\nIs the above reasoning correct? Answer yes/no.",
            prefix=trace,
            max_tokens=self.check_tokens,
        )
        # Placeholder scoring: real mapping parses the yes/no head token.
        score = 0.5
        return VerifierResult(
            score=score, n_tokens=seg.n_tokens, n_forward_passes=seg.n_tokens
        )
