"""Deterministic mock backend + verifier.

Purpose: exercise the full control loop -- branching, budget, cost accounting,
answer aggregation -- on CPU with no model weights. Every test runs against it,
keeping the decision logic verifiable independently of GPU availability.

A simulator, not a model: it produces plausibly-shaped statistics, never real
reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caac.types import Trajectory
from caac.verifier.base import VerifierResult

__all__ = ["MockBackend", "MockVerifier"]


@dataclass
class MockBackend:
    """Simulates segment generation with tunable difficulty.

    ``difficulty`` controls how quickly token confidence rises: easy queries
    converge fast (a good policy should stop early), hard ones stay uncertain
    (a good policy should spend more). Enough structure to make an adaptive
    policy measurably better than a fixed one in tests.
    """

    seed: int = 0
    difficulty: float = 0.5
    finish_after_tokens: int = 256
    answer_pool: tuple = ("42", "7", "-3", "1/2")

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def generate(
        self, prompt: str, prefix: str, max_tokens: int, *, temperature: float | None = None
    ) -> Trajectory:
        """Generate one segment.

        ``temperature`` is accepted for interface parity with the real
        backends. Higher temperature widens the answer distribution, which is
        what BRANCH and sampling baselines rely on.
        """
        n = max_tokens
        progress = min(1.0, len(prefix) / max(1, self.finish_after_tokens))
        base = 0.5 + 0.45 * progress * (1.0 - self.difficulty)
        conf = np.clip(self._rng.normal(base, 0.08, size=n), 0.05, 0.999)
        logprobs = np.log(conf).tolist()
        entropies = (-np.log(conf) * 1.2).tolist()

        finished = len(prefix) + n >= self.finish_after_tokens

        # A trial answer is available at every decision point, matching the
        # trial-answer mechanism the controller relies on: the answer exists
        # before the trajectory finishes, and becomes more reliable as the
        # reasoning progresses. p(correct) rises with progress, falls with
        # difficulty.
        p_correct = min(0.95, (0.35 + 0.6 * progress) * (1.0 - 0.6 * self.difficulty))
        if temperature:
            # Higher temperature flattens the answer distribution toward chance.
            p_correct = 0.5 + (p_correct - 0.5) / (1.0 + float(temperature))
        idx = 0 if self._rng.random() < p_correct else 1
        answer = self.answer_pool[idx % len(self.answer_pool)]

        return Trajectory(
            text="x" * n,
            token_logprobs=logprobs,
            token_entropies=entropies,
            answer=answer,
            finished=finished,
        )

    def reset(self):
        self._rng = np.random.default_rng(self.seed)


@dataclass
class MockVerifier:
    """Noisy but informative verifier, with an explicit cost."""

    seed: int = 0
    accuracy: float = 0.75
    n_tokens: float = 32.0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def verify(self, prompt: str, trace: str) -> VerifierResult:
        signal = self._rng.beta(2, 2)
        score = float(np.clip(signal * self.accuracy + 0.1, 0.0, 1.0))
        return VerifierResult(score=score, n_tokens=self.n_tokens, n_forward_passes=self.n_tokens)
