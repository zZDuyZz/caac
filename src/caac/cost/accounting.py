"""Overhead-honest cost accounting.

The central claim is a Pareto improvement, so the cost side must not quietly
omit anything. Every unit of compute is attributed to a *source*, and the
reported total is always the sum over all sources -- including the controller.

A method that reports token savings while hiding verifier passes is exactly
the failure mode this accounting is designed to make impossible.
"""

from __future__ import annotations

import enum
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from caac.types import CostVector, CostWeights

__all__ = ["CostSource", "CostMeter", "DEPLOYMENT_PROFILES"]


class CostSource(enum.Enum):
    """Where a unit of compute went.

    Keeping REASONING separate from VERIFIER / BRANCH / CONTROLLER is what
    lets the 'gross vs net saving' distinction be reported honestly.
    """

    REASONING = "reasoning"
    VERIFIER = "verifier"
    BRANCH = "branch"
    CONTROLLER = "controller"


@dataclass
class _Tracker:
    """Mutable handle yielded by ``CostMeter.track``."""

    tokens: float = 0.0
    forward_passes: float = 0.0
    kv_bytes: float = 0.0

    def add(self, tokens=0.0, forward_passes=0.0, kv_bytes=0.0) -> None:
        self.tokens += tokens
        self.forward_passes += forward_passes
        self.kv_bytes += kv_bytes


@dataclass
class CostMeter:
    """Accumulates cost per source over one query (or one whole run)."""

    by_source: dict[CostSource, CostVector] = field(
        default_factory=lambda: {s: CostVector.zero() for s in CostSource}
    )

    def add(self, source: CostSource, cost: CostVector) -> None:
        self.by_source[source] = self.by_source[source] + cost

    def total(self) -> CostVector:
        out = CostVector.zero()
        for cost in self.by_source.values():
            out = out + cost
        return out

    def overhead(self) -> CostVector:
        """Everything that is *not* raw reasoning tokens.

        Must be smaller than the compute saved, else the method is a net loss.
        """
        out = CostVector.zero()
        for source, cost in self.by_source.items():
            if source is not CostSource.REASONING:
                out = out + cost
        return out

    def overhead_ratio(self, weights: CostWeights) -> float:
        total = self.total().scalar(weights)
        return self.overhead().scalar(weights) / total if total > 0 else 0.0

    def controller_ratio(self, weights: CostWeights) -> float:
        """Share of total cost spent deciding rather than reasoning.

        Design target from the proposal: below 1-2%.
        """
        total = self.total().scalar(weights)
        if total <= 0:
            return 0.0
        return self.by_source[CostSource.CONTROLLER].scalar(weights) / total

    def merge(self, other: "CostMeter") -> None:
        for source, cost in other.by_source.items():
            self.add(source, cost)

    @contextmanager
    def track(self, source: CostSource) -> Iterator[_Tracker]:
        """Time a block and attribute its cost to ``source``.

        Wall-clock is measured automatically; token/forward-pass counts are
        supplied by the caller because only the caller knows them.
        """
        tracker = _Tracker()
        start = time.perf_counter()
        try:
            yield tracker
        finally:
            elapsed = time.perf_counter() - start
            self.add(
                source,
                CostVector(
                    tokens=tracker.tokens,
                    forward_passes=tracker.forward_passes,
                    kv_bytes=tracker.kv_bytes,
                    latency_s=elapsed,
                ),
            )

    def summary(self, weights: CostWeights) -> dict[str, float]:
        out: dict[str, float] = {}
        for source, cost in self.by_source.items():
            out[f"{source.value}_tokens"] = cost.tokens
            out[f"{source.value}_scalar"] = cost.scalar(weights)
        out["total_scalar"] = self.total().scalar(weights)
        out["overhead_ratio"] = self.overhead_ratio(weights)
        out["controller_ratio"] = self.controller_ratio(weights)
        return out


# Deployment profiles: the same controller re-priced by swapping weights.
DEPLOYMENT_PROFILES: dict[str, CostWeights] = {
    "token_only": CostWeights(tokens=1.0),
    "throughput": CostWeights(tokens=1.0, forward_passes=0.5, kv_bytes=1e-6),
    "latency": CostWeights(tokens=0.2, forward_passes=1.0, latency_s=50.0),
}
