"""Core domain types shared across the CAAC codebase.

The vocabulary of the whole project: an LLM produces reasoning *trajectories*,
a controller observes a *state*, picks an *action*, and pays a *cost*.

Design rule: this module is pure Python + numpy. It never imports torch,
transformers, or vllm, so the decision logic stays testable on CPU.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = [
    "Action",
    "CostWeights",
    "CostVector",
    "Trajectory",
    "ReasoningState",
    "Decision",
    "RolloutResult",
]


class Action(enum.Enum):
    """The four ways to spend (or stop spending) compute.

    STOP     -- terminate reasoning and emit the current answer.
    CONTINUE -- extend the active trajectory by one segment.
    VERIFY   -- pay a verifier pass to sharpen the belief (pure information).
    BRANCH   -- roll back and explore an alternative trajectory.
    """

    STOP = "stop"
    CONTINUE = "continue"
    VERIFY = "verify"
    BRANCH = "branch"

    @classmethod
    def spending(cls) -> tuple["Action", ...]:
        """Actions that consume budget (everything except STOP)."""
        return (cls.CONTINUE, cls.VERIFY, cls.BRANCH)


@dataclass(frozen=True)
class CostWeights:
    """Deployment profile: how each cost dimension is priced.

    Changing these weights re-prices the action space *without retraining* --
    this is what lets one controller serve both latency- and
    throughput-critical deployments.
    """

    tokens: float = 1.0
    forward_passes: float = 0.0
    kv_bytes: float = 0.0
    latency_s: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.tokens, self.forward_passes, self.kv_bytes, self.latency_s],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CostVector:
    """Multi-dimensional cost of one action.

    Kept as a vector rather than a scalar so the same measurement can be
    re-priced under different deployment profiles.
    """

    tokens: float = 0.0
    forward_passes: float = 0.0
    kv_bytes: float = 0.0
    latency_s: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.tokens, self.forward_passes, self.kv_bytes, self.latency_s],
            dtype=np.float64,
        )

    def scalar(self, weights: CostWeights) -> float:
        """Collapse to the scalar cost c(a, s) used in the VOC rule."""
        return float(self.as_array() @ weights.as_array())

    def __add__(self, other: "CostVector") -> "CostVector":
        return CostVector(
            tokens=self.tokens + other.tokens,
            forward_passes=self.forward_passes + other.forward_passes,
            kv_bytes=self.kv_bytes + other.kv_bytes,
            latency_s=self.latency_s + other.latency_s,
        )

    @classmethod
    def zero(cls) -> "CostVector":
        return cls()


@dataclass
class Trajectory:
    """One reasoning path: text plus the token-level statistics it produced.

    token_logprobs and token_entropies are the raw material for every
    uncertainty signal, and come for free during decoding.
    """

    text: str = ""
    token_logprobs: list[float] = field(default_factory=list)
    token_entropies: list[float] = field(default_factory=list)
    answer: str | None = None
    finished: bool = False
    parent_step: int = 0

    @property
    def n_tokens(self) -> int:
        return len(self.token_logprobs)

    def extend(self, other: "Trajectory") -> None:
        """Append a newly generated segment in place."""
        self.text += other.text
        self.token_logprobs.extend(other.token_logprobs)
        self.token_entropies.extend(other.token_entropies)
        if other.answer is not None:
            self.answer = other.answer
        self.finished = other.finished


@dataclass
class ReasoningState:
    """Everything the controller may look at when choosing an action.

    Split into content / belief / resources / features: the first three are
    bookkeeping, and only ``features`` is fed to the learned estimators.
    """

    query_id: str
    prompt: str
    trajectories: list[Trajectory] = field(default_factory=list)
    active: int = 0
    step: int = 0

    budget_remaining: float = float("inf")
    initial_budget: float = float("inf")
    n_verify_used: int = 0
    n_branches: int = 1

    belief: float = 0.5
    features: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def active_trajectory(self) -> Trajectory:
        if not self.trajectories:
            raise ValueError("ReasoningState has no trajectories yet")
        return self.trajectories[self.active]

    @property
    def current_answer(self) -> str | None:
        return self.active_trajectory.answer

    @property
    def total_tokens(self) -> int:
        return sum(t.n_tokens for t in self.trajectories)

    def copy(self, **changes: Any) -> "ReasoningState":
        return replace(self, **changes)


@dataclass(frozen=True)
class Decision:
    """The controller's output at one decision point.

    ``voc`` is kept for every action (not just the winner) because the shape
    of that map across belief bins is itself an experimental result (RQ3/H2).
    """

    action: Action
    voc: dict[Action, float]
    belief: float
    reason: str = ""

    @property
    def best_spending_voc(self) -> float:
        vals = [v for a, v in self.voc.items() if a in Action.spending()]
        return max(vals) if vals else float("-inf")


@dataclass
class RolloutResult:
    """Outcome of running one query to completion under some policy."""

    query_id: str
    answer: str | None
    correct: bool
    cost: CostVector
    n_decisions: int
    action_counts: dict[Action, int] = field(default_factory=dict)
    beliefs: list[float] = field(default_factory=list)

    def to_row(self, weights: CostWeights) -> dict[str, Any]:
        row: dict[str, Any] = {
            "query_id": self.query_id,
            "correct": int(self.correct),
            "tokens": self.cost.tokens,
            "forward_passes": self.cost.forward_passes,
            "kv_bytes": self.cost.kv_bytes,
            "latency_s": self.cost.latency_s,
            "cost_scalar": self.cost.scalar(weights),
            "n_decisions": self.n_decisions,
        }
        for action in Action:
            row[f"n_{action.value}"] = self.action_counts.get(action, 0)
        return row
