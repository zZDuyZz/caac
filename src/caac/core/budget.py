"""Budget management via a dual variable.

The budget constraint in the proposal is an *expectation over the whole
workload*, not a per-query cap. It is enforced by a Lagrange multiplier lambda:
solve the unconstrained problem at a given lambda, then adjust lambda by dual
ascent on a validation set until the average cost hits the target B.

At inference lambda is fixed, so each query is processed independently -- a
property that matters for serving -- while the average constraint still holds.
Sweeping lambda over a grid traces the whole accuracy-cost Pareto frontier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["DualLambda", "lambda_grid"]


@dataclass
class DualLambda:
    """Dual-ascent controller for the workload-level budget constraint."""

    lam: float = 1.0
    step_size: float = 0.1
    lam_min: float = 0.0
    lam_max: float = 1e3
    history: list[float] = field(default_factory=list)

    def update(self, measured_cost: float, target_budget: float) -> float:
        """One dual-ascent step: lam <- [lam + eta (cost - B)]_+.

        If measured cost exceeds the target, lambda rises (compute gets more
        expensive, the policy spends less); if under, lambda falls.
        """
        grad = measured_cost - target_budget
        self.lam = min(self.lam_max, max(self.lam_min, self.lam + self.step_size * grad))
        self.history.append(self.lam)
        return self.lam

    def calibrate(
        self, cost_fn, target_budget: float, n_steps: int = 50, tol: float = 1e-3
    ) -> float:
        """Iterate dual ascent until average cost ~ target.

        ``cost_fn(lam) -> float`` returns the mean cost achieved at a given
        lambda (typically a sweep over the validation set).
        """
        for _ in range(n_steps):
            cost = cost_fn(self.lam)
            self.update(cost, target_budget)
            if abs(cost - target_budget) < tol:
                break
        return self.lam


def lambda_grid(low: float = 1e-3, high: float = 10.0, n: int = 12) -> list[float]:
    """Log-spaced lambda grid for a Pareto sweep."""
    import numpy as np

    return list(np.geomspace(low, high, n))
