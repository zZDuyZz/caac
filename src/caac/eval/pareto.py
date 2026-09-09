"""Pareto frontiers and the summary metrics built on them.

The headline claim is a frontier comparison, so the primitives here are
frontier operations. Two derived numbers matter most:

    cost@iso-accuracy  -- how much cheaper at matched quality
    accuracy@iso-cost  -- how much better at matched spend

Both require interpolation between measured points, done explicitly and
conservatively rather than by fitting a curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "OperatingPoint",
    "dominates",
    "pareto_frontier",
    "cost_at_accuracy",
    "accuracy_at_cost",
    "acc_cost_auc",
    "compare_frontiers",
]


@dataclass(frozen=True)
class OperatingPoint:
    """One (cost, accuracy) pair produced by one setting of a method."""

    cost: float
    accuracy: float
    label: str = ""


def dominates(a: OperatingPoint, b: OperatingPoint) -> bool:
    """True if ``a`` is at least as good on both axes, strictly better on one."""
    not_worse = a.cost <= b.cost and a.accuracy >= b.accuracy
    strictly = a.cost < b.cost or a.accuracy > b.accuracy
    return not_worse and strictly


def pareto_frontier(points: list) -> list:
    """Keep only non-dominated points, sorted by increasing cost."""
    frontier: list = []
    for cand in sorted(points, key=lambda p: (p.cost, -p.accuracy)):
        if any(dominates(k, cand) for k in frontier):
            continue
        frontier = [k for k in frontier if not dominates(cand, k)]
        frontier.append(cand)
    return sorted(frontier, key=lambda p: p.cost)


def cost_at_accuracy(points: list, target: float):
    """Minimum cost required to reach ``target`` accuracy, else None.

    None (not an extrapolated guess) is returned if the target is never met.
    """
    frontier = pareto_frontier(points)
    for p in frontier:
        if p.accuracy >= target:
            return p.cost
    return None


def accuracy_at_cost(points: list, budget: float):
    """Best accuracy achievable at or below ``budget``, else None."""
    affordable = [p for p in points if p.cost <= budget]
    return max((p.accuracy for p in affordable), default=None)


def acc_cost_auc(points: list, cost_min=None, cost_max=None, n_grid: int = 100) -> float:
    """Normalised area under the accuracy-cost curve.

    Meaningful only when the same cost range is used across methods, so callers
    should pass the shared range explicitly.
    """
    frontier = pareto_frontier(points)
    if len(frontier) < 2:
        return float(frontier[0].accuracy) if frontier else 0.0
    costs = np.array([p.cost for p in frontier])
    accs = np.array([p.accuracy for p in frontier])
    lo = costs.min() if cost_min is None else cost_min
    hi = costs.max() if cost_max is None else cost_max
    if hi <= lo:
        return float(accs.max())
    grid = np.linspace(lo, hi, n_grid)
    interp = np.array([accs[costs <= c].max() if np.any(costs <= c) else accs[0] for c in grid])
    return float(np.trapz(interp, grid) / (hi - lo))


def compare_frontiers(method: list, baseline: list, accuracy_targets=None) -> dict:
    """Head-to-head summary. ``cost_ratio`` < 1 means cheaper at matched accuracy."""
    targets = accuracy_targets or [0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for t in targets:
        m, b = cost_at_accuracy(method, t), cost_at_accuracy(baseline, t)
        if m is None or b is None or b == 0:
            continue
        rows.append({"accuracy_target": t, "method_cost": m, "baseline_cost": b,
                     "cost_ratio": m / b, "saving": 1.0 - m / b})
    all_costs = [p.cost for p in method + baseline]
    lo, hi = (min(all_costs), max(all_costs)) if all_costs else (0.0, 1.0)
    return {
        "per_target": rows,
        "mean_cost_ratio": float(np.mean([r["cost_ratio"] for r in rows])) if rows else float("nan"),
        "method_auc": acc_cost_auc(method, lo, hi),
        "baseline_auc": acc_cost_auc(baseline, lo, hi),
    }
