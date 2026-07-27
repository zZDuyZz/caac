"""Lazy-evaluation helper.

The VOCPolicy already supports lazy pruning internally (skip evaluating
VERIFY/BRANCH far from the margin). This module exposes the same predicate so a
scheduler can report, per step, how many action evaluations were saved -- an
ablation on controller overhead.
"""

from __future__ import annotations

__all__ = ["lazy_skip"]


def lazy_skip(belief: float, low: float = 0.15, high: float = 0.97) -> bool:
    """True if information-acquiring actions can be skipped at this belief.

    Only prunes actions that would not have been chosen anyway, so the selected
    action is never changed.
    """
    return belief < low or belief > high
