"""Reproductions of published baselines (phase P1).

Distinct from ``caac.policy.baselines``: those are simplified decision rules for
testing the control loop on CPU, while these follow the mechanism each paper
describes so their numbers can be validated against the published ones.
"""

from caac.baselines.methods import (
    BASELINE_REGISTRY,
    BaselineResult,
    BudgetForcing,
    DeepConf,
    DEER,
    GreedyCoT,
    SelfConsistency,
    build_baseline,
)

__all__ = [
    "BaselineResult", "GreedyCoT", "SelfConsistency", "BudgetForcing",
    "DEER", "DeepConf", "BASELINE_REGISTRY", "build_baseline",
]
