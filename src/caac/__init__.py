"""CAAC -- Cost-Aware Adaptive Computation for Efficient LLM Reasoning.

Deciding *when* and *where* to spend inference-time compute under a limited
budget, by comparing the value of computation against its cost.
"""

__version__ = "0.1.0"

from caac.types import Action, CostVector, CostWeights, Decision, ReasoningState

__all__ = ["Action", "CostVector", "CostWeights", "Decision", "ReasoningState", "__version__"]


def __getattr__(name):
    # Lazy import so `import caac` stays light and torch-free.
    if name == "CAAC":
        from caac.api import CAAC
        return CAAC
    raise AttributeError(f"module 'caac' has no attribute {name!r}")
