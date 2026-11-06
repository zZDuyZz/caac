"""CAAC -- certified, compute-adaptive control of LLM reasoning.

Per question, choose among answering now, drawing more samples, escalating to a
larger model, or abstaining, with the error rate of the answers served
certified by Learn-then-Test. Everything except pool generation runs offline
on CPU by replaying a saved sample pool (``caac.pool``).
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
