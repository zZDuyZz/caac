"""Core types: cost algebra and action taxonomy."""
from caac.types import Action, CostVector, CostWeights


def test_action_spending_excludes_stop():
    assert Action.STOP not in Action.spending()
    assert set(Action.spending()) == {Action.CONTINUE, Action.VERIFY, Action.BRANCH}


def test_cost_vector_add_and_scalar():
    a = CostVector(tokens=10, forward_passes=10)
    b = CostVector(tokens=5, latency_s=1.0)
    c = a + b
    assert c.tokens == 15 and c.forward_passes == 10 and c.latency_s == 1.0
    w = CostWeights(tokens=1.0, latency_s=2.0)
    assert c.scalar(w) == 15 * 1.0 + 1.0 * 2.0


def test_cost_zero_is_identity():
    a = CostVector(tokens=3)
    assert (a + CostVector.zero()).tokens == 3
