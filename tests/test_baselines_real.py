"""The published-baseline reproductions run and are charged honestly."""
import pytest

from caac.backends.mock import MockBackend
from caac.baselines import BASELINE_REGISTRY, build_baseline
from caac.cost.accounting import CostMeter, CostSource
from caac.types import CostWeights

W = CostWeights(tokens=1.0)


@pytest.mark.parametrize("name", sorted(BASELINE_REGISTRY))
def test_baseline_runs_and_costs_something(name):
    meter = CostMeter()
    res = build_baseline(name).run(MockBackend(difficulty=0.4), "q", 4096, meter)
    assert res.n_segments > 0
    assert meter.total().scalar(W) > 0        # nothing is free
    assert meter.by_source[CostSource.REASONING].tokens > 0


def test_sweeping_knob_changes_cost():
    """Each baseline must expose a knob that traces a cost curve."""
    costs = []
    for n in (2, 6):
        m = CostMeter()
        build_baseline("greedy_cot", max_segments=n).run(MockBackend(), "q", 4096, m)
        costs.append(m.total().scalar(W))
    assert costs[1] > costs[0]


def test_deer_early_exit_is_cheaper_than_no_exit():
    lo, hi = CostMeter(), CostMeter()
    build_baseline("deer", threshold=0.0).run(MockBackend(difficulty=0.2), "q", 4096, lo)
    build_baseline("deer", threshold=1.1).run(MockBackend(difficulty=0.2), "q", 4096, hi)
    # threshold 0.0 exits immediately; 1.1 never exits
    assert lo.total().scalar(W) <= hi.total().scalar(W)


def test_self_consistency_scales_with_k():
    small, large = CostMeter(), CostMeter()
    build_baseline("self_consistency", k=2).run(MockBackend(), "q", 4096, small)
    build_baseline("self_consistency", k=6).run(MockBackend(), "q", 4096, large)
    assert large.total().scalar(W) > small.total().scalar(W)
