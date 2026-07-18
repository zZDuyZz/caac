"""Overhead-honest accounting: total is always the sum over sources."""
from caac.cost.accounting import CostMeter, CostSource
from caac.types import CostVector, CostWeights


def test_total_sums_all_sources():
    m = CostMeter()
    m.add(CostSource.REASONING, CostVector(tokens=100))
    m.add(CostSource.VERIFIER, CostVector(tokens=30))
    m.add(CostSource.CONTROLLER, CostVector(tokens=5))
    w = CostWeights(tokens=1.0)
    assert m.total().scalar(w) == 135


def test_overhead_excludes_reasoning():
    m = CostMeter()
    m.add(CostSource.REASONING, CostVector(tokens=100))
    m.add(CostSource.VERIFIER, CostVector(tokens=30))
    m.add(CostSource.CONTROLLER, CostVector(tokens=10))
    w = CostWeights(tokens=1.0)
    assert m.overhead().scalar(w) == 40
    assert abs(m.overhead_ratio(w) - 40 / 140) < 1e-9
    assert abs(m.controller_ratio(w) - 10 / 140) < 1e-9


def test_track_context_measures_wallclock():
    m = CostMeter()
    with m.track(CostSource.REASONING) as t:
        t.add(tokens=50, forward_passes=50)
    assert m.by_source[CostSource.REASONING].tokens == 50
    assert m.by_source[CostSource.REASONING].latency_s >= 0.0
