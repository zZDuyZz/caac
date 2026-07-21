"""Pareto frontier operations and comparison metrics."""
from caac.eval.pareto import (
    OperatingPoint, accuracy_at_cost, compare_frontiers,
    cost_at_accuracy, dominates, pareto_frontier,
)


def test_dominance():
    a = OperatingPoint(cost=1.0, accuracy=0.9)
    b = OperatingPoint(cost=2.0, accuracy=0.8)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_frontier_drops_dominated_points():
    pts = [
        OperatingPoint(1.0, 0.9), OperatingPoint(2.0, 0.8),  # b dominated by a
        OperatingPoint(3.0, 0.95),
    ]
    front = pareto_frontier(pts)
    accs = {p.accuracy for p in front}
    assert 0.8 not in accs  # dominated
    assert 0.9 in accs and 0.95 in accs


def test_cost_at_accuracy_and_accuracy_at_cost():
    pts = [OperatingPoint(1.0, 0.7), OperatingPoint(2.0, 0.85), OperatingPoint(4.0, 0.92)]
    assert cost_at_accuracy(pts, 0.85) == 2.0
    assert cost_at_accuracy(pts, 0.99) is None  # never reached
    assert accuracy_at_cost(pts, 3.0) == 0.85


def test_compare_frontiers_reports_cost_ratio():
    method = [OperatingPoint(1.0, 0.8), OperatingPoint(2.0, 0.9)]
    baseline = [OperatingPoint(2.0, 0.8), OperatingPoint(4.0, 0.9)]
    cmp = compare_frontiers(method, baseline, accuracy_targets=[0.8, 0.9])
    assert cmp["mean_cost_ratio"] < 1.0  # method is cheaper
