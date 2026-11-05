import pytest

import certify as C
from caac.pool.ltt import LambdaResult, certify_bonferroni, certify_fixed_sequence
from caac.pool.replay import RunMetrics
from caac.pool.schema import Completion, PoolRecord


def _pool(n=100):
    return {f"q{i}": PoolRecord(f"q{i}", "d", "m", "x", 10,
                                Completion("x", True, 10, "stop"), []) for i in range(n)}


def test_split_half_is_deterministic_disjoint_and_balanced():
    a, b = C.split_half(_pool(200))
    assert set(a) & set(b) == set() and len(a) + len(b) == 200
    assert 70 < len(a) < 130                      # roughly balanced
    a2, _ = C.split_half(_pool(200))
    assert set(a) == set(a2)                      # same split every run
    # independent of insertion order
    p = _pool(200)
    a3, _ = C.split_half({k: p[k] for k in reversed(list(p))})
    assert set(a) == set(a3)


def test_best_hindsight_picks_max_coverage_within_risk():
    def pair(lam, cov, risk):
        m = RunMetrics(n=100, served=50, correct_served=0, wrong_served=int(50 * risk))
        m.__dict__["served"] = 50
        return (LambdaResult(lam=lam, n_served=50, n_wrong=int(50 * risk), coverage=cov,
                             mean_cost=1.0), m)
    pairs = [pair(0.9, 0.3, 0.02), pair(0.5, 0.6, 0.08), pair(0.1, 0.9, 0.30)]
    best = C.best_hindsight(pairs, alpha=0.10)
    assert best[0].lam == 0.5 and best[0].coverage == 0.6
    assert C.best_hindsight(pairs, alpha=0.001) is None


def test_fixed_sequence_certifies_at_least_as_much_as_bonferroni():
    """The property we rely on (PriceCheck's finding): no multiplicity penalty."""
    def mk():   # conservative first: coverage grows, risk grows
        return [LambdaResult(lam=1 - i / 10, n_served=200, n_wrong=w, coverage=c, mean_cost=1.0)
                for i, (w, c) in enumerate([(2, .2), (6, .4), (12, .6), (30, .8), (60, 1.)])]
    b = certify_bonferroni(mk(), alpha=0.10, delta=0.1)
    f = certify_fixed_sequence(mk(), alpha=0.10, delta=0.1)
    nb = sum(r.certified for r in b)
    nf = sum(r.certified for r in f)
    assert nf >= nb


def test_fixed_sequence_stops_at_the_first_failure():
    rs = [LambdaResult(lam=1.0, n_served=200, n_wrong=2, coverage=.2, mean_cost=1.),
          LambdaResult(lam=0.9, n_served=200, n_wrong=90, coverage=.4, mean_cost=1.),  # fails
          LambdaResult(lam=0.8, n_served=200, n_wrong=2, coverage=.9, mean_cost=1.)]   # would pass
    out = certify_fixed_sequence(rs, alpha=0.10, delta=0.1)
    assert [r.certified for r in out] == [True, False, False]


def test_grid_is_ordered_most_conservative_first():
    grid = [round(1.0 - i / 20, 4) for i in range(21)]
    assert grid[0] == 1.0 and grid[-1] == 0.0
    assert grid == sorted(grid, reverse=True)
