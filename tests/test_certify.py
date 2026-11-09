import pytest

from caac.pool.ltt import LambdaResult, certify_bonferroni, certify_fixed_sequence


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


def test_min_served_matches_the_flawless_run_bound():
    """n_min is the smallest n with (1-alpha)^n <= delta, i.e. the fewest
    questions a ZERO-error run needs before its p-value clears delta."""
    from caac.pool.ltt import min_served_certifiable, pvalue_selective_risk
    for alpha, delta in [(0.1, 0.1), (0.05, 0.1), (0.2, 0.05), (0.1, 0.005)]:
        n = min_served_certifiable(alpha, delta)
        assert pvalue_selective_risk(n, 0, alpha) <= delta
        assert pvalue_selective_risk(n - 1, 0, alpha) > delta


def test_min_served_is_22_at_the_preregistered_levels():
    from caac.pool.ltt import min_served_certifiable
    assert min_served_certifiable(0.10, 0.10) == 22


def test_bonferroni_raises_the_floor():
    """Splitting delta over the grid makes the floor markedly higher -- the
    reason Bonferroni certifies nothing on our smallest calibration sets."""
    from caac.pool.ltt import min_served_certifiable
    assert min_served_certifiable(0.1, 0.1 / 21) > min_served_certifiable(0.1, 0.1)


def test_certification_floor_is_a_coverage_fraction():
    from caac.pool.ltt import certification_floor
    assert certification_floor(220, 0.1, 0.1) == pytest.approx(0.1)
    assert certification_floor(10, 0.1, 0.1) == 1.0      # clamped
    assert certification_floor(0, 0.1, 0.1) == 1.0


def test_min_served_rejects_degenerate_levels():
    from caac.pool.ltt import min_served_certifiable
    for bad in [(0.0, 0.1), (1.0, 0.1), (0.1, 0.0), (0.1, 1.0)]:
        with pytest.raises(ValueError):
            min_served_certifiable(*bad)
