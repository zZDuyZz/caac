"""CAAC-Audit: exact tests, verdicts, and the error-rate guarantee (Monte-Carlo, kept small)."""

import math
import random

import pytest

from caac.pool_v1.audit import (AuditInput, PolicyData, audit, binom_cdf, in_half_a, p_mean_neg,
                                p_mean_nonneg, p_risk_above, p_risk_below, served_needed)


def test_binom_cdf_against_scipy():
    st = pytest.importorskip("scipy.stats")
    for n, p in ((1, .1), (22, .1), (250, .08), (1000, .1), (3000, .05)):
        for k in (0, 1, n // 20, n // 10, n - 1, n):
            assert binom_cdf(k, n, p) == pytest.approx(st.binom.cdf(k, n, p), rel=1e-9, abs=1e-300)


def test_risk_pvalues_directions():
    assert p_risk_below(0, 22, .10) <= .10 < p_risk_below(0, 21, .10)     # the floor 22
    assert p_risk_above(30, 100, .10) < 1e-6 and p_risk_above(5, 100, .10) > .5
    assert p_risk_below(0, 0, .1) == p_risk_above(0, 0, .1) == 1.0


def test_bernstein_pvalue_matches_hand_bound():
    rng = random.Random(3)
    x = [rng.uniform(-1, 1) - 0.4 for _ in range(400)]               # mean ~ -0.4 in [-1.4, 0.6]
    p = p_mean_neg(x, -1.4, 0.6)
    n = len(x); m = sum(x) / n; v = sum((a - m) ** 2 for a in x) / (n - 1); w = 2.0
    slack = lambda d: math.sqrt(2 * v * math.log(2 / d) / n) + 7 * w * math.log(2 / d) / (3 * (n - 1))
    assert m + slack(p * 1.0001) < 0 <= m + slack(p * 0.999)          # p is the boundary level
    assert p_mean_nonneg(x, -1.4, 0.6) == 1.0                          # no evidence the other way


def test_split_is_fixed_and_balanced():
    qs = [f"q{i}" for i in range(4000)]
    a = [in_half_a(q) for q in qs]
    assert a == [in_half_a(q) for q in qs]
    assert 0.45 < sum(a) / len(a) < 0.55
    assert [in_half_a(q, "other") for q in qs] != a


def make(n, policies, big=0.5, seed=0):
    """policies = [(name, q_served, risk, mean_cost)]; costs uniform around the mean in [0, 1]."""
    rng = random.Random(seed)
    qids = [f"q{i}" for i in range(n)]
    pols = []
    for name, q, r, m in policies:
        served = [rng.random() < q for _ in qids]
        err = [s and rng.random() < r for s in served]
        cost = [min(1.0, max(0.0, rng.uniform(m - .2, m + .2))) for _ in qids]
        pols.append(PolicyData(name, served, err, cost))
    bc = [min(1.0, max(0.0, rng.uniform(big - .2, big + .2))) for _ in qids]
    return AuditInput(qids, bc, pols, c_max=1.0)


def test_clear_go_clear_nogo_small_n_inconclusive():
    go = audit(make(1000, [("cheap_safe", .8, .03, .25), ("other", .9, .2, .2)]))
    assert go.verdict == "GO" and go.chosen == "cheap_safe" and not go.conflict
    no = audit(make(1000, [("risky", .8, .25, .25), ("pricey", .9, .02, .48)]))
    assert no.verdict == "NO-GO"
    small = audit(make(40, [("cheap_safe", .8, .03, .25)]))
    assert small.verdict == "INCONCLUSIVE"
    assert set(go.to_dict()) >= {"verdict", "go", "nogo", "n_a", "n_b"}


def test_wrong_verdict_rate_at_most_delta():
    """Infeasible by a small margin (risk .13 > alpha .10, 5 candidates): GO must be rare;
    feasible at the boundary (risk = alpha, saving = s exactly): NO-GO must be rare."""
    reps = 120
    wrong_go = sum(audit(make(400, [(f"p{k}", .8, .13, .25) for k in range(5)], seed=s)).verdict == "GO"
                   for s in range(reps))
    assert wrong_go / reps <= 0.10
    wrong_nogo = sum(audit(make(400, [("b", .8, .10, .40)], seed=1000 + s)).verdict == "NO-GO"
                     for s in range(reps))
    assert wrong_nogo / reps <= 0.10


def test_input_errors():
    inp = make(50, [("p", .8, .1, .3)])
    inp.policies[0].cost[0] = 1.5
    with pytest.raises(ValueError, match="outside"):
        audit(inp)
    inp = make(50, [("p", .8, .1, .3)])
    inp.policies[0].served[0], inp.policies[0].error[0] = False, True
    with pytest.raises(ValueError, match="did not serve"):
        audit(inp)


def test_served_needed():
    assert served_needed(0.0, .10, .10) == 22
    assert served_needed(0.10, .10, .10) is None
    n = served_needed(0.05, .10, .10)
    assert n is not None and binom_cdf(math.floor(.05 * n), n, .10) <= .10
