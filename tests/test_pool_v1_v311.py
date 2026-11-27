"""Amendments v3-10 / v3-11 (relative claim): profiler modes, fixed-n bets, exact conditional test, per-policy range."""
import math
import random

import pytest

from caac.pool_v1.audit import (AuditInput, PolicyData, REL_MODES, REL_TEST, _betting_fixed_n_p, _betting_p,
                                _betting_p_cap, _cp_lower, _cp_upper, _order, audit3, binom_cdf, p_rel_exact_above,
                                p_rel_exact_go, policy_lo, rel_legs)
from caac.pool_v1.certsave import certified_saving
from caac.pool_v1.config import config_from_dict
from caac.pool_v1.gap import gap_relative


def _fixed_n_plain(x, lo, hi, level, cap=0.9):
    """Independent restatement of the v3-11 bet (design DECISION_v311_before_run.md)."""
    n = len(x)
    lmax, prior, c = cap / -lo, (hi - lo) ** 2 / 4, 2 * math.log(1 / level)
    k = best = 0.0
    for t in range(n):
        past = x[:t]
        if t:
            mu = sum(past) / t
            v = (prior + sum((y - mu) ** 2 for y in past)) / (t + 1)
        else:
            v = prior
        k += math.log1p(min(lmax, math.sqrt(c / (n * v))) * x[t])
        best = max(best, k)
    return min(1.0, math.exp(-best))


def test_fixed_n_matches_restatement():
    rng = random.Random(3)
    for _ in range(60):
        n = rng.choice((2, 7, 40, 200)); lo = -rng.uniform(.2, 3); hi = rng.uniform(.2, 3)
        x = [rng.uniform(lo, hi) for _ in range(n)]
        lvl = rng.choice((.1, .02, .004))
        assert abs(_betting_fixed_n_p(x, lo, hi, lvl, None) - _fixed_n_plain(x, lo, hi, lvl)) < 1e-9


def test_fixed_n_size_at_null_boundary():
    rng = random.Random(11)
    for lo, hi in ((-1.0, 1.0), (-4.0, 1.0)):
        w = min(-lo, hi); rej = 0; R = 1500
        for _ in range(R):
            x = [rng.uniform(-w, w) for _ in range(150)]
            rej += _betting_fixed_n_p(x, lo, hi, 0.05, None) <= 0.05
        assert rej / R <= 0.05 + 3 * math.sqrt(.05 * .95 / R)


def test_cap_half_is_v38_betting():
    rng = random.Random(5)
    x = [rng.uniform(-1, 2) for _ in range(300)]
    o = list(range(300)); rng.shuffle(o)
    assert _betting_p_cap(x, -1, 2, o, 0.5) == _betting_p(x, -1, 2, o)
    assert _betting_p_cap(x, -1, 2, o, 0.9) <= 1.0


def test_cp_bounds():
    for m, n in ((0, 50), (3, 50), (49, 50)):
        u = _cp_upper(m, n, 1e-3)
        assert binom_cdf(m, n, u) <= 1e-3 + 1e-12
        if m:
            l = _cp_lower(m, n, 1e-3)
            assert 1 - binom_cdf(m - 1, n, l) <= 1e-3 + 1e-9 and l <= m / n


def _size_exact(fn, eps, n, alpha, pts):
    """Exact size of 'p <= alpha' at trinomial points (pH, pB) by enumeration."""
    worst = 0.0
    for pH, pB in pts:
        p0 = 1 - pH - pB; tot = 0.0
        for h in range(n + 1):
            for b in range(n + 1 - h):
                D = [1] * h + [-1] * b + [0] * (n - h - b)
                if fn(D, eps) <= alpha:
                    lp = (math.lgamma(n + 1) - math.lgamma(h + 1) - math.lgamma(b + 1) - math.lgamma(n - h - b + 1)
                          + (h * math.log(pH) if h else 0) + (b * math.log(pB) if b else 0)
                          + ((n - h - b) * math.log(p0) if n - h - b else 0))
                    tot += math.exp(lp)
        worst = max(worst, tot)
    return worst


def test_exact_tests_valid_small_n():
    eps, n, a = 0.05, 24, 0.1
    go_null = [(pb + eps, pb) for pb in (0.01, 0.1, 0.3)]          # E[D] = eps (GO null boundary)
    assert _size_exact(p_rel_exact_go, eps, n, a, go_null) <= a
    ng_null = [(pb + eps, pb) for pb in (0.02, 0.15, 0.3)] + [(0.05, 0.05)]
    assert _size_exact(p_rel_exact_above, eps, n, a, ng_null) <= a


def _inp(n=400, seed=1, cap=True):
    rng = random.Random(seed)
    qids = [f"q{i}" for i in range(n)]
    big = [rng.uniform(5, 10) for _ in range(n)]
    be = [rng.random() < .2 for _ in range(n)]
    pols = []
    for k, (q, smax) in enumerate(((.7, 1.0), (.9, 2.0))):
        served, err, cost, sp = [], [], [], []
        for i in range(n):
            a = rng.uniform(0, smax)
            if rng.random() < q:                                  # small answers
                e = be[i] if rng.random() < .9 else (not be[i])
                cost.append(a)
            else:                                                 # escalate: big-only answer
                e = be[i]; cost.append(a + big[i])
            served.append(True); err.append(bool(e)); sp.append(a)
        pols.append(PolicyData(f"p{k}", served, err, cost, small_part=sp, small_cap=smax if cap else None))
    return AuditInput(qids, big, pols, c_max=12.0, small_max=2.0, big_max=10.0, big_error=be)


def test_modes_run_and_legacy_unchanged():
    inp = _inp()
    r_legacy = audit3(inp, claim="relative", rel_test="bet")
    assert r_legacy.go["rel_test"] == "bet" and "rel_mode" not in r_legacy.go
    for mode in REL_MODES:
        r = audit3(inp, claim="relative", rel_mode=mode)
        assert r.go["rel_test"] == REL_TEST and r.go["rel_mode"] == mode
        assert r.verdict in ("GO", "NO-GO", "INCONCLUSIVE")
    with pytest.raises(ValueError):
        audit3(inp, claim="relative", rel_mode="peek")


def test_absolute_ignores_mode():
    inp = _inp()
    a = audit3(inp, claim="absolute").to_dict()
    b = audit3(inp, claim="absolute", rel_mode="anytime").to_dict()
    assert a == b


def test_policy_lo_and_guard():
    inp = _inp()
    assert policy_lo(inp, inp.policies[0], .2) == -(1.0 + .2 * 10.0)
    assert policy_lo(_inp(cap=False), _inp(cap=False).policies[0], .2) == -(2.0 + .2 * 10.0)
    bad = _inp()
    bad.policies[0].small_cap = 0.5                               # rows exceed it
    with pytest.raises(ValueError):
        audit3(bad, claim="relative")


def test_anytime_is_betting_only():
    D = [0] * 300 + [1] * 3 + [-1] * 9
    d = [1.0] * len(D)
    o = list(range(len(D)))
    pr_go, ps_go, pr_ng, ps_ng = rel_legs(D, d, 0.02, o, 0.05, -2.0, -3.0, 4.0, "anytime")
    assert pr_go == _betting_p([0.02 - v for v in D], 0.02 - 1, 1.02, o)
    assert pr_ng == _betting_p([v - 0.02 for v in D], -1.02, 0.98, o)
    assert ps_go == _betting_p_cap(d, -2.0, 4.0, o, 0.9)


def test_rel_legs_gate():
    D = [1] * 30 + [0] * 70                                       # mean 0.3 > eps
    d = [1.0] * 100
    o = list(range(100))
    for mode in REL_MODES:
        pr_go, ps_go, pr_ng, ps_ng = rel_legs(D, d, 0.02, o, 0.05, -2.0, -3.0, 4.0, mode)
        assert ps_go == 1.0 and 0 <= pr_ng <= 1


def test_certsave_modes():
    inp = _inp()
    for mode in REL_MODES:
        cs = certified_saving(inp, claim="relative", rel_mode=mode)
        assert cs["rel_mode"] == mode
    assert "rel_mode" not in certified_saving(inp, claim="relative", rel_test="bet")


def test_config_rel_mode():
    base = {"small": "a", "big": "b", "design": "v3", "claim": "relative", "cost": {"lambda": 1.0}}
    assert config_from_dict(base).rel_mode == "fixed_n"
    assert config_from_dict(dict(base, rel_mode="anytime")).rel_mode == "anytime"
    assert config_from_dict(base).to_dict()["rel_mode"] == "fixed_n"
    with pytest.raises(ValueError):
        config_from_dict(dict(base, rel_mode="sometimes"))
    with pytest.raises(ValueError):
        config_from_dict({"small": "a", "big": "b", "design": "v3", "cost": {"lambda": 1.0}, "rel_mode": "anytime"})


def test_gap_modes():
    rng = random.Random(2)
    D = [0] * 1900 + [1] * 40 + [-1] * 60
    d = [rng.uniform(-1, 4) for _ in range(2000)]
    for mode in REL_MODES:
        g = gap_relative(D, 2000, d, -3.0, 0.02, 0.05, 0.8, mode=mode, hi_d=8.0)
        assert g["rel_test"] == mode and g["m_risk_questions"] is not None
