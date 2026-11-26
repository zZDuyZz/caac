import math, random
import pytest
from caac.pool_v1.audit import (AuditInput, PolicyData, audit, audit3, structural_range, certified_frontier,
                                p_risk_below)
from caac.pool_v1.config import config_from_dict
from caac.pool_v1.gap import gap, round_up_2sig


def make(n=400, seed=1, err=0.03, escalate_share=0.3, m=3):
    rng = random.Random(seed)
    qids = [f"q{i}" for i in range(n)]
    big = [rng.uniform(5, 10) for _ in qids]
    pols = []
    for j in range(m):
        served, error, cost, sp = [], [], [], []
        for i in range(n):
            a = rng.uniform(0.5, 2.0)
            esc = rng.random() < escalate_share + 0.1 * j
            served.append(True)
            error.append(rng.random() < err)
            cost.append(a + (big[i] if esc else 0.0))
            sp.append(a)
        pols.append(PolicyData(f"p{j}", served, error, cost, small_part=sp))
    return AuditInput(qids=qids, big_cost=big, policies=pols, c_max=12.0, small_max=2.0, big_max=10.0)


def test_structural_range():
    lo, hi = structural_range(2.0, 10.0, 0.2)
    assert lo == pytest.approx(-(2.0 + 2.0)) and hi == pytest.approx(8.0)
    lo, hi = structural_range(2.0, 10.0, -0.1)
    assert lo == pytest.approx(-2.0)


def test_v3_go_and_bonferroni_level():
    inp = make()
    r = audit3(inp, alpha=0.10, delta=0.10, s=0.20)
    assert r.verdict == "GO" and r.chosen is not None
    assert r.go["level_each_leg"] == pytest.approx(0.10 / 3)
    row = next(x for x in r.nogo["policies"] if x["policy"] == r.chosen)
    assert row["p_risk_go"] <= 0.10 / 3 and row["p_saving_go"] <= 0.10 / 3


def test_v3_structure_violation_is_an_error():
    inp = make(n=50)
    inp.policies[0].cost[3] = inp.policies[0].small_part[3] + 1.0   # neither small part nor + big-only
    with pytest.raises(ValueError):
        audit3(inp, alpha=0.1, delta=0.1, s=0.2)


def test_v3_needs_split_bounds():
    inp = make(n=50)
    inp.big_max = None
    with pytest.raises(ValueError):
        audit3(inp, alpha=0.1, delta=0.1, s=0.2)


def test_v3_nogo_when_risk_high():
    inp = make(err=0.4)
    r = audit3(inp, alpha=0.10, delta=0.10, s=0.20)
    assert r.verdict == "NO-GO"


def test_v3_inconclusive_small_n():
    inp = make(n=30)
    assert audit3(inp, alpha=0.1, delta=0.1, s=0.2).verdict == "INCONCLUSIVE"


def test_v2_untouched():
    inp = make()
    r2 = audit(inp, alpha=0.1, delta=0.1, s=0.2)
    assert r2.verdict in ("GO", "NO-GO", "INCONCLUSIVE")


def test_gap_sanity_and_stable_rounding():
    d = [1.0, 2.0, 0.5, 3.0] * 25
    g = gap(2, 100, 100, d, -4.0, 0.10, 0.033)
    assert g["reachable"]
    assert g["questions_needed"] == round_up_2sig(g["questions_needed"])
    assert gap(20, 100, 100, d, -4.0, 0.10, 0.033)["reachable"] is False
    assert gap(2, 100, 100, [-1.0] * 100, -4.0, 0.10, 0.033)["reachable"] is False
    assert round_up_2sig(1234) == 1300 and round_up_2sig(99) == 99


def test_frontier_bounds_hold_jointly_shape():
    inp = make()
    fr = certified_frontier(inp, delta=0.10)
    assert fr["n_bounds"] == 6 and set(fr["frontier"]) == {"0.05", "0.1", "0.2"}
    for x in fr["policies"]:
        assert 0 <= x["risk_upper"] <= 1


def test_config_design_key():
    base = {"small": "a", "big": "b", "cost": {"measure": "tokens", "lambda": 1.0}, "policies": []}
    assert config_from_dict({**base}).design == "v2"
    assert config_from_dict({**base, "design": "v3"}).design == "v3"
    with pytest.raises(ValueError):
        config_from_dict({**base, "design": "v9"})
