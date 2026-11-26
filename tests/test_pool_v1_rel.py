"""Relative claim (non-inferiority to big-only, paired): audit3(claim='relative'), report, config, lock, gap."""
import json
import random

import pytest

from caac.pool_v1.audit import (AuditInput, PolicyData, REL_TESTS, _order, audit3, p_risk_below, rel_risk_p)
from caac.pool_v1.config import config_from_dict
from caac.pool_v1.gap import gap_relative
from caac.pool_v1.report import dumps, make_report, to_markdown
from caac.pool_v1.schema import Meta, Row

S, B = "org/small", "org/big"


def paired(n, p_big=0.75, flip=0.01, seed=0, cap=1000):
    """Small model agrees with the big one except on a share `flip` of questions (then it is wrong).
    The big model is hard to beat: R_big = 1 - p_big (absolute claim at alpha = .10 is out of reach)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        q = f"q{i:05d}"
        big_ok = rng.random() < p_big
        small_ok = big_ok and not (rng.random() < flip)
        rows.append(Row(q, S, 0, small_ok, answer_key="a" if small_ok else "b", out_tokens=rng.randint(50, 300),
                        in_tokens=40, finish="stop", is_greedy=True, split="test"))
        rows.append(Row(q, B, 0, big_ok, answer_key="a", out_tokens=rng.randint(50, 300),
                        in_tokens=40, finish="stop", is_greedy=True, split="test"))
    meta = Meta(source={"name": "synthetic", "url": "local", "version_or_sha": "1"},
                grader={"kind": "exact", "name": "synthetic"}, answer_equivalence="exact",
                caps={S: cap, B: cap}, answer_keys=True)
    return rows, meta


def cfg(claim="relative", eps=0.02, **kw):
    d = {"small": S, "big": B, "cost": {"lambda": 8.0}, "design": "v3",
         "policies": {"k": [1], "cut": [None, 250], "on_unsure": ["escalate"]}}
    if claim:
        d["claim"] = claim
        d["eps"] = eps
    d.update(kw)
    return config_from_dict(d)


def test_relative_go_where_absolute_cannot():
    rows, meta = paired(3000)
    rel = make_report(rows, meta, cfg())
    ab = make_report(rows, meta, cfg(claim=None))
    assert ab["R1_verdict"]["verdict"] != "GO"                 # big model alone errs ~25% > alpha
    assert rel["R1_verdict"]["verdict"] == "GO"
    assert rel["R1_verdict"]["claim"] == "relative" and rel["R1_verdict"]["eps"] == 0.02
    assert 0.2 < rel["R1_verdict"]["big_only_error"] < 0.3
    assert "claim: relative" in to_markdown(rel)


def test_relative_nogo_when_small_is_much_worse():
    rows, meta = paired(3000, flip=0.30)
    rep = make_report(rows, meta, cfg())
    assert rep["R1_verdict"]["verdict"] == "NO-GO"


def test_relative_report_is_byte_stable():
    rows, meta = paired(800, seed=3)
    assert dumps(make_report(rows, meta, cfg())) == dumps(make_report(list(reversed(rows)), meta, cfg()))


def test_relative_inconclusive_small_n_has_gap():
    rows, meta = paired(150, flip=0.005, seed=5)
    rep = make_report(rows, meta, cfg())
    assert rep["R1_verdict"]["verdict"] == "INCONCLUSIVE"
    c = rep["R4_break_even"]["certification"]
    assert c["claim"] == "relative" and c["reachable"] and c["questions_needed"] > 150


def _inp(n=200):
    qids = [f"q{i}" for i in range(n)]
    big = [5.0] * n
    big_err = [i % 4 == 0 for i in range(n)]
    sp = [1.0] * n
    esc = [i % 2 == 0 for i in range(n)]
    cost = [1.0 + (5.0 if e else 0.0) for e in esc]
    err = [big_err[i] if esc[i] else False for i in range(n)]
    pol = PolicyData("p", [True] * n, err, cost, small_part=sp)
    return AuditInput(qids=qids, big_cost=big, policies=[pol], c_max=6.0, small_max=1.0, big_max=5.0,
                      big_error=big_err)


def test_relative_checks_pairing():
    inp = _inp()
    audit3(inp, claim="relative")                               # consistent: fine
    i = next(i for i in range(200) if i % 2 == 0 and i % 4 != 0)
    inp.policies[0].error[i] = True                             # escalated question disagrees with big-only
    with pytest.raises(ValueError):
        audit3(inp, claim="relative")
    inp = _inp(); inp.big_error = None
    with pytest.raises(ValueError):
        audit3(inp, claim="relative")
    inp = _inp(); inp.policies[0].served[0] = False
    with pytest.raises(ValueError):
        audit3(inp, claim="relative")


def test_rel_risk_p_variants():
    D = [0] * 300 + [1] * 2
    order = _order([f"q{i}" for i in range(len(D))], "s")
    ph = rel_risk_p(D, 0.02, order, "harm")
    assert ph == p_risk_below(2, len(D), 0.02)
    pb = rel_risk_p(D, 0.02, order, "bet")
    pm = rel_risk_p(D, 0.02, order, "mix")
    assert pm == pytest.approx(min(1.0, 2 * min(pb, ph)))
    assert ph < pb                                               # low discordance: exact harm test is stronger
    with pytest.raises(ValueError):
        rel_risk_p(D, 0.02, order, "nope")
    assert set(REL_TESTS) == {"bet", "harm", "mix"}


def test_config_claim_rules():
    assert "claim" not in cfg(claim=None).to_dict()
    d = cfg().to_dict()
    assert d["claim"] == "relative" and d["eps"] == 0.02
    with pytest.raises(ValueError):
        config_from_dict({"small": S, "big": B, "cost": {"lambda": 1.0}, "policies": {}, "claim": "relative"})  # needs v3
    with pytest.raises(ValueError):
        cfg(claim="other")
    with pytest.raises(ValueError):
        cfg(eps=1.5)


def test_gap_relative_reachability():
    d = [1.0] * 400
    g = gap_relative([0] * 398 + [1] * 2, 400, d, -4.0, 0.02, 0.01, test="harm")
    assert g["reachable"] and g["blocking"] in ("risk", "saving")
    assert gap_relative([1] * 40 + [0] * 360, 400, d, -4.0, 0.02, 0.01, test="harm")["reachable"] is False
    assert gap_relative([0] * 400, 400, [-1.0] * 400, -4.0, 0.02, 0.01)["reachable"] is False


def test_lock_v3_relative_fields_and_check():
    from pathlib import Path
    from caac.pool_v1.audit import REL_TEST
    from caac.pool_v1.lock import check_config_against_lock, load_lock, sensitivity_cells_relative
    lp = Path(__file__).resolve().parents[1] / "protocols" / "audit_lock_v3.json"
    L = load_lock(lp)
    a = L["audit"]
    assert a["claim_main"] == "relative" and a["eps"] == 0.02 and a["rel_test"] == REL_TEST
    assert (0.02, a["s"]) not in sensitivity_cells_relative(L) and len(sensitivity_cells_relative(L)) == 11
    base = {"small": S, "big": B, "alpha": a["alpha"], "delta": a["delta"], "s": a["s"], "salt": a["salt"],
            "cost_test": a["cost_test"], "cost": {"lambda": 1.0}, "design": "v3",
            "policies": {"k": [1], "cut": [None], "on_unsure": ["escalate"]}}
    ok = check_config_against_lock(config_from_dict(dict(base, claim="relative", eps=0.02)), lp, caps={S: 1000})
    assert ok == []
    bad = check_config_against_lock(config_from_dict(dict(base, claim="relative", eps=0.03)), lp, caps={S: 1000})
    assert any("eps" in b for b in bad)
