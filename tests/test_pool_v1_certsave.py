"""U-a certified saving: supplementary bound, consistent with the GO verdict, never changes a verdict."""
import pytest

from caac.pool_v1.audit import audit3
from caac.pool_v1.certsave import certified_saving
from caac.pool_v1.replay import replay
from test_pool_v1_rel import _inp, cfg, paired   # tests/ is on sys.path (pytest rootdir import)


def _run(n, flip, seed=0, claim="relative"):
    rows, meta = paired(n, flip=flip, seed=seed)
    c = cfg() if claim == "relative" else cfg(claim=None)
    inp = replay(rows, meta, c).audit_input
    v = audit3(inp, alpha=c.alpha, delta=c.delta, s=c.s, salt=c.salt, claim=claim, eps=0.02).verdict
    return v, certified_saving(inp, claim=claim, alpha=c.alpha, eps=0.02, delta=c.delta, salt=c.salt)


def test_go_implies_certified_saving_at_least_s():
    v, cs = _run(3000, 0.01)
    assert v == "GO" and cs["certified_saving"] is not None and cs["certified_saving"] >= 0.20 - 1e-9
    assert cs["policy"] is not None and cs["saving_hat_of_policy"] >= cs["certified_saving"]


def test_small_sample_has_no_certified_saving():
    v, cs = _run(60, 0.0, seed=2)
    assert v != "GO" and cs["certified_saving"] is None and cs["n_risk_pass"] == 0


def test_absolute_claim_out_of_reach_gives_none():
    v, cs = _run(3000, 0.01, claim="absolute")        # big model alone errs ~25% > alpha
    assert v != "GO" and cs["certified_saving"] is None


def test_checks_and_grid():
    with pytest.raises(ValueError):
        certified_saving(_inp(), claim="other")
    cs = certified_saving(_inp(), claim="relative")
    assert cs["grid"] == [-0.5, 0.98, 0.02] and cs["level_each_leg"] == pytest.approx(0.10)


def test_report_carries_certified_saving_v3_only():
    from caac.pool_v1.report import make_report, to_markdown
    rows, meta = paired(3000, flip=0.01)
    rep = make_report(rows, meta, cfg())
    c = rep["R1_verdict"]["certified_saving"]
    assert (rep["R1_verdict"]["verdict"] == "GO") == (c["value"] is not None and c["value"] >= 0.20 - 1e-9)
    assert "certified saving: at least" in to_markdown(rep)
    rep_abs = make_report(rows, meta, cfg(claim=None))
    assert rep_abs["R1_verdict"]["certified_saving"]["value"] is None
    assert "certified saving: none" in to_markdown(rep_abs)
    from caac.pool_v1.config import config_from_dict
    v2 = config_from_dict({"small": "org/small", "big": "org/big", "cost": {"lambda": 8.0},
                           "policies": {"k": [1], "cut": [None], "on_unsure": ["escalate"]}})
    assert "certified_saving" not in make_report(rows, meta, v2)["R1_verdict"]


def test_lock_v3_has_amendment_v3_7():
    import json
    from pathlib import Path
    L = json.loads((Path(__file__).resolve().parents[1] / "protocols" / "audit_lock_v3.json").read_text())
    assert any(a["id"] == "v3-7" for a in L["amendments"]) and "certified_saving" in L["reporting"]
