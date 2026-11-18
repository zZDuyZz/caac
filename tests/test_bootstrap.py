"""Question-level bootstrap intervals (caac.eval.bootstrap) and the certify.py hook."""
import json
import sys

import numpy as np
import pytest

from caac.eval.bootstrap import bootstrap_ci, paired_diff_ci
from caac.pool.protocol import Table


def _table(served, wrong, tok_small, qids=None):
    served, wrong, ts = (np.asarray(x, float).reshape(-1, 1) for x in (served, wrong, tok_small))
    q = len(served)
    z = np.zeros((q, 1))
    return Table("t", qids or [f"q{i}" for i in range(q)], [0], [[0]], served, wrong, ts, z, z, z, [{}])


def _draw(rng, n, p_serve=0.7, p_wrong=0.1):
    served = rng.random(n) < p_serve
    wrong = served & (rng.random(n) < p_wrong)
    cost = rng.gamma(2.0, 300.0, n)
    return served, wrong, cost


def test_interval_covers_the_true_coverage_about_95_percent_of_the_time():
    rng = np.random.default_rng(0)
    hits = []
    for rep in range(200):
        s, w, c = _draw(rng, 300)
        ci = bootstrap_ci(_table(s, w, c), 0, lam=1.0, n_boot=400, seed=rep)
        hits.append(ci["coverage"]["lo"] <= 0.7 <= ci["coverage"]["hi"])
    assert 0.89 <= np.mean(hits) <= 0.99


def test_point_values_match_the_plain_summary():
    rng = np.random.default_rng(1)
    s, w, c = _draw(rng, 200)
    ci = bootstrap_ci(_table(s, w, c), 0, lam=1.0, n_boot=100)
    assert ci["coverage"]["point"] == pytest.approx(s.mean())
    assert ci["selective_risk"]["point"] == pytest.approx(w.sum() / s.sum())
    assert ci["cost"]["point"] == pytest.approx(c.mean())
    for m in ("coverage", "selective_risk", "accuracy", "cost"):
        assert ci[m]["lo"] <= ci[m]["point"] <= ci[m]["hi"]


def test_nothing_served_has_no_risk_interval():
    ci = bootstrap_ci(_table([0] * 50, [0] * 50, [10] * 50), 0, lam=1.0, n_boot=50)
    assert ci["selective_risk"]["point"] is None and ci["selective_risk"]["lo"] is None
    assert ci["risk_undefined"] == 1.0


def test_paired_difference_of_a_method_with_itself_is_zero():
    rng = np.random.default_rng(2)
    t = _table(*_draw(rng, 300))
    d = paired_diff_ci(t, 0, t, 0, lam=1.0, n_boot=200)
    for m in ("coverage", "cost", "accuracy"):
        assert d[m]["point"] == 0 and d[m]["lo"] == 0 and d[m]["hi"] == 0
        assert not d[m]["excludes_zero"]


def test_paired_difference_sees_a_constant_gap_the_unpaired_intervals_hide():
    """B costs 40 tokens more on every question. The two unpaired cost intervals
    overlap (per-question cost varies by hundreds), the paired one does not."""
    rng = np.random.default_rng(3)
    s, w, c = _draw(rng, 300)
    a, b = _table(s, w, c), _table(s, w, c + 40)
    ca, cb = bootstrap_ci(a, 0, 1.0, n_boot=500), bootstrap_ci(b, 0, 1.0, n_boot=500)
    assert ca["cost"]["hi"] > cb["cost"]["lo"]                    # overlap
    d = paired_diff_ci(a, 0, b, 0, lam=1.0, n_boot=500)
    assert d["cost"]["point"] == pytest.approx(-40)
    assert d["cost"]["excludes_zero"] and d["cost"]["hi"] < 0


def test_paired_needs_the_same_questions_in_the_same_order():
    rng = np.random.default_rng(4)
    s, w, c = _draw(rng, 10)
    a = _table(s, w, c)
    b = _table(s, w, c, qids=[f"q{i}" for i in reversed(range(10))])
    with pytest.raises(ValueError):
        paired_diff_ci(a, 0, b, 0, lam=1.0)


def test_certify_driver_reports_intervals_and_paired_differences(tmp_path, monkeypatch):
    import certify
    from test_protocol import _write_parts
    p = _write_parts(tmp_path, n=450)
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "certify.py", "--train-small", p["train_small"], "--train-big", p["train_big"],
        "--calib-small", p["calib_small"], "--calib-big", p["calib_big"],
        "--test-small", p["test_small"], "--test-big", p["test_big"],
        "--families", "small_abstain,cascade_abstain,caac_staged", "--lams", "5",
        "--coverages", "0.5", "--perms", "2", "--alpha", "0.15", "--bootstrap", "60",
        "--json", str(out)])
    certify.main()
    res = json.loads(out.read_text())
    pick = res["families"]["caac_staged"]["by_lam"]["5"]["max_coverage"]
    assert pick is not None and "ci" in pick["test"]
    assert pick["test"]["ci"]["cost"]["lo"] <= pick["test"]["cost"] <= pick["test"]["ci"]["cost"]["hi"]
    paired = res["paired"]
    assert "5" in paired and "0.5" in paired["5"]
    assert set(paired["5"]["0.5"]) <= {"small_abstain", "cascade_abstain"}
    for d in paired["5"]["0.5"].values():
        assert {"cost", "coverage", "selective_risk"} <= set(d)


def test_bootstrap_off_by_default_keeps_the_old_output(tmp_path, monkeypatch):
    import certify
    from test_protocol import _write_parts
    p = _write_parts(tmp_path, n=300)
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "certify.py", "--train-small", p["train_small"], "--train-big", p["train_big"],
        "--calib-small", p["calib_small"], "--calib-big", p["calib_big"],
        "--test-small", p["test_small"], "--test-big", p["test_big"],
        "--families", "small_abstain", "--lams", "1", "--coverages", "0.5", "--perms", "2",
        "--alpha", "0.2", "--json", str(out)])
    certify.main()
    res = json.loads(out.read_text())
    assert "paired" not in res
    m = res["families"]["small_abstain"]["by_lam"]["1"]["max_coverage"]
    assert m is None or "ci" not in m["test"]
