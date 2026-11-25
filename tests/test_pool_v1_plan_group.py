"""caac plan (sample-size estimate) and per-group risk in the report."""

import json

from caac.pool_v1.audit import binom_cdf
from caac.pool_v1.cli import main
from caac.pool_v1.config import config_from_dict
from caac.pool_v1.plan import critical_errors, plan, power
from caac.pool_v1.report import dumps, make_report, to_markdown
from caac.pool_v1.schema import Meta, Row


def test_power_matches_the_audit_test_and_the_documented_table():
    # documented: true risk 8%, alpha = delta = 0.10 -> n=1000: 81%, n=500: 54.2%, n=2000: 96.6%
    assert abs(power(1000, 0.08, 0.10, 0.10) - 0.81) < 0.01
    assert abs(power(500, 0.08, 0.10, 0.10) - 0.542) < 0.01
    assert abs(power(2000, 0.08, 0.10, 0.10) - 0.966) < 0.01
    # zero-error floor: 22 served answers at alpha = delta = .10 pass with 0 errors, 21 do not
    assert critical_errors(22, 0.10, 0.10) >= 0 and critical_errors(21, 0.10, 0.10) < 0
    k = critical_errors(300, 0.10, 0.10)
    assert binom_cdf(k, 300, 0.10) <= 0.10 < binom_cdf(k + 1, 300, 0.10)


def test_plan_reports_stable_sample_sizes():
    p = plan(0.10, 0.10, 0.08, coverage=0.8)
    t = {x["power"]: x for x in p["targets"]}
    assert 400 <= t[0.5]["served_needed"] <= 470
    assert 950 <= t[0.8]["served_needed"] <= 1100
    for x in p["targets"]:                                  # power stays above target from there on
        assert power(x["served_needed"] + 7, 0.08, 0.10, 0.10) >= x["power"] - 1e-9
    assert plan(0.10, 0.10, 0.12)["targets"] == []          # false claim: nothing to plan


def test_plan_cli(capsys):
    assert main(["plan", "--risk", "0.05"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["floor_served_zero_errors"] == 22
    assert main(["plan"]) == 2                              # no risk given


def _pool(n=400):
    rows = []
    for i in range(n):
        q = f"q{i:04d}"
        g = "easy" if i % 4 else "hard"
        ok_small = (i % 20 != 0) if g == "easy" else (i % 3 != 0)
        rows.append(Row(q, "s", 0, ok_small, out_tokens=100, in_tokens=0, finish="stop", group=g))
        rows.append(Row(q, "b", 0, True, out_tokens=100, in_tokens=0, finish="stop", group=g))
    meta = Meta(source={"name": "t"}, grader={"kind": "other", "name": "x"}, answer_equivalence="none",
                caps={"s": 4000, "b": 4000})
    cfg = config_from_dict({"small": "s", "big": "b", "cost": {"lambda": 4.0},
                            "policies": {"k": [1], "cut": [None], "on_unsure": ["escalate"]}})
    return rows, meta, cfg


def test_report_lists_groups_and_stays_byte_identical():
    # overall risk ~8% < alpha => GO, while the "hard" group has risk ~1/3: the group table must show it
    rows, meta, cfg = _pool(2000)
    a = dumps(make_report(rows, meta, cfg))
    assert a == dumps(make_report(list(reversed(rows)), meta, cfg))
    rep = json.loads(a)
    assert rep["R1_verdict"]["verdict"] == "GO"
    bg = rep["R2_estimates"]["by_group"]
    assert set(bg["groups"]) == {"easy", "hard"} and "descriptive" in bg["note"]
    assert bg["groups"]["hard"]["lower_bound_above_alpha"] is True
    assert bg["groups"]["easy"]["lower_bound_above_alpha"] is False
    assert "Risk by group" in to_markdown(rep) and "lower bound above alpha" in to_markdown(rep)


def test_no_group_column_means_no_group_section():
    rows, meta, cfg = _pool()
    for r in rows:
        r.group = None
    assert "by_group" not in make_report(rows, meta, cfg)["R2_estimates"]


def test_pmf_walk_does_not_underflow():
    # P(X = 0) = 0.5**4000 underflows to 0.0 in plain floats; the log-space walk must still sum to 1
    from caac.pool_v1.plan import _pmf_walk
    assert abs(sum(pk for _, pk in _pmf_walk(4000, 0.5)) - 1.0) < 1e-9
    assert critical_errors(4000, 0.5, 0.10) < 2000          # not "every k passes"
    assert power(3000, 0.45, 0.5, 0.1) > 0.999


def test_plan_from_a_report_without_policy_is_a_clean_error(tmp_path):
    bad = tmp_path / "r.json"
    bad.write_text(json.dumps({"R1_verdict": {"verdict": "NOT RUN"}}), encoding="utf-8")
    assert main(["plan", "--from-report", str(bad)]) == 2


def test_validator_flags_mixed_group_labels():
    from caac.pool_v1.validate import validate
    rows, meta, _ = _pool(40)
    rows[1].group = "other"                                  # big model row of q0000 disagrees
    codes = {i.code for i in validate(rows, meta).issues}
    assert "W_GROUP_MIXED" in codes
