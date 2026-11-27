"""U-c action guidance (amendment v3-8): derived from report fields only, never changes a verdict."""
import copy

from caac.pool_v1.action import action_from_report
from caac.pool_v1.report import make_report, to_markdown
from test_pool_v1_rel import cfg, paired   # tests/ is on sys.path (pytest rootdir import)

KIND = {"GO": "deploy", "NO-GO": "keep_big", "INCONCLUSIVE": "collect_or_stop"}


def test_go_report_says_deploy_and_budget_with_s_star():
    rows, meta = paired(3000, flip=0.01)
    rep = make_report(rows, meta, cfg())
    a = rep["R1_verdict"]["action"]
    assert rep["R1_verdict"]["verdict"] == "GO" and a["kind"] == "deploy"
    assert a["policy"] == rep["R1_verdict"]["chosen"]
    assert a["budget_saving"] == rep["R1_verdict"]["certified_saving"]["value"]
    md = to_markdown(rep)
    assert "## What to do" in md and "Budget with the certified saving" in md


def test_small_pool_is_inconclusive_below_floor():
    rows, meta = paired(120, flip=0.0, seed=2)
    rep = make_report(rows, meta, cfg())
    a = rep["R1_verdict"]["action"]
    assert rep["R1_verdict"]["verdict"] == "INCONCLUSIVE" and a["kind"] == "collect_or_stop"
    assert a["below_floor"] is True and a["type"] in ("reachable", "unreachable")
    if a["type"] == "reachable":
        c = rep["R4_break_even"]["certification"]
        assert a["blocking"] == c["blocking"] and a["more_questions_needed"] == c["more_questions_needed"]


def test_absolute_claim_out_of_reach_is_not_go_and_names_quality():
    rows, meta = paired(3000, flip=0.01)
    rep = make_report(rows, meta, cfg(claim=None))      # big model alone errs ~25% > alpha
    a = rep["R1_verdict"]["action"]
    assert a["kind"] == KIND[rep["R1_verdict"]["verdict"]] and a["kind"] != "deploy"
    if a["kind"] == "keep_big":
        assert a["rejected_by_quality"] + a["rejected_by_both"] == len(rep["R2_estimates"]["all_policies"])
    else:
        assert a["type"] == "unreachable" and a["blocking"] == "risk"


def test_pure_function_and_verdict_untouched():
    rows, meta = paired(3000, flip=0.01)
    rep = make_report(rows, meta, cfg())
    a = rep["R1_verdict"].pop("action")
    before = copy.deepcopy(rep)
    assert action_from_report(rep) == a and rep == before


def test_unreachable_blocking_rules():
    rep = {"R1_verdict": {"verdict": "INCONCLUSIVE", "claim": "relative", "eps": 0.02, "alpha": 0.1, "s": 0.2,
                          "delta": 0.1, "n_questions": 1000, "certified_saving": {"value": 0.1, "policy": "p"}},
           "R2_estimates": {"all_policies": [{"policy": "p", "delta_hat": 0.01, "saving": 0.1},
                                             {"policy": "q", "delta_hat": 0.05, "saving": 0.5}]},
           "R4_break_even": {"certification": {"reachable": False}, "minimum_questions_zero_errors": 600}}
    a = action_from_report(rep)
    assert a["type"] == "unreachable" and a["blocking"] == "trade_off" and a["below_floor"] is False
    assert a["smaller_saving_certified"] == {"value": 0.1, "policy": "p"}
    rep["R2_estimates"]["all_policies"][1]["saving"] = 0.15          # nobody saves 20%
    assert action_from_report(rep)["blocking"] == "saving"
    rep["R2_estimates"]["all_policies"][0]["saving"] = 0.3           # p meets both: margin too thin
    assert action_from_report(rep)["blocking"] == "power"
    rep["R2_estimates"]["all_policies"][1]["delta_hat"] = 0.03
    rep["R2_estimates"]["all_policies"][0]["delta_hat"] = 0.03
    assert action_from_report(rep)["blocking"] == "risk"


def test_v2_reports_have_no_action():
    from caac.pool_v1.config import config_from_dict
    rows, meta = paired(600, flip=0.01)
    v2 = config_from_dict({"small": "org/small", "big": "org/big", "cost": {"lambda": 8.0},
                           "policies": {"k": [1], "cut": [None], "on_unsure": ["escalate"]}})
    rep = make_report(rows, meta, v2)
    assert "action" not in rep["R1_verdict"] and "What to do" not in to_markdown(rep)


def test_lock_v3_has_amendment_v3_8():
    import json
    from pathlib import Path
    L = json.loads((Path(__file__).resolve().parents[1] / "protocols" / "audit_lock_v3.json").read_text())
    assert any(a["id"] == "v3-8" for a in L["amendments"]) and "action" in L["reporting"]
