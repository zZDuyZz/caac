"""Replay semantics on tiny hand-made pools (every number checkable by hand), config checks,
answer equivalence."""

import pytest

from caac.pool_v1.config import config_from_dict
from caac.pool_v1.replay import Policy, normalize_key, policy_set, replay
from caac.pool_v1.schema import Meta, Row

S, B = "small", "big"


def meta(eq="math", caps=None):
    return Meta(source={"name": "t"}, grader={"kind": "exact", "name": "t"}, answer_equivalence=eq,
                caps=caps if caps is not None else {S: 100, B: 100}, answer_keys=True)


def cfg(**kw):
    d = {"small": S, "big": B, "cost": {"lambda": 2.0},
         "policies": {"k": [1], "tau": [1.0], "cut": [None], "on_unsure": ["escalate"]}}
    d.update(kw)
    return config_from_dict(d)


def rows_q(q, small, big):
    """small = [(correct, key, tokens, finish)], big = (correct, tokens)."""
    out = [Row(q, S, i, c, answer_key=k, out_tokens=t, in_tokens=0, finish=f)
           for i, (c, k, t, f) in enumerate(small)]
    out.append(Row(q, B, 0, big[0], answer_key="b", out_tokens=big[1], in_tokens=0, finish="stop"))
    return out


def test_k1_serve_escalate_and_costs():
    rows = (rows_q("q1", [(True, "x", 10, "stop")], (True, 30))          # served small, right
            + rows_q("q2", [(False, "y", 20, "stop")], (True, 40))      # served small, wrong
            + rows_q("q3", [(False, None, 100, "length")], (False, 50)))  # truncated -> big, wrong
    rr = replay(rows, meta(), cfg())
    p = rr.audit_input.policies[0]
    assert rr.qids == ["q1", "q2", "q3"]
    assert p.served == [True, True, True]
    assert p.error == [False, True, True]
    assert p.cost == [10, 20, 100 + 2 * 50]                      # lambda = 2 -> big token costs 2
    assert rr.audit_input.big_cost == [60, 80, 100]
    assert rr.audit_input.c_max == 1 * 100 + 2 * 100           # k * cap_s + lambda * cap_b


def test_early_stop_cuts_and_escalates():
    rows = (rows_q("q1", [(True, "x", 10, "stop")], (True, 30))
            + rows_q("q2", [(True, "y", 80, "stop")], (False, 40)))
    rr = replay(rows, meta(), cfg(policies={"k": [1], "cut": [50]}))
    p = rr.audit_input.policies[0]
    assert p.name == "k1_tau1_cut50_escalate"
    assert p.cost == [10, 50 + 2 * 40]       # cut at 50 tokens, then the big model
    assert p.error == [False, True]          # big model's (wrong) answer is served
    assert rr.small_tokens[p.name] == [10, 50]


def test_vote_tau_and_abstain():
    rows = (rows_q("q1", [(True, "a", 5, "stop"), (True, "a", 5, "stop"), (False, "b", 5, "stop")],
                   (True, 10))
            + rows_q("q2", [(True, "a", 5, "stop"), (False, "b", 5, "stop"), (False, "c", 5, "stop")],
                     (True, 10)))
    c = cfg(policies={"k": [3], "tau": [0.6], "on_unsure": ["escalate", "abstain"]})
    rr = replay(rows, meta(), c)
    esc, abst = rr.audit_input.policies
    assert esc.served == [True, True] and esc.error == [False, False]
    assert esc.cost == [15, 15 + 20]          # q2: 1/3 < 0.6 -> big model
    assert abst.served == [True, False] and abst.error == [False, False] and abst.cost == [15, 15]


def test_vote_tie_takes_earliest_draw():
    rows = rows_q("q1", [(False, "b", 5, "stop"), (True, "a", 5, "stop")], (True, 10))
    rr = replay(rows, meta(), cfg(policies={"k": [2], "tau": [0.5]}))
    assert rr.audit_input.policies[0].error == [True]     # "b" came first


def test_exact_equivalence_votes_across_spacing_and_case():
    assert normalize_key("  Paris  France ", "exact") == normalize_key("paris france", "exact")
    assert normalize_key("x", "none") is None and normalize_key("\\frac12", "math") == "\\frac12"
    rows = rows_q("q1", [(True, "Paris", 5, "stop"), (True, " paris", 5, "stop")], (True, 10))
    rr = replay(rows, meta("exact"), cfg(policies={"k": [2], "tau": [1.0]}))
    assert rr.audit_input.policies[0].cost == [10]     # agreed -> no escalation


def test_errors_are_explicit():
    rows = rows_q("q1", [(True, "a", 5, "stop")], (True, 10))
    with pytest.raises(ValueError, match="fewer than k=2"):
        replay(rows, meta(), cfg(policies={"k": [2]}))
    with pytest.raises(ValueError, match="answer_equivalence is 'none'"):
        replay(rows + [Row("q1", S, 1, True, out_tokens=5)], meta("none"), cfg(policies={"k": [2]}))
    with pytest.raises(ValueError, match="meta.caps"):
        replay(rows, meta(caps={}), cfg())


def test_config_rejects_unpriced_tokens_and_unbounded_money():
    with pytest.raises(ValueError, match="lambda or cost.price_out"):
        config_from_dict({"small": S, "big": B})
    with pytest.raises(ValueError, match="c_max"):
        config_from_dict({"small": S, "big": B, "cost": {"measure": "usd"}})
    with pytest.raises(ValueError, match="unknown key"):
        config_from_dict({"small": S, "big": B, "cost": {"lambda": 1}, "alfa": 0.1})
    c = config_from_dict({"small": S, "big": B, "cost": {"price_out": {"small": 0.5, "big": 2.0}}})
    assert c.cost.lam == 4.0


def test_policy_set_is_fixed_and_dedups_k1():
    c = cfg(policies={"k": [2, 1], "tau": [1.0, 0.5], "cut": [None]})
    names = [p.name for p in policy_set(c)]
    assert names == ["k1_tau1_cutnone_escalate", "k2_tau0.5_cutnone_escalate", "k2_tau1_cutnone_escalate"]
    assert Policy(1, 1.0, None, "escalate").name == names[0]


def test_row_order_does_not_matter():
    rows = (rows_q("q2", [(True, "x", 10, "stop")], (True, 30))
            + rows_q("q1", [(False, "y", 20, "stop")], (True, 40)))
    a = replay(rows, meta(), cfg()).audit_input
    b = replay(list(reversed(rows)), meta(), cfg()).audit_input
    assert a.qids == b.qids and a.big_cost == b.big_cost and a.policies[0].cost == b.policies[0].cost
