"""Tests for the three-stage controller.

The behaviour that matters: it must pay for a gate before reading its evidence,
bill every failed gate, prefer the cheapest gate first, and be able to decline
after paying -- a controller that always served once it had spent the tokens
would be sunk-cost reasoning with a risk guarantee bolted on.
"""

from __future__ import annotations

import pytest

from caac.pool.estimator import PerActionEstimator, SecondStageEstimator
from caac.pool.ledger import cost_profiles
from caac.pool.policies import caac_certified, caac_staged, greedy_only
from caac.pool.replay import make_orders, run_policy
from caac.pool.schema import Completion, PoolRecord
from caac.pool.synth import make_pools

SMALL = "Qwen/Qwen2.5-1.5B-Instruct"
BIG = "Qwen/Qwen2.5-7B-Instruct"
ORDER = list(range(16))


def _scorer(**p):
    def fn(small, big):
        d = {"answer_now": 0.0, "sample_more": 0.0, "escalate": 0.0, "abstain": 1.0}
        d.update(p)
        return d
    return fn


def _stage2(value):
    return lambda rec, idxs, p1: value


def _rec(model, correct, answer, n_samples=16, sample_correct=False, decode=100):
    g = Completion(answer=answer, correct=correct, decode_tokens=decode,
                   finish_reason="stop", confidence=0.5, conf_prefix32=0.5)
    ss = [Completion(answer="G" if sample_correct else "W1", correct=sample_correct,
                     decode_tokens=decode, finish_reason="stop", confidence=0.5)
          for _ in range(n_samples)]
    return PoolRecord(query_id="q", dataset="d", model=model, gold="G",
                      prompt_tokens=80, greedy=g, samples=ss)


# --------------------------------------------------------------------------- #
# gate order and short-circuiting
# --------------------------------------------------------------------------- #

def test_stage1_gate_serves_greedy_without_paying_for_anything_else():
    s, b = _rec(SMALL, True, "G"), _rec(BIG, True, "G")
    out = caac_staged(s, b, _scorer(answer_now=0.9), _stage2(1.0), tau=0.5,
                      order=ORDER)
    assert out.action == "answer_now" and out.meta["gate"] == "stage1"
    assert out.meta["tried"] == []
    assert cost_profiles(out.calls) == cost_profiles(greedy_only(s).calls)


def test_stage2_reads_evidence_the_policy_has_already_paid_for():
    """The samples decide, and they are billed whether or not they decide yes."""
    s = _rec(SMALL, False, "W1", sample_correct=True)
    b = _rec(BIG, True, "G")
    out = caac_staged(s, b, _scorer(answer_now=0.1, escalate=0.1), _stage2(0.99),
                      tau=0.5, order=ORDER, k=8)
    assert out.action == "sample_more" and out.meta["gate"] == "stage2"
    assert out.served and out.correct
    assert sum(len(c.decode_tokens) for c in out.calls) == 9   # greedy + 8


def test_it_can_still_abstain_after_paying_for_the_samples():
    """No sunk-cost serving: having spent the tokens is not a reason to answer."""
    s = _rec(SMALL, False, "W1", sample_correct=False)
    b = _rec(BIG, True, "G")
    out = caac_staged(s, b, _scorer(answer_now=0.1, escalate=0.1), _stage2(0.2),
                      tau=0.9, order=ORDER, k=8)
    assert out.action == "abstain" and not out.served
    assert "sample_more" in out.meta["tried"]
    assert sum(len(c.decode_tokens) for c in out.calls) == 9    # still billed


def test_a_failed_stage2_gate_falls_through_to_escalation_and_bills_both():
    s = _rec(SMALL, False, "W1", sample_correct=False)
    b = _rec(BIG, True, "G")
    out = caac_staged(s, b, _scorer(answer_now=0.1, escalate=0.95), _stage2(0.1),
                      tau=0.9, order=ORDER, k=8)
    assert out.action == "escalate" and out.served and out.correct
    assert out.meta["tried"] == ["sample_more", "escalate"]
    # greedy + 8 samples + the big model's greedy
    assert sum(len(c.decode_tokens) for c in out.calls) == 10
    assert cost_profiles(out.calls)["rounds"] == 3


def test_the_cheaper_correcting_gate_is_tried_first():
    """Order comes from what each action costs on THIS question, not a hardcoded
    preference, so a long prompt or a big k can change the order."""
    s = _rec(SMALL, False, "W1", decode=10)       # 8 cheap samples
    b = _rec(BIG, True, "G", decode=10_000)       # a very expensive big model
    out = caac_staged(s, b, _scorer(answer_now=0.0, escalate=0.99), _stage2(0.0),
                      tau=0.9, order=ORDER, k=8)
    assert out.meta["tried"][0] == "sample_more"

    s2 = _rec(SMALL, False, "W1", decode=10_000)  # 8 very expensive samples
    b2 = _rec(BIG, True, "G", decode=10)
    out2 = caac_staged(s2, b2, _scorer(answer_now=0.0, escalate=0.99), _stage2(0.0),
                       tau=0.9, order=ORDER, k=8)
    assert out2.meta["tried"][0] == "escalate"
    assert out2.action == "escalate"


def test_restricting_the_action_set_removes_its_gate():
    s = _rec(SMALL, False, "W1", sample_correct=True)
    b = _rec(BIG, True, "G")
    out = caac_staged(s, b, _scorer(answer_now=0.1, escalate=0.99), _stage2(0.99),
                      tau=0.9, order=ORDER, actions=("answer_now",))
    assert out.action == "abstain" and out.meta["tried"] == []


def test_coverage_is_monotone_in_tau():
    small, big = make_pools(150, seed=51)
    q = sorted(small)
    sc = _scorer(answer_now=0.5, sample_more=0.6, escalate=0.7)
    cov = []
    for tau in [i / 10 for i in range(11)]:
        served = sum(caac_staged(small[k], big[k], sc, _stage2(0.65), tau, ORDER).served
                     for k in q)
        cov.append(served / len(q))
    assert all(a >= b - 1e-9 for a, b in zip(cov, cov[1:]))


def test_missing_stage2_estimator_never_serves_on_a_guess():
    s = _rec(SMALL, False, "W1", sample_correct=True)
    b = _rec(BIG, True, "G")
    out = caac_staged(s, b, _scorer(answer_now=0.1, escalate=0.1), None,
                      tau=0.5, order=ORDER)
    assert out.action == "abstain"


def test_handles_a_pool_with_no_samples():
    s = _rec(SMALL, False, "W1", n_samples=0)
    b = _rec(BIG, True, "G")
    out = caac_staged(s, b, _scorer(answer_now=0.1, escalate=0.95), _stage2(0.99),
                      tau=0.9, order=ORDER)
    assert out.action == "escalate"


# --------------------------------------------------------------------------- #
# end to end, with fitted estimators
# --------------------------------------------------------------------------- #

def test_staged_controller_runs_with_real_estimators_on_held_out_questions():
    small, big = make_pools(500, seed=52)
    q = sorted(small)
    fit, held = q[:320], q[320:]
    est = PerActionEstimator().fit(fit, small, big)
    orders = make_orders(min(r.n_samples for r in small.values()), 3, seed=0)
    est2 = SecondStageEstimator().fit(fit, small, orders, stage1=lambda r: est(r))

    m = run_policy(held, lambda qid, o: caac_staged(small[qid], big[qid], est, est2,
                                                    0.7, o, k=8), orders)
    assert m.n == len(held)
    assert 0.0 <= m.coverage <= 1.0
    assert set(m.action_dist) <= {"answer_now", "sample_more", "escalate", "abstain"}


def test_stage2_separates_right_from_wrong_majorities():
    """If the post-sample head carried no signal the third stage would be pure
    extra cost, so this is the precondition for the whole design."""
    from collections import Counter

    small, big = make_pools(600, seed=53)
    q = sorted(small)
    est = PerActionEstimator().fit(q[:400], small, big)
    orders = make_orders(32, 4, seed=0)
    est2 = SecondStageEstimator().fit(q[:400], small, orders, stage1=lambda r: est(r))

    hi, lo = [], []
    for key in q[400:]:
        rec = small[key]
        idxs = list(range(8))
        p = est2(rec, idxs, est(rec))
        canon = rec.canonical_sample_answers()
        votes = [canon[i] for i in idxs if canon[i] is not None]
        ok = bool(votes) and rec.label_correct(Counter(votes).most_common(1)[0][0])
        (hi if ok else lo).append(p)
    assert sum(hi) / len(hi) > sum(lo) / len(lo) + 0.2


def test_staged_covers_at_least_as_much_as_single_stage_at_equal_tau():
    """Stage 2 can only ADD a way to clear the gate: any question the two-stage
    controller serves via answer_now or escalate is still served here."""
    small, big = make_pools(300, seed=54)
    q = sorted(small)
    est = PerActionEstimator().fit(q[:200], small, big)
    orders = make_orders(32, 3, seed=0)
    est2 = SecondStageEstimator().fit(q[:200], small, orders, stage1=lambda r: est(r))
    tau = 0.7
    a = sum(caac_certified(small[k], big[k], est, tau, ORDER).served for k in q[200:])
    b = sum(caac_staged(small[k], big[k], est, est2, tau, ORDER).served for k in q[200:])
    assert b >= a
