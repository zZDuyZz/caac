"""Tests for the CAAC controller: cheapest predicted-safe action, else abstain.

The last test is the paper's central claim turned into an assertion: on a
question the small model gets wrong, a serve/abstain-only action set can only
choose between being wrong and abstaining, while the correcting action set can
serve a CORRECT answer. Everything else here guards the mechanics that claim
rests on.
"""

from __future__ import annotations

import pytest

from caac.pool.ledger import cost_profiles
from caac.pool.policies import (_calls_for_action, caac_certified, greedy_only,
                                pricecheck)
from caac.pool.schema import Completion, PoolRecord
from caac.pool.synth import make_pools

ORDER = list(range(16))
SMALL = "Qwen/Qwen2.5-1.5B-Instruct"   # must be in ledger.MODEL_REGISTRY
BIG = "Qwen/Qwen2.5-7B-Instruct"


def _flat(**p):
    """A scorer with fixed probabilities, so the test controls eligibility."""
    def scorer(small, big):
        d = {"answer_now": 0.0, "sample_more": 0.0, "escalate": 0.0, "abstain": 1.0}
        d.update(p)
        return d
    return scorer


def _rec(model, correct, answer, conf=0.5, n_samples=16, sample_correct=False):
    model = {"small": SMALL, "big": BIG}.get(model, model)
    g = Completion(answer=answer, correct=correct, decode_tokens=100,
                   finish_reason="stop", confidence=conf, conf_prefix32=conf)
    ss = [Completion(answer=("G" if sample_correct else "W1"),
                     correct=sample_correct, decode_tokens=100,
                     finish_reason="stop", confidence=conf)
          for _ in range(n_samples)]
    return PoolRecord(query_id="q", dataset="d", model=model, gold="G",
                      prompt_tokens=80, greedy=g, samples=ss)


# --------------------------------------------------------------------------- #
# the knob
# --------------------------------------------------------------------------- #

def test_tau_zero_serves_the_cheapest_action():
    s = _rec("small", True, "G")
    b = _rec("big", True, "G")
    out = caac_certified(s, b, _flat(answer_now=0.5, sample_more=0.9, escalate=0.99),
                         tau=0.0, order=ORDER)
    assert out.action == "answer_now"      # cheapest, even though it scores worst
    assert out.served and out.correct


def test_tau_above_one_always_abstains_and_still_pays_for_greedy():
    s = _rec("small", True, "G")
    b = _rec("big", True, "G")
    out = caac_certified(s, b, _flat(answer_now=1.0, sample_more=1.0, escalate=1.0),
                         tau=1.01, order=ORDER)
    assert out.action == "abstain"
    assert not out.served and out.answer is None
    assert cost_profiles(out.calls)["flops"] > 0


def test_coverage_is_monotone_non_increasing_in_tau():
    small, big = make_pools(150, seed=21)
    q = sorted(small)

    def scorer(s, b):
        c = s.greedy.confidence or 0.5
        return {"answer_now": c, "sample_more": min(1.0, c + 0.1),
                "escalate": min(1.0, c + 0.2), "abstain": 1.0}

    cov = []
    for tau in [i / 10 for i in range(11)]:
        served = sum(caac_certified(small[k], big[k], scorer, tau=tau,
                                    order=ORDER).served for k in q)
        cov.append(served / len(q))
    assert all(a >= b - 1e-9 for a, b in zip(cov, cov[1:]))
    assert cov[0] == 1.0


def test_cheapest_eligible_wins_even_when_a_pricier_action_scores_higher():
    s = _rec("small", True, "G")
    b = _rec("big", True, "G")
    costs = {a: cost_profiles(_calls_for_action(s, b, a, 8, ORDER))["flops"]
             for a in ("answer_now", "sample_more", "escalate")}
    assert costs["answer_now"] < costs["escalate"]
    assert costs["answer_now"] < costs["sample_more"]
    out = caac_certified(s, b, _flat(answer_now=0.8, escalate=1.0), tau=0.75,
                         order=ORDER)
    assert out.action == "answer_now"


def test_an_ineligible_cheap_action_is_skipped_for_an_eligible_dear_one():
    s = _rec("small", False, "W1")          # small is wrong
    b = _rec("big", True, "G")              # big is right
    out = caac_certified(s, b, _flat(answer_now=0.2, sample_more=0.3, escalate=0.95),
                         tau=0.9, order=ORDER)
    assert out.action == "escalate"
    assert out.served and out.correct and out.answer == "G"


def test_action_set_can_be_restricted_for_the_ablation():
    s = _rec("small", False, "W1")
    b = _rec("big", True, "G")
    sc = _flat(answer_now=0.2, sample_more=0.3, escalate=0.95)
    full = caac_certified(s, b, sc, tau=0.9, order=ORDER)
    limited = caac_certified(s, b, sc, tau=0.9, order=ORDER, actions=("answer_now",))
    assert full.action == "escalate" and limited.action == "abstain"


def test_meta_records_the_decision():
    s = _rec("small", True, "G")
    b = _rec("big", True, "G")
    out = caac_certified(s, b, _flat(answer_now=0.9), tau=0.5, order=ORDER)
    assert out.meta["tau"] == 0.5
    assert out.meta["eligible"] == ["answer_now"]
    assert out.meta["phat"]["answer_now"] == 0.9


def test_sample_more_is_replayed_and_billed_for_k_samples():
    s = _rec("small", False, "W1", n_samples=16, sample_correct=True)
    b = _rec("big", False, "W1")
    out = caac_certified(s, b, _flat(sample_more=0.99), tau=0.9, order=ORDER, k=8)
    assert out.action == "sample_more"
    assert out.correct                       # the samples carry the gold answer
    assert cost_profiles(out.calls)["flops"] > cost_profiles(
        greedy_only(s).calls)["flops"]


# --------------------------------------------------------------------------- #
# the central claim
# --------------------------------------------------------------------------- #

def test_correcting_actions_serve_correctly_where_checks_can_only_abstain():
    """The Chow argument, as code.

    One question: the small model's greedy answer is WRONG, the big model's is
    right. PriceCheck's action set can serve only the greedy answer, so its
    choices are "wrong" or "abstain" -- it can never be both covered and
    correct. CAAC's action set can.
    """
    s = _rec("small", False, "W1", conf=0.2, sample_correct=True)
    b = _rec("big", True, "G")

    # the baseline, at every threshold it has
    for lam in [i / 10 for i in range(11)]:
        pc = pricecheck(s, lam=lam, check="resolve_vote", k=8, order=ORDER)
        assert not (pc.served and pc.correct)

    for action, scores in (("escalate", {"escalate": 0.95}),
                           ("sample_more", {"sample_more": 0.95})):
        out = caac_certified(s, b, _flat(**scores), tau=0.9, order=ORDER)
        assert out.action == action
        assert out.served and out.correct


def test_controller_with_a_fitted_estimator_runs_end_to_end():
    """Smoke test of the real wiring: fitted estimator -> controller -> replay,
    with the estimator fitted on questions the evaluation never sees."""
    from caac.pool.estimator import PerActionEstimator
    from caac.pool.replay import make_orders, run_policy

    small, big = make_pools(400, seed=22)
    q = sorted(small)
    fit, held = q[:250], q[250:]
    est = PerActionEstimator().fit(fit, small, big)
    orders = make_orders(min(r.n_samples for r in small.values()), 3, seed=0)

    def fn(qid, order):
        return caac_certified(small[qid], big[qid], est, tau=0.6, order=order)

    m = run_policy(held, fn, orders)
    assert m.n == len(held)
    assert 0.0 <= m.coverage <= 1.0
    assert sum(m.action_dist.values()) > 0
    if m.served:
        assert 0.0 <= m.selective_risk <= 1.0
