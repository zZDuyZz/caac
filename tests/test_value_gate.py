"""The gate that decides whether to buy compute at all.

A confidence gate and a net-value gate disagree exactly where the paper's
claim lives: on a question the small model is unsure about but nothing can
repair. These tests pin that difference down, and check that refusing to buy
means the big model was never billed.
"""

from __future__ import annotations

import pytest

from caac.pool.estimator import RecoveryEstimator
from caac.pool.families import conformal_cascade
from caac.pool.ledger import Call
from caac.pool.policies import RHO_MODES, caac_staged, calls_for_action
from caac.pool.schema import Completion, PoolRecord
from caac.pool.synth import make_pools


def _rec(ok: bool, samples: list[bool], model="small", gold="G") -> PoolRecord:
    def comp(o):
        return Completion(answer=gold if o else "W1", correct=o, decode_tokens=100,
                          finish_reason="stop", confidence=0.6)
    return PoolRecord(query_id="q", dataset="t", model=model, gold=gold, prompt_tokens=10,
                      greedy=comp(ok), samples=[comp(o) for o in samples])


def _scorer(**p):
    base = {"answer_now": 0.0, "sample_more": 0.0, "escalate": 0.0, "abstain": 1.0}
    base.update(p)
    return lambda s, b: dict(base)


def _tokens(calls: list[Call]) -> float:
    return sum(c.prefill_tokens + sum(c.decode_tokens) for c in calls)


# --------------------------------------------------------------------------- #
# the gate itself
# --------------------------------------------------------------------------- #

def test_rho_mode_must_be_known():
    s, b = _rec(False, [False] * 4), _rec(True, [True], model="big")
    with pytest.raises(ValueError):
        caac_staged(s, b, _scorer(), None, tau=0.9, order=[0, 1, 2, 3], rho=0.0,
                    rho_mode="confidence", cost_fn=_tokens)
    assert set(RHO_MODES) == {"phat", "delta", "value"}


def test_value_gate_refuses_an_action_that_cannot_repair():
    """P(correct | escalate) is high because the question is easy for the big
    model -- but the small model is already right, so escalating can only lose.
    The confidence gate buys it; the net-value gate does not."""
    s = _rec(True, [True] * 4)
    b = _rec(False, [False], model="big")
    order = [0, 1, 2, 3]
    # tau above the stage-1 estimate, so the policy must consider correcting
    args = dict(stage2=None, tau=0.95, order=order, k=4, k_steps=(4,),
                escalate_gate=lambda *a, **k: 1.0, cost_fn=_tokens)
    phat = caac_staged(s, b, _scorer(answer_now=0.5, escalate=0.9), rho=0.6,
                       rho_mode="phat", **args)
    assert phat.action == "escalate"
    value = caac_staged(s, b, _scorer(answer_now=0.5, escalate=0.9), rho=0.0,
                        rho_mode="delta", recovery=lambda ss, bb: {"escalate": -1.0},
                        **args)
    assert value.action == "abstain"


def test_refusing_to_buy_never_bills_the_big_model():
    """Early abstention has to be early: the whole point is the token bill."""
    s = _rec(False, [False] * 4)
    b = _rec(True, [True], model="big")
    out = caac_staged(s, b, _scorer(answer_now=0.1), None, tau=0.9, order=[0, 1, 2, 3],
                      k=4, k_steps=(4,), rho=0.5, rho_mode="delta", cost_fn=_tokens,
                      recovery=lambda ss, bb: {"sample_more": 0.0, "escalate": 0.0})
    assert out.action == "abstain"
    assert [c.model for c in out.calls] == ["small"]
    assert out.meta["rho_mode"] == "delta"


def _value_args(rho, cost_fn=_tokens):
    return dict(stage2=lambda *a, **k: 1.0, tau=0.5, order=[0, 1, 2, 3], k=4, k_steps=(4,),
                escalate_gate=lambda *a, **k: 1.0, rho=rho, rho_mode="value", cost_fn=cost_fn,
                recovery=lambda ss, bb: {"sample_more": 0.3, "escalate": 0.3})


def test_value_mode_prices_the_action_it_is_considering():
    """A repair worth 0.3 is bought at a low exchange rate and refused at a high
    one. Escalation costs one extra greedy turn here (ratio 0.3), four samples
    cost four (ratio 0.075), so a rate between the two buys only escalation."""
    s = _rec(False, [True] * 4)
    b = _rec(True, [True], model="big")
    cheap = caac_staged(s, b, _scorer(answer_now=0.1), **_value_args(0.01))
    assert cheap.served
    mid = caac_staged(s, b, _scorer(answer_now=0.1), **_value_args(0.2))
    assert mid.action == "escalate" and mid.meta["tried"] == ["escalate"]
    dear = caac_staged(s, b, _scorer(answer_now=0.1), **_value_args(0.5))
    assert dear.action == "abstain" and [c.model for c in dear.calls] == ["small"]


def test_value_gate_does_not_depend_on_the_cost_unit():
    """The regression this guards: with cost in raw tokens the old gate
    subtracted hundreds from a number in [-1, 1] and never bought anything."""
    s = _rec(False, [True] * 4)
    b = _rec(True, [True], model="big")
    for scale in (1.0, 1000.0):
        out = caac_staged(s, b, _scorer(answer_now=0.1),
                          **_value_args(0.2, cost_fn=lambda calls, k=scale: k * _tokens(calls)))
        assert out.action == "escalate"


def test_phat_mode_is_unchanged_by_the_new_arguments():
    """Default behaviour must not move: the old arm is the comparison."""
    small, big = make_pools(40, n_small=16, n_big=4, seed=41)
    for q in sorted(small)[:15]:
        s, b, order = small[q], big[q], list(range(16))
        kw = dict(stage2=lambda *a, **k: 0.99, tau=0.9, order=order, k=8,
                  k_steps=(4, 8), escalate_gate=lambda *a, **k: 0.99, rho=0.2)
        a = caac_staged(s, b, _scorer(answer_now=0.5, sample_more=0.5, escalate=0.5), **kw)
        bb = caac_staged(s, b, _scorer(answer_now=0.5, sample_more=0.5, escalate=0.5),
                         rho_mode="phat", recovery=None, **kw)
        assert (a.action, a.served, a.correct) == (bb.action, bb.served, bb.correct)


def test_a_fitted_recovery_estimator_drops_into_the_gate():
    """End to end: the estimator from TRAIN is callable as the gate's scorer."""
    small, big = make_pools(240, n_small=16, n_big=4, seed=43)
    q = sorted(small)
    rec = RecoveryEstimator(k=8).fit(q, small, big)
    out = caac_staged(small[q[0]], big[q[0]], _scorer(answer_now=0.2), lambda *a, **k: 0.99,
                      tau=0.9, order=list(range(16)), k=8, k_steps=(2, 4, 8),
                      escalate_gate=lambda *a, **k: 0.99, rho=-1.0, rho_mode="value",
                      recovery=rec)
    assert out.action in ("sample_more", "escalate", "abstain")


def test_k_step_two_is_reachable():
    """A 2-sample vote is a real step, not rounded up to 4."""
    s = _rec(False, [True, True] + [False] * 14)
    b = _rec(True, [True], model="big")
    out = caac_staged(s, b, _scorer(answer_now=0.1, sample_more=0.9),
                      stage2=lambda ss, drawn, p1: 1.0 if len(drawn) == 2 else 0.0,
                      tau=0.5, order=list(range(16)), k=16, k_steps=(2, 4, 8, 16),
                      cost_fn=_tokens)
    assert out.action == "sample_more"
    assert out.meta["n_drawn"] == 2


def test_calls_for_action_is_the_shared_pricer():
    s = _rec(False, [True] * 8)
    b = _rec(True, [True], model="big")
    order = list(range(8))
    now = _tokens(calls_for_action(s, b, "answer_now", 4, order))
    sampled = _tokens(calls_for_action(s, b, "sample_more", 4, order))
    escalated = _tokens(calls_for_action(s, b, "escalate", 4, order))
    assert 0 < now < sampled
    assert now < escalated
    assert _tokens(calls_for_action(s, b, "abstain", 4, order)) == now


def test_driver_builds_both_arms_with_different_gates():
    """The confidence arm and the recoverability arm must both be reachable,
    and must not be the same policy under two names."""
    import certify

    small, big = make_pools(240, n_small=16, n_big=4, seed=47)
    q = sorted(small)
    ests = certify.fit_estimators(small, big, (2, 4, 8))
    assert len(ests) == 4
    phat = certify.caac_family("caac_staged", ests, 1.5, "big", (2, 4, 8))
    value = certify.caac_family("caac_staged_value", ests, 1.5, "big", (2, 4, 8))
    rhos = lambda fam: sorted({c[0] for c in fam.configs})
    assert rhos(phat) != rhos(value)
    assert min(rhos(value)) < 0, "a net-value gate needs a grid that reaches below zero"
    delta = certify.caac_family("caac_staged_delta", ests, 1.5, "big", (2, 4, 8))
    assert delta.make(delta.configs[0])(small[q[0]], big[q[0]], list(range(16))).meta["rho_mode"] == "delta"
    out = value.make(value.configs[0])(small[q[0]], big[q[0]], list(range(16)))
    assert out.meta["rho_mode"] == "value"


# --------------------------------------------------------------------------- #
# Conformal Cascade
# --------------------------------------------------------------------------- #

def test_conformal_cascade_commits_on_a_singleton_and_never_abstains():
    fam = conformal_cascade(n_small=4, n_big=4)
    fn = fam.make(0.9)
    agree = _rec(False, [True] * 4)                       # unanimous vote
    b = _rec(True, [True] * 4, model="big")
    out = fn(agree, b, list(range(4)))
    assert (out.served, out.action, out.meta["tier"]) == (True, "sample_more", "small")
    split = _rec(False, [True, False, True, False])        # no singleton
    out2 = fn(split, b, list(range(4)))
    assert out2.served and out2.action == "escalate"


def test_conformal_cascade_abstain_variant_can_decline():
    fam = conformal_cascade(n_small=4, n_big=4, abstain_last=True)
    fn = fam.make(0.9)
    split = _rec(False, [True, False, True, False])
    b_split = _rec(False, [True, False, True, False], model="big")
    out = fn(split, b_split, list(range(4)))
    assert not out.served and out.action == "abstain"
    assert fam.name == "conformal_cascade_abstain"


def test_conformal_cascade_draws_a_fixed_number_of_samples():
    """No adaptation: the small tier always pays for n_small completions."""
    fam = conformal_cascade(n_small=4, n_big=4)
    for cut in (0.0, 0.5, 1.0):
        out = fam.make(cut)(_rec(False, [True] * 8), _rec(True, [True] * 4, model="big"),
                            list(range(8)))
        assert len(out.calls[0].decode_tokens) == 4


def test_value_arm_buys_compute_on_a_real_shaped_pool():
    """End to end at token-scale costs: some configuration of the value arm must
    spend more than the greedy turn, or the arm is dead."""
    import certify
    from caac.pool.protocol import build_table

    small, big = make_pools(300, n_small=16, n_big=4, seed=53)
    q = sorted(small)
    ests = certify.fit_estimators(small, big, (2, 4, 8))
    fam = certify.caac_family("caac_staged_value", ests, 1.5, "big", (2, 4, 8))
    t = build_table(fam, q, small, big)
    bought = [m for m in t.actions if m.get("sample_more", 0) + m.get("escalate", 0) > 0]
    assert bought, "no configuration of the value arm ever buys a correcting action"
    assert t.actions[-1].get("sample_more", 0) + t.actions[-1].get("escalate", 0) == 0
