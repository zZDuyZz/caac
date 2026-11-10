"""Tests for the recoverability measurement and the net-value estimator.

The claims a reviewer would attack, in order: that "sampling repairs this" is
the deployable vote and not an oracle; that the four groups are a distribution;
that the region split really does move with the price; that the net value
counts the damage an action can do; and that the estimator still cannot see the
outcome it predicts.
"""

from __future__ import annotations

import pytest

from caac.pool.estimator import RecoveryEstimator, recovery_targets
from caac.pool.recovery import (PRIMITIVE_GROUPS, REGIONS, delta_probs, group_probs,
                                harm_probs, oracle_rescue, recovery_table,
                                region_of_group, region_probs, rescue_probs,
                                sample_correct_prob, sample_correct_realised)
from caac.pool.schema import Completion, PoolRecord
from caac.pool.synth import make_pair, make_pools


def _rec(greedy_ok: bool, sample_ok: list[bool], model="small", gold="G") -> PoolRecord:
    def comp(ok):
        return Completion(answer=gold if ok else "W1", correct=ok, decode_tokens=100,
                          finish_reason="stop", confidence=0.7)
    return PoolRecord(query_id="q", dataset="t", model=model, gold=gold, prompt_tokens=10,
                      greedy=comp(greedy_ok), samples=[comp(o) for o in sample_ok])


# --------------------------------------------------------------------------- #
# deployable, not oracle
# --------------------------------------------------------------------------- #

def test_rescue_is_the_vote_not_the_best_sample():
    """One correct sample among many identical wrong ones does NOT count: the
    majority a deployed system serves is wrong. This is the whole distinction
    between recoverability and an oracle upper bound."""
    s = _rec(False, [True, False, False, False])
    assert sample_correct_prob(s, 4) == 0.0
    assert oracle_rescue(s, 4) == 1.0          # the right answer IS in the pool


def test_oracle_bound_is_never_below_the_deployable_number():
    small, big = make_pools(60, n_small=16, n_big=4, seed=5)
    for q, s in small.items():
        assert oracle_rescue(s, 8) >= sample_correct_prob(s, 8) - 1e-12


def test_realised_vote_matches_the_exact_probability_when_unanimous():
    s = _rec(False, [True] * 8)
    assert sample_correct_prob(s, 4) == pytest.approx(1.0)
    assert sample_correct_realised(s, list(range(8)), 4) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# the four groups and the three regions
# --------------------------------------------------------------------------- #

def test_groups_are_a_distribution_over_greedy_wrong_questions():
    small, big = make_pools(40, n_small=16, n_big=4, seed=7)
    for q, s in small.items():
        g = group_probs(s, big[q], 8)
        if s.greedy.correct:
            assert g == {}                     # nothing to rescue
        else:
            assert set(g) == set(PRIMITIVE_GROUPS)
            assert sum(g.values()) == pytest.approx(1.0)


def test_region_split_follows_the_price():
    """A question both actions can repair changes region when sampling stops
    being the cheaper repair -- so a single cost proxy cannot settle it."""
    assert region_of_group("both", sample_cost=1.0, escalate_cost=5.0) == "cheap_recoverable"
    assert region_of_group("both", sample_cost=5.0, escalate_cost=1.0) == "expensive_only"
    assert region_of_group("sampling_only", 9.0, 1.0) == "cheap_recoverable"
    assert region_of_group("escalation_only", 1.0, 9.0) == "expensive_only"
    assert region_of_group("neither", 1.0, 1.0) == "unrecoverable"
    with pytest.raises(ValueError):
        region_of_group("nonsense", 1.0, 1.0)


def test_unrecoverable_only_when_neither_action_repairs():
    s, b = _rec(False, [False] * 8), _rec(False, [False], model="big")
    r = region_probs(s, b, 8, 1.0, 2.0)
    assert r["unrecoverable"] == pytest.approx(1.0)
    b2 = _rec(True, [True], model="big")
    assert region_probs(s, b2, 8, 1.0, 2.0)["unrecoverable"] == pytest.approx(0.0)


def test_region_probs_sum_to_the_group_mass():
    small, big = make_pools(30, n_small=16, n_big=4, seed=11)
    for q, s in small.items():
        if s.greedy.correct:
            continue
        r = region_probs(s, big[q], 8, 1.0, 3.0)
        assert set(r) == set(REGIONS)
        assert sum(r.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# net value: an action can also break a right answer
# --------------------------------------------------------------------------- #

def test_harm_is_measured_on_greedy_correct_questions():
    s = _rec(True, [False] * 8)                       # every sample disagrees
    assert rescue_probs(s, None, 8) == {}
    assert harm_probs(s, None, 8)["sample_more"] == pytest.approx(1.0)
    assert delta_probs(s, None, 8)["sample_more"] == pytest.approx(-1.0)


def test_delta_equals_after_minus_now():
    """The identity the estimator is built on, checked on the synthetic pools."""
    small, big = make_pools(50, n_small=16, n_big=4, seed=13)
    for q, s in small.items():
        b = big[q]
        now = float(s.greedy.correct)
        d = delta_probs(s, b, 8)
        assert d["sample_more"] == pytest.approx(sample_correct_prob(s, 8) - now)
        assert d["escalate"] == pytest.approx(float(b.greedy.correct) - now)


def test_escalating_a_right_answer_can_lose_it():
    s = _rec(True, [True] * 4)
    b = _rec(False, [False], model="big")
    assert delta_probs(s, b, 4)["escalate"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #

def test_recovery_table_shapes_and_bounds():
    small, big = make_pools(80, n_small=16, n_big=4, seed=17)
    t = recovery_table(sorted(small), small, big, k=8)
    assert t["n"] == 80
    assert t["n_greedy_wrong"] + t["n_greedy_right"] == 80
    assert set(t["groups"]) == set(PRIMITIVE_GROUPS)
    assert set(t["regions"]) == set(REGIONS)
    assert sum(t["groups"].values()) == pytest.approx(1.0)
    assert sum(t["regions"].values()) == pytest.approx(1.0)
    assert t["oracle_minus_deployable"] >= -1e-12
    for v in t["mean_delta"].values():
        assert -1.0 <= v <= 1.0


def test_recovery_table_without_a_big_pool_reports_no_groups():
    small, _ = make_pools(40, n_small=16, seed=19)
    t = recovery_table(sorted(small), small, None, k=8)
    assert sum(t["groups"].values()) == pytest.approx(0.0)
    assert "sample_more" in t["mean_delta"]


def test_recovery_table_price_changes_only_the_regions():
    """The groups are a measurement; the regions are a reading of them at one
    price. Making a vote dearer than an escalation must move mass from
    cheap_recoverable to expensive_only and leave the groups alone."""
    small, big = make_pools(60, n_small=16, n_big=4, seed=23)
    flat = recovery_table(sorted(small), small, big, k=8, price=lambda c: 1.0)
    # one unit per completion: 8 samples cost more than one big-model answer
    per_draw = recovery_table(sorted(small), small, big, k=8,
                              price=lambda cs: sum(len(c.decode_tokens) for c in cs))
    assert flat["groups"] == per_draw["groups"]
    assert flat["groups"]["both"] > 0.0
    assert per_draw["regions"]["cheap_recoverable"] < flat["regions"]["cheap_recoverable"]
    assert per_draw["regions"]["unrecoverable"] == pytest.approx(
        flat["regions"]["unrecoverable"])


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #

def test_recovery_targets_put_all_mass_on_the_possible_half():
    s_wrong, b = _rec(False, [True] * 8), _rec(True, [True], model="big")
    t = recovery_targets(s_wrong, b, 8)
    assert t["escalate"] == {"gain": 1.0, "harm": 0.0}
    s_right = _rec(True, [True] * 8)
    t2 = recovery_targets(s_right, b, 8)
    assert t2["escalate"] == {"gain": 0.0, "harm": 0.0}
    assert t2["sample_more"]["gain"] == 0.0


def test_recovery_estimator_fits_and_predicts_in_range():
    small, big = make_pools(240, n_small=16, n_big=4, seed=29)
    q = sorted(small)
    est = RecoveryEstimator(k=8).fit(q, small, big)
    assert est.heads, "no head was fitted"
    d = est(small[q[0]])
    assert set(d) <= {"sample_more", "escalate"}
    for v in d.values():
        assert -1.0 <= v <= 1.0


def test_recovery_estimator_cannot_see_the_action_outcome():
    """Same guard as the per-action estimator: deleting every sample and
    flipping the big model must not move the prediction."""
    small, big = make_pools(240, n_small=16, n_big=4, seed=31)
    q = sorted(small)
    est = RecoveryEstimator(k=8).fit(q, small, big)
    rec = small[q[0]]
    before = est(rec, big[q[0]])
    rec.samples = []
    big[q[0]].greedy.correct = not big[q[0]].greedy.correct
    assert est(rec, big[q[0]]) == before


def test_recovery_estimator_round_trips_through_json():
    small, big = make_pools(240, n_small=16, n_big=4, seed=37)
    q = sorted(small)
    est = RecoveryEstimator(k=8).fit(q, small, big)
    back = RecoveryEstimator.from_json(est.to_json())
    for qid in q[:20]:
        assert back(small[qid]) == pytest.approx(est(small[qid]))


def test_unfitted_recovery_estimator_scores_nothing():
    """An unfitted head must not produce a guess the gate could act on."""
    s, _ = make_pair("q0", "synth", "small", "big", n_small=16, n_big=4, seed=3)
    assert RecoveryEstimator()(s) == {}
