"""Tests for the calibrated per-action estimator.

The invariants here are the ones a reviewer would attack: that the features
cannot see the outcome they predict, that calibration is fitted out-of-fold,
and that an action with no fitted head is treated as unusable rather than as a
coin flip (the second would quietly break the risk guarantee).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from caac.eval.calibration import expected_calibration_error
from caac.pool.estimator import (SERVING_ACTIONS, Isotonic, LogisticHead,
                                 PerActionEstimator, action_targets, feature_names,
                                 features, realised_targets)
from caac.pool.schema import Completion
from caac.pool.synth import make_pair, make_pools


def _pair():
    return make_pair("q0", "synth", "small", "big", n_small=16, n_big=4, seed=3)


# --------------------------------------------------------------------------- #
# features: no peeking
# --------------------------------------------------------------------------- #

def test_feature_vector_matches_names():
    s, _ = _pair()
    for groups in (("qonly",), ("greedy",), ("qonly", "greedy")):
        assert len(features(s, groups)) == len(feature_names(groups))


def test_features_ignore_samples_and_big_model():
    """The decisive property: sample_more's features must not look at the
    samples, or the estimator would be perfect offline and useless online."""
    s, b = _pair()
    before = features(s)
    s.samples = []                                  # delete every sample
    assert features(s) == before
    b.greedy.correct = not b.greedy.correct         # and perturb the big model
    assert features(s) == before


def test_features_tolerate_old_pools_without_the_proxy():
    s, _ = _pair()
    s.greedy.conf_prefix32 = None
    s.greedy.mean_logprob = None
    v = features(s)
    assert all(math.isfinite(x) for x in v)


def test_extra_features_extend_the_vector():
    s, _ = _pair()
    assert len(features(s, ("qonly",), extra=[0.1, 0.2])) == \
        len(features(s, ("qonly",))) + 2
    assert feature_names(("qonly",), extra_dim=2)[-1] == "extra_1"


def test_unknown_feature_group_is_rejected():
    s, _ = _pair()
    with pytest.raises(ValueError):
        features(s, ("nope",))


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #

def test_soft_target_is_exact_maj_at_k():
    s, b = _pair()
    t = action_targets(s, b, k=8)
    assert t["answer_now"] == float(s.greedy.correct)
    assert t["escalate"] == float(b.greedy.correct)
    assert t["sample_more"] == pytest.approx(s.maj_at_k(8))
    assert 0.0 <= t["sample_more"] <= 1.0


def test_escalate_target_absent_without_a_big_pool():
    s, _ = _pair()
    assert "escalate" not in action_targets(s, None)


def test_realised_target_is_a_hard_vote_and_order_dependent():
    """Evaluation must use an achievable outcome, so it is 0/1 and depends on
    which samples were drawn."""
    s, b = _pair()
    seen = set()
    for order in ([0, 1, 2, 3, 4, 5, 6, 7], list(range(15, 7, -1))):
        v = realised_targets(s, b, order, k=8)["sample_more"]
        assert v in (0.0, 1.0)
        seen.add(v)
    assert realised_targets(s, b, list(range(8)), k=8)["answer_now"] == \
        float(s.greedy.correct)


# --------------------------------------------------------------------------- #
# logistic head
# --------------------------------------------------------------------------- #

def test_logistic_head_learns_a_monotone_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 1))
    y = (rng.random(600) < 1 / (1 + np.exp(-2.5 * x[:, 0]))).astype(float)
    h = LogisticHead(l2=0.1).fit(x, y)
    p = h.predict([[-2.0], [0.0], [2.0]])
    assert p[0] < p[1] < p[2]
    assert p[0] < 0.3 and p[2] > 0.7


def test_logistic_head_accepts_soft_labels_and_matches_the_base_rate():
    x = np.zeros((200, 1))                     # no signal at all
    y = np.full(200, 0.37)                     # soft labels
    p = LogisticHead(l2=1.0).fit(x, y).predict([[0.0]])[0]
    assert p == pytest.approx(0.37, abs=0.02)  # intercept is unpenalised


def test_logistic_head_rejects_labels_outside_the_unit_interval():
    with pytest.raises(ValueError):
        LogisticHead().fit(np.zeros((5, 1)), [0, 1, 2, 0, 1])


def test_logistic_head_survives_a_constant_column():
    x = np.column_stack([np.ones(50), np.linspace(0, 1, 50)])
    y = (np.linspace(0, 1, 50) > 0.5).astype(float)
    p = LogisticHead().fit(x, y).predict(x)
    assert np.all(np.isfinite(p)) and p[0] < p[-1]


def test_logistic_head_json_round_trip():
    x = np.random.default_rng(1).normal(size=(50, 3))
    y = (x[:, 0] > 0).astype(float)
    h = LogisticHead().fit(x, y)
    h2 = LogisticHead.from_dict(h.to_dict())
    assert np.allclose(h.predict(x), h2.predict(x))


def test_unfitted_head_raises():
    with pytest.raises(RuntimeError):
        LogisticHead().predict([[0.0]])


# --------------------------------------------------------------------------- #
# isotonic calibration
# --------------------------------------------------------------------------- #

def test_isotonic_is_monotone():
    rng = np.random.default_rng(2)
    s = rng.random(300)
    y = (rng.random(300) < s).astype(float)
    iso = Isotonic().fit(s, y)
    xs = np.linspace(0, 1, 50)
    p = iso.predict(xs)
    assert np.all(np.diff(p) >= -1e-9)
    assert np.all((p >= 0) & (p <= 1))


def test_isotonic_fixes_a_miscalibrated_score():
    """A score that ranks well but is systematically over-confident should come
    back calibrated -- this is the whole reason the threshold is trustworthy."""
    rng = np.random.default_rng(3)
    true_p = rng.random(1500)
    y = (rng.random(1500) < true_p).astype(float)
    raw = np.clip(true_p * 0.5 + 0.5, 0, 1)          # squashed into [0.5, 1]
    iso = Isotonic().fit(raw, y)
    before = expected_calibration_error(raw, y.astype(int))
    after = expected_calibration_error(iso.predict(raw), y.astype(int))
    assert after < before / 2


def test_isotonic_empty_fit_is_the_identity():
    iso = Isotonic().fit([], [])
    assert iso.predict([0.3])[0] == pytest.approx(0.3)


def test_isotonic_rejects_misaligned_input():
    with pytest.raises(ValueError):
        Isotonic().fit([0.1, 0.2], [1])


# --------------------------------------------------------------------------- #
# the estimator as a whole
# --------------------------------------------------------------------------- #

def test_estimator_fits_every_action_and_stays_in_range():
    small, big = make_pools(240, seed=5)
    q = sorted(small)
    est = PerActionEstimator().fit(q, small, big)
    assert set(est.heads) == set(SERVING_ACTIONS)
    p = est(small[q[0]], big[q[0]])
    assert p["abstain"] == 1.0
    for a in SERVING_ACTIONS:
        assert 0.0 <= p[a] <= 1.0


def test_estimator_ranks_better_than_chance_on_held_out_questions():
    small, big = make_pools(600, seed=6)
    q = sorted(small)
    est = PerActionEstimator().fit(q[:400], small, big)
    ps = [est(small[k])["answer_now"] for k in q[400:]]
    ys = [small[k].greedy.correct for k in q[400:]]
    hi = np.mean([p for p, y in zip(ps, ys) if y])
    lo = np.mean([p for p, y in zip(ps, ys) if not y])
    assert hi > lo + 0.05          # correct questions get higher scores


def test_estimator_calibration_is_fitted_out_of_fold():
    """The calibrator may only see scores from folds its head did not fit."""
    small, big = make_pools(200, seed=7)
    q = sorted(small)
    est = PerActionEstimator(n_folds=4).fit(q, small, big)
    assert set(est.oof["answer_now"]) == set(q)
    # out-of-fold scores differ from the refitted head's in-fold scores
    infold = [float(est.heads["answer_now"].predict([features(small[k])])[0]) for k in q]
    oof = [est.oof["answer_now"][k] for k in q]
    assert not np.allclose(infold, oof)


def test_unfitted_action_is_unusable_not_a_coin_flip():
    """Safety: with no big pool there is no escalate head, and the scorer must
    report 0.0 so the controller can never pick it."""
    small, _ = make_pools(60, seed=8)
    q = sorted(small)
    est = PerActionEstimator().fit(q, small, None)
    assert "escalate" not in est.heads
    assert est(small[q[0]])["escalate"] == 0.0


def test_estimator_ignores_the_big_record_it_is_handed():
    """escalate is predicted from the question, because at decision time the
    big model has not run."""
    small, big = make_pools(120, seed=9)
    q = sorted(small)
    est = PerActionEstimator().fit(q, small, big)
    a = est(small[q[0]], big[q[0]])
    other = big[q[1]]
    b = est(small[q[0]], other)
    assert a == b


def test_estimator_json_round_trip_preserves_predictions():
    small, big = make_pools(150, seed=10)
    q = sorted(small)
    est = PerActionEstimator().fit(q, small, big)
    back = PerActionEstimator.from_json(est.to_json())
    for k in q[:20]:
        assert est(small[k]) == pytest.approx(back(small[k]))


def test_question_only_estimator_uses_fewer_features():
    small, big = make_pools(150, seed=11)
    q = sorted(small)
    est = PerActionEstimator(groups=("qonly",)).fit(q, small, big)
    assert est.heads["answer_now"].w.shape[0] == len(feature_names(("qonly",))) + 1
