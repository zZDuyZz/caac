"""What counts as a vote: truncated samples never do, and greedy always does."""

from __future__ import annotations

import pytest

from caac.pool.families import sc_abstain
from caac.pool.schema import Completion, PoolRecord
from caac.pool.voting import greedy_vote_prob, serve_with_greedy


def _c(ans, ok, fin="stop"):
    return Completion(answer=ans, correct=ok, decode_tokens=50, finish_reason=fin, confidence=0.6)


def _rec(greedy, samples, gold="G"):
    return PoolRecord(query_id="q", dataset="t", model="small", gold=gold, prompt_tokens=10,
                      greedy=greedy, samples=samples)


def test_truncated_sample_is_not_a_vote():
    s = _rec(_c("W", False), [_c("G", True), _c("W2", False, "length"), _c("W2", False, "length")])
    assert s.canonical_sample_answers()[1:] == [None, None]
    assert s.canonical_sample_answers()[0] is not None


def test_truncation_changes_the_exact_majority():
    cut = _rec(_c("W", False), [_c("G", True), _c("X", False, "length"), _c("X", False, "length")])
    kept = _rec(_c("W", False), [_c("G", True), _c("X", False), _c("X", False)])
    assert cut.maj_at_k(3) == pytest.approx(1.0)
    assert kept.maj_at_k(3) == pytest.approx(0.0)


def test_greedy_is_one_vote_and_wins_ties():
    s = _rec(_c("G", True), [_c("W1", False), _c("W2", False)])
    labels = s.canonical_sample_answers()
    assert serve_with_greedy(s, [0, 1]) == s.greedy_label()
    assert labels[0] != labels[1]


def test_two_agreeing_samples_overturn_greedy():
    s = _rec(_c("W", False), [_c("G", True), _c("G", True), _c("W", False)])
    assert s.label_correct(serve_with_greedy(s, [0, 1]))
    assert not s.label_correct(serve_with_greedy(s, [0, 2]))


def test_plain_majority_can_break_a_right_greedy_but_the_new_vote_cannot_at_k2():
    s = _rec(_c("G", True), [_c("A", False), _c("B", False)] + [_c("G", True)] * 6)
    assert greedy_vote_prob(s, 2) > s.maj_at_k(2)


def test_greedy_vote_prob_is_deterministic():
    s = _rec(_c("W", False), [_c("G", True)] * 9 + [_c("W", False)] * 9)
    assert greedy_vote_prob(s, 4) == greedy_vote_prob(s, 4)


def test_sc_abstain_greedy_is_a_distinct_family():
    assert sc_abstain().name == "sc_abstain"
    assert sc_abstain(greedy_vote=True).name == "sc_abstain_greedy"
    s = _rec(_c("G", True), [_c("A", False), _c("B", False)])
    plain = sc_abstain(ks=(2,), grid=(0.0,)).make((2, 0.0))
    withg = sc_abstain(ks=(2,), grid=(0.0,), greedy_vote=True).make((2, 0.0))
    assert withg(s, None, [0, 1]).correct
    assert plain(s, None, [0, 1]) is not None


def test_sc_abstain_greedy_pays_for_the_greedy_turn():
    s = _rec(_c("G", True), [_c("G", True), _c("G", True)])
    plain = sc_abstain(ks=(2,), grid=(0.0,)).make((2, 0.0))(s, None, [0, 1])
    withg = sc_abstain(ks=(2,), grid=(0.0,), greedy_vote=True).make((2, 0.0))(s, None, [0, 1])
    tokens = lambda out: sum(c.prefill_tokens + sum(c.decode_tokens) for c in out.calls)
    assert tokens(withg) == tokens(plain) + s.prompt_tokens + s.greedy.decode_tokens


def test_single_stage_caac_serves_the_vote_its_estimator_was_fitted_on():
    from caac.pool.policies import caac_certified
    s = _rec(_c("G", True), [_c("A", False), _c("B", False)])
    out = caac_certified(s, None, lambda a, b: {"sample_more": 1.0}, tau=0.5, order=[0, 1], k=2,
                         actions=("sample_more",), cost_fn=lambda calls: 1.0)
    assert out.action == "sample_more" and out.correct


def test_adaptive_consistency_survives_draws_with_no_usable_answer():
    from caac.pool.adaptive import adaptive_consistency
    cut = [_c("X", False, "length")] * 3
    s = _rec(_c("W", False), cut + [_c("G", True)] * 5)
    out = adaptive_consistency(s, list(range(8)), max_k=8)
    assert out.answer is not None and out.correct
    none = adaptive_consistency(_rec(_c("W", False), cut), [0, 1, 2], max_k=3)
    assert none.answer is None and not none.correct


def test_oracle_is_a_ceiling_whatever_the_voter():
    from caac.pool.families import oracle
    s = _rec(_c("W", False), [_c("G", True), _c("W", False), _c("W", False), _c("W", False)])
    assert oracle(["q"], {"q": s}, None, 1.0, (4,))["coverage"] == 1.0


def test_draw_counting_baselines_use_the_draws_every_split_holds():
    import certify
    fam = certify.baseline_family("conformal_cascade", n_big=8, n_small=16)
    s = _rec(_c("W", False), [_c("A", False), _c("B", False)] * 8)
    b = PoolRecord(query_id="q", dataset="t", model="big", gold="G", prompt_tokens=10,
                   greedy=_c("G", True), samples=[_c("G", True)] * 16)
    out = fam.make(1.0)(s, b, list(range(16)))
    assert len(out.calls[1].decode_tokens) == 8
