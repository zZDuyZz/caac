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
