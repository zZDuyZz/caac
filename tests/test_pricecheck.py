"""The PriceCheck baseline must never serve a corrected answer."""
import pytest

from caac.pool.policies import agreement_with_greedy, pricecheck, self_consistency
from caac.pool.schema import Completion, PoolRecord

M = "Qwen/Qwen2.5-1.5B-Instruct"


def _c(ok, ans=None, dec=100, conf=0.5):
    return Completion(answer=ans if ans is not None else ("x" if ok else "w"),
                      correct=ok, decode_tokens=dec, finish_reason="stop", confidence=conf)


def _rec(greedy_ok, samples, conf=0.5, greedy_ans=None):
    g = _c(greedy_ok, greedy_ans, conf=conf)
    return PoolRecord("q", "d", M, "x", 50, g,
                      [s if isinstance(s, Completion) else _c(s) for s in samples])


def test_agreement_compares_answer_classes():
    # greedy correct: agreement = share of correct samples
    r = _rec(True, [True, True, False, False])
    assert agreement_with_greedy(r, [0, 1, 2, 3]) == pytest.approx(0.5)
    # greedy wrong: only samples with the SAME wrong string agree
    r = _rec(False, [_c(False, "w"), _c(False, "other"), _c(True)], greedy_ans="w")
    assert agreement_with_greedy(r, [0, 1, 2]) == pytest.approx(1 / 3)
    assert agreement_with_greedy(r, []) == 0.0


def test_serves_greedy_even_when_the_vote_knows_better():
    """The defining limitation: 7/8 re-solves are right, greedy is wrong.
    Self-consistency fixes the answer; PriceCheck can only serve or abstain."""
    r = _rec(False, [True] * 7 + [False], greedy_ans="w")
    sc = self_consistency(r, 8, list(range(8)))
    assert sc.correct is True                      # SC corrects the answer

    served = pricecheck(r, lam=0.0, check="resolve_vote", k=8)   # threshold so low it serves
    assert served.served and served.correct is False             # still the WRONG greedy answer
    assert served.answer == "w"
    abstained = pricecheck(r, lam=0.5, check="resolve_vote", k=8)
    assert not abstained.served and abstained.action == "abstain"


def test_confidence_check_costs_nothing_extra():
    r = _rec(True, [True] * 8, conf=0.9)
    out = pricecheck(r, lam=0.8, check="confidence")
    assert out.served and len(out.calls) == 1                     # greedy call only
    assert not pricecheck(r, lam=0.95, check="confidence").served


def test_schedule_skips_the_expensive_check_when_confident():
    r = _rec(True, [True] * 8, conf=0.9)
    cheap = pricecheck(r, lam=0.5, check="both", k=8, conf_threshold=0.8)
    assert cheap.served and len(cheap.calls) == 1                 # re-solves not paid for
    assert cheap.meta["decided_by"] == "confidence"
    expensive = pricecheck(r, lam=0.5, check="both", k=8, conf_threshold=0.95)
    assert expensive.served and len(expensive.calls) == 2         # had to pay for re-solves


def test_abstaining_still_pays_for_what_it_ran():
    # re-solves land on a DIFFERENT wrong answer -> they disagree -> abstain,
    # but the greedy run and all 8 re-solves were still paid for
    r = _rec(False, [_c(False, "other")] * 8, conf=0.1, greedy_ans="w")
    out = pricecheck(r, lam=0.9, check="resolve_vote", k=8)
    assert not out.served
    assert sum(len(c.decode_tokens) for c in out.calls) == 9      # greedy + 8 re-solves


def test_confidently_wrong_fools_the_check():
    """Honest failure mode to report: when every re-solve repeats the SAME wrong
    answer, agreement is 1.0 and PriceCheck serves a wrong answer at any lam.
    No serve/abstain check can detect this -- it is the pi_pop ceiling in action."""
    r = _rec(False, [_c(False, "w")] * 8, greedy_ans="w")
    out = pricecheck(r, lam=1.0, check="resolve_vote", k=8)
    assert out.served and out.correct is False


def test_rejects_unknown_check():
    with pytest.raises(ValueError):
        pricecheck(_rec(True, [True]), lam=0.5, check="magic")
