import pytest

from caac.pool.frontier import (cap_correct, cap_serve, cost_ratio_at, frontier_curve,
                                pi_after, recovery_rate, wilson)
from caac.pool.schema import Completion, PoolRecord

S, B = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"


def _c(ok, dec=100):
    return Completion(answer="x" if ok else "w", correct=ok, decode_tokens=dec,
                      finish_reason="stop")


def _rec(model, ok, samples):
    return PoolRecord("q", "d", model, "x", 50, _c(ok), [_c(s) for s in samples])


def _pools(spec):
    """spec: list of (greedy_ok, [sample_oks], big_greedy_ok)"""
    small, big = {}, {}
    for i, (g, ss, bg) in enumerate(spec):
        q = f"q{i}"
        small[q] = PoolRecord(q, "d", S, "x", 50, _c(g), [_c(s) for s in ss])
        big[q] = PoolRecord(q, "d", B, "x", 50, _c(bg), [])
    return small, big


def test_chow_caps_match_the_formula():
    assert cap_serve(0.49, 0.10) == pytest.approx(0.49 / 0.9)
    assert cap_serve(0.957, 0.10) == pytest.approx(1.0)          # saturates
    assert pi_after(0.5, 0.4) == pytest.approx(0.7)
    assert cap_correct(0.5, 0.10, 0.4) == pytest.approx(0.7 / 0.9)
    assert cap_correct(0.5, 0.10, 0.0) == cap_serve(0.5, 0.10)   # r=0 => no lift


def test_recovery_rate_counts_only_greedy_wrong():
    # q0 greedy right; q1 wrong but maj@8 right; q2 wrong, maj wrong, 7B right;
    # q3 wrong and nothing works
    small, big = _pools([
        (True,  [True] * 8,  True),
        (False, [True] * 8,  False),
        (False, [False] * 8, True),
        (False, [False] * 8, False),
    ])
    rr = recovery_rate(small, big, k=8)
    assert rr.n_total == 4 and rr.n_greedy_wrong == 3
    assert rr.pi_greedy == pytest.approx(0.25)
    assert rr.n_recovered_any == 2 and rr.r == pytest.approx(2 / 3)
    assert rr.by_action == {"sample_more": 1, "escalate": 1, "both": 0}
    lo, hi = rr.r_ci
    assert lo < rr.r < hi


def test_recovery_without_big_model_uses_sampling_only():
    small, big = _pools([(False, [True] * 8, True), (False, [False] * 8, True)])
    rr = recovery_rate(small, None, k=8)
    assert rr.n_recovered_any == 1 and rr.by_action["escalate"] == 0


def test_majority_needs_a_real_majority():
    # 3/8 correct: gold does not win a majority -> not recovered
    small, _ = _pools([(False, [True] * 3 + [False] * 5, False)])
    assert recovery_rate(small, None, k=8).n_recovered_any == 0
    small, _ = _pools([(False, [True] * 6 + [False] * 2, False)])
    assert recovery_rate(small, None, k=8).n_recovered_any == 1


def test_frontier_curve_monotone_and_gap_closes_at_the_top():
    curve = frontier_curve(0.10, r=0.5, pis=[0.1, 0.3, 0.5, 0.7, 0.95])
    assert all(c["cap_correct"] >= c["cap_serve"] for c in curve)
    caps = [c["cap_serve"] for c in curve]
    assert caps == sorted(caps)                       # monotone in pi
    assert curve[-1]["gap"] == pytest.approx(0.0)     # both saturate at high pi
    assert curve[0]["gap"] > 0.1                      # binds hard at low pi


def test_cost_ratio_is_relative_to_answer_now():
    small, big = _pools([(True, [True] * 8, True)])
    cr = cost_ratio_at(small, big, k=8, profile="tokens")
    assert cr["answer_now"] == pytest.approx(1.0)
    assert cr["sample_more"] > cr["answer_now"]
    assert cr["escalate"] > cr["answer_now"]


def test_wilson_edges():
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 0.4
