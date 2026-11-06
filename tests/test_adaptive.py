"""Tests for the adaptive-sampling baselines.

Two classes of invariant: the stopping rules do what the papers say, and the
ledger bills every token that was generated -- including the traces DeepConf
throws away, which is the one place a cost comparison could be quietly rigged
in our favour.
"""

from __future__ import annotations

import pytest

from caac.pool.adaptive import (adaptive_consistency, beta_win_probability,
                                deepconf_offline, esc, trace_confidence)
from caac.pool.ledger import cost_profiles
from caac.pool.policies import self_consistency
from caac.pool.schema import GOLD, Completion, PoolRecord
from caac.pool.synth import make_pools

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ORDER = list(range(16))


def _rec(sample_spec, decode=100):
    """sample_spec: list of (answer, correct, confidence)."""
    g = Completion(answer="G", correct=True, decode_tokens=decode,
                   finish_reason="stop", confidence=0.8)
    ss = [Completion(answer=a, correct=c, decode_tokens=decode,
                     finish_reason="stop", confidence=conf)
          for a, c, conf in sample_spec]
    return PoolRecord(query_id="q", dataset="d", model=MODEL, gold="G",
                      prompt_tokens=80, greedy=g, samples=ss)


# --------------------------------------------------------------------------- #
# the stopping rule itself
# --------------------------------------------------------------------------- #

def test_beta_win_probability_is_symmetric_and_bounded():
    assert beta_win_probability(0, 0) == pytest.approx(0.5)
    assert beta_win_probability(3, 3) == pytest.approx(0.5)
    assert 0.0 <= beta_win_probability(5, 0) <= 1.0


def test_beta_win_probability_grows_with_the_margin():
    vals = [beta_win_probability(n, 0) for n in range(1, 8)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    assert beta_win_probability(5, 0) > beta_win_probability(5, 3)


def test_beta_win_probability_matches_the_closed_form():
    # P(Beta(3,1) > 1/2) = 1 - (1/2)^3 = 0.875
    assert beta_win_probability(2, 0) == pytest.approx(0.875)
    # P(Beta(2,2) > 1/2) = 0.5 by symmetry
    assert beta_win_probability(1, 1) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# adaptive consistency
# --------------------------------------------------------------------------- #

def test_adaptive_consistency_stops_early_when_samples_agree():
    rec = _rec([("G", True, 0.9)] * 16)
    out = adaptive_consistency(rec, ORDER, threshold=0.9, max_k=8)
    assert out.meta["stopped_early"]
    assert out.meta["n_drawn"] < 8
    assert out.served and out.correct


def test_adaptive_consistency_spends_the_full_budget_when_split():
    rec = _rec([("G", True, 0.9), ("W", False, 0.9)] * 8)
    out = adaptive_consistency(rec, ORDER, threshold=0.99, max_k=8)
    assert not out.meta["stopped_early"]
    assert out.meta["n_drawn"] == 8


def test_adaptive_consistency_never_exceeds_max_k():
    rec = _rec([("G", True, 0.9), ("W", False, 0.9)] * 8)
    for max_k in (2, 4, 8, 16):
        assert adaptive_consistency(rec, ORDER, 0.999, max_k).meta["n_drawn"] <= max_k


def test_adaptive_consistency_is_billed_per_sequential_round():
    """Its saving is in tokens; the draws are serial, so it is the WORST policy
    on the latency axis. The ledger must show that, not hide it."""
    rec = _rec([("G", True, 0.9)] * 16)
    out = adaptive_consistency(rec, ORDER, threshold=0.9, max_k=8)
    assert cost_profiles(out.calls)["rounds"] == out.meta["n_drawn"]
    sc = self_consistency(rec, 8, ORDER)
    assert cost_profiles(sc.calls)["rounds"] == 1
    assert cost_profiles(out.calls)["tokens"] < cost_profiles(sc.calls)["tokens"]


def test_a_lower_threshold_never_draws_more():
    rec = _rec([("G", True, 0.9)] * 6 + [("W", False, 0.9)] * 10)
    n = [adaptive_consistency(rec, ORDER, t, 8).meta["n_drawn"]
         for t in (0.80, 0.90, 0.95, 0.99)]
    assert all(a <= b for a, b in zip(n, n[1:]))


# --------------------------------------------------------------------------- #
# ESC
# --------------------------------------------------------------------------- #

def test_esc_stops_at_the_first_unanimous_window():
    rec = _rec([("G", True, 0.9)] * 16)
    out = esc(rec, ORDER, window=4, max_k=8)
    assert out.meta["n_drawn"] == 4 and out.meta["stopped_early"]
    assert cost_profiles(out.calls)["rounds"] == 1


def test_esc_continues_past_a_split_window():
    spec = [("G", True, 0.9), ("W", False, 0.9)] * 2 + [("G", True, 0.9)] * 12
    out = esc(_rec(spec), ORDER, window=4, max_k=8)
    assert out.meta["n_drawn"] == 8
    assert cost_profiles(out.calls)["rounds"] == 2


def test_esc_draws_in_window_sized_steps():
    spec = [("G", True, 0.9), ("W", False, 0.9)] * 8
    for w in (2, 4):
        assert esc(_rec(spec), ORDER, window=w, max_k=8).meta["n_drawn"] % w == 0


# --------------------------------------------------------------------------- #
# DeepConf offline
# --------------------------------------------------------------------------- #

def test_deepconf_filtering_can_flip_a_wrong_majority():
    """5 confident-correct against 11 unconfident-wrong: plain voting loses,
    confidence filtering wins. This is the whole claim of the method."""
    spec = [("G", True, 0.95)] * 5 + [("W", False, 0.20)] * 11
    rec = _rec(spec)
    assert not self_consistency(rec, 16, ORDER).correct
    assert deepconf_offline(rec, ORDER, k=16, eta=0.3).correct


def test_deepconf_pays_for_every_trace_it_discards():
    """Offline filtering buys accuracy, never compute."""
    rec = _rec([("G", True, 0.9)] * 16)
    out = deepconf_offline(rec, ORDER, k=8, eta=0.25)
    assert out.meta["n_kept"] == 2 and out.meta["n_drawn"] == 8
    assert cost_profiles(out.calls) == cost_profiles(self_consistency(rec, 8, ORDER).calls)


def test_deepconf_is_one_parallel_round():
    rec = _rec([("G", True, 0.9)] * 16)
    assert cost_profiles(deepconf_offline(rec, ORDER, k=8).calls)["rounds"] == 1


def test_deepconf_eta_one_unweighted_is_plain_self_consistency():
    spec = [("G", True, 0.9)] * 5 + [("W", False, 0.8)] * 3
    rec = _rec(spec)
    a = deepconf_offline(rec, ORDER, k=8, eta=1.0, weighted=False)
    b = self_consistency(rec, 8, ORDER)
    assert a.answer == b.answer and a.correct == b.correct


def test_deepconf_weighting_uses_confidence():
    """4 wrong at low confidence vs 3 correct at high: weighting flips it."""
    spec = [("W", False, 0.30)] * 4 + [("G", True, 0.99)] * 3
    rec = _rec(spec)
    assert not deepconf_offline(rec, ORDER, k=7, eta=1.0, weighted=False).correct
    assert deepconf_offline(rec, ORDER, k=7, eta=1.0, weighted=True).correct


def test_trace_confidence_handles_a_pool_without_confidence():
    rec = _rec([("G", True, None)] * 4)
    assert trace_confidence(rec, 0) == 0.0
    assert deepconf_offline(rec, ORDER, k=4).served


# --------------------------------------------------------------------------- #
# all three, on a synthetic pool
# --------------------------------------------------------------------------- #

def test_every_baseline_always_answers_and_costs_no_more_than_full_voting():
    """They trade cost at coverage 1.0 -- they cannot abstain, which is why
    they never appear in the risk-certification table."""
    small, _ = make_pools(120, seed=31)
    full = adaptive = 0.0
    for rec in small.values():
        sc = self_consistency(rec, 8, ORDER)
        full += cost_profiles(sc.calls)["tokens"]
        for fn in (lambda r: adaptive_consistency(r, ORDER, 0.95, 8),
                   lambda r: esc(r, ORDER, 4, 8),
                   lambda r: deepconf_offline(r, ORDER, 8, 0.5)):
            out = fn(rec)
            assert out.served and out.answer is not None
        adaptive += cost_profiles(adaptive_consistency(rec, ORDER, 0.95, 8).calls)["tokens"]
    assert adaptive < full


def test_baselines_accept_a_short_pool():
    rec = _rec([("G", True, 0.9)] * 2)
    for out in (adaptive_consistency(rec, ORDER, 0.95, 8),
                esc(rec, ORDER, 4, 8),
                deepconf_offline(rec, ORDER, 8, 0.5)):
        assert out.meta["n_drawn"] <= 2


def test_prefill_sharing_is_an_explicit_convention_not_a_hidden_win():
    """The baselines get the generous convention (prompt billed once across
    sequential draws). Flipping it must change the number, so the paper can
    report which one it used."""
    rec = _rec([("G", True, 0.9), ("W", False, 0.9)] * 8, decode=50)
    shared = adaptive_consistency(rec, ORDER, 0.99, 8, share_prefill=True)
    repaid = adaptive_consistency(rec, ORDER, 0.99, 8, share_prefill=False)
    assert shared.meta["n_drawn"] == repaid.meta["n_drawn"]
    t_shared = cost_profiles(shared.calls)["tokens"]
    t_repaid = cost_profiles(repaid.calls)["tokens"]
    assert t_repaid - t_shared == rec.prompt_tokens * (shared.meta["n_drawn"] - 1)
    assert cost_profiles(shared.calls)["rounds"] == cost_profiles(repaid.calls)["rounds"]


def test_shared_prefill_never_beats_full_voting_on_tokens():
    """With the prompt billed once, drawing fewer samples can only cost less."""
    small, _ = make_pools(80, seed=32)
    for rec in small.values():
        ac = cost_profiles(adaptive_consistency(rec, ORDER, 0.95, 8).calls)["tokens"]
        sc = cost_profiles(self_consistency(rec, 8, ORDER).calls)["tokens"]
        assert ac <= sc
