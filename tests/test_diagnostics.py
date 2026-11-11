import math

from caac.pool.diagnostics import (action_costs, auroc, perfect_stop, recoverable_width,
                                   repetition_score, signal_auroc, truncation_kind)
from caac.pool.synth import make_pools


def _pools(n=120, seed=7):
    return make_pools(n, n_small=16, n_big=4, seed=seed)


def test_action_costs_are_positive_and_relative_to_the_greedy_turn():
    small, big = _pools()
    r = action_costs(sorted(small), small, big, lam=1.5)
    assert r["ratio_to_greedy"]["answer_now"] == 1.0
    assert all(v > 0 for v in r["mean_cost"].values())
    # sampling more can only cost more than sampling less
    m = r["mean_cost"]
    assert m["sample_2"] <= m["sample_4"] <= m["sample_8"] <= m["sample_16"]
    # a dearer big-model token makes escalation dearer
    assert action_costs(sorted(small), small, big, 4.5)["mean_cost"]["escalate"] > m["escalate"]


def test_perfect_stop_partitions_questions_and_never_costs_more_than_trying_everything():
    small, big = _pools()
    r = perfect_stop(sorted(small), small, big, lam=1.5)
    assert math.isclose(r["right_now"] + r["repairable"] + r["unrecoverable"], 1.0, abs_tol=1e-9)
    assert r["greedy_turn_cost"] <= r["perfect_stop_cost"] <= r["try_everything_cost"]
    assert 0.0 <= r["repair_waste_share"] <= 1.0
    assert 0.0 <= r["saving_vs_try_everything"] <= 1.0
    greedy_acc = sum(s.greedy.correct for s in small.values()) / len(small)
    assert r["coverage_ceiling"] >= greedy_acc - 1e-12


def test_perfect_stop_with_no_big_model_cannot_escalate():
    small, _ = _pools()
    r = perfect_stop(sorted(small), small, None, lam=1.0)
    assert r["unrecoverable"] >= 0.0
    assert r["try_everything_cost"] >= r["perfect_stop_cost"]


def test_auroc_basics():
    assert auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert auroc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert math.isnan(auroc([1, 1], [0.1, 0.2]))


def test_signal_auroc_reports_ranges_or_skips_instead_of_guessing():
    small, big = _pools(300)
    r = signal_auroc(sorted(small), small, big, n_orders=5)
    assert r["n_greedy_wrong"] > 0
    for t in r["targets"].values():
        if "skipped" in t:
            continue
        for v in t["signals"].values():
            assert 0.0 <= v["auroc"] <= 1.0
            assert v["ci95"][0] <= v["ci95"][1]


def test_signal_auroc_skips_a_tiny_population():
    small, big = _pools(10)
    r = signal_auroc(sorted(small)[:3], small, big)
    assert "skipped" in r or all("skipped" in t for t in r["targets"].values())


def test_recoverable_width_covers_every_budget_and_price():
    small, big = _pools()
    r = recoverable_width(sorted(small), small, big, lams=(1.0, 4.5), ks=(2, 8))
    assert set(r) == {"k=2", "k=8"}
    for row in r.values():
        assert set(row) == {"1", "4.5"}
        for regions in row.values():
            assert math.isclose(sum(regions.values()), 1.0, abs_tol=1e-6) or sum(regions.values()) == 0.0


def test_repetition_score_separates_loops_from_fresh_text():
    loop = ("so the answer is 42 and then we check it " * 40).strip()
    fresh = " ".join(f"word{i}" for i in range(300))
    assert repetition_score(loop) > 0.8
    assert repetition_score(fresh) == 0.0
    assert repetition_score("too short to judge") == 0.0


def test_truncation_kind_splits_loops_from_long_answers():
    small, _ = _pools(6)
    qs = sorted(small)
    for q in qs:
        small[q].greedy.finish_reason = "stop"
    small[qs[0]].greedy.finish_reason = "length"
    small[qs[1]].greedy.finish_reason = "length"
    texts = {qs[0]: {"greedy_text": ("try again and again and again " * 60)},
             qs[1]: {"greedy_text": " ".join(f"step{i}" for i in range(400))}}
    r = truncation_kind(small, texts)
    assert r["n_truncated_with_text"] == 2
    assert r["loop_share"] == 0.5
    assert r["loop_examples"] == [qs[0]] and r["long_examples"] == [qs[1]]
