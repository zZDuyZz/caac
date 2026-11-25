"""Conformance suite, kept small for CI (the full table: python -m caac.pool_v1.conformance 300).
Monte-Carlo with few repetitions: the bound checked is loose (<= 0.25 at 40 reps) so the test
is not flaky; the full run checks <= delta."""

from caac.pool_v1.conformance import SCENARIOS, make_pool, overhead_us_per_question, verdict_rate
from caac.pool_v1.replay import replay


def test_truth_of_the_scenarios_is_what_the_docstring_says():
    rows, meta, cfg = make_pool("boundary", 40_000, seed=1)
    rr = replay(rows, meta, cfg)
    p = rr.audit_input.policies[0]
    risk = sum(p.error) / sum(p.served)
    saving = 1 - sum(p.cost) / sum(rr.audit_input.big_cost)
    assert abs(risk - 0.10) < 0.005 and abs(saving - 0.20) < 0.01
    rows, meta, cfg = make_pool("heavy_tail", 40_000, seed=2)
    rr = replay(rows, meta, cfg)
    saving = 1 - sum(rr.audit_input.policies[0].cost) / sum(rr.audit_input.big_cost)
    assert saving < 0.15                                   # clearly below s = 0.20


def test_wrong_verdicts_are_rare_small_run():
    for sc, spec in SCENARIOS.items():
        wrong = spec[4]
        if wrong is None:
            continue
        r = verdict_rate(sc, 1000, 40, "betting")
        assert r[wrong] <= 0.25, (sc, r)


def test_clear_go_has_power():
    assert verdict_rate("clear_go", 2000, 20, "betting")["GO"] >= 0.8


def test_overhead_is_measured():
    o = overhead_us_per_question(n=200, k=4, repeat=1)
    assert o["policies"] > 1 and o["replay_us_per_question_all_policies"] > 0
