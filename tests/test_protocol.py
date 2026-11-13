"""The shared certification protocol, the method families, and the two drivers."""
import json
import sys

import numpy as np
import pytest

from caac.pool import families as F
from caac.pool.ledger import Call
from caac.pool.protocol import (Family, Table, build_table, certify, pick_max_coverage,
                                pick_min_cost, plugin_select, prune, split_cost, summarize)
from caac.pool.policies import PolicyOutcome
from caac.pool.schema import Completion, PoolRecord, write_records
from caac.pool.synth import make_pools

SM, BM = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"


def _table(served, wrong, paths=None):
    served, wrong = np.asarray(served, float), np.asarray(wrong, float)
    q, c = served.shape
    z = np.zeros((q, c))
    return Table("t", [f"q{i}" for i in range(q)], list(range(c)),
                 paths or [list(range(c))], served, wrong, z, z, z, z, [{}] * c)


def _comp(ans, ok, dec=10):
    return Completion(answer=ans, correct=ok, decode_tokens=dec, finish_reason="stop",
                      confidence=0.9)


# --------------------------------------------------------------------------- #
# certification
# --------------------------------------------------------------------------- #

def test_certify_refuses_averaged_counts():
    t = _table([[0.5]] * 40, [[0.0]] * 40)
    with pytest.raises(ValueError):
        certify(t, None, 0.1, 0.1)


def test_fixed_sequence_path_stops_at_first_failure_and_paths_split_delta():
    # config 0: 100 served, 0 wrong (passes); 1: 100 served, 50 wrong (fails); 2 would pass
    served = [[1, 1, 1]] * 100
    wrong = [[0, 1, 0]] * 50 + [[0, 0, 0]] * 50
    ok, _ = certify(_table(served, wrong), None, 0.1, 0.1)
    assert ok.tolist() == [True, False, False]
    ok, _ = certify(_table(served, wrong, paths=[[0], [1], [2]]), None, 0.1, 0.1)
    assert ok.tolist() == [True, False, True]          # Bonferroni: each rule on its own


def test_more_paths_cost_power():
    # 30 served, 0 wrong: passes at delta=0.1 (n_min 22) but not at 0.1/30 (n_min 55)
    served, wrong = [[1] * 30] * 30, [[0] * 30] * 30
    ok1, _ = certify(_table(served, wrong), None, 0.1, 0.1)
    okb, _ = certify(_table(served, wrong, paths=[[j] for j in range(30)]), None, 0.1, 0.1)
    assert ok1.all() and not okb.any()


def test_plugin_takes_calibration_risk_at_face_value():
    t = _table([[1]] * 10, [[1]] + [[0]] * 9)
    assert plugin_select(t, None, 0.1).tolist() == [True]
    assert certify(t, None, 0.1, 0.1)[0].tolist() == [False]


def test_selectors():
    t = _table([[1, 1, 0]] * 50 + [[0, 1, 1]] * 50, [[0, 0, 0]] * 100)
    t.tok_small[:] = np.array([1.0, 5.0, 2.0])
    ok = np.array([True, True, True])
    assert pick_max_coverage(t, None, ok) == 1
    assert pick_min_cost(t, None, ok, lam=1.0, min_coverage=0.5) == 0
    assert pick_min_cost(t, None, ok, lam=1.0, min_coverage=0.9) == 1
    assert pick_min_cost(t, None, ok, lam=1.0, min_coverage=1.1) is None


def test_prune_uses_training_coverage_only():
    fam = Family("f", [0, 1, 2], make=lambda c: None, paths=[[0, 1, 2]])
    train = _table([[0, 1, 1]] * 10 + [[0, 0, 1]] * 90, [[0, 0, 0]] * 100)
    out = prune(fam, train, n_calib=100, alpha=0.1, delta=0.1)
    assert out.configs == [2] and out.paths == [[0]]


def _non_monotone():
    """Config 0 is the 'most conservative' one but makes errors; 1 and 2 serve
    more and are clean. Same pattern on train and on calibration."""
    served = [[1, 1, 1]] * 120 + [[0, 1, 1]] * 30 + [[0, 0, 1]] * 30
    wrong = [[1, 0, 0]] * 12 + [[0, 0, 0]] * 168
    return served, wrong


def test_family_order_stalls_when_the_conservative_end_is_not_the_safest():
    served, wrong = _non_monotone()
    fam = Family("f", [0, 1, 2], make=lambda c: None, paths=[[0, 1, 2]])
    out = prune(fam, _table(served, wrong), n_calib=180, alpha=0.1, delta=0.1, order="family")
    cal = _table(served, wrong, paths=out.paths)
    ok, _ = certify(cal, None, 0.1, 0.1)
    assert not ok.any()          # stops at config 0 and never reaches the clean ones


def test_train_pvalue_order_reaches_the_clean_configurations():
    served, wrong = _non_monotone()
    fam = Family("f", [0, 1, 2], make=lambda c: None, paths=[[0], [1], [2]])
    out = prune(fam, _table(served, wrong), n_calib=180, alpha=0.1, delta=0.1)
    assert len(out.paths) == 1 and out.paths[0][-1] == 0     # one path, the risky config last
    cal = _table(served, wrong, paths=out.paths)
    ok, _ = certify(cal, None, 0.1, 0.1)
    assert ok[1] and ok[2] and not ok[0]


def test_prune_order_reads_only_the_training_table():
    served, wrong = _non_monotone()
    fam = Family("f", [0, 1, 2], make=lambda c: None)
    a = prune(fam, _table(served, wrong), n_calib=180, alpha=0.1, delta=0.1)
    b = prune(fam, _table(served, wrong), n_calib=180, alpha=0.1, delta=0.1)
    assert a.paths == b.paths and a.configs == b.configs
    with pytest.raises(ValueError):
        prune(fam, _table(served, wrong), n_calib=180, alpha=0.1, delta=0.1, order="calib")


# --------------------------------------------------------------------------- #
# costs and tables
# --------------------------------------------------------------------------- #

def test_cost_splits_by_model_and_scales_with_lam():
    calls = [Call(SM, 10, [5, 5], 0), Call(BM, 10, [20], 1)]
    assert split_cost(calls, BM) == (20.0, 30.0)
    assert F.lam_cost(2.0, BM)(calls) == 80.0


def test_order_free_family_is_replayed_once():
    small, big = make_pools(20, seed=3)
    t = build_table(F.small_abstain([0.5]), sorted(small), small, big,
                    orders=[[0, 1], [1, 0], [0, 1]])
    assert np.allclose(t.served, np.round(t.served))    # integers: one replay


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #

def test_pricecheck_only_ever_serves_the_small_greedy_answer():
    small, big = make_pools(80, seed=5, n_big=8)
    fam = F.pricecheck(n_big=8)
    for cfg in fam.configs:
        fn = fam.make(cfg)
        for q in sorted(small):
            out = fn(small[q], big[q], list(range(32)))
            if out.served:
                assert out.answer == small[q].greedy.answer
                assert out.correct == small[q].greedy.correct


def test_race_serves_at_k_agreements_and_stops_at_j_refusals():
    g = _comp("5", True)
    s = PoolRecord("q", "d", SM, "5", 10, g, [_comp("5", True)] * 4)
    b = PoolRecord("q", "d", BM, "5", 10, g, [_comp("5", True), _comp("4", False),
                                              _comp("5", True), _comp("5", True)])
    fn = F.pricecheck(n_big=4).make(("race", 2, 2))
    out = fn(s, b, [0, 1, 2, 3])
    assert out.served and out.meta["draws"] == 3
    assert len({c.round_index for c in out.calls}) == 4    # greedy + one round per draw
    out = F.pricecheck(n_big=4).make(("race", 3, 1))(s, b, [0, 1, 2, 3])
    assert not out.served                                  # first refusal ends it


def test_big_votes_are_compared_with_the_small_answer_label_free():
    s = PoolRecord("q", "d", SM, "x", 10, _comp("1/2", False), [])
    b = PoolRecord("q", "d", BM, "x", 10, _comp("3", True), [_comp("0.5", False), _comp("3", True)])
    assert F._big_agrees(s, b) == [True, False]


def test_staged_adaptive_steps_bill_only_what_was_drawn():
    class AlwaysSample:
        def __call__(self, s, b=None):
            return {"answer_now": 0.0, "sample_more": 0.9, "escalate": 0.0, "abstain": 1.0}

    def stage2(s, idxs, phat):
        return 0.99 if len(idxs) >= 8 else 0.0       # convinced only after 8 draws

    s = PoolRecord("q", "d", SM, "5", 10, _comp("5", True), [_comp("5", True, dec=7)] * 16)
    from caac.pool.policies import caac_staged
    out = caac_staged(s, None, AlwaysSample(), stage2, tau=0.95, order=list(range(16)),
                      actions=("answer_now", "sample_more"), k_steps=(4, 8, 16))
    assert out.served and out.meta["n_drawn"] == 8
    drawn = sum(len(c.decode_tokens) for c in out.calls) - 1     # minus the greedy turn
    assert drawn == 8


# --------------------------------------------------------------------------- #
# drivers, end to end on synthetic pools
# --------------------------------------------------------------------------- #

def _write_parts(tmp_path, n=600, seed=11):
    small, big = make_pools(n, seed=seed, n_small=16, n_big=8)
    q = sorted(small)
    parts = {"train": q[: n // 3], "calib": q[n // 3: 2 * n // 3], "test": q[2 * n // 3:]}
    paths = {}
    for nm, qs in parts.items():
        for side, pool in (("small", small), ("big", big)):
            p = tmp_path / f"{nm}_{side}.jsonl.gz"
            write_records(p, [pool[x] for x in qs])
            paths[f"{nm}_{side}"] = str(p)
    return paths


def test_certify_driver_end_to_end(tmp_path, monkeypatch):
    import certify
    p = _write_parts(tmp_path)
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "certify.py", "--train-small", p["train_small"], "--train-big", p["train_big"],
        "--calib-small", p["calib_small"], "--calib-big", p["calib_big"],
        "--test-small", p["test_small"], "--test-big", p["test_big"],
        "--families", "small_abstain,pricecheck,caac_staged,caac_staged_no_sample",
        "--lams", "1,5", "--coverages", "0.5,0.7", "--perms", "2", "--alpha", "0.15",
        "--json", str(out)])
    certify.main()
    res = json.loads(out.read_text())
    assert set(res["families"]) == {"small_abstain", "pricecheck", "caac_staged",
                                    "caac_staged_no_sample"}
    assert set(res["phase_map"]) == {"1", "5"}
    staged = res["families"]["caac_staged"]["by_lam"]["5"]["max_coverage"]
    assert staged is not None and staged["test"]["coverage"] > 0


def test_certify_driver_rejects_overlapping_splits(tmp_path, monkeypatch):
    import certify
    p = _write_parts(tmp_path, n=90)
    monkeypatch.setattr(sys, "argv", [
        "certify.py", "--train-small", p["train_small"], "--calib-small", p["train_small"],
        "--test-small", p["test_small"], "--families", "small_abstain"])
    with pytest.raises(SystemExit):
        certify.main()


def test_certificate_holds_across_resplits():
    """The guarantee, checked: over many random calibration/test splits the
    certified pick violates alpha on at most about delta of them."""
    import resplit
    small, big = make_pools(1200, seed=21, n_small=16, n_big=8)
    q = sorted(small)
    fam = F.cascade_abstain()
    table = build_table(fam, q, small, big)
    r = resplit.run(table, len(q), alpha=0.15, delta=0.10, splits=200, frac=0.5,
                    rule="max_coverage", lam=5.0, min_cov=0.0, seed=0)
    assert r["certify"]["violation_rate"] <= 0.15
    assert r["certify"]["mean_coverage"] > 0


def test_escalation_is_judged_after_paying_for_it():
    """With a post-escalation gate the big answer is paid for first and served
    only if the gate clears tau; it is billed either way."""
    from caac.pool.policies import caac_staged

    class Pre:
        def __call__(self, s, b=None):
            return {"answer_now": 0.0, "sample_more": 0.0, "escalate": 0.5, "abstain": 1.0}

    s = PoolRecord("q", "d", SM, "5", 10, _comp("5", True), [])
    b = PoolRecord("q", "d", BM, "5", 10, _comp("5", True), [])
    acts = ("answer_now", "escalate")
    yes = caac_staged(s, b, Pre(), None, tau=0.9, order=[], actions=acts,
                      escalate_gate=lambda *_: 0.95, rho=0.4)
    no = caac_staged(s, b, Pre(), None, tau=0.9, order=[], actions=acts,
                     escalate_gate=lambda *_: 0.5, rho=0.4)
    skip = caac_staged(s, b, Pre(), None, tau=0.9, order=[], actions=acts,
                       escalate_gate=lambda *_: 0.95, rho=0.6)
    assert yes.served and yes.action == "escalate"
    assert not no.served and any(c.model == BM for c in no.calls)     # paid, then declined
    assert not skip.served and all(c.model != BM for c in skip.calls)  # not worth trying


def test_escalate_gate_reads_agreement_between_models():
    from caac.pool.estimator import EscalateGate, post_escalate_features
    s = PoolRecord("q", "d", SM, "5", 10, _comp("5", True), [_comp("5", True)] * 8)
    b_same = PoolRecord("q", "d", BM, "5", 10, _comp("5", True), [])
    b_diff = PoolRecord("q", "d", BM, "5", 10, _comp("6", False), [])
    f_same = post_escalate_features(s, b_same, list(range(8)))
    f_diff = post_escalate_features(s, b_diff, list(range(8)))
    assert f_same[3] == 1.0 and f_same[4] == 1.0
    assert f_diff[3] == 0.0 and f_diff[4] == 0.0
    small, big = make_pools(300, seed=9, n_small=16, n_big=8)
    gate = EscalateGate().fit(sorted(small), small, big)
    assert gate.head is not None and 0.0 <= gate(s, b_same, []) <= 1.0
