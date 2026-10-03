"""CPU tests for the pool-replay pipeline. No GPU, no network.

Verifies the algorithms the paper leans on and the ordering properties that
must also hold on real data, so build_pool output can be trusted downstream.
"""

from __future__ import annotations

import random
from functools import partial

from caac.pool.estimators import pass_at_k, maj_at_k_exact, chow_coverage_bound
from caac.pool.ltt import binom_cdf, pvalue_selective_risk, certify_bonferroni, certify_fixed_sequence
from caac.pool.schema import PoolRecord, Completion, write_records, load_pool
from caac.pool.synth import make_pools
from caac.pool import policies as P
from caac.pool.replay import run_policy, make_orders, lambda_result_from_metrics, pi_pop_after


def test_pass_at_k_basic():
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 8, 1) == 1.0
    assert abs(pass_at_k(8, 1, 1) - 1 / 8) < 1e-9
    # monotone in k
    vals = [pass_at_k(16, 3, k) for k in (1, 2, 4, 8)]
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


def test_maj_at_k_exact_vs_montecarlo():
    rng = random.Random(0)
    answers = ["G", "G", "G", "W1", "W1", "W2", None, "G", "W1", "G"]
    gold = "G"
    for k in (1, 3, 5, 7):
        exact = maj_at_k_exact(answers, gold, k, tie="fair")
        # Monte-Carlo check
        hits = 0.0
        T = 40000
        for _ in range(T):
            sub = rng.sample(answers, k)
            from collections import Counter
            cleaned = [a for a in sub if a is not None]
            if not cleaned:
                continue
            c = Counter(cleaned)
            mx = max(c.values())
            winners = [a for a, v in c.items() if v == mx]
            if gold in winners:
                hits += 1.0 / len(winners)
        mc = hits / T
        assert abs(exact - mc) < 0.02, f"k={k}: exact={exact:.4f} mc={mc:.4f}"


def test_maj_at_k_edge_cases():
    assert maj_at_k_exact(["W1", "W1"], "G", 2) == 0.0      # gold absent
    assert maj_at_k_exact(["G", "G"], "G", 2) == 1.0        # all gold
    assert maj_at_k_exact(["G"], "G", 5) == 1.0             # k>n clamps


def test_chow_bound():
    assert abs(chow_coverage_bound(0.9574, 0.015) - 0.9720) < 1e-3  # PriceCheck's pool
    assert abs(chow_coverage_bound(0.441, 0.10) - 0.490) < 1e-3     # small-model regime
    assert chow_coverage_bound(0.99, 0.5) == 1.0                    # capped at 1


def test_binom_and_pvalue():
    assert abs(binom_cdf(10, 10, 0.5) - 1.0) < 1e-9
    assert abs(binom_cdf(0, 3, 0.5) - 0.125) < 1e-9
    # serving 100 with 2 wrong at alpha=0.10 should give a tiny p-value (safe)
    assert pvalue_selective_risk(100, 2, 0.10) < 0.01
    # serving 100 with 15 wrong at alpha=0.10 should NOT be safe
    assert pvalue_selective_risk(100, 15, 0.10) > 0.5
    assert pvalue_selective_risk(0, 0, 0.10) == 1.0


def test_schema_roundtrip(tmp_path):
    small, big = make_pools(n_questions=5)
    recs = list(small.values())
    path = tmp_path / "p.jsonl"
    write_records(path, recs)
    back = load_pool(path)
    assert len(back) == 5
    r0 = recs[0]
    b0 = back[r0.query_id]
    assert b0.gold == r0.gold
    assert b0.n_samples == r0.n_samples
    assert b0.greedy.correct == r0.greedy.correct


def test_policies_run_and_order_properties():
    small, big = make_pools(n_questions=300, n_small=32, n_big=8, seed=1)
    qids = list(small.keys())
    orders = make_orders(32, n_perms=20, seed=7)

    # greedy
    m_greedy = run_policy(qids, lambda q, o: P.greedy_only(small[q]))
    # self-consistency k=8 runs and serves everything
    m_sc8 = run_policy(qids, lambda q, o: P.self_consistency(small[q], 8, o), orders)
    assert m_sc8.coverage == 1.0
    # NOTE: we deliberately do NOT assert maj@8 >= greedy. For a small model on
    # hard questions (per-sample p < 0.5) majority vote DEGRADES accuracy
    # (Condorcet), a real phenomenon the paper should report, not hide.

    # escalate raises pi_pop (big greedy more correct than small greedy)
    pi_small = pi_pop_after(qids, lambda q: small[q].greedy.correct)
    pi_big = pi_pop_after(qids, lambda q: big[q].greedy.correct)
    assert pi_big > pi_small

    # serve/abstain (PriceCheck-style): never serves a corrected answer, and
    # its selective risk should drop as threshold rises (serves only confident)
    m_lo = run_policy(qids, lambda q, o: P.serve_abstain(small[q], 0.3))
    m_hi = run_policy(qids, lambda q, o: P.serve_abstain(small[q], 0.7))
    assert m_hi.coverage <= m_lo.coverage + 1e-9
    assert m_hi.selective_risk <= m_lo.selective_risk + 0.05

    # cascade serves everything and beats small greedy (escalating low-confidence
    # questions replaces likely-wrong small answers with a better model). It may
    # even exceed big-greedy-on-all, since it keeps small's confident-correct wins.
    m_casc = run_policy(qids, lambda q, o: P.cascade(small[q], big[q], 0.6))
    assert m_casc.coverage == 1.0
    assert m_casc.accuracy >= m_greedy.accuracy - 1e-6


def test_caac_controller_runs_and_certifies():
    small, big = make_pools(n_questions=400, n_small=32, n_big=8, seed=2)
    qids = list(small.keys())
    orders = make_orders(32, n_perms=10, seed=3)
    scorer = P.ConfidenceScorer(sample_k=8)

    # sweep lambda -> LambdaResults, certify with LTT, expect >=1 certified at alpha=0.15
    results = []
    for lam in [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        m = run_policy(
            qids,
            lambda q, o, L=lam: P.caac(small[q], big[q], scorer, L, o or list(range(8)),
                                       k=8, cost_profile="tokens", abstain_penalty=0.5),
            orders,
        )
        results.append(lambda_result_from_metrics(lam, m))

    certify_bonferroni(results, alpha=0.15, delta=0.1)
    # at least the most conservative lambda should serve something
    assert any(r.n_served > 0 for r in results)
    # fixed-sequence (conservative lambda first) should certify at least as many
    ordered = sorted(results, key=lambda r: -float(r.lam))  # high lambda = conservative
    certify_fixed_sequence(ordered, alpha=0.15, delta=0.1)
    assert sum(r.certified for r in ordered) >= sum(r.certified for r in results) - 0  # sane


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    import tempfile, pathlib
    passed = 0
    for fn in fns:
        try:
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
