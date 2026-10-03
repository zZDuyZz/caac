"""End-to-end CPU demo on a synthetic pool: the two headline analyses.

Run: PYTHONPATH=src python demo_offline.py
Shows (1) the Chow-bound story -- serve/abstain-only is capped by pi_pop, while
answer-correcting actions lift it -- and (2) an LTT-certified CAAC frontier.
Numbers are from synthetic data; on the real pool the same code produces the
paper's figures.
"""

from __future__ import annotations

from caac.pool.synth import make_pools
from caac.pool.estimators import chow_coverage_bound
from caac.pool import policies as P
from caac.pool.replay import run_policy, make_orders, lambda_result_from_metrics, pi_pop_after
from caac.pool.ltt import certify_fixed_sequence, select_max_coverage

small, big = make_pools(n_questions=400, n_small=32, n_big=8, seed=11)
qids = list(small)
orders = make_orders(32, n_perms=20, seed=5)

pi_greedy = pi_pop_after(qids, lambda q: small[q].greedy.correct)
pi_maj = pi_pop_after(qids, lambda q: P.self_consistency(small[q], 16, orders[0]).correct)
pi_big = pi_pop_after(qids, lambda q: big[q].greedy.correct)

print("=== Chow bound: max certifiable coverage by candidate-answer quality ===")
print(f"{'action producing the answer':32} {'pi_pop':>7} {'cap@a=.10':>10} {'cap@a=.05':>10}")
for name, pi in [("greedy 1.5B (serve/abstain-only)", pi_greedy),
                 ("+ sample-more maj@16", pi_maj),
                 ("+ escalate to 7B", pi_big)]:
    print(f"{name:32} {pi:7.3f} {chow_coverage_bound(pi,0.10):10.3f} {chow_coverage_bound(pi,0.05):10.3f}")

print("\n=== LTT-certified CAAC frontier (alpha=0.15, delta=0.1, fixed-sequence) ===")
scorer = P.ConfidenceScorer(sample_k=8)
results = []
for lam in [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 0.0]:  # conservative -> liberal
    m = run_policy(qids, lambda q, o, L=lam: P.caac(small[q], big[q], scorer, L,
                   o or list(range(8)), k=8, cost_profile="tokens", abstain_penalty=0.5), orders)
    r = lambda_result_from_metrics(lam, m)
    r.acc = m.accuracy; r.risk = m.selective_risk; r.adist = m.action_dist
    results.append(r)

certify_fixed_sequence(results, alpha=0.15, delta=0.1)
print(f"{'lambda':>8} {'cov':>6} {'risk':>6} {'acc':>6} {'cost':>9} {'cert':>5}  action_dist")
for r in results:
    ad = {k: round(v, 2) for k, v in r.adist.items()}
    print(f"{r.lam:8.0e} {r.coverage:6.2f} {r.risk:6.3f} {r.acc:6.3f} {r.mean_cost:9.0f} "
          f"{'Y' if r.certified else '.':>5}  {ad}")

best = select_max_coverage(results)
print(f"\nSelected (max coverage among certified): lambda={best.lam:.0e}, "
      f"coverage={best.coverage:.2f}, risk={best.risk:.3f}" if best else
      "\n(Nothing certified: the stand-in scorer never abstains, so risk stays "
      "above alpha. Abstention is what buys the certificate -- see the sweep below.)")

print("\n=== PriceCheck-style serve/abstain, LTT-certified (alpha=0.15) ===")
print("(shows LTT certifying, and the Chow cap biting: coverage can't beat pi_pop/(1-a))")
sa = []
for thr in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:  # conservative = high threshold first
    m = run_policy(qids, lambda q, o, T=thr: P.serve_abstain(small[q], T))
    r = lambda_result_from_metrics(thr, m); r.risk = m.selective_risk; r.acc = m.accuracy
    sa.append(r)
sa.sort(key=lambda r: -float(r.lam))  # high threshold (conservative) first
certify_fixed_sequence(sa, alpha=0.15, delta=0.1)
print(f"{'thr':>6} {'cov':>6} {'risk':>6} {'acc':>6} {'cert':>5}")
for r in sa:
    print(f"{r.lam:6.2f} {r.coverage:6.2f} {r.risk:6.3f} {r.acc:6.3f} {'Y' if r.certified else '.':>5}")
bsa = select_max_coverage(sa)
print(f"serve/abstain best certified coverage: {bsa.coverage:.2f} (acc {bsa.acc:.3f})"
      if bsa else "serve/abstain: nothing certified")
print(f"Chow cap at pi_pop={pi_greedy:.3f}, a=0.15: {chow_coverage_bound(pi_greedy,0.15):.3f}")
