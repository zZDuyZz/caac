"""Conformance suite: synthetic pools with KNOWN truth, run through the whole pipeline
(rows -> replay -> audit -> report). A tool that claims "wrong with probability <= delta"
must show it here, including at the boundary and with heavy-tailed costs.

Truth is known because the candidate set is the single policy k=1, no cut, escalate,
and no generation is truncated: then risk = 1 - P(small correct) and
saving = 1 - E[small tokens] / (lambda * E[big tokens]) exactly (uniform token counts).

  scenario         truth                         wrong verdict counted
  risk_high        risk .13 > alpha, saving .50  GO
  saving_low       risk .03, saving .15 < s      GO
  boundary         risk = alpha, saving = s      NO-GO   (claim true at the boundary)
  heavy_tail       risk .03, saving ~.10 < s     GO      (lognormal tokens, clipped at cap)
  clear_go         risk .02, saving .50          (power, not an error)

Run: python -m caac.pool_v1.conformance [reps]   (writes a table to stdout)
"""

from __future__ import annotations

import math
import random
import sys
import time

from caac.pool_v1.config import config_from_dict
from caac.pool_v1.replay import replay
from caac.pool_v1.audit import audit
from caac.pool_v1.schema import Meta, Row

S, B = "syn/small", "syn/big"
CAP = 4000

SCENARIOS = {
    #            p_small_correct, small tokens, big tokens, lambda, wrong verdict
    "risk_high": (0.87, (50, 150), (50, 150), 2.0, "GO"),
    "saving_low": (0.97, (70, 270), (50, 150), 2.0, "GO"),
    "boundary": (0.90, (60, 260), (50, 150), 2.0, "NO-GO"),
    "heavy_tail": (0.97, "lognormal:180", "lognormal:100", 2.0, "GO"),
    "clear_go": (0.98, (50, 150), (50, 150), 2.0, None),
}


def _tok(rng, spec):
    if isinstance(spec, tuple):
        return rng.randint(*spec)
    mean = float(spec.split(":")[1])
    return max(1, min(CAP, int(rng.lognormvariate(math.log(mean) - 0.5, 1.0))))


def make_pool(scenario: str, n: int, seed: int):
    p, st, bt, lam, _ = SCENARIOS[scenario]
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        q = f"c{i:06d}"
        rows.append(Row(q, S, 0, rng.random() < p, out_tokens=_tok(rng, st), in_tokens=0, finish="stop", is_greedy=True))
        rows.append(Row(q, B, 0, rng.random() < 0.97, out_tokens=_tok(rng, bt), in_tokens=0, finish="stop", is_greedy=True))
    meta = Meta(source={"name": "conformance"}, grader={"kind": "other", "name": "synthetic"},
                answer_equivalence="none", caps={S: CAP, B: CAP})
    cfg = config_from_dict({"small": S, "big": B, "cost": {"lambda": lam},
                            "policies": {"k": [1], "cut": [None], "on_unsure": ["escalate"]}})
    return rows, meta, cfg


def verdict_rate(scenario: str, n: int, reps: int, cost_test: str = "betting", seed0: int = 0) -> dict:
    counts = {"GO": 0, "NO-GO": 0, "INCONCLUSIVE": 0}
    for r in range(reps):
        rows, meta, cfg = make_pool(scenario, n, seed0 + 7919 * r + n)
        rr = replay(rows, meta, cfg)
        res = audit(rr.audit_input, alpha=cfg.alpha, delta=cfg.delta, s=cfg.s, salt=cfg.salt, cost_test=cost_test)
        counts[res.verdict] += 1
    return {k: v / reps for k, v in counts.items()}


def run_suite(reps: int = 200, ns=(250, 1000, 4000), cost_tests=("betting", "bernstein")) -> list[dict]:
    out = []
    for ct in cost_tests:
        for sc, spec in SCENARIOS.items():
            for n in ns:
                rates = verdict_rate(sc, n, reps, ct)
                wrong = spec[4]
                out.append({"cost_test": ct, "scenario": sc, "n": n, "reps": reps, **rates,
                            "wrong_verdict": wrong, "wrong_rate": rates[wrong] if wrong else None})
    return out


def overhead_us_per_question(n: int = 2000, k: int = 8, repeat: int = 3) -> dict:
    """C18: CPU cost of the controller itself (replay of the candidate set + audit), per question."""
    rng = random.Random(0)
    rows = []
    for i in range(n):
        q = f"o{i:06d}"
        for d in range(k):
            ok = rng.random() < 0.8
            rows.append(Row(q, S, d, ok, answer_key=str(rng.randint(0, 3)), out_tokens=rng.randint(50, 900),
                            in_tokens=50, finish="stop"))
        rows.append(Row(q, B, 0, True, answer_key="0", out_tokens=rng.randint(50, 900), in_tokens=50, finish="stop"))
    meta = Meta(source={"name": "overhead"}, grader={"kind": "other", "name": "synthetic"},
                answer_equivalence="exact", caps={S: 1000, B: 1000}, answer_keys=True)
    cfg = config_from_dict({"small": S, "big": B, "cost": {"lambda": 4.0},
                            "policies": {"k": [x for x in (1, 2, 4, 8) if x <= k], "tau": [0.5, 0.75, 1.0],
                                         "cut": [None, 250, 500]}})
    best_r = best_a = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        rr = replay(rows, meta, cfg)
        t1 = time.perf_counter()
        audit(rr.audit_input, cost_test="betting")
        t2 = time.perf_counter()
        best_r, best_a = min(best_r, t1 - t0), min(best_a, t2 - t1)
    npol = len(rr.policies)
    return {"questions": n, "draws_per_question": k, "policies": npol,
            "replay_us_per_question_all_policies": round(1e6 * best_r / n, 2),
            "replay_us_per_question_per_policy": round(1e6 * best_r / n / npol, 3),
            "audit_us_per_question": round(1e6 * best_a / n, 2)}


def _main(argv):
    reps = int(argv[1]) if len(argv) > 1 else 200
    print(f"CAAC conformance suite (SIMULATION, known truth), alpha=delta=0.10, s=0.20, reps={reps}")
    print(f"{'test':<10}{'scenario':<12}{'n':>6}{'GO':>7}{'NO-GO':>7}{'INC':>7}  wrong<=0.10?")
    for r in run_suite(reps):
        w = "" if r["wrong_verdict"] is None else (
            f"{r['wrong_verdict']} {r['wrong_rate']:.3f} {'ok' if r['wrong_rate'] <= 0.10 else 'FAIL'}")
        print(f"{r['cost_test']:<10}{r['scenario']:<12}{r['n']:>6}{r['GO']:>7.3f}{r['NO-GO']:>7.3f}"
              f"{r['INCONCLUSIVE']:>7.3f}  {w}", flush=True)
    print("overhead:", overhead_us_per_question())


if __name__ == "__main__":
    _main(sys.argv)
