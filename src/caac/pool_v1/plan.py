"""`caac plan`: how many questions to collect before a GO is possible (risk leg).

Given alpha, delta, the risk you expect (or a pilot's) and the share of questions the policy
serves, it answers: how many served answers on the tested half, and how many questions in all,
give power P that the exact binomial risk test passes. Uses the SAME test as the audit:
pass iff P(Bin(n, alpha) <= errors) <= delta.

This is an ESTIMATE under an assumed true risk. It covers the RISK leg only: the cost leg
(saving >= s) can need more questions, mainly when costs have a heavy tail (see the report's
cost_bound_ratio). Nothing here is a guarantee.
"""

from __future__ import annotations

import math

from caac.pool_v1.validate import certification_floor


def _pmf_walk(n: int, p: float):
    """Yield (k, P(Bin(n, p) = k)) for k = 0, 1, ... In log space, so a P(X = 0) that
    underflows (large n, large p) does not turn every later term into 0."""
    if p <= 0.0:
        yield 0, 1.0
        return
    if p >= 1.0:
        for k in range(n):
            yield k, 0.0
        yield n, 1.0
        return
    lp = n * math.log1p(-p)                    # log P(X = 0)
    lr = math.log(p) - math.log1p(-p)
    for k in range(n + 1):
        yield k, math.exp(lp)
        if k < n:
            lp += math.log((n - k) / (k + 1)) + lr


def critical_errors(n: int, alpha: float, delta: float) -> int:
    """Largest k with P(Bin(n, alpha) <= k) <= delta; -1 if even 0 errors does not pass."""
    cum, best = 0.0, -1
    for k, pk in _pmf_walk(n, alpha):
        cum += pk
        if cum > delta:
            break
        best = k
    return best


def power(n: int, risk: float, alpha: float, delta: float) -> float:
    """P(the risk leg passes) with n served answers on the tested half and true risk `risk`."""
    k = critical_errors(n, alpha, delta)
    if k < 0:
        return 0.0
    cum = 0.0
    for j, pj in _pmf_walk(n, risk):
        if j > k:
            break
        cum += pj
    return min(1.0, cum)


def plan(alpha: float, delta: float, risk: float, coverage: float = 1.0,
         powers=(0.5, 0.8, 0.9), n_max: int = 4000) -> dict:
    """Smallest served-answer count after which the power stays >= each target (scan to n_max)."""
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must be in (0, 1)")
    if not 0 <= risk < 1 or not 0 < coverage <= 1:
        raise ValueError("risk must be in [0, 1) and coverage in (0, 1]")
    floor = certification_floor(alpha, delta)
    out = {"alpha": alpha, "delta": delta, "assumed_risk": risk, "assumed_coverage": coverage,
           "floor_served_zero_errors": floor, "n_max_scanned": n_max, "targets": []}
    if risk >= alpha:
        out["note"] = ("assumed risk >= alpha: the claim is false, so no number of questions will "
                       "certify it (power stays at most delta-level)")
        return out
    pw = [power(n, risk, alpha, delta) for n in range(1, n_max + 1)]
    for target in powers:
        n_star = None
        for n in range(n_max, 0, -1):              # last n where power < target, then +1
            if pw[n - 1] < target:
                n_star = n + 1
                break
        else:
            n_star = 1
        if n_star > n_max:
            out["targets"].append({"power": target, "served_needed": None, "questions_needed": None})
            continue
        out["targets"].append({"power": target, "served_needed": n_star,
                               "questions_needed": math.ceil(2 * n_star / coverage)})
    out["note"] = ("RISK leg only, under the assumed risk; the cost leg may need more "
                   "(heavy-tailed costs). 'questions_needed' = 2 x served / coverage "
                   "(half the questions choose the policy, half test it).")
    return out
