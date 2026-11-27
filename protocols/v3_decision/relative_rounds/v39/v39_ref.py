"""Amendment v3-9 candidate tests, plain Python (no numpy/scipy), built on the official audit helpers.
Design: DECISION_v39_before_run.md (rev. 1). Used as the independent reference (X5) and as the source of the code
that goes into audit.py if a variant is adopted."""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CAAC_REPO", os.path.join(HERE, "..", "..", "pchk8"))
sys.path.insert(0, os.path.join(REPO, "src"))
from caac.pool_v1.audit import binom_cdf, _betting_p        # noqa: E402

GAMMA = 1e-4
MIX_LAMBDAS = 5


def _sf(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Bin(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, max(0.0, 1.0 - binom_cdf(k - 1, n, p)))


def cp_upper(m: int, n: int, gamma: float) -> float:
    """Smallest theta with P(Bin(n, theta) <= m) <= gamma (one-sided exact upper bound)."""
    if m >= n:
        return 1.0
    lo, hi = m / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(m, n, mid) <= gamma:
            hi = mid
        else:
            lo = mid
    return hi


def cp_lower(m: int, n: int, gamma: float) -> float:
    """Largest theta with P(Bin(n, theta) >= m) <= gamma (one-sided exact lower bound); 0 if m = 0."""
    if m <= 0:
        return 0.0
    lo, hi = 0.0, m / n
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sf(m, n, mid) <= gamma:
            lo = mid
        else:
            hi = mid
    return lo


def counts(D):
    h = sum(1 for v in D if v > 0)
    b = sum(1 for v in D if v < 0)
    return h, h + b


def p_risk_go_xc(D, eps: float, gamma: float = GAMMA) -> float:
    """H0: E[D] >= eps (reject for small H). See the design file."""
    n = len(D)
    h, m = counts(D)
    th = cp_upper(m, n, gamma)
    if th < eps:
        return min(1.0, gamma)
    pi_min = 0.5 + eps / (2.0 * th)
    return min(1.0, binom_cdf(h, m, pi_min) + gamma)


def p_risk_above_xc(D, eps: float, gamma: float = GAMMA) -> float:
    """H0: E[D] <= eps (reject for large H)."""
    n = len(D)
    h, m = counts(D)
    tl = cp_lower(m, n, gamma)
    pi_max = 1.0 if tl <= eps else 0.5 + eps / (2.0 * tl)
    pi_max = min(1.0, pi_max)
    return min(1.0, _sf(h, m, pi_max) + gamma)


def betting_p_v(x, lo: float, hi: float, order=None, cap: float = 0.5, mix: bool = False) -> float:
    """H0: E[x] <= 0, x in [lo, hi]. cap = bet cap as a fraction of 1/|lo|. mix: average of the plug-in wealth (weight 1/2)
    and five fixed-lambda wealths lambda_j = cap/|lo| * 2^-j (weight 1/10 each); p = 1 / max_t of the mixture wealth.
    cap = 0.5, mix = False reproduces audit._betting_p exactly."""
    n = len(x)
    if n < 2 or lo >= 0 or hi <= 0:
        return 1.0
    if not mix:
        if cap == 0.5:
            return _betting_p(x, lo, hi, order)
    idx = order if order is not None else list(range(n))
    lmax = cap / (-lo)
    lams = [lmax * 2.0 ** (-j) for j in range(MIX_LAMBDAS)]
    logk = 0.0
    logf = [0.0] * MIX_LAMBDAS
    best = 0.0
    s1 = s2 = 0.0
    m = 0
    for i in idx:
        mu = s1 / m if m else 0.0
        var = (s2 / m - mu * mu) if m else 0.0
        lam = 0.0 if mu <= 0 else min(lmax, mu / (var + mu * mu + 1e-12))
        logk += math.log1p(lam * x[i])
        if mix:
            for j, l in enumerate(lams):
                logf[j] += math.log1p(l * x[i])
            terms = [math.log(0.5) + logk] + [math.log(0.1) + v for v in logf]
            mx = max(terms)
            cur = mx + math.log(sum(math.exp(t - mx) for t in terms))
        else:
            cur = logk
        best = max(best, cur)
        s1 += x[i]; s2 += x[i] * x[i]; m += 1
    return min(1.0, math.exp(-best))
