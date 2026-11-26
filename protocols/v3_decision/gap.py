"""T1 fix candidate: the 'evidence gap' -- questions needed for a GO, BOTH legs, same tests as the audit.

v2's R4 covers the risk leg only, and with a point estimate (n such that the test passes IF the
observed error rate holds exactly, i.e. about 50% power). On the real secondary map, 34 of 132
INCONCLUSIVE configs pass the risk leg and fail only the saving leg, so v2 reports a 'questions
needed' that is already met while the blocking leg is ignored.

gap(policy) = 2 x max(m_risk / coverage, m_saving)   [x2: half the questions select, half test]
  m_risk   : served answers for the exact binomial test (same as the audit) to pass with power P,
             at the observed error rate  (caac.pool_v1.plan.power)
  m_saving : questions for the betting test (same bets as the audit) to pass with power P,
             by bootstrap of the observed d = (1-s)B - C on the tested half
  unreachable if observed rate >= alpha or mean(d) <= 0 (more questions of the same kind cannot help).
This is a PLANNING estimate (plug-in); it never changes the verdict.
"""
from __future__ import annotations
import math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pchk6", "src"))
from caac.pool_v1.plan import power as risk_power  # noqa: E402


def betting_pass_rate(d: np.ndarray, m: int, lo: float, delta: float, B: int = 300, seed: int = 0) -> float:
    """P(betting test on m i.i.d. resamples of d passes at level delta). Same bets as audit._betting_p."""
    rng = np.random.default_rng(seed)
    X = rng.choice(d, size=(m, B), replace=True)
    lmax = 0.5 / (-lo)
    s1 = np.zeros(B); s2 = np.zeros(B); logk = np.zeros(B); best = np.zeros(B)
    thr = math.log(1.0 / delta)
    for t in range(m):
        if t:
            mu = s1 / t; var = s2 / t - mu * mu
            lam = np.where(mu <= 0, 0.0, np.minimum(lmax, mu / (var + mu * mu + 1e-12)))
        else:
            lam = np.zeros(B)
        x = X[t]
        logk += np.log1p(lam * x)
        np.maximum(best, logk, out=best)
        s1 += x; s2 += x * x
    return float(np.mean(best >= thr))


def m_saving(d, lo, delta, target=0.8, m_max=200_000, B=300):
    d = np.asarray(d, float)
    if len(d) < 2 or d.mean() <= 0:
        return None
    m = 16
    while m <= m_max and betting_pass_rate(d, m, lo, delta, B) < target:
        m *= 2
    if m > m_max:
        return None
    lo_m, hi_m = m // 2, m
    while hi_m - lo_m > max(8, lo_m // 20):          # 5% resolution
        mid = (lo_m + hi_m) // 2
        if betting_pass_rate(d, mid, lo, delta, B) >= target:
            hi_m = mid
        else:
            lo_m = mid
    return hi_m


def m_risk(rate, alpha, delta, target=0.8, m_max=200_000):
    if rate is None or rate >= alpha:
        return None
    m = 8
    while m <= m_max and risk_power(m, rate, alpha, delta) < target:
        m *= 2
    if m > m_max:
        return None
    lo_m, hi_m = m // 2, m
    while hi_m - lo_m > 1:
        mid = (lo_m + hi_m) // 2
        if risk_power(mid, rate, alpha, delta) >= target:
            hi_m = mid
        else:
            lo_m = mid
    return hi_m


def gap(errors, served, n_tested, d, lo, alpha, delta, target=0.8, split_factor=2):
    """Total questions needed for a GO with power `target`, or None (unreachable).
    split_factor=2 for the v2 split design (half selects, half tests); 1 for the no-split design,
    where `delta` is the per-policy level (delta/|Pi|) and `lo` the structural range."""
    rate = errors / served if served else None
    cov = served / n_tested if n_tested else 0
    mr = m_risk(rate, alpha, delta, target)
    ms = m_saving(d, lo, delta, target)
    out = {"rate": rate, "coverage": cov, "mean_d": float(np.mean(d)) if len(d) else None,
           "m_risk_served": mr, "m_saving_questions": ms}
    if mr is None or ms is None or cov <= 0:
        out.update(reachable=False, questions_needed=None,
                   blocking="risk" if mr is None else ("saving" if ms is None else "coverage"))
        return out
    q_r, q_s = math.ceil(mr / cov), ms
    out.update(reachable=True, questions_needed=split_factor * max(q_r, q_s),
               blocking="risk" if q_r >= q_s else "saving")
    return out
