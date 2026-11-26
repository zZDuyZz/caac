"""Evidence gap (design v3): the number of questions still needed for a GO, for BOTH legs, planned
with the same tests the audit runs. Planning only: it never changes a verdict.

  risk leg    served answers m_r such that the exact binomial test (level delta/|Pi|) passes with power P
              if the error rate stays at the observed value (caac.pool_v1.plan.power);
  saving leg  questions m_s such that the betting test (same bets as the audit, structural range)
              passes with power P, by bootstrap of the observed d = (1-s)B - C (fixed seed);
  total       max(ceil(m_r / coverage), m_s)   (no split: every question is tested).
Unreachable when the observed rate >= alpha or the mean of d <= 0 (more questions of the same kind
cannot help). The count is rounded UP to two significant digits so the report is byte-stable.
"""
from __future__ import annotations

import math

import numpy as np

from .plan import power as risk_power

_SEED = 20261126


def betting_pass_rate(d, m: int, lo: float, delta: float, B: int = 300, seed: int = _SEED) -> float:
    """P(the betting test passes at level delta on m i.i.d. resamples of d). Same bets as audit._betting_p."""
    rng = np.random.default_rng(seed)
    X = rng.choice(np.asarray(d, float), size=(m, B), replace=True)
    lmax = 0.5 / (-lo)
    s1 = np.zeros(B); s2 = np.zeros(B); logk = np.zeros(B); best = np.zeros(B)
    thr = math.log(1.0 / delta)
    for t in range(m):
        if t:
            mu = s1 / t
            var = s2 / t - mu * mu
            lam = np.where(mu <= 0, 0.0, np.minimum(lmax, mu / (var + mu * mu + 1e-12)))
        else:
            lam = np.zeros(B)
        x = X[t]
        logk += np.log1p(lam * x)
        np.maximum(best, logk, out=best)
        s1 += x; s2 += x * x
    return float(np.mean(best >= thr))


def _search(fn, target, m0, m_max):
    m = m0
    while m <= m_max and fn(m) < target:
        m *= 2
    if m > m_max:
        return None
    lo, hi = m // 2, m
    while hi - lo > max(1, lo // 20):                # 5% resolution
        mid = (lo + hi) // 2
        if fn(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def m_saving(d, lo, delta, target=0.8, m_max=200_000, B=300):
    d = np.asarray(d, float)
    if len(d) < 2 or d.mean() <= 0:
        return None
    return _search(lambda m: betting_pass_rate(d, m, lo, delta, B), target, 16, m_max)


def m_risk(rate, alpha, delta, target=0.8, m_max=200_000):
    if rate is None or rate >= alpha:
        return None
    return _search(lambda m: risk_power(m, rate, alpha, delta), target, 8, m_max)


def round_up_2sig(x: int) -> int:
    if x < 100:
        return int(x)
    k = 10 ** (len(str(int(x))) - 2)
    return int(math.ceil(x / k) * k)


def gap(errors, served, n_tested, d, lo, alpha, delta_each, target=0.8):
    """delta_each = delta/|Pi| (level of each leg); lo = lower end of the structural range of d."""
    rate = errors / served if served else None
    cov = served / n_tested if n_tested else 0.0
    mr = m_risk(rate, alpha, delta_each, target)
    ms = m_saving(d, lo, delta_each, target)
    out = {"rate": rate, "coverage": cov, "mean_d": float(np.mean(d)) if len(d) else None,
           "m_risk_served": mr, "m_saving_questions": ms}
    if mr is None or ms is None or cov <= 0:
        out.update(reachable=False, questions_needed=None,
                   blocking="risk" if mr is None else ("saving" if ms is None else "coverage"))
        return out
    qr, qs = math.ceil(mr / cov), ms
    out.update(reachable=True, questions_needed=round_up_2sig(max(qr, qs)),
               blocking="risk" if qr >= qs else "saving")
    return out


def gap_relative(D, n_tested, d, lo_d, eps, delta_each, target=0.8, test="bet"):
    """Relative claim, both legs, same tests as the audit (planning only).
      risk leg  bet: bootstrap of x = eps - D (range [eps - 1, eps + 1]);
                harm: exact binomial power of the harm count at the observed harm rate;
                mix: the better of the two at level delta_each / 2.
      cost leg  bootstrap of d (structural range), as for the absolute claim.
    Unreachable if mean(D) >= eps (bet), harm rate >= eps (harm), or mean(d) <= 0."""
    D = np.asarray(D, float)
    n = len(D)
    mean_D = float(D.mean()) if n else None
    h = float((D > 0).mean()) if n else None

    def m_bet(level):
        return m_saving(eps - D, eps - 1.0, level, target) if (mean_D is not None and mean_D < eps) else None

    def m_harm(level):
        return m_risk(h, eps, level, target) if (h is not None and eps > 0) else None

    if test == "bet":
        mr = m_bet(delta_each)
    elif test == "harm":
        mr = m_harm(delta_each)
    else:
        c = [m for m in (m_bet(delta_each / 2), m_harm(delta_each / 2)) if m is not None]
        mr = min(c) if c else None
    ms = m_saving(d, lo_d, delta_each, target)
    out = {"delta_hat": mean_D, "harm_rate": h, "eps": eps, "rel_test": test,
           "mean_d": float(np.mean(d)) if len(d) else None, "m_risk_questions": mr, "m_saving_questions": ms}
    if mr is None or ms is None:
        out.update(reachable=False, questions_needed=None, blocking="risk" if mr is None else "saving")
        return out
    out.update(reachable=True, questions_needed=round_up_2sig(max(mr, ms)),
               blocking="risk" if mr >= ms else "saving")
    return out
