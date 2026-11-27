"""Plain-Python reference for the v3-11 fixed-n betting p-value (independent of numpy)."""
import math


def wsr_p(xs, lo, hi, cap, alpha):
    n = len(xs)
    if n < 2 or hi <= 0 or lo >= 0:
        return 1.0
    lmax = cap / (-lo)
    prior = (hi - lo) ** 2 / 4.0
    c = 2.0 * math.log(1.0 / alpha)
    logk = best = s1 = s2 = 0.0
    for m, x in enumerate(xs):
        if m:
            mu = s1 / m
            v = (prior + s2 - m * mu * mu) / (m + 1)
        else:
            v = prior
        lam = min(lmax, math.sqrt(c / (n * v)))
        logk += math.log1p(lam * x)
        best = max(best, logk)
        s1 += x; s2 += x * x
    return min(1.0, math.exp(-best))
