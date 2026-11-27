"""Candidates of amendment v3-11 (design: DECISION_v311_before_run.md): fixed-n mode of the relative audit.
Reuses the validated pieces of v3-9 / v3-10 (betting_rows, xc_p, H1)."""
from __future__ import annotations
import numpy as np
import v39_methods as V39
from v39_methods import betting_rows, xc_p, _cheapest, EPS, DELTA, BM

# name -> (GO quality test, GO cost test); every candidate keeps H1's NO-GO legs and the per-policy cost range P
CANDS = {"F3": (("wsr", 0.9), ("agrapa", 0.9)), "F2": (("wsr", 0.9), ("wsr", 0.9)), "F1": (("wsr", 0.5), ("wsr", 0.9))}
ALL = ("V0", "H1") + tuple(CANDS)


def wsr_rows(X: np.ndarray, lo, hi: float, cap: float, alpha: float) -> np.ndarray:
    """Fixed-n betting p-values (H0: E[x] <= 0), rows of X (N, n). lambda_t = min(cap/|lo|, sqrt(2 log(1/alpha) / (n v_{t-1}))),
    v_{t-1} = ((hi - lo)^2/4 + sum_{i<t} (x_i - xbar_{t-1})^2) / t; p = min(1, 1 / max_t K_t). lo scalar or per row."""
    N, n = X.shape
    lo = np.broadcast_to(np.asarray(lo, float), (N,)).copy()
    if n < 2 or hi <= 0:
        return np.ones(N)
    bad = lo >= 0
    lo_s = np.where(bad, -1.0, lo)
    lmax = cap / (-lo_s)
    prior = (hi - lo_s) ** 2 / 4.0
    c = 2.0 * np.log(1.0 / alpha)
    logk = np.zeros(N); best = np.zeros(N); s1 = np.zeros(N); s2 = np.zeros(N)
    for m in range(n):
        x = X[:, m]
        if m:
            mu = s1 / m
            v = (prior + s2 - m * mu * mu) / (m + 1)
        else:
            v = prior
        lam = np.minimum(lmax, np.sqrt(c / (n * v)))
        logk = logk + np.log1p(lam * x)
        best = np.maximum(best, logk)
        s1 = s1 + x; s2 = s2 + x * x
    p = np.minimum(1.0, np.exp(-best))
    return np.where(bad, 1.0, p)


def run_cands(D, d, mc, lo_pol, lo_glob, hi, names=ALL, eps=EPS):
    Rr, P, n = D.shape
    D2 = D.reshape(Rr * P, n); d2 = d.reshape(Rr * P, n)
    mD = D.mean(-1)
    gate = mD <= eps
    lo_rows = np.tile(np.asarray(lo_pol, float), Rr)
    a = DELTA / P
    rg_b = betting_rows(eps - D2, eps - 1, eps + 1).reshape(Rr, P)
    rn_b = betting_rows(D2 - eps, -1 - eps, 1 - eps).reshape(Rr, P)
    rg_x, rn_x = xc_p(D, eps)
    rn_h = np.minimum(1.0, 2 * np.minimum(rn_b, rn_x))                    # H1 NO-GO quality (kept by all F*)
    sn = betting_rows(-d2, -hi, -lo_glob).reshape(Rr, P)                # NO-GO cost leg unchanged since v3-8
    cache = {}

    def q(kind, cap):
        k = ("q", kind, cap)
        if k not in cache:
            cache[k] = wsr_rows(eps - D2, eps - 1, eps + 1, cap, a).reshape(Rr, P)
        return cache[k]

    def s(kind, cap, perpol=True, glob_cap=None):
        k = ("s", kind, cap, perpol)
        if k not in cache:
            lo_ = lo_rows if perpol else lo_glob
            p = betting_rows(d2, lo_, hi, cap, False) if kind == "agrapa" else wsr_rows(d2, lo_, hi, cap, a)
            cache[k] = np.where(gate, p.reshape(Rr, P), 1.0)
        return cache[k]

    out = {}
    for v in names:
        if v == "V0":
            rg, rn, s_go = rg_b, rn_b, s("agrapa", 0.5, False)
        elif v == "H1":
            rg, rn, s_go = np.minimum(1.0, 2 * np.minimum(rg_b, rg_x)), rn_h, s("agrapa", 0.9)
        else:
            (qk, qc), (sk, sc) = CANDS[v]
            rg, rn, s_go = q(qk, qc), rn_h, s(sk, sc)
        p = (rg <= a) & (s_go <= a)
        nogo = np.all((rn <= DELTA / 2) | (sn <= DELTA / 2), axis=1)
        go = p.any(axis=1)
        verdict = np.where(go & nogo, "INCONCLUSIVE", np.where(go, "GO", np.where(nogo, "NO-GO", "INCONCLUSIVE")))
        out[v] = (verdict, np.where(verdict == "GO", _cheapest(p, mc), -1))
    return out
