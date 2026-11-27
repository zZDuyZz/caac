"""Vectorised variants of the relative-claim audit for amendment v3-9 (design: DECISION_v39_before_run.md rev. 1).
Batches: D (R, P, n) in {-1, 0, 1}, d (R, P, n), columns in the test order; lo_pol (P,) per-policy lower range of d (GO cost leg);
lo_glob, hi: the global structural range. Variants: name -> (risk test, per-policy range, cap, mixture)."""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import beta, binom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bl"))
import bl_methods as BM                                  # noqa: E402  (sets up the repo path, constants)
from bl_methods import R                                 # noqa: E402

EPS, S, DELTA = BM.EPS, BM.S, BM.DELTA
GAMMA = 1e-4
MIXJ = 5

# name -> (risk, perpol, cap, mix)
VARIANTS = {
    "V0": ("bet", False, 0.5, False),
    "V1": ("xc", False, 0.5, False),
    "V2": ("xc", True, 0.5, False),
    "V3": ("xc", True, 0.9, False),
    "V4": ("xc", True, 0.9, True),
    "Ponly": ("bet", True, 0.5, False),
    "Conly": ("bet", False, 0.9, False),
    "Monly": ("bet", False, 0.5, True),
}


def betting_rows(X: np.ndarray, lo, hi: float, cap: float = 0.5, mix: bool = False) -> np.ndarray:
    """Row-wise p-values (H0: E[x] <= 0). lo scalar or per-row array. cap = 0.5, mix = False is arithmetic-identical to
    sim_rel.betting_p_rows / audit._betting_p."""
    N, n = X.shape
    lo = np.broadcast_to(np.asarray(lo, float), (N,)).copy()
    bad = (n < 2) | (lo >= 0) | (hi <= 0)
    if n < 2 or hi <= 0:
        return np.ones(N)
    lmax = cap / (-np.where(lo < 0, lo, -1.0))
    logk = np.zeros(N); best = np.zeros(N); s1 = np.zeros(N); s2 = np.zeros(N)
    if mix:
        lj = lmax[:, None] * (2.0 ** -np.arange(MIXJ))[None, :]
        logf = np.zeros((N, MIXJ))
    for m in range(n):
        x = X[:, m]
        if m:
            mu = s1 / m
            var = s2 / m - mu * mu
            lam = np.where(mu <= 0, 0.0, np.minimum(lmax, mu / (var + mu * mu + 1e-12)))
        else:
            lam = np.zeros(N)
        logk = logk + np.log1p(lam * x)
        if mix:
            logf = logf + np.log1p(lj * x[:, None])
            terms = np.concatenate([(np.log(0.5) + logk)[:, None], np.log(0.1) + logf], axis=1)
            mx = terms.max(axis=1)
            cur = mx + np.log(np.exp(terms - mx[:, None]).sum(axis=1))
        else:
            cur = logk
        best = np.maximum(best, cur)
        s1 = s1 + x; s2 = s2 + x * x
    p = np.minimum(1.0, np.exp(-best))
    return np.where(bad, 1.0, p)


def xc_p(D: np.ndarray, eps: float = EPS, gamma: float = GAMMA):
    """R1 p-values (GO: small H; NO-GO: large H) for D (..., n)."""
    n = D.shape[-1]
    H = (D > 0).sum(-1).astype(float); Bn = (D < 0).sum(-1).astype(float)
    m = H + Bn
    th_u = np.where(m >= n, 1.0, beta.ppf(1 - gamma, m + 1, np.maximum(n - m, 1)))
    th_l = np.where(m <= 0, 0.0, beta.ppf(gamma, np.maximum(m, 1), n - m + 1))
    pi_min = 0.5 + eps / (2.0 * np.maximum(th_u, 1e-300))
    p_go = np.where(th_u < eps, gamma, np.minimum(1.0, binom.cdf(H, m, np.minimum(pi_min, 1.0)) + gamma))
    pi_max = np.where(th_l <= eps, 1.0, np.minimum(1.0, 0.5 + eps / (2.0 * np.maximum(th_l, 1e-300))))
    p_ng = np.minimum(1.0, binom.sf(H - 1, m, pi_max) + gamma)
    return p_go, p_ng


def _cheapest(passes, mc):
    m = np.where(passes, mc, np.inf)
    ch = m.argmin(axis=1)
    return np.where(passes.any(axis=1), ch, -1)


def run_variants(D, d, mc, lo_pol, lo_glob, hi, names=tuple(VARIANTS), eps=EPS):
    """-> {variant: (verdict (R,), chosen (R,))}. Shared pieces are computed once."""
    Rr, P, n = D.shape
    D2 = D.reshape(Rr * P, n); d2 = d.reshape(Rr * P, n)
    mD = D.mean(-1)
    lo_rows = np.tile(np.asarray(lo_pol, float), Rr)
    need = [VARIANTS[v] for v in names]
    cache = {}

    def sg(perpol, cap, mix):
        k = (perpol, cap, mix)
        if k not in cache:
            p = betting_rows(d2, lo_rows if perpol else lo_glob, hi, cap, mix).reshape(Rr, P)
            cache[k] = np.where(mD <= eps, p, 1.0)
        return cache[k]

    rg_bet = rn_bet = rg_xc = rn_xc = None
    if any(v[0] == "bet" for v in need):
        rg_bet = betting_rows(eps - D2, eps - 1, eps + 1).reshape(Rr, P)
        rn_bet = betting_rows(D2 - eps, -1 - eps, 1 - eps).reshape(Rr, P)
    if any(v[0] == "xc" for v in need):
        rg_xc, rn_xc = xc_p(D, eps)
    sn = betting_rows(-d2, -hi, -lo_glob).reshape(Rr, P)         # NO-GO cost leg: unchanged
    out = {}
    for v in names:
        risk, perpol, cap, mix = VARIANTS[v]
        rg, rn = (rg_bet, rn_bet) if risk == "bet" else (rg_xc, rn_xc)
        p = (rg <= DELTA / P) & (sg(perpol, cap, mix) <= DELTA / P)
        nogo = np.all((rn <= DELTA / 2) | (sn <= DELTA / 2), axis=1)
        go = p.any(axis=1)
        verdict = np.where(go & nogo, "INCONCLUSIVE", np.where(go, "GO", np.where(nogo, "NO-GO", "INCONCLUSIVE")))
        out[v] = (verdict, np.where(verdict == "GO", _cheapest(p, mc), -1))
    return out
