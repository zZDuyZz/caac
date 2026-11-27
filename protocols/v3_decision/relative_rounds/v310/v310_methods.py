"""Candidates of amendment v3-10 (design: DECISION_v310_before_run.md). Reuses the validated pieces of v39_methods."""
from __future__ import annotations
import numpy as np
import v39_methods as V39
from v39_methods import betting_rows, xc_p, _cheapest, EPS, DELTA, BM

# name -> (quality test, cap of the cost-leg bets); every candidate uses the per-policy cost range P
CANDS = {"B1": ("bet", 0.9), "B2": ("bet", 0.99), "X1": ("xc", 0.9), "X2": ("xc", 0.99), "H1": ("hyb", 0.9), "H2": ("hyb", 0.99)}
ALL = ("V0",) + tuple(CANDS)


def run_cands(D, d, mc, lo_pol, lo_glob, hi, names=ALL, eps=EPS):
    Rr, P, n = D.shape
    D2 = D.reshape(Rr * P, n); d2 = d.reshape(Rr * P, n)
    mD = D.mean(-1)
    lo_rows = np.tile(np.asarray(lo_pol, float), Rr)
    rg_b = betting_rows(eps - D2, eps - 1, eps + 1).reshape(Rr, P)
    rn_b = betting_rows(D2 - eps, -1 - eps, 1 - eps).reshape(Rr, P)
    need_x = any(v != "V0" and CANDS[v][0] in ("xc", "hyb") for v in names)
    if need_x:
        rg_x, rn_x = xc_p(D, eps)
    sn = betting_rows(-d2, -hi, -lo_glob).reshape(Rr, P)          # NO-GO cost leg unchanged
    cache = {}

    def sg(key, perpol, cap):
        if key not in cache:
            p = betting_rows(d2, lo_rows if perpol else lo_glob, hi, cap, False).reshape(Rr, P)
            cache[key] = np.where(mD <= eps, p, 1.0)
        return cache[key]

    out = {}
    for v in names:
        if v == "V0":
            rg, rn, s_go = rg_b, rn_b, sg(("V0",), False, 0.5)
        else:
            q, cap = CANDS[v]
            s_go = sg(("P", cap), True, cap)
            if q == "bet":
                rg, rn = rg_b, rn_b
            elif q == "xc":
                rg, rn = rg_x, rn_x
            else:
                rg, rn = np.minimum(1.0, 2 * np.minimum(rg_b, rg_x)), np.minimum(1.0, 2 * np.minimum(rn_b, rn_x))
        p = (rg <= DELTA / P) & (s_go <= DELTA / P)
        nogo = np.all((rn <= DELTA / 2) | (sn <= DELTA / 2), axis=1)
        go = p.any(axis=1)
        verdict = np.where(go & nogo, "INCONCLUSIVE", np.where(go, "GO", np.where(nogo, "NO-GO", "INCONCLUSIVE")))
        out[v] = (verdict, np.where(verdict == "GO", _cheapest(p, mc), -1))
    return out
