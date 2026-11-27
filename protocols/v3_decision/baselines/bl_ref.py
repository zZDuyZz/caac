"""Independent plain-Python implementation of every rule (check X1). No numpy, no scipy; written separately
from bl_methods.py. Betting uses the OFFICIAL audit._betting_p; the CAAC verdict logic is re-derived here.
One study: D[k][i], d[k][i] (lists), columns in test order, mc[k] mean cost."""
from __future__ import annotations

import math
from statistics import NormalDist

import bl_methods as _bm                     # only for the path setup and constants
from caac.pool_v1.audit import _betting_p

EPS, S, DELTA = _bm.EPS, _bm.S, _bm.DELTA


def _mean(v):
    return sum(v) / len(v)


def _sd(v):
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _pick(ok, mc):
    best = -1
    for k, o in enumerate(ok):
        if o and (best < 0 or mc[k] < mc[best]):
            best = k
    return best


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    tot = 0.0
    for j in range(0, k + 1):
        tot += math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                        + (j * math.log(p) if j else 0.0) + ((n - j) * math.log1p(-p) if n - j else 0.0))
    return min(1.0, tot)


def hb(xbar, n, mu0):
    a = min(xbar, mu0)
    def t(u, v):
        return 0.0 if u == 0 else u * math.log(u / v)
    h = t(a, mu0) + t(1 - a, 1 - mu0)
    return min(1.0, math.exp(-n * h), math.e * binom_cdf(math.ceil(n * xbar - 1e-9), n, mu0))


def decide(D, d, mc, lo, hi, eps=EPS):
    P, n = len(D), len(D[0])
    out = {}
    mD = [_mean(r) for r in D]; md = [_mean(r) for r in d]
    out["naive"] = _pick([mD[k] <= eps and md[k] >= 0 for k in range(P)], mc)
    for name, lvl in (("clt", DELTA), ("clt_bonf", DELTA / P)):
        z = NormalDist().inv_cdf(1 - lvl)
        out[name] = _pick([mD[k] + z * _sd(D[k]) / math.sqrt(n) <= eps and md[k] - z * _sd(d[k]) / math.sqrt(n) >= 0
                           for k in range(P)], mc)
    t = math.sqrt(math.log(P / DELTA) / (2 * n))
    out["hoeff_bonf"] = _pick([mD[k] + 2 * t <= eps and md[k] - (hi - lo) * t >= 0 for k in range(P)], mc)
    out["ltt_hb"] = _pick([hb((mD[k] + 1) / 2, n, (eps + 1) / 2) <= DELTA / P
                           and hb((hi - md[k]) / (hi - lo), n, hi / (hi - lo)) <= DELTA / P for k in range(P)], mc)
    out["crc"] = _pick([(n * mD[k] + 1) / (n + 1) <= eps and (-n * md[k] - lo) / (n + 1) <= 0 for k in range(P)], mc)
    h = n // 2
    sel_ok = [_mean(D[k][:h]) <= eps and _mean(d[k][:h]) >= 0 for k in range(P)]
    sel = _pick(sel_ok, [-_mean(d[k][:h]) for k in range(P)])
    if sel >= 0:
        z = NormalDist().inv_cdf(1 - DELTA)
        B, b = D[sel][h:], d[sel][h:]
        nb = len(B)
        ok = _mean(B) + z * _sd(B) / math.sqrt(nb) <= eps and _mean(b) - z * _sd(b) / math.sqrt(nb) >= 0
        out["holdout"] = sel if ok else -1
    else:
        out["holdout"] = -1
    idx = list(range(n))
    go_ok, all_rej = [], True
    for k in range(P):
        prg = _betting_p([eps - v for v in D[k]], eps - 1.0, eps + 1.0, idx)
        psg = _betting_p(list(d[k]), lo, hi, idx) if mD[k] <= eps else 1.0
        prn = _betting_p([v - eps for v in D[k]], -1.0 - eps, 1.0 - eps, idx)
        psn = _betting_p([-v for v in d[k]], -hi, -lo, idx)
        go_ok.append(prg <= DELTA / P and psg <= DELTA / P)
        all_rej = all_rej and (prn <= DELTA / 2 or psn <= DELTA / 2)
    go = any(go_ok)
    v = "INCONCLUSIVE" if (go and all_rej) else ("GO" if go else ("NO-GO" if all_rej else "INCONCLUSIVE"))
    out["caac_verdict"] = v
    out["caac"] = _pick(go_ok, mc) if v == "GO" else -1
    return out
