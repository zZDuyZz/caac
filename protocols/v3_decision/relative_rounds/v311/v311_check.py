"""Checks for v3-11 (design: DECISION_v311_before_run.md): X0, X5, X7. Runs on import (used by v311_w1.py)."""
import math, random, sys
import numpy as np
sys.path.insert(0, "../bl")
import v311_methods as C, v310_methods as C10, v39_ref as F, v311_ref as RF
import bl_w1 as W, bl_methods as BM
from bl_methods import R

rng = random.Random(11)
names = list(R.SCEN)


def lo_pol_of(name):
    pols, kind, _ = R.SCEN[name]
    b_max = R.cap(kind, R.MEAN_B)
    return np.array([-(c * R.cap(kind, R.MEAN_A) + BM.S * b_max) for c, *_ in pols])


# X0: V0 == official vectorised CAAC, H1 == the v3-10 implementation
bad = 0
for t in range(60):
    nm = rng.choice(names); n = rng.choice((150, 300, 600)); rep = rng.randrange(1000)
    D, d, mc, lo, hi = W.draw(nm, n, rep, tag="rel")
    o = C.run_cands(D[None], d[None], mc[None], lo_pol_of(nm), lo, hi, names=("V0", "H1"))
    o10 = C10.run_cands(D[None], d[None], mc[None], lo_pol_of(nm), lo, hi, names=("H1",))
    ref = BM.caac(D[None], d[None], mc[None], lo, hi)
    bad += (o["V0"][0][0] != ref[2][0]) or (o["V0"][1][0] != ref[1][0])
    bad += (o["H1"][0][0] != o10["H1"][0][0]) or (o["H1"][1][0] != o10["H1"][1][0])
print("X0 V0 == caac and H1 == v3-10 H1, mismatches:", bad, "/ 60"); assert bad == 0

# X5: plain-Python decision for every candidate
def decide_plain(D, d, mc, lo_p, lo_g, hi, v, eps=BM.EPS):
    P, n = len(D), len(D[0]); idx = list(range(n)); ok = []; allrej = True; a = BM.DELTA / P
    for k in range(P):
        mD = sum(D[k]) / n
        pb_g = F._betting_p([eps - x for x in D[k]], eps - 1.0, eps + 1.0, idx)
        pb_n = F._betting_p([x - eps for x in D[k]], -1.0 - eps, 1.0 - eps, idx)
        px_g, px_n = F.p_risk_go_xc(D[k], eps), F.p_risk_above_xc(D[k], eps)
        rn = min(1.0, 2 * min(pb_n, px_n))
        if v == "V0":
            rg, rn = pb_g, pb_n
            sgp = F._betting_p(list(d[k]), lo_g, hi, idx) if mD <= eps else 1.0
        elif v == "H1":
            rg = min(1.0, 2 * min(pb_g, px_g))
            sgp = F.betting_p_v(list(d[k]), lo_p[k], hi, idx, 0.9, False) if mD <= eps else 1.0
        else:
            (qk, qc), (sk, sc) = C.CANDS[v]
            rg = RF.wsr_p([eps - x for x in D[k]], eps - 1.0, eps + 1.0, qc, a)
            if mD <= eps:
                sgp = F.betting_p_v(list(d[k]), lo_p[k], hi, idx, sc, False) if sk == "agrapa" else RF.wsr_p(list(d[k]), lo_p[k], hi, sc, a)
            else:
                sgp = 1.0
        sn = F._betting_p([-x for x in d[k]], -hi, -lo_g, idx)
        ok.append(rg <= a and sgp <= a)
        allrej = allrej and (rn <= BM.DELTA / 2 or sn <= BM.DELTA / 2)
    go = any(ok)
    verdict = "INCONCLUSIVE" if (go and allrej) else ("GO" if go else ("NO-GO" if allrej else "INCONCLUSIVE"))
    best = -1
    if verdict == "GO":
        for k in range(P):
            if ok[k] and (best < 0 or mc[k] < mc[best]):
                best = k
    return verdict, best


mis = 0
for t in range(150):
    nm = rng.choice(names); n = rng.choice((150, 300)); rep = rng.randrange(1000)
    D, d, mc, lo, hi = W.draw(nm, n, rep, tag="chk11")
    lp = lo_pol_of(nm)
    out = C.run_cands(D[None], d[None], mc[None], lp, lo, hi)
    for v in C.ALL:
        pv, pb = decide_plain(D.tolist(), d.tolist(), mc.tolist(), lp, lo, hi, v)
        if out[v][0][0] != pv or int(out[v][1][0]) != pb:
            mis += 1; print("MISMATCH", v, nm, n, rep, out[v][0][0], pv, out[v][1][0], pb)
print("X5 mismatches", mis, "/ 150 draws x", len(C.ALL), "methods"); assert mis == 0

# X5b: p-values themselves, vectorised vs plain, random bounded data
mx = 0.0
for t in range(200):
    n = rng.choice((2, 5, 50, 300)); lo = -rng.uniform(0.1, 3); hi = rng.uniform(0.1, 3); cap = rng.choice((0.5, 0.9)); al = rng.choice((0.1, 0.02, 0.005))
    xs = [rng.uniform(lo, hi) for _ in range(n)]
    pv = C.wsr_rows(np.array([xs]), lo, hi, cap, al)[0]; pp = RF.wsr_p(xs, lo, hi, cap, al)
    mx = max(mx, abs(pv - pp))
print("X5b max |vectorised - plain| p-value:", mx); assert mx < 1e-12

# X7: size at the null boundary (E[x] = 0), worst of several bounded distributions, 20,000 sequences each
nrng = np.random.default_rng(7); worst = 0.0
for lo, hi in ((-1.0, 1.0), (-0.98, 1.02), (-9.0, 1.0), (-1.0, 9.0)):
    for kind in ("two-point", "uniform"):
        if kind == "two-point":
            p_hi = -lo / (hi - lo)
            X = np.where(nrng.random((20000, 300)) < p_hi, hi, lo)
        else:
            w = min(-lo, hi); X = nrng.uniform(-w, w, (20000, 300))
        for al in (0.1, 0.02):
            r = (C.wsr_rows(X, lo, hi, 0.9, al) <= al).mean() / al
            worst = max(worst, r)
print("X7 worst size / alpha at the null boundary:", round(worst, 3)); assert worst <= 1.0 + 4 * math.sqrt(1 / (20000 * 0.02))
print("ALL CHECKS PASS")
