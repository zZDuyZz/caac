"""X0/X5 for the v3-10 candidates (design: DECISION_v310_before_run.md)."""
import random, sys
import numpy as np
sys.path.insert(0, "../bl")
import v310_methods as C, v39_ref as F
import bl_w1 as W, bl_methods as BM
from bl_methods import R

rng = random.Random(9)
names = list(R.SCEN)


def lo_pol_of(name):
    pols, kind, _ = R.SCEN[name]
    b_max = R.cap(kind, R.MEAN_B)
    return np.array([-(c * R.cap(kind, R.MEAN_A) + BM.S * b_max) for c, *_ in pols])


bad = 0
for t in range(80):
    nm = rng.choice(names[:11]); n = rng.choice((150, 300, 600)); rep = rng.randrange(1000)
    D, d, mc, lo, hi = W.draw(nm, n, rep, tag="rel")
    o = C.run_cands(D[None], d[None], mc[None], lo_pol_of(nm), lo, hi, names=("V0",))
    ref = BM.caac(D[None], d[None], mc[None], lo, hi)
    bad += (o["V0"][0][0] != ref[2][0]) or (o["V0"][1][0] != ref[1][0])
print("X0 V0 vs bl_methods.caac mismatches:", bad, "/ 80"); assert bad == 0


def decide_plain(D, d, mc, lo_p, lo_g, hi, v, eps=BM.EPS):
    P, n = len(D), len(D[0]); idx = list(range(n)); ok = []; allrej = True
    for k in range(P):
        mD = sum(D[k]) / n
        pb_g = F._betting_p([eps - x for x in D[k]], eps - 1.0, eps + 1.0, idx)
        pb_n = F._betting_p([x - eps for x in D[k]], -1.0 - eps, 1.0 - eps, idx)
        if v == "V0":
            rg, rn, sgp = pb_g, pb_n, (F._betting_p(list(d[k]), lo_g, hi, idx) if mD <= eps else 1.0)
        else:
            q, cap = C.CANDS[v]
            sgp = F.betting_p_v(list(d[k]), lo_p[k], hi, idx, cap, False) if mD <= eps else 1.0
            px_g, px_n = F.p_risk_go_xc(D[k], eps), F.p_risk_above_xc(D[k], eps)
            if q == "bet":
                rg, rn = pb_g, pb_n
            elif q == "xc":
                rg, rn = px_g, px_n
            else:
                rg, rn = min(1.0, 2 * min(pb_g, px_g)), min(1.0, 2 * min(pb_n, px_n))
        sn = F._betting_p([-x for x in d[k]], -hi, -lo_g, idx)
        ok.append(rg <= BM.DELTA / P and sgp <= BM.DELTA / P)
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
    D, d, mc, lo, hi = W.draw(nm, n, rep)
    lp = lo_pol_of(nm)
    out = C.run_cands(D[None], d[None], mc[None], lp, lo, hi)
    for v in C.ALL:
        pv, pb = decide_plain(D.tolist(), d.tolist(), mc.tolist(), lp, lo, hi, v)
        if out[v][0][0] != pv or int(out[v][1][0]) != pb:
            mis += 1; print("MISMATCH", v, nm, n, rep, out[v][0][0], pv, out[v][1][0], pb)
print("X5 mismatches", mis, "/ 150 draws x", len(C.ALL), "candidates"); assert mis == 0
print("ALL CHECKS PASS")
