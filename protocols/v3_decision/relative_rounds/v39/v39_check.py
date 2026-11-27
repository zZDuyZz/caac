"""X0 (V0 == bl_methods.caac on W1 draws), X5 (vectorised == plain Python for every variant), p-value cross-checks."""
import random, math
import numpy as np
import v39_methods as V, v39_ref as F
import bl_w1 as W
from bl_methods import R
import bl_methods as BM

rng = random.Random(5)
names = list(R.SCEN)
def lo_pol_of(name, hi_cap_s=BM.S):
    pols, kind, _ = R.SCEN[name]
    b_max = R.cap(kind, R.MEAN_B)
    return np.array([-(c * R.cap(kind, R.MEAN_A) + hi_cap_s * b_max) for c, *_ in pols])

# X0
bad = 0
for t in range(80):
    nm = rng.choice(names); n = rng.choice((150, 300, 600)); rep = rng.randrange(1000)
    D, d, mc, lo, hi = W.draw(nm, n, rep)
    o = V.run_variants(D[None], d[None], mc[None], lo_pol_of(nm), lo, hi, names=("V0",))
    ref = BM.caac(D[None], d[None], mc[None], lo, hi)
    bad += (o["V0"][0][0] != ref[2][0]) or (o["V0"][1][0] != ref[1][0])
print("X0 V0 vs bl_methods.caac mismatches:", bad, "/ 80"); assert bad == 0

# betting vectorised == plain (cap, mix, per-row lo)
worst = 0
for t in range(60):
    n = rng.randrange(5, 200); lo = -rng.uniform(0.3, 50); hi = rng.uniform(0.3, 10)
    X = np.array([[rng.uniform(lo, hi) if rng.random() < .3 else 0.0 for _ in range(n)] for _ in range(3)])
    for cap, mix in ((0.5, False), (0.9, False), (0.5, True), (0.9, True)):
        v = V.betting_rows(X, lo, hi, cap, mix)
        for r in range(3):
            ref = F.betting_p_v(list(X[r]), lo, hi, None, cap, mix)
            worst = max(worst, abs(v[r] - ref) / max(ref, 1e-300) if ref > 1e-200 else abs(v[r]-ref))
print("betting vectorised vs plain, max rel diff:", worst); assert worst < 1e-9

# xc p-values vectorised vs plain
worst = 0
for t in range(300):
    n = rng.randrange(10, 600); eps = rng.choice((0.02, 0.1))
    D = np.array([rng.choices((-1, 0, 1), weights=(rng.random()*.1, 1, rng.random()*.1))[0] for _ in range(n)])
    pg, pn = V.xc_p(D[None, None, :], eps)
    rg, rn = F.p_risk_go_xc(list(D), eps), F.p_risk_above_xc(list(D), eps)
    worst = max(worst, abs(pg[0, 0] - rg), abs(pn[0, 0] - rn))
print("xc vectorised vs plain, max abs diff:", worst); assert worst < 1e-9

# X5: every variant, full decision, on 150 random draws (plain-Python decision re-derived here)
def decide_plain(D, d, mc, lo_p, lo_g, hi, variant, eps=BM.EPS):
    risk, perpol, cap, mix = V.VARIANTS[variant]
    P, n = len(D), len(D[0]); idx = list(range(n)); ok = []; allrej = True
    for k in range(P):
        mD = sum(D[k]) / n
        rg = F.p_risk_go_xc(D[k], eps) if risk == "xc" else F._betting_p([eps - v for v in D[k]], eps - 1.0, eps + 1.0, idx)
        rn = F.p_risk_above_xc(D[k], eps) if risk == "xc" else F._betting_p([v - eps for v in D[k]], -1.0 - eps, 1.0 - eps, idx)
        sgp = F.betting_p_v(list(d[k]), lo_p[k] if perpol else lo_g, hi, idx, cap, mix) if mD <= eps else 1.0
        sn = F._betting_p([-v for v in d[k]], -hi, -lo_g, idx)
        ok.append(rg <= BM.DELTA / P and sgp <= BM.DELTA / P)
        allrej = allrej and (rn <= BM.DELTA / 2 or sn <= BM.DELTA / 2)
    go = any(ok)
    v = "INCONCLUSIVE" if (go and allrej) else ("GO" if go else ("NO-GO" if allrej else "INCONCLUSIVE"))
    best = -1
    if v == "GO":
        for k in range(P):
            if ok[k] and (best < 0 or mc[k] < mc[best]):
                best = k
    return v, best

mis = 0
for t in range(150):
    nm = rng.choice(names); n = rng.choice((150, 300)); rep = rng.randrange(1000)
    D, d, mc, lo, hi = W.draw(nm, n, rep)
    lp = lo_pol_of(nm)
    out = V.run_variants(D[None], d[None], mc[None], lp, lo, hi)
    for v in V.VARIANTS:
        pv, pb = decide_plain(D.tolist(), d.tolist(), mc.tolist(), lp, lo, hi, v)
        if out[v][0][0] != pv or int(out[v][1][0]) != pb:
            mis += 1; print("MISMATCH", v, nm, n, rep, out[v][0][0], pv, out[v][1][0], pb)
print("X5 mismatches", mis, "/ 150 draws x", len(V.VARIANTS), "variants"); assert mis == 0
print("ALL CHECKS PASS")
