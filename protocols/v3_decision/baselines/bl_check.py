"""Checks X1 (independent re-implementation), X2 (p-values vs scipy), and CAAC == sim_rel 'bonf'."""
import random, sys
import numpy as np
from scipy.stats import binom, norm
import bl_methods as M, bl_ref as F, bl_w1 as W
from bl_methods import R
import caac; print("caac from", caac.__file__)

rng = random.Random(7)
names = list(R.SCEN)
# CAAC vectorised == sim_rel.one 'bonf' (same draws, tag 'rel')
bad = 0
for t in range(120):
    nm = rng.choice(names[:11]); n = rng.choice((150, 300, 600)); rep = rng.randrange(1000)
    D, d, mc, lo, hi = W.draw(nm, n, rep, tag="rel")
    v = M.caac(D[None], d[None], mc[None], lo, hi)[2][0]
    ref = R.one((nm, n, rep))[2]["bonf"]
    bad += v != ref
print("CAAC vs sim_rel bonf: mismatches", bad, "/ 120"); assert bad == 0

# X1: 300 random draws, every method, vectorised vs plain Python
mis = 0; cnt = {m: 0 for m in M.METHODS}
for t in range(300):
    nm = rng.choice(names); n = rng.choice(R.NS[:4]); rep = rng.randrange(1000)
    D, d, mc, lo, hi = W.draw(nm, n, rep)
    v = M.run_all(D[None], d[None], mc[None], lo, hi)
    f = F.decide(D.tolist(), d.tolist(), mc.tolist(), lo, hi)
    for m in M.METHODS:
        cnt[m] += v[m][0] >= 0
        if int(v[m][0]) != int(f[m]):
            mis += 1; print("MISMATCH", m, nm, n, rep, v[m][0], f[m])
    if v["caac_verdict"][0] != f["caac_verdict"]:
        mis += 1; print("MISMATCH verdict", nm, n, rep)
print("X1 mismatches", mis, "GO counts in sample", cnt); assert mis == 0

# X2: HB p vs scipy and vs plain binomial
worst = 0
for t in range(500):
    n = rng.randrange(5, 800); mu0 = rng.uniform(0.05, 0.95); xb = rng.randrange(0, n + 1) / n
    a = M.hb_p(np.array([xb]), n, mu0)[0]
    b = F.hb(xb, n, mu0)
    import math
    k = math.ceil(n * xb - 1e-9)
    c = min(1.0, math.exp(-n * ((min(xb,mu0) and min(xb,mu0)*math.log(min(xb,mu0)/mu0)) + (1-min(xb,mu0))*math.log((1-min(xb,mu0))/(1-mu0)))), math.e * binom.cdf(k, n, mu0))
    worst = max(worst, abs(a - b) / max(b, 1e-300) if b > 1e-250 else abs(a-b), abs(a - c))
print("X2 max rel diff HB", worst); assert worst < 1e-6
print("z", norm.ppf(0.9), F.NormalDist().inv_cdf(0.9))
print("ALL CHECKS PASS")
