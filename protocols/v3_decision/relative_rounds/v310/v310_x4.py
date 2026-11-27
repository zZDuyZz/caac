"""X4: exact validity of the quality-leg test R1 by enumeration of the multinomial (design: DECISION_v39_before_run.md).
For n, eps and a grid of (p_h, p_b) on and beyond the boundary of each null, the exact rejection probability must be <= level.
Uses the plain-Python p-values of v39_ref (the code that would enter audit.py)."""
import sys
import numpy as np
from scipy.stats import multinomial
import v39_ref as V

LEVELS_GO = tuple(l / 2 for l in (0.1, 0.1 / 3, 0.1 / 30, 0.1 / 100))
LEVELS_NG = tuple(l / 2 for l in (0.05, 0.1 / 4))
worst = {"go": (0.0, None), "nogo": (0.0, None)}
ncell = 0


def tables(n, eps):
    """p_go[h, b], p_ng[h, b] via (h, m) caches."""
    th_u, th_l = {}, {}
    for m in range(n + 1):
        th_u[m] = V.cp_upper(m, n, V.GAMMA); th_l[m] = V.cp_lower(m, n, V.GAMMA)
    cg, cn = {}, {}
    pg = np.ones((n + 1, n + 1)); pn = np.ones((n + 1, n + 1))
    for h in range(n + 1):
        for b in range(n + 1 - h):
            m = h + b
            if (h, m) not in cg:
                tu = th_u[m]
                cg[(h, m)] = V.GAMMA if tu < eps else min(1.0, V.binom_cdf(h, m, 0.5 + eps / (2 * tu)) + V.GAMMA)
                tl = th_l[m]
                pm = min(1.0, 1.0 if tl <= eps else 0.5 + eps / (2 * tl))
                cn[(h, m)] = min(1.0, V._sf(h, m, pm) + V.GAMMA)
            pg[h, b] = cg[(h, m)]; pn[h, b] = cn[(h, m)]
    return pg, pn


for n in (30, 60, 120):
    H, B = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    valid = (H + B) <= n
    Hv, Bv = H[valid], B[valid]
    X = np.stack([Hv, Bv, n - Hv - Bv], axis=1)
    for eps in (0.02, 0.10):
        pg, pn = tables(n, eps)
        pgv, pnv = pg[valid], pn[valid]
        for pb in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45):
            for extra in (0.0, 0.01, 0.05, 0.15):
                ph = pb + eps + extra                      # GO null: E[D] >= eps
                if ph + pb <= 1.0 + 1e-12:
                    pmf = multinomial.pmf(X, n, [ph, pb, max(0.0, 1 - ph - pb)])
                    for L in LEVELS_GO:
                        r = pmf[pgv <= L].sum() / L; ncell += 1
                        if r > worst["go"][0]:
                            worst["go"] = (r, (n, eps, pb, ph, L))
                ph = pb + eps - extra                      # NO-GO null: E[D] <= eps
                if ph >= -1e-12 and ph + pb <= 1.0:
                    pmf = multinomial.pmf(X, n, [max(0.0, ph), pb, 1 - max(0.0, ph) - pb])
                    for L in LEVELS_NG:
                        r = pmf[pnv <= L].sum() / L; ncell += 1
                        if r > worst["nogo"][0]:
                            worst["nogo"] = (r, (n, eps, pb, ph, L))
print("cells", ncell)
print("max rejection prob / level, GO null :", worst["go"])
print("max rejection prob / level, NO-GO null:", worst["nogo"])
ok = worst["go"][0] <= 1.0 and worst["nogo"][0] <= 1.0
print("X4", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
