"""Exact unconditional p-value table for the paired non-inferiority test (GO quality leg), ESTIMATION ONLY
(grid sup without a discretisation correction). H0: pH - pB >= eps; small Wald Z = evidence against H0.
For fixed p_B the rejection probability is decreasing in p_H (shown in the design), so the sup is on the boundary
pH = pB + eps, pB in [0, (1 - eps)/2]. Usage: python eu_table.py N EPS GRID OUT.npy"""
import sys
import numpy as np
from scipy.special import gammaln


def wald_z(h, b, n, eps):
    d = (h - b) / n
    v = (h + b + 1.0) / n - d * d          # +1 count keeps v > 0
    return (d - eps) / np.sqrt(np.maximum(v, 1.0 / n) / n)


def table(n, eps, G):
    h, b = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    ok = h + b <= n
    h = h[ok].astype(float); b = b[ok].astype(float); z0 = n - h - b
    Z = wald_z(h, b, n, eps)
    order = np.argsort(Z, kind="stable")
    Zs = Z[order]
    # end index of each tie group
    last = np.searchsorted(Zs, Zs, side="right") - 1
    lc = gammaln(n + 1) - gammaln(h + 1) - gammaln(b + 1) - gammaln(z0 + 1)
    lc, hs, bs, zs = lc[order], h[order], b[order], z0[order]
    best = np.zeros(len(Zs))
    top = (1 - eps) / 2
    for pB in np.linspace(0, top, G):
        pH = pB + eps; p0 = 1 - pH - pB
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = lc + np.where(hs > 0, hs * np.log(pH), 0) + np.where(bs > 0, bs * np.log(pB) if pB > 0 else -np.inf, 0) \
                 + np.where(zs > 0, zs * np.log(p0) if p0 > 0 else -np.inf, 0)
        cdf = np.cumsum(np.exp(lp))
        best = np.maximum(best, cdf[last])
    T = np.ones((n + 1, n + 1))
    T[hs.astype(int), bs.astype(int)] = np.minimum(1.0, best)
    return T


if __name__ == "__main__":
    n, eps, G, out = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    np.save(out, table(n, eps, G))
