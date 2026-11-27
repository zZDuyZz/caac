"""#170 baselines (design: DECISION_bl_before_run.md, hashed before any run).

Vectorised decision rules for the relative claim. Inputs are batches of studies:
  D  (R, P, n) in {-1, 0, 1}    paired error difference per policy and question
  d  (R, P, n) in [lo, hi]      (1 - s) B_i - C_pi,i
  mc (R, P)                     mean cost of each policy on the study's questions (for "cheapest passing")
Columns are already in the test order. Every rule returns (passes (R, P) bool, chosen (R,) int, -1 = not GO);
CAAC also returns verdicts.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.special import xlogy
from scipy.stats import binom, norm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CAAC_REPO", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "protocols", "v3_decision", "relative"))
import sim_rel as R                                    # noqa: E402  (validated vectorised betting)

EPS, S, DELTA = 0.02, 0.20, 0.10
METHODS = ("naive", "clt", "clt_bonf", "hoeff_bonf", "ltt_hb", "crc", "holdout", "caac")


def _cheapest(passes: np.ndarray, mc: np.ndarray) -> np.ndarray:
    m = np.where(passes, mc, np.inf)
    ch = m.argmin(axis=1)
    return np.where(passes.any(axis=1), ch, -1)


def _mean_sd(X):
    n = X.shape[-1]
    mu = X.mean(axis=-1)
    sd = X.std(axis=-1, ddof=1) if n > 1 else np.zeros_like(mu)
    return mu, sd, n


def naive(D, d, mc, lo, hi, eps=EPS):
    p = (D.mean(-1) <= eps) & (d.mean(-1) >= 0)
    return p, _cheapest(p, mc)


def clt(D, d, mc, lo, hi, eps=EPS, level=DELTA):
    z = norm.ppf(1 - level)
    mD, sD, n = _mean_sd(D)
    md, sdd, _ = _mean_sd(d)
    p = (mD + z * sD / np.sqrt(n) <= eps) & (md - z * sdd / np.sqrt(n) >= 0)
    return p, _cheapest(p, mc)


def clt_bonf(D, d, mc, lo, hi, eps=EPS):
    return clt(D, d, mc, lo, hi, eps, DELTA / D.shape[1])


def hoeff_bonf(D, d, mc, lo, hi, eps=EPS):
    P, n = D.shape[1], D.shape[-1]
    t = np.sqrt(np.log(P / DELTA) / (2 * n))
    p = (D.mean(-1) + 2.0 * t <= eps) & (d.mean(-1) - (hi - lo) * t >= 0)
    return p, _cheapest(p, mc)


def hb_p(xbar: np.ndarray, n: int, mu0: float) -> np.ndarray:
    """Hoeffding-Bentkus p-value for H0: E[x] >= mu0 (x in [0, 1]); small p = evidence that E[x] < mu0."""
    a = np.minimum(xbar, mu0)
    h = xlogy(a, a / mu0) + xlogy(1 - a, (1 - a) / (1 - mu0))
    p_h = np.exp(-n * h)
    p_b = np.e * binom.cdf(np.ceil(n * xbar - 1e-9), n, mu0)
    return np.minimum(1.0, np.minimum(p_h, p_b))


def ltt_hb(D, d, mc, lo, hi, eps=EPS):
    P, n = D.shape[1], D.shape[-1]
    x = (D.mean(-1) + 1) / 2
    y = (hi - d.mean(-1)) / (hi - lo)
    pr = hb_p(x, n, (eps + 1) / 2)
    ps = hb_p(y, n, hi / (hi - lo))
    p = (pr <= DELTA / P) & (ps <= DELTA / P)
    return p, _cheapest(p, mc)


def crc(D, d, mc, lo, hi, eps=EPS):
    n = D.shape[-1]
    p = ((n * D.mean(-1) + 1.0) / (n + 1) <= eps) & ((n * (-d).mean(-1) - lo) / (n + 1) <= 0)
    return p, _cheapest(p, mc)


def holdout(D, d, mc, lo, hi, eps=EPS):
    """Select on the first half by the naive rule (cheapest on that half), test the one policy on the second
    half with the CLT interval at level delta per leg. passes[r, k] is True only for the tested policy."""
    n = D.shape[-1]
    h = n // 2
    DA, dA, DB, dB = D[..., :h], d[..., :h], D[..., h:], d[..., h:]
    sel_pass = (DA.mean(-1) <= eps) & (dA.mean(-1) >= 0)
    # cheapest on A: d = (1 - s) B - C and B is common to all policies, so ordering by -mean(dA)
    # is the same as ordering by mean cost on A
    sel = _cheapest(sel_pass, -dA.mean(-1))
    z = norm.ppf(1 - DELTA)
    mD, sD, nb = _mean_sd(DB)
    md, sdd, _ = _mean_sd(dB)
    test = (mD + z * sD / np.sqrt(nb) <= eps) & (md - z * sdd / np.sqrt(nb) >= 0)
    p = np.zeros_like(test)
    r = np.arange(D.shape[0])
    ok = sel >= 0
    p[r[ok], sel[ok]] = test[r[ok], sel[ok]]
    ch = np.where(ok & p[r, np.maximum(sel, 0)], sel, -1)
    return p, ch


def caac(D, d, mc, lo, hi, eps=EPS):
    """Official v3 relative audit, vectorised (same arithmetic as audit3 / sim_rel.one 'bonf')."""
    Rr, P, n = D.shape
    D2 = D.reshape(Rr * P, n); d2 = d.reshape(Rr * P, n)
    p_rg = R.betting_p_rows(eps - D2, eps - 1, eps + 1).reshape(Rr, P)
    p_sg = R.betting_p_rows(d2, lo, hi).reshape(Rr, P)
    p_rn = R.betting_p_rows(D2 - eps, -1 - eps, 1 - eps).reshape(Rr, P)
    p_sn = R.betting_p_rows(-d2, -hi, -lo).reshape(Rr, P)
    mD = D.mean(-1)
    p_sg = np.where(mD <= eps, p_sg, 1.0)
    p = (p_rg <= DELTA / P) & (p_sg <= DELTA / P)
    nogo = np.all((p_rn <= DELTA / 2) | (p_sn <= DELTA / 2), axis=1)
    go = p.any(axis=1)
    verdict = np.where(go & nogo, "INCONCLUSIVE", np.where(go, "GO", np.where(nogo, "NO-GO", "INCONCLUSIVE")))
    ch = np.where(verdict == "GO", _cheapest(p, mc), -1)
    return p, ch, verdict


def run_all(D, d, mc, lo, hi, eps=EPS):
    out = {}
    for m in METHODS[:-1]:
        out[m] = globals()[m](D, d, mc, lo, hi, eps)[1]
    _, ch, v = caac(D, d, mc, lo, hi, eps)
    out["caac"] = ch
    out["caac_verdict"] = v
    return out
