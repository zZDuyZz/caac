"""Simulation for the RELATIVE claim (non-inferiority to big-only, paired) and the two-direction
fixed-sequence M1. Design and criteria: DECISION_rel_before_real.md (written before this ran).

Usage:  python sim_rel.py check                 (criterion d: vectorised betting == audit._betting_p)
        python sim_rel.py run REPS OUT.json [scenario,...]
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v3work", "src"))
from caac.pool_v1.audit import _betting_p, _order, DEFAULT_SALT  # noqa: E402

EPS, S, DELTA, R_B = 0.02, 0.20, 0.10, 0.25
NS = (150, 300, 600, 1200, 2400)


# --------------------------------------------------------------------------- betting (vectorised)
def betting_p_rows(X: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Row-wise p-values of _betting_p (H0: E[x] <= 0), columns ALREADY in the test order.
    Same arithmetic, same order of operations as caac.pool_v1.audit._betting_p."""
    P, n = X.shape
    if n < 2 or lo >= 0 or hi <= 0:
        return np.ones(P)
    lmax = 0.5 / (-lo)
    logk = np.zeros(P); best = np.zeros(P); s1 = np.zeros(P); s2 = np.zeros(P)
    for m in range(n):
        x = X[:, m]
        if m:
            mu = s1 / m
            var = s2 / m - mu * mu
            lam = np.where(mu <= 0, 0.0, np.minimum(lmax, mu / (var + mu * mu + 1e-12)))
        else:
            lam = np.zeros(P)
        logk = logk + np.log1p(lam * x)
        best = np.maximum(best, logk)
        s1 = s1 + x; s2 = s2 + x * x
    return np.minimum(1.0, np.exp(-best))


def check(nchk=200):
    rng = np.random.default_rng(1)
    worst = 0.0
    for t in range(nchk):
        n = int(rng.integers(2, 400)); lo = -float(rng.uniform(0.2, 3)); hi = float(rng.uniform(0.2, 3))
        X = rng.uniform(lo, hi, size=(3, n))
        if t % 3 == 0:                      # discrete {-1,0,1}-like rows as in the risk leg
            X = (EPS - rng.choice([-1.0, 0.0, 1.0], p=[.05, .9, .05], size=(3, n)))
            lo, hi = EPS - 1, EPS + 1
        v = betting_p_rows(X, lo, hi)
        for r in range(3):
            ref = _betting_p(list(X[r]), lo, hi, list(range(n)))
            worst = max(worst, abs(ref - v[r]))
    print("criterion (d): max |vectorised - _betting_p| over", nchk, "checks =", worst)
    assert worst <= 1e-12, worst
    return worst


# --------------------------------------------------------------------------- scenarios (fixed in DECISION)
DEC = [(1, .95, .02, .08), (1, .40, .03, .035), (2, .85, .03, .06), (2, .90, .02, .07)]


def chains(delta_mid, swap=False):
    pols, ch = [], []
    for j in range(8):
        c = 1 if j < 4 else 2
        f = (1.0, 1.1, 1.2, 1.25)[j % 4]
        idx = []
        for q, dl in zip((.45, .65, .85), (.002, delta_mid, .06)):
            a = .03; b = a + dl * f / q
            idx.append(len(pols)); pols.append((c, q, a, b))
        if swap:
            idx[0], idx[1] = idx[1], idx[0]
        ch.append(idx)
    return pols, ch


SCEN = {
    "S1_rel_feasible":     ([(1, .70, .04, .05)] + DEC, "uniform", None),
    "S2_rel_tight":        ([(1, .60, .03, .058)] + DEC, "uniform", None),
    "S3_rel_bound_false":  ([(1, .70, .03, .03 + .025 / .70), (1, .40, .03, .035), (1, .95, .02, .08)], "uniform", None),
    "S4_rel_bound_true":   ([(1, .70, .03, .03 + .02 / .70), (1, .40, .03, .035), (1, .95, .02, .08)], "uniform", None),
    "S5_rel_false_risk":   ([(1, .80, .02, .07), (1, .95, .02, .08), (2, .90, .02, .07), (1, .40, .03, .035)], "uniform", None),
    "S6_rel_false_saving": ([(1, .45, .03, .035), (1, .50, .04, .05), (2, .80, .03, .04)], "uniform", None),
    "S7_rel_heavy":        ([(1, .70, .04, .05)] + DEC, "lognorm", None),
    "S8_rel_discordant":   ([(1, .70, .15, .16)] + DEC, "uniform", None),
}
for nm, (dm, sw) in {"S9_rel_chains_true": (.016, False), "S10_rel_chains_false": (.03, False),
                     "S11_rel_nonmono": (.016, True)}.items():
    p, ch = chains(dm, sw)
    SCEN[nm] = (p, "uniform", ch)

MEAN_A, MEAN_B, RATIO = 1.0, 3.0, 10.0


def draw_cost(rng, kind, mean, size):
    if kind == "uniform":
        return rng.uniform(0, 2 * mean, size)
    return np.minimum(RATIO * mean, rng.lognormal(math.log(mean) - 0.8, 1.265, size))


def cap(kind, mean):
    return 2 * mean if kind == "uniform" else RATIO * mean


_EAB = {}


def ea_eb(kind):
    if kind not in _EAB:
        r = np.random.default_rng(12345)
        _EAB[kind] = (draw_cost(r, kind, MEAN_A, 2_000_000).mean(), draw_cost(r, kind, MEAN_B, 2_000_000).mean())
    return _EAB[kind]


def truth(name, eps=EPS, s=S):
    pols, kind, _ = SCEN[name]
    Ea, Eb = ea_eb(kind)
    rows = [(q * (b - a), q - c * Ea / Eb) for c, q, a, b in pols]
    for c, q, a, b in pols:
        assert 0 <= a <= R_B and 0 <= b <= 1 - R_B, (name, a, b)
    claim = any(dl <= eps + 1e-12 and sv >= s - 1e-12 for dl, sv in rows)
    return claim, rows


# --------------------------------------------------------------------------- one replicate
def one(args):
    name, n, rep = args
    pols, kind, ch = SCEN[name]
    seed = int(hashlib.sha256(f"rel|{name}|{n}|{rep}".encode()).hexdigest()[:15], 16)
    rng = np.random.default_rng(seed)
    P = len(pols)
    A = draw_cost(rng, kind, MEAN_A, n); B = draw_cost(rng, kind, MEAN_B, n)
    big_err = rng.random(n) < R_B
    D = np.zeros((P, n)); C = np.zeros((P, n))
    for k, (c, q, a, b) in enumerate(pols):
        small = rng.random(n) < q
        u = rng.random(n)
        dk = np.where(big_err, np.where(u < a / R_B, -1.0, 0.0), np.where(u < b / (1 - R_B), 1.0, 0.0))
        D[k] = np.where(small, dk, 0.0)
        C[k] = c * A + np.where(small, 0.0, B)
    qids = [f"q{seed:x}_{i}" for i in range(n)]
    order = np.array(_order(qids, DEFAULT_SALT))
    D = D[:, order]; C = C[:, order]; Bo = B[order]
    a_max = max(c for c, *_ in pols) * cap(kind, MEAN_A); b_max = cap(kind, MEAN_B)
    lo_d, hi_d = -(a_max + S * b_max), (1 - S) * b_max
    d = (1 - S) * Bo[None, :] - C
    assert d.min() >= lo_d - 1e-9 and d.max() <= hi_d + 1e-9
    # legs
    p_risk_go = betting_p_rows(EPS - D, EPS - 1, EPS + 1)
    p_sav_go = betting_p_rows(d, lo_d, hi_d)
    p_risk_ng = betting_p_rows(D - EPS, -1 - EPS, 1 - EPS)
    p_sav_ng = betting_p_rows(-d, -hi_d, -lo_d)
    meanD = D.mean(axis=1); mc = C.mean(axis=1)
    nogo = bool(np.all((p_risk_ng <= DELTA / 2) | (p_sav_ng <= DELTA / 2)))
    out = {}
    # bonf (with the same gate as audit3: the saving leg counts only if mean D <= eps)
    ok = (p_risk_go <= DELTA / P) & (p_sav_go <= DELTA / P) & (meanD <= EPS)
    out["bonf"] = _verdict(ok, nogo)
    if ch is not None:
        Cn = len(ch); lvl = DELTA / (2 * Cn)
        r_ok = np.zeros(P, bool); s_ok = np.zeros(P, bool)
        for idx in ch:
            for k in idx:                    # risk: conservative -> liberal
                if p_risk_go[k] <= lvl:
                    r_ok[k] = True
                else:
                    break
            for k in reversed(idx):          # saving: liberal -> conservative
                if p_sav_go[k] <= lvl:
                    s_ok[k] = True
                else:
                    break
        out["fs2"] = _verdict(r_ok & s_ok, nogo)
    return name, n, out


def _verdict(ok, nogo):
    go = bool(ok.any())
    if go and nogo:
        return "INCONCLUSIVE"
    return "GO" if go else ("NO-GO" if nogo else "INCONCLUSIVE")


# --------------------------------------------------------------------------- runner
def run(reps, outp, only=None):
    names = [k for k in SCEN if (not only or k in only)]
    jobs = [(nm, n, r) for nm in names for n in NS for r in range(reps)]
    agg = {}
    t0 = time.time()
    with Pool(2) as p:
        for nm, n, o in p.imap_unordered(one, jobs, chunksize=16):
            for v, verd in o.items():
                agg.setdefault(f"{nm}|{n}|{v}", {"GO": 0, "NO-GO": 0, "INCONCLUSIVE": 0})[verd] += 1
    info = {}
    for nm in names:
        cl, rows = truth(nm)
        info[nm] = {"claim_true": cl, "delta_saving": [(round(a, 4), round(b, 4)) for a, b in rows],
                    "chains": SCEN[nm][2] is not None}
    json.dump({"reps": reps, "eps": EPS, "s": S, "delta": DELTA, "R_B": R_B, "rates": agg, "truth": info,
               "seconds": round(time.time() - t0)}, open(outp, "w"), indent=1)
    for nm in names:
        cl = info[nm]["claim_true"]
        print(nm, "claim", cl)
        for n in NS:
            row = f"  n={n:5d}"
            for v in ("bonf", "fs2"):
                c = agg.get(f"{nm}|{n}|{v}")
                if c:
                    wrong = (c["GO"] if not cl else c["NO-GO"]) / reps
                    row += f" | {v}: GO {c['GO']/reps:.3f} NG {c['NO-GO']/reps:.3f} wrong {wrong:.3f}"
            print(row)
    print("seconds", round(time.time() - t0))


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        check()
        run(int(sys.argv[2]), sys.argv[3], set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None)
