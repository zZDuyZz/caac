"""Revision 2 of the relative-claim simulation: adds the exact harm-count test ('harm') and the
Bonferroni mix of harm and betting ('mix') for the GO risk leg. sim_rel.py is imported unchanged.
Usage: python sim_rel2.py run REPS OUT.json [scenario,...]"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import sim_rel as R
from caac.pool_v1.audit import _order, p_risk_below, DEFAULT_SALT

VARS = ("bet", "harm", "mix")


def one(args):
    name, n, rep = args
    pols, kind, ch = R.SCEN[name]
    seed = int(hashlib.sha256(f"rel|{name}|{n}|{rep}".encode()).hexdigest()[:15], 16)   # same data as rev. 1
    rng = np.random.default_rng(seed)
    P = len(pols)
    A = R.draw_cost(rng, kind, R.MEAN_A, n); B = R.draw_cost(rng, kind, R.MEAN_B, n)
    big_err = rng.random(n) < R.R_B
    D = np.zeros((P, n)); C = np.zeros((P, n))
    for k, (c, q, a, b) in enumerate(pols):
        small = rng.random(n) < q
        u = rng.random(n)
        dk = np.where(big_err, np.where(u < a / R.R_B, -1.0, 0.0), np.where(u < b / (1 - R.R_B), 1.0, 0.0))
        D[k] = np.where(small, dk, 0.0)
        C[k] = c * A + np.where(small, 0.0, B)
    qids = [f"q{seed:x}_{i}" for i in range(n)]
    order = np.array(_order(qids, DEFAULT_SALT))
    D = D[:, order]; C = C[:, order]; Bo = B[order]
    a_max = max(c for c, *_ in pols) * R.cap(kind, R.MEAN_A); b_max = R.cap(kind, R.MEAN_B)
    lo_d, hi_d = -(a_max + R.S * b_max), (1 - R.S) * b_max
    d = (1 - R.S) * Bo[None, :] - C
    p_bet = R.betting_p_rows(R.EPS - D, R.EPS - 1, R.EPS + 1)
    harms = (D > 0).sum(axis=1)
    p_harm = np.array([p_risk_below(int(h), n, R.EPS) for h in harms])
    p_mix = np.minimum(1.0, 2 * np.minimum(p_bet, p_harm))
    p_sav_go = R.betting_p_rows(d, lo_d, hi_d)
    p_risk_ng = R.betting_p_rows(D - R.EPS, -1 - R.EPS, 1 - R.EPS)
    p_sav_ng = R.betting_p_rows(-d, -hi_d, -lo_d)
    meanD = D.mean(axis=1)
    nogo = bool(np.all((p_risk_ng <= R.DELTA / 2) | (p_sav_ng <= R.DELTA / 2)))
    out = {}
    for v, pr in (("bet", p_bet), ("harm", p_harm), ("mix", p_mix)):
        ok = (pr <= R.DELTA / P) & (p_sav_go <= R.DELTA / P) & (meanD <= R.EPS)
        out[v] = R._verdict(ok, nogo)
        if ch is not None:
            Cn = len(ch); lvl = R.DELTA / (2 * Cn)
            r_ok = np.zeros(P, bool); s_ok = np.zeros(P, bool)
            for idx in ch:
                for k in idx:
                    if pr[k] <= lvl:
                        r_ok[k] = True
                    else:
                        break
                for k in reversed(idx):
                    if p_sav_go[k] <= lvl:
                        s_ok[k] = True
                    else:
                        break
            out[v + "+fs2"] = R._verdict(r_ok & s_ok, nogo)
    return name, n, out


def run(reps, outp, only=None):
    names = [k for k in R.SCEN if (not only or k in only)]
    jobs = [(nm, n, r) for nm in names for n in R.NS for r in range(reps)]
    agg = {}; t0 = time.time()
    with Pool(2) as p:
        for nm, n, o in p.imap_unordered(one, jobs, chunksize=16):
            for v, verd in o.items():
                agg.setdefault(f"{nm}|{n}|{v}", {"GO": 0, "NO-GO": 0, "INCONCLUSIVE": 0})[verd] += 1
    info = {nm: {"claim_true": R.truth(nm)[0]} for nm in names}
    json.dump({"reps": reps, "rates": agg, "truth": info, "seconds": round(time.time() - t0)}, open(outp, "w"), indent=1)
    for nm in names:
        cl = info[nm]["claim_true"]; print(nm, "claim", cl)
        keys = sorted({k.split("|")[2] for k in agg if k.startswith(nm + "|")})
        for n in R.NS:
            row = f"  n={n:5d}"
            for v in keys:
                c = agg[f"{nm}|{n}|{v}"]
                wrong = (c["GO"] if not cl else c["NO-GO"]) / reps
                row += f" | {v}: GO {c['GO']/reps:.3f} wr {wrong:.3f}"
            print(row)
    print("seconds", round(time.time() - t0))


if __name__ == "__main__":
    R.check()
    run(int(sys.argv[2]), sys.argv[3], set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None)
