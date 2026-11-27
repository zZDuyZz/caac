"""W1 synthetic world (design: DECISION_bl_before_run.md). Scenarios S1-S11 from sim_rel.py unchanged;
S12-S14 written in the design. Draws as in sim_rel.one with seed tag 'bl'.
Usage: python bl_w1.py REPS OUT.json [workers]"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import bl_methods as M
from bl_methods import R
from caac.pool_v1.audit import _order, DEFAULT_SALT

_grid = [(1, q, .03, .03 + .03 / q) for q in [round(.60 + .01 * i, 2) for i in range(30)]]
R.SCEN["S12_near_miss_grid"] = (_grid, "uniform", None)
R.SCEN["S13_heavy_saving_false"] = ([(1, q, .03, .03 + .005 / q) for q in (.50, .505, .51, .515, .52)], "lognorm", None)
R.SCEN["S14_one_feasible_in_grid"] = (_grid[1:] + [(1, .65, .03, .03)], "uniform", None)
NEW = ("S12_near_miss_grid", "S13_heavy_saving_false", "S14_one_feasible_in_grid")


def feasible(name):
    cl, rows = R.truth(name)
    return cl, np.array([dl <= M.EPS + 1e-12 and sv >= M.S - 1e-12 for dl, sv in rows])


def draw(name, n, rep, tag="bl"):
    """Same arithmetic as sim_rel.one; returns D, d (P, n) in the audit order, mean cost, lo, hi."""
    pols, kind, _ = R.SCEN[name]
    seed = int(hashlib.sha256(f"{tag}|{name}|{n}|{rep}".encode()).hexdigest()[:15], 16)
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
    lo, hi = -(a_max + M.S * b_max), (1 - M.S) * b_max
    d = (1 - M.S) * Bo[None, :] - C
    return D, d, C.mean(axis=1), lo, hi


BATCH = 50


def cell(args):
    name, n, r0, r1 = args
    Ds, ds, mcs = [], [], []
    for rep in range(r0, r1):
        D, d, mc, lo, hi = draw(name, n, rep)
        Ds.append(D); ds.append(d); mcs.append(mc)
    out = M.run_all(np.stack(Ds), np.stack(ds), np.stack(mcs), lo, hi)
    return name, n, r0, {k: v.tolist() for k, v in out.items()}


def summarise(name, n, res, reps):
    cl, feas = feasible(name)
    row = {"claim_true": bool(cl), "n": n, "reps": reps}
    for m in M.METHODS:
        ch = np.array(res[m])
        go = ch >= 0
        viol = go & ~feas[np.maximum(ch, 0)]
        row[m] = {"go": int(go.sum()), "false_go": int(viol.sum()), "power": int((go & ~viol).sum())}
    v = np.array(res["caac_verdict"])
    row["caac"]["nogo"] = int((v == "NO-GO").sum())
    return row


def main():
    reps, outp = int(sys.argv[1]), sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    names = list(R.SCEN)
    jobs = [(nm, n, r0, min(reps, r0 + BATCH)) for nm in names for n in R.NS for r0 in range(0, reps, BATCH)]
    acc = {}
    t0 = time.time()
    with Pool(workers) as P:
        for nm, n, r0, o in P.imap_unordered(cell, jobs, chunksize=1):
            a = acc.setdefault((nm, n), {})
            a[r0] = o
    cells = []
    for nm in names:
        for n in R.NS:
            parts = acc[(nm, n)]
            res = {k: sum((parts[r0][k] for r0 in sorted(parts)), []) for k in parts[0]}
            cells.append({"scenario": nm, **summarise(nm, n, res, reps)})
    json.dump({"reps": reps, "eps": M.EPS, "s": M.S, "delta": M.DELTA, "cells": cells,
               "seconds": round(time.time() - t0)}, open(outp, "w"), indent=1, sort_keys=True)
    print("seconds", round(time.time() - t0))


if __name__ == "__main__":
    main()
