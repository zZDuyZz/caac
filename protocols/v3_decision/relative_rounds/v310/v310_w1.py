"""W1 for the v3-10 candidates. Usage: python v310_w1.py REPS OUT.json SEED_TAG [workers]   (dev: bl, confirmation: bl2)"""
import json, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "../bl")
import v310_methods as C
import bl_w1 as W
from bl_methods import R
import bl_methods as BM
from v310_check import lo_pol_of  # noqa  (runs the checks once on import; cheap)

BATCH = 50
TAG = "bl"


def cell(args):
    name, n, r0, r1 = args
    Ds, ds, mcs = [], [], []
    for rep in range(r0, r1):
        D, d, mc, lo, hi = W.draw(name, n, rep, tag=TAG)
        Ds.append(D); ds.append(d); mcs.append(mc)
    out = C.run_cands(np.stack(Ds), np.stack(ds), np.stack(mcs), lo_pol_of(name), lo, hi)
    return name, n, r0, {v: (o[0].tolist(), o[1].tolist()) for v, o in out.items()}


def main():
    global TAG
    reps, outp, TAG = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    names = list(R.SCEN)
    jobs = [(nm, n, r0, min(reps, r0 + BATCH)) for nm in names for n in R.NS for r0 in range(0, reps, BATCH)]
    acc = {}; t0 = time.time()
    with Pool(workers) as P:
        for nm, n, r0, o in P.imap_unordered(cell, jobs, chunksize=1):
            acc.setdefault((nm, n), {})[r0] = o
    cells = []
    for nm in names:
        cl, feas = W.feasible(nm)
        for n in R.NS:
            parts = acc[(nm, n)]
            row = {"scenario": nm, "n": n, "reps": reps, "claim_true": bool(cl), "seed_tag": TAG}
            for v in C.ALL:
                verd = sum((parts[r0][v][0] for r0 in sorted(parts)), [])
                ch = np.array(sum((parts[r0][v][1] for r0 in sorted(parts)), []))
                go = ch >= 0
                viol = go & ~feas[np.maximum(ch, 0)]
                row[v] = {"go": int(go.sum()), "false_go": int(viol.sum()), "power": int((go & ~viol).sum()),
                          "nogo": int(sum(x == "NO-GO" for x in verd))}
            cells.append(row)
    json.dump({"reps": reps, "seed_tag": TAG, "cells": cells, "seconds": round(time.time() - t0)}, open(outp, "w"), indent=1, sort_keys=True)
    print("seconds", round(time.time() - t0))


if __name__ == "__main__":
    main()
