"""T2/T4: error rates and power of the v2 GO procedure and the U1 candidates, known truth.
Usage: python run_sim.py REPS OUT.json"""
import json, sys, time
from multiprocessing import Pool
import random, hashlib
import sim_core
from sim_core import Truth, VARIANTS, nogo, ALPHA, S, DELTA

# saving_k = q - c * Ea/Eb  (Ea/Eb ~ 1/3);  (c, q, r)
DECOYS = [(1, .95, .15), (1, .90, .12), (1, .40, .03), (2, .80, .06), (2, .90, .11)]
SCEN = {
    "feasible":       ([(1, .75, .05)] + DECOYS, "uniform"),
    "boundary_true":  ([(1, .5333, .10), (1, .95, .15), (1, .40, .03), (2, .9, .11)], "uniform"),
    "false_risk":     ([(1, .80, .115), (1, .95, .15), (1, .40, .03), (2, .9, .11)], "uniform"),
    "false_saving":   ([(1, .48, .03), (1, .95, .15), (1, .90, .12), (2, .70, .04)], "uniform"),
    "heavy_feasible": ([(1, .75, .05)] + DECOYS, "lognorm"),
    "heavy_false":    ([(1, .48, .03), (1, .95, .15), (1, .90, .12), (2, .70, .04)], "lognorm"),
    "pi24_feasible":  ([(1, .75, .05)] + DECOYS + [(c, q, r) for c in (1, 2) for q, r in
                       ((.97, .2), (.85, .09 + .0), (.6, .04), (.92, .13), (.7, .07), (.5, .02), (.88, .105),
                        (.99, .25), (.3, .025))], "uniform"),
    # stress multiplicity: many cheap policies just above alpha, safe ones save < s  -> claim false
    "pi24_false":     ([(c, q, r) for c in (1, 2) for q, r in ((.80, .105), (.85, .11), (.75, .104), (.9, .12),
                        (.7, .103), (.95, .13), (.82, .106), (.88, .108))] +
                       [(1, .45, .03), (1, .50, .05), (2, .75, .04), (2, .80, .06), (1, .52, .08), (2, .85, .07),
                        (1, .40, .02), (2, .70, .02)], "uniform"),
}
NS = (150, 300, 600, 1200, 2400)
USE = ("v2", "bonf")
KEYS = ("v2", "v2+U2", "bonf", "bonf+U2")


def one(args):
    name, n, rep = args
    pols, kind = SCEN[name]
    T = TRUTHS[name]
    rng = random.Random(int(hashlib.sha256(f"{name}|{n}|{rep}".encode()).hexdigest()[:12], 16))
    data = T.sample(n, rng)
    out = {}
    for st in (False, True):
        sim_core.STRUCT = st
        ng = nogo(data)
        for v in USE:
            g = VARIANTS[v](data)
            key = v + ("+U2" if st else "")
            out[key] = "INCONCLUSIVE" if (g is not None and ng) else ("GO" if g is not None else ("NO-GO" if ng else "INCONCLUSIVE"))
    return name, n, out


TRUTHS = {k: Truth(p, kind, ratio=10.0) for k, (p, kind) in SCEN.items()}
# usd-like: the small model is 10x cheaper per question (saving = q - 0.1 c), heavy tail
USD = {"usd_feasible": [(1, .45, .05), (1, .90, .15), (1, .20, .03), (2, .60, .11)],
       "usd_false":    [(1, .25, .03), (1, .90, .15), (2, .28, .05)]}
for k, p in USD.items():
    SCEN[k] = (p, "lognorm")
    TRUTHS[k] = Truth(p, "lognorm", ratio=10.0, mean_a=0.3, mean_b=3.0)

if __name__ == "__main__":
    reps = int(sys.argv[1]); outp = sys.argv[2]
    jobs = [(nm, n, r) for nm in SCEN for n in NS for r in range(reps)]
    t0 = time.time()
    agg = {}
    with Pool(2) as p:
        for nm, n, o in p.imap_unordered(one, jobs, chunksize=8):
            for v, verd in o.items():
                agg.setdefault(f"{nm}|{n}|{v}", {"GO": 0, "NO-GO": 0, "INCONCLUSIVE": 0})[verd] += 1
    info = {k: {"claim_true": T.claim(), "best": [(round(T.risk(i), 3), round(T.saving(i), 3)) for i in range(len(T.pols))],
                "c_max_over_Eb": round(T.c_max / T.Eb, 1), "n_policies": len(T.pols)} for k, T in TRUTHS.items()}
    json.dump({"reps": reps, "alpha": ALPHA, "delta": DELTA, "s": S, "rates": agg, "truth": info,
               "seconds": round(time.time() - t0)}, open(outp, "w"), indent=1)
    for k, T in TRUTHS.items():
        print(k, "claim", T.claim(), "c_max/Eb", round(T.c_max / T.Eb, 1))
        for n in NS:
            row = "  n=%5d " % n
            for v in KEYS:
                c = agg[f"{k}|{n}|{v}"]
                row += f" {v}: GO {c['GO']/reps:.3f} NG {c['NO-GO']/reps:.3f} |"
            print(row)
    print("seconds", round(time.time() - t0))
