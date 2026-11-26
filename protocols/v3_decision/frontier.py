"""U3: certified frontier. For every policy pi in Pi, on ALL questions:
  UCB_risk(pi)   exact one-sided Clopper-Pearson upper bound, level delta/(2|Pi|)
  LCB_saving(pi) largest s on a fixed grid (ascending, fixed-sequence) whose saving test (betting on
                 d_s = (1-s)B - C, structural range) rejects 'saving < s' at level delta/(2|Pi|)
Simultaneous guarantee (union bound over 2|Pi| bounds): with prob >= 1 - delta, every pi has
R_pi <= UCB and saving_pi >= LCB. Hence every (alpha, s) with some pi having UCB <= alpha and
LCB >= s is certified at once: the certified frontier F(alpha) = max{LCB_pi : UCB_pi <= alpha}.
The fixed-sequence over the s-grid needs no correction (nested nulls, stop at first non-rejection).
Usage: python frontier.py REPS"""
import json, math, random, hashlib, sys
from multiprocessing import Pool
import sim_core
from sim_core import Truth, d_range, _betting_p, _order, ALPHA, DELTA, S

sys.path.insert(0, "../pchk6/src")
from caac.pool_v1.audit import binom_cdf

GRID = [round(0.02 * i, 2) for i in range(-25, 50)]     # s from -0.50 to 0.98


def ucb_risk(e, n, level):
    """Smallest r with P(Bin(n, r) <= e) <= level (one-sided exact upper bound); 1 if n = 0."""
    if n == 0:
        return 1.0
    lo, hi = e / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(e, n, mid) <= level:
            hi = mid
        else:
            lo = mid
    return hi


def lcb_saving(data, k, level, order):
    n = len(data["qids"])
    cost = data["pols"][k][2]
    best = None
    for s in GRID:
        d = [(1 - s) * data["big"][i] - cost[i] for i in range(n)]
        if s >= 0:
            lo, hi = d_range(data, s, True)
        else:       # s < 0: escalated d = -sB - a >= -a_max; served d <= (1-s) b_max
            lo, hi = -data["a_part_max"], (1 - s) * data["b_part_max"]
        if _betting_p(d, lo, hi, order) <= level:
            best = s
        else:
            break
    return best


def certify(data, delta=DELTA):
    m = len(data["pols"]); lev = delta / (2 * m)
    order = _order(data["qids"], "caac-audit-v1")
    out = []
    for k, (served, err, cost) in enumerate(data["pols"]):
        u = ucb_risk(sum(err), sum(served), lev)
        l = lcb_saving(data, k, lev, order) if u < 0.5 else None    # skip hopeless (saves time; no effect on validity)
        out.append((u, l))
    return out


def frontier(cert, alpha):
    vals = [l for u, l in cert if u <= alpha and l is not None]
    return max(vals) if vals else None


DECOYS = [(1, .95, .15), (1, .90, .12), (1, .40, .03), (2, .80, .06), (2, .90, .11)]
SC = {"feasible": Truth([(1, .75, .05)] + DECOYS, "uniform", ratio=10.0),
      "false_saving": Truth([(1, .48, .03), (1, .95, .15), (1, .90, .12), (2, .70, .04)], "uniform", ratio=10.0),
      "usd_feasible": Truth([(1, .45, .05), (1, .90, .15), (1, .20, .03), (2, .60, .11)], "lognorm",
                            ratio=10.0, mean_a=0.3, mean_b=3.0)}


def one(args):
    name, n, rep = args
    T = SC[name]
    rng = random.Random(int(hashlib.sha256(f"fr|{name}|{n}|{rep}".encode()).hexdigest()[:12], 16))
    data = T.sample(n, rng)
    cert = certify(data)
    viol = any((u < T.risk(k)) or (l is not None and l > T.saving(k) + 1e-12) for k, (u, l) in enumerate(cert))
    return name, n, viol, frontier(cert, ALPHA), {a: frontier(cert, a) for a in (0.05, 0.10, 0.20)}


if __name__ == "__main__":
    reps = int(sys.argv[1])
    jobs = [(nm, n, r) for nm in SC for n in (600, 2400) for r in range(reps)]
    agg = {}
    with Pool(2) as p:
        for nm, n, viol, f10, fa in p.imap_unordered(one, jobs, chunksize=2):
            a = agg.setdefault(f"{nm}|{n}", {"reps": 0, "violations": 0, "go_at_main_cell": 0, "frontier": {0.05: [], 0.10: [], 0.20: []}})
            a["reps"] += 1; a["violations"] += viol
            a["go_at_main_cell"] += (f10 is not None and f10 >= S)
            for al, v in fa.items():
                a["frontier"][al].append(v if v is not None else float("nan"))
    out = {}
    for key, a in sorted(agg.items()):
        T = SC[key.split("|")[0]]
        fr = {}
        for al, v in a["frontier"].items():
            vv = sorted(x for x in v if x == x)
            fr[str(al)] = {"certified_share": round(len(vv) / a["reps"], 3), "median_s": vv[len(vv) // 2] if vv else None}
        best_true = {str(al): max([T.saving(k) for k in range(len(T.pols)) if T.risk(k) <= al] or [None]) for al in (0.05, 0.10, 0.20)}
        out[key] = {"reps": a["reps"], "P(any bound wrong)": round(a["violations"] / a["reps"], 3),
                    "GO at (0.10, 0.20) via frontier": round(a["go_at_main_cell"] / a["reps"], 3),
                    "frontier": fr, "true_frontier": best_true}
        print(key, json.dumps(out[key]), flush=True)
    json.dump(out, open("frontier.json", "w"), indent=1)
