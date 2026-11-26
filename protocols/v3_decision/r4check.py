"""T2 test of the planning estimate: 'collect the questions the report says, and the GO comes'.
Pilot of n0 questions (known truth). (a) v2's R4: risk leg only, point estimate, split design;
followed by the v2 procedure. (b) the evidence gap: both legs, power .8, no-split design at
delta/|Pi| with the structural range; followed by the no-split procedure (bonf+U2).
Fresh sample of the stated size from the same truth; report how often the GO actually comes.
Usage: python r4check.py REPS"""
import json, math, random, hashlib, sys
from multiprocessing import Pool
import sim_core
from sim_core import Truth, split, candidates_on, go_v2, go_bonf, d_range, ALPHA, DELTA, S
from gap import gap
sys.path.insert(0, "../pchk6/src")
from caac.pool_v1.audit import served_needed

DECOYS = [(1, .95, .15), (1, .90, .12), (1, .40, .03), (2, .80, .06), (2, .90, .11)]
SC = {"feasible": Truth([(1, .75, .05)] + DECOYS, "uniform", ratio=10.0),
      "tight": Truth([(1, .62, .075)] + DECOYS, "uniform", ratio=10.0),
      "usd_feasible": Truth([(1, .45, .05), (1, .90, .15), (1, .20, .03), (2, .60, .11)], "lognorm",
                            ratio=10.0, mean_a=0.3, mean_b=3.0)}
CAP = 30_000


def old_r4(data):
    A, B = split(data)
    c = candidates_on(data, A, ALPHA)
    if not c:
        return None
    served, err, _ = data["pols"][c[0][1]]
    sv = sum(served[i] for i in B); e = sum(err[i] for i in B)
    sn = served_needed(e / sv, ALPHA, DELTA) if sv else None
    return math.ceil(2 * sn / (sv / len(B))) if sn else None


def new_gap(data):
    n = len(data["qids"]); m = len(data["pols"])
    lo, _ = d_range(data, S, True)
    best = None
    for k, (served, err, cost) in enumerate(data["pols"]):
        sv, e = sum(served), sum(err)
        d = [(1 - S) * data["big"][i] - cost[i] for i in range(n)]
        if not sv or e / sv >= ALPHA or sum(d) <= 0:
            continue
        g = gap(e, sv, n, d, lo, ALPHA, DELTA / m, 0.8, split_factor=1)
        if g["reachable"] and (best is None or g["questions_needed"] < best["questions_needed"]):
            best = g
    return best


def one(args):
    name, n0, rep = args
    T = SC[name]
    rng = random.Random(int(hashlib.sha256(f"r4|{name}|{n0}|{rep}".encode()).hexdigest()[:12], 16))
    data = T.sample(n0, rng)
    res = {"old": old_r4(data)}
    g = new_gap(data)
    res["new"] = g["questions_needed"] if g else None
    res["blocking"] = g["blocking"] if g else None
    for tag, fn, st in (("old", go_v2, False), ("new", go_bonf, True)):
        N = res[tag]
        if N is None or N > CAP:
            res[tag + "_go"] = None
            continue
        fresh = T.sample(max(N, 2), random.Random(rng.getrandbits(32)), prefix="f")
        sim_core.STRUCT = st
        res[tag + "_go"] = fn(fresh) is not None
    return name, n0, res


if __name__ == "__main__":
    reps = int(sys.argv[1])
    jobs = [(nm, n0, r) for nm in SC for n0 in (150, 400) for r in range(reps)]
    agg = {}
    with Pool(2) as p:
        for nm, n0, r in p.imap_unordered(one, jobs, chunksize=2):
            agg.setdefault(f"{nm}|{n0}", []).append(r)
    summ = {}
    for key in sorted(agg):
        rs = agg[key]
        s = {"pilots": len(rs)}
        for tag in ("old", "new"):
            got = [r[tag + "_go"] for r in rs if r[tag + "_go"] is not None]
            Ns = sorted(r[tag] for r in rs if r[tag])
            s[tag] = {"pilots_with_number": len(Ns), "median_N": Ns[len(Ns) // 2] if Ns else None,
                      "GO_rate_when_followed": round(sum(got) / len(got), 3) if got else None}
        s["new_blocking_saving"] = sum(1 for r in rs if r["blocking"] == "saving")
        summ[key] = s
        print(key, json.dumps(s), flush=True)
    json.dump(summ, open("r4check.json", "w"), indent=1)
