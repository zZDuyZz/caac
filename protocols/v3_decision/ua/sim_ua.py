"""U-a simulation: validity of the certified saving S* (DECISION_ua_before_run.md). Reuses the data model
and scenarios S1-S8 of ../rel/sim_rel.py unchanged.  python sim_ua.py REPS OUT.json"""
import hashlib, json, os, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "relative"))
import sim_rel as R  # noqa: E402
from sim_rel import _betting_p, _order, DEFAULT_SALT  # noqa: E402

S_GRID = np.array([round(0.02 * i, 2) for i in range(-25, 50)])
EPS, DELTA = R.EPS, R.DELTA
NAMES = [k for k in R.SCEN if R.SCEN[k][2] is None]          # S1-S8 (no chains)


def bet_rows(X, lo):
    """Row-wise _betting_p with a per-row lower bound lo (< 0); same arithmetic as audit._betting_p."""
    P, n = X.shape
    lmax = 0.5 / (-lo)
    logk = np.zeros(P); best = np.zeros(P); s1 = np.zeros(P); s2 = np.zeros(P)
    for m in range(n):
        x = X[:, m]
        if m:
            mu = s1 / m; var = s2 / m - mu * mu
            lam = np.where(mu <= 0, 0.0, np.minimum(lmax, mu / (var + mu * mu + 1e-12)))
        else:
            lam = np.zeros(P)
        logk = logk + np.log1p(lam * x); best = np.maximum(best, logk)
        s1 = s1 + x; s2 = s2 + x * x
    return np.minimum(1.0, np.exp(-best))


def check(k=200):
    rng = np.random.default_rng(7); worst = 0.0
    for _ in range(k):
        n = int(rng.integers(2, 300)); los = -rng.uniform(.2, 3, 3); his = rng.uniform(.2, 3, 3)
        X = np.stack([rng.uniform(los[r], his[r], n) for r in range(3)])
        v = bet_rows(X, los)
        for r in range(3):
            worst = max(worst, abs(v[r] - _betting_p(list(X[r]), los[r], his[r], list(range(n)))))
    print("criterion (b): max diff", worst); assert worst <= 1e-12
    return worst


def one(args):
    name, n, rep = args
    pols, kind, _ = R.SCEN[name]
    seed = int(hashlib.sha256(f"ua|{name}|{n}|{rep}".encode()).hexdigest()[:15], 16)
    rng = np.random.default_rng(seed); P = len(pols)
    A = R.draw_cost(rng, kind, R.MEAN_A, n); B = R.draw_cost(rng, kind, R.MEAN_B, n)
    big_err = rng.random(n) < R.R_B
    D = np.zeros((P, n)); C = np.zeros((P, n))
    for k, (c, q, a, b) in enumerate(pols):
        small = rng.random(n) < q; u = rng.random(n)
        dk = np.where(big_err, np.where(u < a / R.R_B, -1.0, 0.0), np.where(u < b / (1 - R.R_B), 1.0, 0.0))
        D[k] = np.where(small, dk, 0.0); C[k] = c * A + np.where(small, 0.0, B)
    order = np.array(_order([f"q{seed:x}_{i}" for i in range(n)], DEFAULT_SALT))
    D = D[:, order]; C = C[:, order]; Bo = B[order]
    a_max = max(c for c, *_ in pols) * R.cap(kind, R.MEAN_A); b_max = R.cap(kind, R.MEAN_B)
    lvl = DELTA / P
    pr = R.betting_p_rows(EPS - D, EPS - 1, EPS + 1)
    rpass = (pr <= lvl) & (D.mean(axis=1) <= EPS)
    G = len(S_GRID)
    X = np.concatenate([(1 - s) * Bo[None, :] - C for s in S_GRID])               # (G*P, n), block g = s_g
    lo = np.repeat(-(a_max + np.maximum(S_GRID, 0) * b_max), P)
    p = bet_rows(X, lo).reshape(G, P)
    lcb = np.full(P, np.nan)
    for k in range(P):
        for g in range(G):
            if p[g, k] <= lvl:
                lcb[k] = S_GRID[g]
            else:
                break
    # GO at s = .20 (as audit3): risk pass and saving test at s=.20
    g20 = int(np.argmin(abs(S_GRID - 0.20)))
    go20 = bool(np.any(rpass & (p[g20] <= lvl)))
    ok = rpass & ~np.isnan(lcb)
    if not ok.any():
        return name, n, None, go20
    k = int(np.nanargmax(np.where(ok, lcb, -np.inf)))
    return name, n, (k, float(lcb[k])), go20


def run(reps, outp):
    _, rows_t = None, None
    truth = {nm: R.truth(nm)[1] for nm in NAMES}
    jobs = [(nm, n, r) for nm in NAMES for n in R.NS for r in range(reps)]
    agg = {}; t0 = time.time()
    with Pool(2) as pl:
        for nm, n, res, go20 in pl.imap_unordered(one, jobs, chunksize=8):
            a = agg.setdefault(f"{nm}|{n}", {"defined": 0, "false": 0, "sum_S": 0.0, "go20": 0,
                                             "go20_S_below": 0, "S_ge20_no_go": 0})
            a["go20"] += go20
            if res is None:
                continue
            k, S = res; dl, sv = truth[nm][k]
            a["defined"] += 1; a["sum_S"] += S
            a["false"] += int(dl > EPS + 1e-12 or sv < S - 1e-12)
            a["go20_S_below"] += int(go20 and S < 0.20 - 1e-9)
            a["S_ge20_no_go"] += int((not go20) and S >= 0.20 - 1e-9)
    json.dump({"reps": reps, "agg": agg, "truth": truth, "seconds": round(time.time() - t0)}, open(outp, "w"), indent=1)
    for nm in NAMES:
        print(nm, [(round(dl, 4), round(sv, 3)) for dl, sv in truth[nm]])
        for n in R.NS:
            a = agg[f"{nm}|{n}"]
            print(f"  n={n:5d} false {a['false']/reps:.3f} defined {a['defined']/reps:.3f} "
                  f"meanS {a['sum_S']/max(1,a['defined']):.3f} go20 {a['go20']/reps:.3f} "
                  f"go20&S<.2 {a['go20_S_below']} S>=.2&noGO {a['S_ge20_no_go']}")
    print("seconds", round(time.time() - t0))


if __name__ == "__main__":
    check()
    run(int(sys.argv[1]), sys.argv[2])
