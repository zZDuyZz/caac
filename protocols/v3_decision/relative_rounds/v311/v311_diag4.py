"""POST-HOC DIAGNOSTIC for v3-11 planning (not a test of any claim). DEV pools and DEV seed tag only ('DG', new, never
used for selection or confirmation). Decomposes the power gap between H1 (v3-10 chosen) and CLT + Bonferroni.
Usage: python v311_diag.py POOLS MAP DEV_W2_JSONL OUT.json [workers] [reps]"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bl"))
import v39_methods as V
import v39_w2 as X
import bl_w2 as W2
import bl_methods as BM
EU = {n: np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eu", f"T_{n}.npy")) for n in (250, 500, 1000)}

M4 = W2.M4
HELD = ("lrb_medqa_test", "lrb_winogrande_valid", "lurume_mmlu_pro")
EPS, DELTA = BM.EPS, BM.DELTA
REPS = 100
NS = (250, 500, 1000)


def bet_custom(X_, lo, hi, cap, lamrule, alpha=None, lam_const=None):
    """Betting p-value with Ville max; lamrule in {'agrapa','wsr','const'}. lo per-row."""
    N, n = X_.shape
    lo = np.broadcast_to(np.asarray(lo, float), (N,))
    lmax = cap / np.abs(lo)
    logk = np.zeros(N); best = np.zeros(N); s1 = np.zeros(N); s2 = np.zeros(N)
    for m in range(n):
        x = X_[:, m]
        if lamrule == "const":
            lam = lam_const
        elif m == 0:
            lam = np.zeros(N) if lamrule == "agrapa" else np.minimum(lmax, np.sqrt(2 * np.log(1 / alpha) / (n * (hi - lo) ** 2 / 4)))
        else:
            mu = s1 / m; var = s2 / m - mu * mu
            if lamrule == "agrapa":
                lam = np.where(mu <= 0, 0.0, np.minimum(lmax, mu / (var + mu * mu + 1e-12)))
            else:   # fixed-n predictable plug-in (Waudby-Smith & Ramdas), variance shrunk to a prior of range^2/4
                v = ((hi - lo) ** 2 / 4 + s2 - m * mu * mu) / (m + 1)
                lam = np.minimum(lmax, np.sqrt(2 * np.log(1 / alpha) / (n * v)))
        logk = logk + np.log1p(lam * x)
        best = np.maximum(best, logk)
        s1 += x; s2 += x * x
    return np.minimum(1.0, np.exp(-best))


def const_lam(x_pop, lmax):
    g = np.linspace(0, 1, 201)[1:] * lmax
    val = [np.mean(np.log1p(l * x_pop)) for l in g]
    return g[int(np.argmax(val))]


def run_config(x):
    D, d, Cc, lo, hi, lo_pol, names, cfg = X.extract(x)
    P, N = D.shape
    Df = D.astype(float)
    tD = D.sum(1) / N; td = d.mean(1)
    feas = (tD <= EPS + 1e-12) & (td >= -1e-9 * max(1.0, hi))
    if not feas.any():
        return None
    mcost = Cc.mean(1)
    best = int(np.argmin(np.where(feas, mcost, np.inf)))
    lamc = np.array([const_lam(d[k], 0.9 / abs(lo_pol[k])) for k in range(P)])
    out = {"config": M4.config_id(x), "pool": x["pool"], "P": P, "n_feasible": int(feas.sum())}
    for n in NS:
        idx = np.stack([np.random.default_rng(W2.seed_of("DG4", out["config"], n, r)).integers(0, N, n) for r in range(REPS)])
        Ds = np.transpose(Df[:, idx], (1, 0, 2)); ds = np.transpose(d[:, idx], (1, 0, 2))
        mc = np.transpose(Cc[:, idx], (1, 0, 2)).mean(-1)
        R_ = REPS
        D2 = Ds.reshape(R_ * P, n); d2 = ds.reshape(R_ * P, n)
        lo_rows = np.tile(lo_pol, R_)
        mD = Ds.mean(-1)
        q_b = V.betting_rows(EPS - D2, EPS - 1, EPS + 1).reshape(R_, P)
        q_x, _ = V.xc_p(Ds, EPS)
        q_h = np.minimum(1.0, 2 * np.minimum(q_b, q_x))
        gate = mD <= EPS
        c_ag = np.where(gate, bet_custom(d2, lo_rows, hi, 0.9, "agrapa").reshape(R_, P), 1.0)
        c_ws = {a: np.where(gate, bet_custom(d2, lo_rows, hi, 0.9, "wsr", alpha=a).reshape(R_, P), 1.0) for a in (DELTA / P, DELTA)}
        c_or = np.where(gate, bet_custom(d2, lo_rows, hi, 0.9, "const", lam_const=np.tile(lamc, R_)).reshape(R_, P), 1.0)
        # CLT + Bonferroni on the same draws
        sD = Ds.std(-1, ddof=1); sd_ = ds.std(-1, ddof=1); md = ds.mean(-1)
        def clt(a):
            z = norm.ppf(1 - a)
            return (mD + z * sD / np.sqrt(n) <= EPS) & (md - z * sd_ / np.sqrt(n) >= 0)
        def pw(passes):
            ch = V._cheapest(passes, mc)
            return int(((ch >= 0) & feas[np.maximum(ch, 0)]).sum())
        a = DELTA / P
        qrows = EPS - D2
        qw = {cap: bet_custom(qrows, np.full(R_ * P, EPS - 1), EPS + 1, cap, "wsr", alpha=a).reshape(R_, P) for cap in (0.5, 0.9)}
        qa9 = bet_custom(qrows, np.full(R_ * P, EPS - 1), EPS + 1, 0.9, "agrapa").reshape(R_, P)
        qx = {g: V.xc_p(Ds, EPS, g)[0] for g in (1e-3, a / 10)}
        cw = np.where(gate, bet_custom(d2, lo_rows, hi, 0.9, "wsr", alpha=a).reshape(R_, P), 1.0)
        def h2(*ps):
            k = len(ps); return np.minimum(1.0, k * np.minimum.reduce(ps))
        Hc = (Ds > 0).sum(-1); Bc = (Ds < 0).sum(-1)
        qeu = EU[n][Hc, Bc]
        cell = {
            "H1": pw((q_h <= a) & (c_ag <= a)),
            "Q_wsr9_Cwsr": pw((qw[0.9] <= a) & (cw <= a)),
            "Q_eu_Cag": pw((qeu <= a) & (c_ag <= a)),
            "Q_eu_Cwsr": pw((qeu <= a) & (cw <= a)),
            "Q_hyb_eu_wsr9_Cwsr": pw((h2(qeu, qw[0.9]) <= a) & (cw <= a)),
            "Qclt_Cag": pw((mD + norm.ppf(1 - a) * sD / np.sqrt(n) <= EPS) & (c_ag <= a)),
            "Qclt_Cwsr": pw((mD + norm.ppf(1 - a) * sD / np.sqrt(n) <= EPS) & (cw <= a)),
            "cltB": pw(clt(a)),
        }
        out[str(n)] = cell
    return out


def _init(pools, lock):
    W2._init(pools, lock)


def main():
    pools, mp, devj, outp = sys.argv[1:5]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    global REPS
    if len(sys.argv) > 6:
        REPS = int(sys.argv[6])
    keep = {json.loads(l)["config"] for l in open(devj) if json.loads(l)["any_feasible"] and json.loads(l)["pool"] not in HELD}
    configs = [c for c in M4.select(mp) if M4.config_id(c) in keep]
    lock = os.path.join(BM.REPO, "protocols", "audit_lock_v3.json")
    t0 = time.time(); res = []
    with Pool(workers, initializer=_init, initargs=(pools, lock)) as Pl:
        for i, r in enumerate(Pl.imap_unordered(run_config, configs), 1):
            if r: res.append(r)
            if i % 50 == 0: print(i, len(configs), round((time.time() - t0) / 60, 1), "min", flush=True)
    res.sort(key=lambda r: r["config"])
    tot = {n: {} for n in map(str, NS)}
    for r in res:
        for n in tot:
            for k, v in r[n].items():
                tot[n][k] = tot[n].get(k, 0) + v
    summ = {n: {k: round(v / (REPS * len(res)), 4) for k, v in t.items()} for n, t in tot.items()}
    json.dump({"configs": len(res), "reps": REPS, "summary": summ, "rows": res}, open(outp, "w"), indent=1)
    for n, s in summ.items():
        print(n, s)


if __name__ == "__main__":
    main()
