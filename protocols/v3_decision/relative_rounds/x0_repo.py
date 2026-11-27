"""X0-repo: the integrated repository audit3 reproduces, on the FULL data of every configuration,
  rel_test='bet' (legacy)  -> the official map verdict and chosen policy (lock v3-8),
  rel_mode='anytime'       -> B1 of the v3-10 experiments, recomputed here with the vectorised v3-10 code on the rows in
                              the order given (the anytime mode tests in arrival order; deviation v3-11-D2),
  rel_mode='fixed_n'       -> F2 of the v3-11 experiments (W2_V311_JSONL).
Usage: python x0_repo.py REPO POOLS MAP W2_V311_JSONL OUT.jsonl [workers] [limit]
(needs scipy and, on the path, protocols/v3_decision/baselines and relative_rounds/v310)"""
import json, os, sys, time
from multiprocessing import Pool

REPO = sys.argv[1]
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "protocols", "v3_decision", "m4"))
import m4_run as M4                                     # noqa: E402
from caac.pool_v1.audit import audit3                   # noqa: E402
from caac.pool_v1.replay import replay                  # noqa: E402
from caac.pool_v1.audit import policy_lo, structural_range   # noqa: E402
import numpy as np                                      # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "protocols", "v3_decision", "baselines"))
sys.path.insert(0, os.path.join(HERE, "v310"))
import v310_methods as C10                              # noqa: E402

EXP = {}


def run(x):
    sub, meta, cfg, F = M4._build(x)
    inp = replay(sub, meta, cfg).audit_input
    cid = M4.config_id(x)
    out = {"config": cid}
    kw = dict(alpha=cfg.alpha, delta=cfg.delta, s=cfg.s, salt=cfg.salt, claim="relative", eps=cfg.eps)
    for tag, extra in (("V0", {"rel_test": "bet"}), ("B1", {"rel_mode": "anytime"}), ("F2", {"rel_mode": "fixed_n"})):
        r = audit3(inp, **kw, **extra)
        out[tag] = [r.verdict, r.chosen]
    e = EXP[cid]
    n = len(inp.qids)
    be = np.array([int(bool(v)) for v in inp.big_error])
    D = np.array([[int(bool(v)) for v in p.error] for p in inp.policies]) - be[None, :]
    B = np.array(inp.big_cost, float); Cc = np.array([p.cost for p in inp.policies], float)
    d = (1 - cfg.s) * B[None, :] - Cc
    lo, hi = structural_range(inp.small_max, inp.big_max, cfg.s)
    lop = np.array([policy_lo(inp, p, cfg.s) for p in inp.policies])
    v, ch = C10.run_cands(D.astype(float)[None], d[None], Cc.mean(1)[None], lop, lo, hi, names=("B1",))["B1"]
    b1 = [str(v[0]), inp.policies[ch[0]].name if ch[0] >= 0 else None]
    g = lambda v: [v[0], v[1] if v[0] == "GO" else None]      # experiments record a policy only for GO (repo also on conflict)
    out["ok"] = {"V0": out["V0"] == [x["verdict"], x.get("chosen")],
                 "B1": g(out["B1"]) == b1, "F2": g(out["F2"]) == e["F2"][:2]}
    return out


def _init(pools, lock, w2):
    M4._init(pools, lock)
    for l in open(w2):
        r = json.loads(l)
        EXP[r["config"]] = r["w3"]


def main():
    pools, mp, w2, outp = sys.argv[2:6]
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    limit = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    configs = M4.select(mp)
    if limit:
        import random
        random.Random(7).shuffle(configs); configs = configs[:limit]
    lock = os.path.join(REPO, "protocols", "audit_lock_v3.json")
    t0 = time.time(); bad = {"V0": 0, "B1": 0, "F2": 0}; n = 0
    with Pool(workers, initializer=_init, initargs=(pools, lock, w2)) as P, open(outp, "w") as f:
        for r in P.imap_unordered(run, configs, chunksize=1):
            n += 1
            for k, v in r["ok"].items():
                if not v:
                    bad[k] += 1; print("MISMATCH", k, r["config"], r[k], flush=True)
            f.write(json.dumps(r, sort_keys=True) + "\n")
            if n % 100 == 0:
                print(n, len(configs), round((time.time() - t0) / 60, 1), "min", flush=True)
    print("done", n, "mismatches", bad, "minutes", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
