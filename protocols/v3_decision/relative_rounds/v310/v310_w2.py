"""W2 + W3 + X0 for the v3-10 candidates. Usage: python v310_w2.py POOLS MAP OUT_DIR SEED_TAG [workers] [limit]  (dev: BL, confirmation: BL2)"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, "../bl")
import v310_methods as C
import v39_w2 as X
import bl_w2 as W2
import bl_methods as BM

M4 = W2.M4
TAG = "BL"


def run_config(x):
    t0 = time.time()
    D, d, Cc, lo, hi, lo_pol, names, cfg = X.extract(x)
    P, N = D.shape
    Df = D.astype(float)
    truth_D = D.sum(axis=1) / N; truth_d = d.mean(axis=1)
    feas = (truth_D <= BM.EPS + 1e-12) & (truth_d >= -1e-9 * max(1.0, hi))
    full = C.run_cands(Df[None], d[None], Cc.mean(axis=1)[None], lo_pol, lo, hi)
    res = {"config": M4.config_id(x), "pool": x["pool"], "n_full": N, "n_policies": P, "map_verdict": x["verdict"],
           "any_feasible": bool(feas.any()), "seed_tag": TAG,
           "w3": {v: [str(o[0][0]), names[o[1][0]] if o[1][0] >= 0 else None,
                      (bool(feas[o[1][0]]) if o[1][0] >= 0 else None)] for v, o in full.items()},
           "x0_ok": bool(str(full["V0"][0][0]) == x["verdict"] and (names[full["V0"][1][0]] if full["V0"][1][0] >= 0 else None) == x.get("chosen")),
           "w2": {}}
    for n in X.N_SUBS:
        idx = np.stack([np.random.default_rng(W2.seed_of(TAG, res["config"], n, r)).integers(0, N, n) for r in range(X.REPS)])
        Ds = np.transpose(Df[:, idx], (1, 0, 2)); ds = np.transpose(d[:, idx], (1, 0, 2))
        mc = np.transpose(Cc[:, idx], (1, 0, 2)).mean(axis=-1)
        out = C.run_cands(Ds, ds, mc, lo_pol, lo, hi)
        cell = {}
        for v, (verd, ch) in out.items():
            go = ch >= 0
            viol = go & ~feas[np.maximum(ch, 0)]
            cell[v] = [int(go.sum()), int(viol.sum()), int((go & ~viol).sum()), int((verd == "NO-GO").sum())]
        res["w2"][str(n)] = cell
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _init(pools, lock, tag):
    global TAG
    TAG = tag
    W2._init(pools, lock)


def main():
    pools, mp, out, tag = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    lock = os.path.join(BM.REPO, "protocols", "audit_lock_v3.json")
    configs = M4.select(mp)
    if limit:
        configs = configs[:limit]
    out.mkdir(parents=True, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump({"n_configs": len(configs), "seed_tag": tag, "map_sha256": M4.sha256_file(mp),
               "design_sha256": M4.sha256_file(os.path.join(here, "DECISION_v310_before_run.md")),
               "scripts": {f: M4.sha256_file(os.path.join(here, f)) for f in ("v310_methods.py", "v310_w2.py", "v310_w1.py", "v39_methods.py", "v39_ref.py")}},
              open(out / "v310_w2_header.json", "w"), indent=1, sort_keys=True)
    t0, res = time.time(), []
    with Pool(workers, initializer=_init, initargs=(pools, lock, tag)) as Pl:
        for i, r in enumerate(Pl.imap_unordered(run_config, configs, chunksize=1), 1):
            res.append(r)
            if not r["x0_ok"]:
                print("X0 MISMATCH", r["config"], flush=True)
            if i % 100 == 0 or i == len(configs):
                el = time.time() - t0
                print(f"{i}/{len(configs)} {el/60:.1f} min eta {el/i*(len(configs)-i)/60:.0f} min", flush=True)
    res.sort(key=lambda r: r["config"])
    with open(out / "v310_w2.jsonl", "w", newline="\n") as f:
        for r in res:
            r = dict(r); r.pop("seconds", None)
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print("done", len(res), "X0 mismatches", sum(not r["x0_ok"] for r in res))


if __name__ == "__main__":
    main()
