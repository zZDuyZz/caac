"""W2 + W3 + X0 for the v3-9 variants. Usage: python v39_w2.py POOLS MAP OUT_DIR [workers] [limit]"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import v39_methods as V
import bl_w2 as W2
import bl_methods as BM
from caac.pool_v1.audit import _order, structural_range
from caac.pool_v1.replay import replay

M4 = W2.M4
N_SUBS, REPS = W2.N_SUBS, W2.REPS


def extract(x):
    sub, meta, cfg, F = M4._build(x)
    rr = replay(sub, meta, cfg)
    inp = rr.audit_input
    order = _order(inp.qids, cfg.salt)
    be = np.array([bool(v) for v in inp.big_error], dtype=np.int8)
    B = np.array(inp.big_cost, dtype=float)
    D = np.array([[int(bool(v)) for v in p.error] for p in inp.policies], dtype=np.int8) - be[None, :]
    C = np.array([p.cost for p in inp.policies], dtype=float)
    d = (1 - cfg.s) * B[None, :] - C
    lo, hi = structural_range(inp.small_max, inp.big_max, cfg.s)
    lo_pol = np.array([-(p.small_cap + cfg.s * inp.big_max) if p.small_cap is not None else lo for p in inp.policies])
    assert (lo_pol >= lo - 1e-9).all()
    for k, p in enumerate(inp.policies):                       # structural bound must contain every observation
        if p.small_cap is not None:
            assert max(p.small_part) <= p.small_cap * (1 + 1e-9), (x, p.name)
            assert d[k].min() >= lo_pol[k] - 1e-6 * abs(lo_pol[k]), (x, p.name)
    return D[:, order], d[:, order], C[:, order], lo, hi, lo_pol, [p.name for p in inp.policies], cfg


def run_config(x):
    t0 = time.time()
    D, d, C, lo, hi, lo_pol, names, cfg = extract(x)
    P, N = D.shape
    Df = D.astype(float)
    truth_D = D.sum(axis=1) / N; truth_d = d.mean(axis=1)
    feas = (truth_D <= BM.EPS + 1e-12) & (truth_d >= -1e-9 * max(1.0, hi))
    full = V.run_variants(Df[None], d[None], C.mean(axis=1)[None], lo_pol, lo, hi)
    res = {"config": M4.config_id(x), "n_full": N, "n_policies": P, "map_verdict": x["verdict"], "any_feasible": bool(feas.any()),
           "w3": {v: [str(o[0][0]), names[o[1][0]] if o[1][0] >= 0 else None,
                      (bool(feas[o[1][0]]) if o[1][0] >= 0 else None)] for v, o in full.items()},
           "x0_ok": bool(str(full["V0"][0][0]) == x["verdict"] and (names[full["V0"][1][0]] if full["V0"][1][0] >= 0 else None) == x.get("chosen")),
           "w2": {}}
    for n in N_SUBS:
        idx = np.stack([np.random.default_rng(W2.seed_of("BL", res["config"], n, r)).integers(0, N, n) for r in range(REPS)])
        Ds = np.transpose(Df[:, idx], (1, 0, 2)); ds = np.transpose(d[:, idx], (1, 0, 2))
        mc = np.transpose(C[:, idx], (1, 0, 2)).mean(axis=-1)
        out = V.run_variants(Ds, ds, mc, lo_pol, lo, hi)
        cell = {}
        for v, (verd, ch) in out.items():
            go = ch >= 0
            viol = go & ~feas[np.maximum(ch, 0)]
            cell[v] = [int(go.sum()), int(viol.sum()), int((go & ~viol).sum()), int((verd == "NO-GO").sum())]
        res["w2"][str(n)] = cell
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    pools, mp, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    lock = os.path.join(BM.REPO, "protocols", "audit_lock_v3.json")
    configs = M4.select(mp)
    if limit:
        configs = configs[:limit]
    out.mkdir(parents=True, exist_ok=True)
    hdr = {"n_configs": len(configs), "n_subs": N_SUBS, "reps": REPS, "map_sha256": M4.sha256_file(mp),
           "design_sha256": M4.sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DECISION_v39_before_run.md")),
           "scripts": {f: M4.sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                       for f in ("v39_methods.py", "v39_w2.py", "v39_w1.py", "v39_ref.py")}}
    json.dump(hdr, open(out / "v39_w2_header.json", "w"), indent=1, sort_keys=True)
    t0, res = time.time(), []
    with Pool(workers, initializer=W2._init, initargs=(pools, lock)) as Pl:
        for i, r in enumerate(Pl.imap_unordered(run_config, configs, chunksize=1), 1):
            res.append(r)
            if not r["x0_ok"]:
                print("X0 MISMATCH", r["config"], flush=True)
            if i % 100 == 0 or i == len(configs):
                el = time.time() - t0
                print(f"{i}/{len(configs)} {el/60:.1f} min eta {el/i*(len(configs)-i)/60:.0f} min", flush=True)
    res.sort(key=lambda r: r["config"])
    with open(out / "v39_w2.jsonl", "w", newline="\n") as f:
        for r in res:
            r = dict(r); r.pop("seconds", None)
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print("done", len(res), "X0 mismatches", sum(not r["x0_ok"] for r in res))


if __name__ == "__main__":
    main()
