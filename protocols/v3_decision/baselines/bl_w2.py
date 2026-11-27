"""W2 plasmode on real data + W3 full-data decisions + check X0 (design: DECISION_bl_before_run.md).
Usage: python bl_w2.py POOLS_DIR MAP_REL_JSONL OUT_DIR [workers] [limit]
Output: OUT_DIR/bl_w2.jsonl (one line per configuration, sorted), OUT_DIR/bl_w2_header.json."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import bl_methods as M
from caac.pool_v1.audit import _order, structural_range
from caac.pool_v1.replay import replay

sys.path.insert(0, os.path.join(M.REPO, "protocols", "v3_decision", "m4"))
import m4_run as M4                                    # noqa: E402  (same configuration builder as M4)

N_SUBS = (250, 500, 1000)
REPS = 200


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


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
    names = [p.name for p in inp.policies]
    return D[:, order], d[:, order], C[:, order], lo, hi, names, cfg


def run_config(x):
    t0 = time.time()
    D, d, C, lo, hi, names, cfg = extract(x)
    assert abs(cfg.eps - M.EPS) < 1e-12 and abs(cfg.s - M.S) < 1e-12 and abs(cfg.delta - M.DELTA) < 1e-12
    P, N = D.shape
    Df = D.astype(float)
    # X0 + W3: every method on the full data (official order, official mean cost for the cheapest)
    full = M.run_all(Df[None], d[None], C.mean(axis=1)[None], lo, hi)
    v0 = str(full["caac_verdict"][0])
    ch0 = names[full["caac"][0]] if full["caac"][0] >= 0 else None
    x0_ok = v0 == x["verdict"] and ch0 == x.get("chosen")
    truth_D = D.sum(axis=1) / N
    truth_d = d.mean(axis=1)
    feas = (truth_D <= M.EPS + 1e-12) & (truth_d >= -1e-9 * max(1.0, hi))
    res = {"config": M4.config_id(x), "tier": x["tier"], "n_full": N, "n_policies": P, "map_verdict": x["verdict"],
           "x0_verdict": v0, "x0_chosen": ch0, "x0_ok": bool(x0_ok), "any_feasible": bool(feas.any()),
           "n_feasible": int(feas.sum()),
           "w3": {m: (names[full[m][0]] if full[m][0] >= 0 else None) for m in M.METHODS},
           "w3_feasible": {m: (bool(feas[full[m][0]]) if full[m][0] >= 0 else None) for m in M.METHODS},
           "w2": {}}
    for n in N_SUBS:
        idx = np.stack([np.random.default_rng(seed_of("BL", res["config"], n, r)).integers(0, N, n) for r in range(REPS)])
        Ds = np.transpose(Df[:, idx], (1, 0, 2))           # (R, P, n)
        ds = np.transpose(d[:, idx], (1, 0, 2))
        mc = np.transpose(C[:, idx], (1, 0, 2)).mean(axis=-1)
        out = M.run_all(Ds, ds, mc, lo, hi)
        cell = {}
        for m in M.METHODS:
            ch = out[m]
            go = ch >= 0
            viol = go & ~feas[np.maximum(ch, 0)]
            cell[m] = [int(go.sum()), int(viol.sum()), int((go & ~viol).sum())]     # go, false_go, power
        v = out["caac_verdict"]
        cell["caac_nogo"] = int((v == "NO-GO").sum())
        cell["caac_inc"] = int((v == "INCONCLUSIVE").sum())
        res["w2"][str(n)] = cell
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _init(pools, lock):
    M4._init(pools, lock)


def main():
    pools, mp, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    lock = os.path.join(M.REPO, "protocols", "audit_lock_v3.json")
    configs = M4.select(mp)
    if limit:
        configs = configs[:limit]
    out.mkdir(parents=True, exist_ok=True)
    hdr = {"n_configs": len(configs), "n_subs": N_SUBS, "reps": REPS, "map_sha256": M4.sha256_file(mp),
           "lock_sha256_file": M4.sha256_file(lock),
           "scripts": {f: M4.sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                       for f in ("bl_methods.py", "bl_w2.py", "bl_ref.py", "bl_w1.py")},
           "design_sha256": M4.sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DECISION_bl_before_run.md"))}
    json.dump(hdr, open(out / "bl_w2_header.json", "w"), indent=1, sort_keys=True)
    t0, res = time.time(), []
    part = open(out / "bl_w2.partial.jsonl", "w")
    with Pool(workers, initializer=_init, initargs=(pools, lock)) as P:
        for i, r in enumerate(P.imap_unordered(run_config, configs, chunksize=1), 1):
            res.append(r); part.write(json.dumps(r, sort_keys=True) + "\n"); part.flush()
            if not r["x0_ok"]:
                print("X0 MISMATCH", r["config"], r["map_verdict"], r["x0_verdict"], r["x0_chosen"], flush=True)
            if i % 100 == 0 or i == len(configs):
                el = time.time() - t0
                print(f"{i}/{len(configs)} {el/60:.1f} min eta {el/i*(len(configs)-i)/60:.0f} min", flush=True)
    part.close()
    res.sort(key=lambda r: r["config"])
    with open(out / "bl_w2.jsonl", "w", newline="\n") as f:
        for r in res:
            r = dict(r); r.pop("seconds", None)
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print("done", len(res), "X0 mismatches", sum(not r["x0_ok"] for r in res))


if __name__ == "__main__":
    main()
