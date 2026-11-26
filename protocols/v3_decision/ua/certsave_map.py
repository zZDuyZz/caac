"""Run certified saving (U-a) on every configuration, both claims, ONCE. Recomputes audit3 first and requires
its verdict to equal the map.jsonl verdict of the official run.  python certsave_map.py POOLS LOCK OUT MAPS..."""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
from caac.pool_v1.audit import audit3
from caac.pool_v1.certsave import certified_saving
from caac.pool_v1.config import config_from_dict
from caac.pool_v1.crossmap import common_questions, included, measures_for, pairs_for, policy_grid
from caac.pool_v1.io import load_pool_v1
from caac.pool_v1.lock import load_lock
from caac.pool_v1.replay import replay

POOLS, LOCK, OUT = sys.argv[1:4]
MAPS = sys.argv[4:]
REF = {}
for mp in MAPS:
    for l in open(mp):
        x = json.loads(l)
        REF[(x["claim"], x["pool"], x["small"], x["big"], x["measure"])] = x["verdict"]


def one_pool(d):
    L = load_lock(LOCK); a = L["audit"]
    rows, rerr, meta, merr = load_pool_v1(Path(d))
    out = []
    inc, _ = included(Path(d).name, L)
    if not inc:
        return out
    for pair in pairs_for(Path(d).name, rows, L):
        if pair["tier"] not in ("primary", "secondary"):
            continue
        q = common_questions(rows, pair["small"], pair["big"])
        sub = [r for r in rows if r.model in (pair["small"], pair["big"]) and r.qid in q]
        grid = policy_grid(sub, meta, pair, L)
        for ms in measures_for(Path(d).name, sub, meta, pair, L):
            for claim in ("relative", "absolute"):
                key = (claim, Path(d).name, pair["small"], pair["big"], ms["name"])
                ref = REF.get(key)
                if ref is None or ref == "NOT RUN":
                    continue
                cfgd = {"small": pair["small"], "big": pair["big"], "alpha": a["alpha"], "delta": a["delta"],
                        "s": a["s"], "salt": a["salt"], "cost_test": a["cost_test"], "cost": ms["cost"],
                        "policies": grid, "design": "v3"}
                if claim == "relative":
                    cfgd.update({"claim": "relative", "eps": a["eps"]})
                cfg = config_from_dict(cfgd)
                inp = replay(sub, meta, cfg).audit_input
                v = audit3(inp, alpha=cfg.alpha, delta=cfg.delta, s=cfg.s, salt=cfg.salt, claim=claim,
                           eps=a["eps"]).verdict
                cs = certified_saving(inp, claim=claim, alpha=cfg.alpha, eps=a["eps"], delta=cfg.delta, salt=cfg.salt)
                cs.pop("policies")
                out.append({"claim": claim, "pool": Path(d).name, "small": pair["small"], "big": pair["big"],
                            "measure": ms["name"], "tier": pair["tier"], "n": len(inp.qids), "verdict": v,
                            "verdict_matches_map": v == ref, **cs})
    return out


if __name__ == "__main__":
    t0 = time.time()
    ds = sorted(str(p) for p in Path(POOLS).iterdir() if (p / "meta.json").exists())
    res = []
    with Pool(2) as pl:
        for o in pl.imap_unordered(one_pool, ds):
            res.extend(o); print(len(res), round(time.time() - t0), flush=True)
    res.sort(key=lambda x: (x["claim"], x["pool"], x["small"], x["big"], x["measure"]))
    with open(OUT, "w") as f:
        for x in res:
            f.write(json.dumps(x, sort_keys=True) + "\n")
    print("done", len(res), "mismatch", sum(not x["verdict_matches_map"] for x in res), round(time.time() - t0))
