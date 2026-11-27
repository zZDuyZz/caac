"""X6: absolute-claim (design v3) and design-v2 reports must be byte-identical before and after the v3-10/v3-11
integration. Run once per repository; compare the outputs. Usage: python x6_reports.py REPO POOLS MAP OUT.jsonl N"""
import hashlib, json, os, random, sys
REPO = sys.argv[1]
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "protocols", "v3_decision", "m4"))
import m4_run as M4                                                   # noqa: E402
from caac.pool_v1.config import config_from_dict                     # noqa: E402
from caac.pool_v1.report import make_report                           # noqa: E402
from caac.pool_v1.crossmap import measures_for, policy_grid, common_questions  # noqa: E402

pools, mp, outp, N = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
M4._init(pools, os.path.join(REPO, "protocols", "audit_lock_v3.json"))
L = M4._W["L"]
configs = M4.select(mp)
random.Random(66).shuffle(configs)
with open(outp, "w") as f:
    for x in configs[:N]:
        rows, meta = M4._pool(x["pool"])
        pair = {"small": x["small"], "big": x["big"]}
        q = common_questions(rows, pair["small"], pair["big"])
        sub = [r for r in rows if r.model in pair.values() and r.qid in q]
        grid = policy_grid(sub, meta, pair, L)
        ms = [m for m in measures_for(x["pool"], sub, meta, pair, L) if m["name"] == x["measure"]][0]
        a = L["audit"]
        out = {"config": M4.config_id(x)}
        for tag, extra in (("abs_v3", {"design": "v3", "claim": "absolute"}), ("v2", {"design": "v2"})):
            cfgd = {"small": pair["small"], "big": pair["big"], "alpha": a["alpha"], "delta": a["delta"], "s": a["s"],
                    "salt": a["salt"], "cost_test": a["cost_test"], "cost": ms["cost"], "policies": grid, **extra}
            rep = make_report(sub, meta, config_from_dict(cfgd), input_info={"pool": x["pool"], "lock": "X6"})
            out[tag] = hashlib.sha256(json.dumps(rep, sort_keys=True).encode()).hexdigest()
        f.write(json.dumps(out, sort_keys=True) + "\n")
print("done", N)
