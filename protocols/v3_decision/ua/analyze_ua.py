import json, statistics as st, sys
from collections import Counter
X = [json.loads(l) for l in open(sys.argv[1])]
out = {}
for claim in ("relative", "absolute"):
    for tier in ("primary", "secondary"):
        xs = [x for x in X if x["claim"] == claim and x["tier"] == tier]
        dfn = [x for x in xs if x["certified_saving"] is not None]
        S = [x["certified_saving"] for x in dfn]
        go = [x for x in xs if x["verdict"] == "GO"]
        r = {"n": len(xs), "verdicts": dict(Counter(x["verdict"] for x in xs)),
             "S_defined": len(dfn), "S_ge_0.20": sum(s >= 0.2 - 1e-9 for s in S),
             "S_in_(0,0.20)": sum(0 < s < 0.2 - 1e-9 for s in S), "S_le_0": sum(s <= 1e-9 for s in S),
             "S_median": st.median(S) if S else None,
             "GO_but_S_below_0.20": sum(1 for x in go if (x["certified_saving"] or -9) < 0.2 - 1e-9),
             "S_ge_0.20_not_GO": sum(1 for x in dfn if x["certified_saving"] >= 0.2 - 1e-9 and x["verdict"] != "GO"),
             "S_by_verdict": {v: dict(Counter(("none" if x["certified_saving"] is None else ">=.20" if x["certified_saving"] >= .2 - 1e-9 else "(0,.20)" if x["certified_saving"] > 1e-9 else "<=0") for x in xs if x["verdict"] == v)) for v in ("GO", "INCONCLUSIVE", "NO-GO")}}
        out[f"{claim}|{tier}"] = r
prim = sorted([x for x in X if x["tier"] == "primary" and x["certified_saving"] is not None and x["certified_saving"] > 0],
              key=lambda x: (x["claim"], -x["certified_saving"]))
out["primary_positive"] = [{k: x[k] for k in ("claim", "pool", "small", "big", "measure", "n", "verdict", "certified_saving", "saving_hat_of_policy", "policy")} for x in prim]
out["running_example"] = [{k: x[k] for k in ("claim", "measure", "verdict", "certified_saving", "saving_hat_of_policy", "policy", "n_risk_pass")}
                          for x in X if x["pool"] == "lurume_gpqa_diamond" and x["small"] == "deepseek-v4-flash"]
json.dump(out, open(sys.argv[2], "w"), indent=1)
for k, v in out.items():
    if k in ("primary_positive", "running_example"):
        print(k); [print("  ", json.dumps(e)) for e in v]
    else:
        print(k, json.dumps(v))
