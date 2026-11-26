"""T1: on the real v2 maps, which leg blocks each INCONCLUSIVE and which leg drives each NO-GO."""
import json, sys, collections, statistics as st
D = 0.10
for tier in ("map_v2_primary", "map_v2_secondary"):
    rows = [json.loads(l) for l in open(f"{tier}/map.jsonl")]
    c = collections.Counter(); nogo = collections.Counter(); ratios = collections.defaultdict(list)
    for r in rows:
        if r["verdict"] not in ("INCONCLUSIVE", "NO-GO", "GO"):
            continue
        rep = json.load(open(f"{tier}/reports/{r['report']}"))
        e = rep.get("R2_estimates", {})
        if r["verdict"] == "INCONCLUSIVE":
            if e.get("policy") is None:
                c["no policy passed half A (selection step)"] += 1; ratios["selection"].append(r["cost_bound_ratio"])
            else:
                pr, ps = e["p_risk"], e["p_saving"]
                k = ("risk fails" if pr > D else "risk ok") + " / " + ("saving fails" if ps > D else "saving ok")
                c[k] += 1; ratios[k].append(r["cost_bound_ratio"])
        if r["verdict"] == "NO-GO":
            pols = e.get("all_policies", [])
            byr = sum(p["p_risk_above"] <= D/2 for p in pols); bys = sum(p["p_saving_below"] <= D/2 for p in pols)
            onlys = sum(p["p_saving_below"] <= D/2 and p["p_risk_above"] > D/2 for p in pols)
            nogo["policies"] += len(pols); nogo["rejected by risk"] += byr; nogo["rejected by saving"] += bys
            nogo["rejected ONLY by saving"] += onlys
    print(tier, "INCONCLUSIVE:", dict(c))
    for k, v in ratios.items():
        v = [x for x in v if isinstance(x, (int, float))]
        if v: print("   ", k, "cost_bound_ratio median", round(st.median(v), 1), "max", round(max(v), 1))
    print(tier, "NO-GO policy-level:", dict(nogo))
