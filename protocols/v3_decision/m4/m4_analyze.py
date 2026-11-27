"""M4 analysis (written before the real run; see DECISION_m4_before_run.md).
python m4_analyze.py OUT_DIR/m4_runs.jsonl SUMMARY.json
Rates are ratios of counts pooled over runs; 95% intervals by cluster bootstrap over configurations
(2,000 resamples, seed 0), because the subsamples of one configuration are dependent."""
import json, random, sys
from collections import Counter, defaultdict

EPS, S, ALPHA_CLEAR = 0.02, 0.20, 0.05
runs_path, out_path = sys.argv[1], sys.argv[2]
cfgs = [json.loads(l) for l in open(runs_path, encoding="utf-8")]
if any(not c["c0_ok"] for c in cfgs):
    bad = [c["config"] for c in cfgs if not c["c0_ok"]]
    sys.exit(f"STOP: {len(bad)} configurations do not reproduce the official map (C0), e.g. {bad[:3]}")


def events(c, n_sub):
    """Per configuration: numerators/denominators of every rate (dict name -> [num, den])."""
    e = defaultdict(lambda: [0, 0])
    for x in c["runs"]:
        if n_sub is not None and x["n_sub"] != n_sub:
            continue
        s, f, v = x["sub"], x["follow"], x["sub"]["verdict"]
        e["runs"][1] += 1
        e["below_floor"][1] += v == "INCONCLUSIVE"
        e["below_floor"][0] += bool(s.get("below_floor"))
        if v == "INCONCLUSIVE" and s["type"] == "reachable":
            e["E1_uncheckable_share"][1] += 1
            e["E1_uncheckable_share"][0] += f["kind"] == "uncheckable"
            if f["kind"] == "extension":
                go = f["verdict"] == "GO"
                e["E1_reachable_goes_GO"][1] += 1; e["E1_reachable_goes_GO"][0] += go
                b = s.get("blocking") or "?"
                e[f"E1_reachable_goes_GO|blocking={b}"][1] += 1; e[f"E1_reachable_goes_GO|blocking={b}"][0] += go
        if v == "INCONCLUSIVE" and s["type"] == "unreachable":
            e["E2_unreachable_full_GO"][1] += 1; e["E2_unreachable_full_GO"][0] += c["full_verdict"] == "GO"
        if v == "GO":
            pv = (f["delta_hat"] is not None and f["delta_hat"] > EPS) or (f["saving"] is not None and f["saving"] < S)
            cv = ((f["p_risk_above"] is not None and f["p_risk_above"] <= ALPHA_CLEAR)
                  or (f["p_saving_below"] is not None and f["p_saving_below"] <= ALPHA_CLEAR))
            e["E3_GO_point_violation"][1] += 1; e["E3_GO_point_violation"][0] += pv
            e["E3_GO_clear_violation"][1] += 1; e["E3_GO_clear_violation"][0] += cv
            e["E3_GO_full_verdict_GO"][1] += 1; e["E3_GO_full_verdict_GO"][0] += c["full_verdict"] == "GO"
        if v == "NO-GO":
            e["E4_NOGO_full_GO"][1] += 1; e["E4_NOGO_full_GO"][0] += c["full_verdict"] == "GO"
    return e


def rates(cs, n_sub):
    tot = defaultdict(lambda: [0, 0])
    for c in cs:
        for k, (a, b) in events(c, n_sub).items():
            tot[k][0] += a; tot[k][1] += b
    return tot


def summary(n_sub):
    per = [events(c, n_sub) for c in cfgs]
    tot = rates(cfgs, n_sub)
    rng = random.Random(0)
    boots = defaultdict(list)
    for _ in range(2000):
        acc = defaultdict(lambda: [0, 0])
        for _ in range(len(per)):
            for k, (a, b) in per[rng.randrange(len(per))].items():
                acc[k][0] += a; acc[k][1] += b
        for k in tot:
            if acc[k][1]:
                boots[k].append(acc[k][0] / acc[k][1])
    out = {}
    for k, (a, b) in sorted(tot.items()):
        if k == "runs":
            out[k] = b; continue
        bs = sorted(boots[k])
        ci = [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]] if len(bs) > 20 else None
        out[k] = {"num": a, "den": b, "rate": (a / b if b else None), "ci95_cluster": ci}
    trans = Counter()
    for c in cfgs:
        for x in c["runs"]:
            if n_sub is None or x["n_sub"] == n_sub:
                trans[f"{x['sub']['verdict']} -> {c['full_verdict']}"] += 1
    out["transitions_sub_to_full"] = dict(sorted(trans.items()))
    return out


res = {"configs": len(cfgs), "c0_mismatches": sum(not c["c0_ok"] for c in cfgs),
       "by_n_sub": {str(n): summary(n) for n in sorted({x["n_sub"] for c in cfgs for x in c["runs"]})},
       "all": summary(None)}
json.dump(res, open(out_path, "w"), indent=1, sort_keys=True)
for k in ("by_n_sub",):
    for n, s in res[k].items():
        print(f"n_sub={n}: runs {s['runs']}")
        for m in sorted(s):
            if m.startswith("E") and "|" not in m:
                v = s[m]; print(f"  {m}: {v['num']}/{v['den']} = {v['rate'] if v['rate'] is None else round(v['rate'], 3)} CI {v['ci95_cluster']}")
print("C0 mismatches:", res["c0_mismatches"])
