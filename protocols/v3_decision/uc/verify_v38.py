"""Verify amendment v3-8 (U-c) against the pre-run criteria in DECISION_uc_before_run.md.
C1/C2: each new report equals the v3-7 report after removing R1_verdict.action and input.lock; map lines equal
ignoring report hashes.  C4: an independent derivation of every structured field of the action from the OLD
report (no import of caac.pool_v1.action).  C5: consistency with verdict and R4.  Also prints the distribution.
python verify_v38.py CLAIM NEW_MAP OLD_MAP [OLD_MAP ...]"""
import json, statistics, sys
from collections import Counter
from pathlib import Path

claim, new, olds = sys.argv[1], Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]


def find_old(rel):
    for o in olds:
        g = o / "reports" / rel
        if g.exists():
            return g
    return None


def derive(r):
    """Independent re-derivation (structured fields only)."""
    v = r["R1_verdict"]; ver = v["verdict"]
    if ver == "NOT RUN":
        return {"kind": "fix_data"}
    cl = v.get("claim") or "absolute"
    pols = r["R2_estimates"].get("all_policies") or []
    S = (v.get("certified_saving") or {}).get("value")
    a = {}
    if ver == "GO":
        a["kind"] = "deploy"; a["policy"] = v["chosen"]; a["budget_saving"] = S
        groups = []
        if cl == "absolute" and r["R2_estimates"].get("by_group"):
            groups = sorted(k for k, x in r["R2_estimates"]["by_group"]["groups"].items()
                            if x.get("lower_bound_above_alpha"))
        a["groups_to_check"] = groups
        a["verdict_changes_with_measure"] = bool(r["R3_cost_measures"]["verdict_changes_with_measure"])
        return a
    half = v["delta"] / 2
    if ver == "NO-GO":
        q = [p.get("p_risk_above") is not None and p["p_risk_above"] <= half for p in pols]
        s_ = [p.get("p_saving_below") is not None and p["p_saving_below"] <= half for p in pols]
        a = {"kind": "keep_big", "rejected_by_quality": sum(x and not y for x, y in zip(q, s_)),
             "rejected_by_saving": sum(y and not x for x, y in zip(q, s_)),
             "rejected_by_both": sum(x and y for x, y in zip(q, s_))}
    elif v.get("conflict"):
        a = {"kind": "check_data"}
    else:
        c = r["R4_break_even"]["certification"]
        fl = r["R4_break_even"].get("minimum_questions_zero_errors")
        a = {"kind": "collect_or_stop", "below_floor": fl is not None and v["n_questions"] < fl}
        if c.get("reachable"):
            a.update(type="reachable", blocking=c["blocking"], policy=c["policy"],
                     more_questions_needed=c["more_questions_needed"], questions_needed=c["questions_needed"])
        else:
            if cl == "relative":
                okq = {p["policy"] for p in pols if p.get("delta_hat") is not None and p["delta_hat"] < v["eps"]}
            else:
                okq = {p["policy"] for p in pols if p.get("served") and p.get("risk") is not None and p["risk"] < v["alpha"]}
            oks = {p["policy"] for p in pols if p.get("saving") is not None and p["saving"] > v["s"]}
            b = "risk" if not okq else ("saving" if not oks else ("trade_off" if not (okq & oks) else "power"))
            a.update(type="unreachable", blocking=b)
    if S is not None and S > 0:
        a["smaller_saving_certified"] = {"value": S, "policy": v["certified_saving"].get("policy")}
    return a


n = bad2 = bad4 = bad5 = 0
dist = {"primary": Counter(), "secondary": Counter()}
more = {"primary": [], "secondary": []}
tier = {}
for o in olds:
    for l in open(o / "map.jsonl"):
        x = json.loads(l); tier[x["report"]] = x.get("tier")
for f in sorted((new / "reports").rglob("*.json")):
    rel = f.relative_to(new / "reports")
    g = find_old(rel)
    b = json.load(open(f))
    act = b["R1_verdict"].pop("action", None)
    if g is None or act is None:
        bad2 += 1; print("MISSING", rel, g is None, act is None); continue
    a = json.load(open(g))
    a["input"].pop("lock", None); b["input"].pop("lock", None)
    n += 1
    if a != b:
        bad2 += 1
        if bad2 <= 3: print("C2 DIFF", rel)
    d = derive(a)
    if {k: v for k, v in act.items() if k != "text"} != d:
        bad4 += 1
        if bad4 <= 3: print("C4 DIFF", rel, act, d)
    ver = a["R1_verdict"]["verdict"]
    exp = {"GO": {"deploy"}, "NO-GO": {"keep_big"}, "INCONCLUSIVE": {"collect_or_stop", "check_data"},
           "NOT RUN": {"fix_data"}}[ver]
    c = a.get("R4_break_even", {}).get("certification") or {}
    ok5 = act["kind"] in exp and isinstance(act.get("text"), list) and act["text"]
    if act["kind"] == "collect_or_stop":
        ok5 = ok5 and ((act["type"] == "reachable") == bool(c.get("reachable")))
        if act["type"] == "reachable":
            ok5 = ok5 and act["blocking"] == c.get("blocking")
    if not ok5:
        bad5 += 1; print("C5 FAIL", rel)
    if "sensitivity" in rel.parts:
        continue
    t = tier.get(str(rel.name)) or tier.get(rel.name)
    if t not in dist:
        continue
    D = dist[t]
    D["verdict:" + ver] += 1
    if act["kind"] == "keep_big":
        for k in ("quality", "saving", "both"):
            D[f"nogo_rejected_by_{k}(policies)"] += act[f"rejected_by_{k}"]
        D["nogo_any_saving_rejection"] += (act["rejected_by_saving"] + act["rejected_by_both"]) > 0
        D["nogo_any_quality_rejection"] += (act["rejected_by_quality"] + act["rejected_by_both"]) > 0
    if act["kind"] == "collect_or_stop":
        D["inc:" + act["type"]] += 1
        D[f"inc:{act['type']}:blocking={act['blocking']}"] += 1
        D["inc:below_floor"] += act["below_floor"]
        if act["type"] == "reachable":
            more[t].append(act["more_questions_needed"])
    if act["kind"] == "check_data":
        D["inc:conflict"] += 1
    if "smaller_saving_certified" in act:
        D[f"smaller_saving_certified:{ver}"] += 1
    if act["kind"] == "deploy":
        D["go:groups_flagged"] += bool(act["groups_to_check"])
        D["go:verdict_changes_with_measure"] += act["verdict_changes_with_measure"]


def strip(x):
    x = dict(x); x.pop("report_sha16", None)
    if x.get("sensitivity"):
        x["sensitivity"] = {c: {k: v for k, v in r.items() if k != "report_sha16"} for c, r in x["sensitivity"].items()}
    return json.dumps(x, sort_keys=True)


ol = set()
for o in olds:
    ol |= {strip(json.loads(l)) for l in open(o / "map.jsonl")}
nl = {strip(json.loads(l)) for l in open(new / "map.jsonl")}
print(f"== {claim}: reports {n}; C2 differing {bad2}; C4 differing {bad4}; C5 failing {bad5}; "
      f"C1 map lines equal ignoring report hashes: {ol == nl} ({len(nl)} lines)")
out = {}
for t in ("primary", "secondary"):
    m = more[t]
    out[t] = dict(sorted(dist[t].items()))
    out[t]["reachable_more_questions_median"] = statistics.median(m) if m else None
    out[t]["reachable_more_questions_quartiles"] = (statistics.quantiles(m, n=4) if len(m) > 1 else m)
print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).with_name(f"uc_summary_{claim}.json"), "w"), indent=1, sort_keys=True)
