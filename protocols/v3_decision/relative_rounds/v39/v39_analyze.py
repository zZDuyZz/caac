"""Analysis for amendment v3-9 (written before the results were read). Implements criteria A, B, C and the selection rule of
DECISION_v39_before_run.md rev. 1.   python v39_analyze.py W1.json W2_DIR SUMMARY.json"""
import json, random, sys
from collections import defaultdict
from scipy.stats import binom
import v39_methods as V

w1p, w2d, outp = sys.argv[1], sys.argv[2], sys.argv[3]
DELTA = 0.10
ELIG = ("V1", "V2", "V3", "V4")
ALLV = tuple(V.VARIANTS)
REPS = 200


def fails(k, n):
    return bool(k > DELTA * n and binom.sf(k - 1, n, DELTA) < 0.01)


w1 = json.load(open(w1p)); reps1 = w1["reps"]
W1 = {v: {"fail_false_go": [], "fail_wrong_nogo": [], "max_false_go": 0.0, "mean_power_true_cells": 0.0} for v in ALLV}
for v in ALLV:
    pw = []
    for c in w1["cells"]:
        r = c[v]
        W1[v]["max_false_go"] = max(W1[v]["max_false_go"], r["false_go"] / reps1)
        if fails(r["false_go"], reps1):
            W1[v]["fail_false_go"].append(f"{c['scenario']}@{c['n']}")
        if c["claim_true"]:
            pw.append(r["power"] / reps1)
            if fails(r["nogo"], reps1):
                W1[v]["fail_wrong_nogo"].append(f"{c['scenario']}@{c['n']}")
    W1[v]["mean_power_true_cells"] = sum(pw) / len(pw)

cfgs = [json.loads(l) for l in open(f"{w2d}/v39_w2.jsonl")]
x0_bad = [c["config"] for c in cfgs if not c["x0_ok"]]
if x0_bad:
    sys.exit(f"STOP: X0 mismatches {len(x0_bad)}")
NS = sorted({int(n) for c in cfgs for n in c["w2"]})
HEAVY = "lrb_arcc_test|Llama-3.1-8B-Instruct|Llama-3.1-8B-UltraMedical|tokens_lambda1"


def pooled(cs, n, v):
    fg = pw = ng = den_f = den_p = 0
    for c in cs:
        go, f, p, nogo = c["w2"][str(n)][v]
        fg += f; den_f += REPS
        if c["any_feasible"]:
            pw += p; den_p += REPS; ng += nogo
    return fg, den_f, pw, den_p, ng


W2 = {}
for n in NS:
    W2[str(n)] = {}
    for v in ALLV:
        fg, df, pw, dp, ng = pooled(cfgs, n, v)
        rng = random.Random(0); bf = []; bp = []
        per = [(c["w2"][str(n)][v], c["any_feasible"]) for c in cfgs]
        for _ in range(1000):
            s = [per[rng.randrange(len(per))] for _ in per]
            a = sum(x[0][1] for x in s); b = sum(REPS for _ in s)
            bf.append(a / b)
            pp = sum(x[0][2] for x in s if x[1]); dd = sum(REPS for x in s if x[1])
            bp.append(pp / dd if dd else 0)
        bf.sort(); bp.sort()
        W2[str(n)][v] = {"false_go": fg / df, "false_go_ci": [bf[25], bf[974]], "power": pw / dp, "power_ci": [bp[25], bp[974]],
                         "wrong_nogo": ng / dp,
                         "configs_failing_validity": sum(fails(c["w2"][str(n)][v][1], REPS) for c in cfgs),
                         "max_config_false_go": max(c["w2"][str(n)][v][1] / REPS for c in cfgs),
                         "heavy_config_false_go": next(c["w2"][str(n)][v][1] / REPS for c in cfgs if c["config"] == HEAVY)}

W3 = {}
for v in ALLV:
    go0 = [c for c in cfgs if c["w3"]["V0"][0] == "GO"]
    lost = sum(c["w3"][v][0] != "GO" for c in go0)
    flips = sum((c["w3"]["V0"][0] == "GO" and c["w3"][v][0] == "NO-GO") or (c["w3"]["V0"][0] == "NO-GO" and c["w3"][v][0] == "GO") for c in cfgs)
    trans = defaultdict(int)
    for c in cfgs:
        trans[f"{c['w3']['V0'][0]} -> {c['w3'][v][0]}"] += 1
    W3[v] = {"go": sum(c["w3"][v][0] == "GO" for c in cfgs), "v0_go_lost": lost, "v0_go": len(go0), "flips": flips,
             "go_infeasible_policy": sum(c["w3"][v][2] is False for c in cfgs), "transitions": dict(sorted(trans.items()))}

decision = {}
for v in ELIG:
    A = (not W1[v]["fail_false_go"] and not W1[v]["fail_wrong_nogo"]
         and all(W2[str(n)][v]["false_go"] <= DELTA and W2[str(n)][v]["configs_failing_validity"] == 0
                 and W2[str(n)][v]["wrong_nogo"] <= DELTA for n in NS))
    gains = {n: W2[str(n)][v]["power"] - W2[str(n)]["V0"]["power"] for n in NS}
    B = sum(g >= 0.05 for g in gains.values()) >= 2 and all(g >= 0 for g in gains.values())
    C = W3[v]["v0_go_lost"] <= 10 and W3[v]["flips"] == 0
    decision[v] = {"A_valid": bool(A), "B_gain": bool(B), "C_compat": bool(C), "gains": gains}
cand = [v for v in ELIG if all(decision[v][k] for k in ("A_valid", "B_gain", "C_compat"))]
chosen = None
if cand:
    best = max(cand, key=lambda v: (W2["500"][v]["power"], -ELIG.index(v)))
    chosen = best
    for v in cand:                                     # simpler variant within 0.01 wins
        if ELIG.index(v) < ELIG.index(chosen) and W2["500"][chosen]["power"] - W2["500"][v]["power"] <= 0.01:
            chosen = v
summary = {"design": "DECISION_v39_before_run.md rev.1", "W1": W1, "W2": W2, "W3": W3, "decision": decision, "chosen": chosen,
           "configs": len(cfgs), "configs_with_feasible": sum(c["any_feasible"] for c in cfgs)}
json.dump(summary, open(outp, "w"), indent=1, sort_keys=True)
print("W1 max false GO:", {v: round(W1[v]["max_false_go"], 3) for v in ALLV})
print("W1 fails:", {v: (len(W1[v]["fail_false_go"]), len(W1[v]["fail_wrong_nogo"])) for v in ALLV})
print("W1 mean power (claim-true cells):", {v: round(W1[v]["mean_power_true_cells"], 3) for v in ALLV})
for n in NS:
    print(f"W2 n={n}")
    for v in ALLV:
        r = W2[str(n)][v]
        print(f"  {v:6s} falseGO {r['false_go']:.4f} power {r['power']:.4f} [{r['power_ci'][0]:.3f},{r['power_ci'][1]:.3f}] "
              f"wrongNOGO {r['wrong_nogo']:.4f} cfg_fail {r['configs_failing_validity']} maxcfg {r['max_config_false_go']:.3f} heavy {r['heavy_config_false_go']:.3f}")
print("W3:", {v: (W3[v]["go"], W3[v]["v0_go_lost"], W3[v]["flips"], W3[v]["go_infeasible_policy"]) for v in ALLV})
print("decision:", json.dumps(decision, default=float)); print("CHOSEN:", chosen)
