"""Analysis for amendment v3-10 (written before the dev results were read). Rules: DECISION_v310_before_run.md.
  python v310_analyze.py dev  W1_DEV.json W2_DEV_DIR OUT.json
  python v310_analyze.py conf CHOSEN W1_CONF.json W2_CONF_DIR OUT.json
Selection (dev) reads ONLY the 8 development pools; the 3 held-out pools are never used for the choice."""
import json, random, sys
from collections import defaultdict
from binom_helper import fails, DELTA

HELD = ("lrb_medqa_test", "lrb_winogrande_valid", "lurume_mmlu_pro")
CANDS = ("B1", "B2", "X1", "X2", "H1", "H2")           # simplicity order
REPS = 200
HEAVY = "lrb_arcc_test|Llama-3.1-8B-Instruct|Llama-3.1-8B-UltraMedical|tokens_lambda1"
mode = sys.argv[1]


def load(w1p, w2d):
    w1 = json.load(open(w1p))
    cfgs = [json.loads(l) for l in open(f"{w2d}/v310_w2.jsonl")]
    bad = [c["config"] for c in cfgs if not c["x0_ok"]]
    if bad:
        sys.exit(f"STOP: X0 mismatches {len(bad)}")
    return w1, cfgs


def w1_stats(w1, v):
    reps = w1["reps"]; fgmax = 0.0; fail = []; pw = []; pw0 = []
    for c in w1["cells"]:
        r = c[v]
        fgmax = max(fgmax, r["false_go"] / reps)
        if fails(r["false_go"], reps): fail.append(f"{c['scenario']}@{c['n']}:fg")
        if c["claim_true"]:
            pw.append(r["power"] / reps); pw0.append(c["V0"]["power"] / reps)
            if fails(r["nogo"], reps): fail.append(f"{c['scenario']}@{c['n']}:nogo")
    return {"max_false_go": fgmax, "fails": fail, "gain": sum(a - b for a, b in zip(pw, pw0)) / len(pw),
            "power": sum(pw) / len(pw)}


def w2_stats(cfgs, v, n):
    fg = pw = ng = dp = df = 0; failing = 0; maxc = 0.0
    for c in cfgs:
        go, f, p, nogo = c["w2"][str(n)][v]
        fg += f; df += REPS; maxc = max(maxc, f / REPS); failing += fails(f, REPS)
        if c["any_feasible"]:
            pw += p; dp += REPS; ng += nogo
    return {"false_go": fg / df, "power": pw / dp if dp else None, "wrong_nogo": ng / dp if dp else None,
            "configs_failing": failing, "max_config_false_go": maxc}


def w3_stats(cfgs, v):
    go0 = [c for c in cfgs if c["w3"]["V0"][0] == "GO"]
    return {"go": sum(c["w3"][v][0] == "GO" for c in cfgs), "v0_go": len(go0),
            "v0_go_lost": sum(c["w3"][v][0] != "GO" for c in go0),
            "flips": sum((c["w3"]["V0"][0] == "GO" and c["w3"][v][0] == "NO-GO") or (c["w3"]["V0"][0] == "NO-GO" and c["w3"][v][0] == "GO") for c in cfgs),
            "go_infeasible_policy": sum(c["w3"][v][2] is False for c in cfgs)}


def cluster_ci(cfgs, v, n, B=1000):
    rng = random.Random(0); gains = []
    per = [(c["w2"][str(n)][v][2], c["w2"][str(n)]["V0"][2]) for c in cfgs if c["any_feasible"]]
    for _ in range(B):
        s = [per[rng.randrange(len(per))] for _ in per]
        gains.append((sum(a for a, _ in s) - sum(b for _, b in s)) / (REPS * len(s)))
    gains.sort()
    return [gains[int(.025 * B)], gains[int(.975 * B) - 1]]


NS = (250, 500, 1000)
if mode == "dev":
    w1, cfgs = load(sys.argv[2], sys.argv[3]); outp = sys.argv[4]
    dev = [c for c in cfgs if c["pool"] not in HELD]
    print("dev configurations:", len(dev), "of", len(cfgs))
    res = {}
    for v in CANDS:
        a1 = w1_stats(w1, v)
        a2 = {n: w2_stats(dev, v, n) for n in NS}; b2 = {n: w2_stats(dev, "V0", n) for n in NS}
        a3 = w3_stats(dev, v)
        gW2 = sum(a2[n]["power"] - b2[n]["power"] for n in NS) / 3
        gains = {n: a2[n]["power"] - b2[n]["power"] for n in NS}
        A = (not a1["fails"]) and all(a2[n]["false_go"] <= DELTA and a2[n]["configs_failing"] == 0 and a2[n]["wrong_nogo"] <= DELTA for n in NS)
        Cc = a3["v0_go_lost"] <= 0.03 * a3["v0_go"] and a3["flips"] == 0
        gate = sum(g >= 0.05 for g in gains.values()) >= 2
        score = 0.5 * gW2 + 0.5 * a1["gain"]
        res[v] = {"W1": a1, "W2": a2, "W3": a3, "gW2": gW2, "gW1": a1["gain"], "score": score, "gains": gains,
                  "eligible": bool(A and Cc and gate), "A": bool(A), "C": bool(Cc), "gate": bool(gate)}
        print(f"{v}: score {score:+.4f} gW2 {gW2:+.4f} gW1 {a1['gain']:+.4f} gains {[round(gains[n],3) for n in NS]} "
              f"A {A} C {Cc} gate {gate} | W1 maxFG {a1['max_false_go']:.3f} W3 lost {a3['v0_go_lost']}/{a3['v0_go']} GO {a3['go']}")
    elig = [v for v in CANDS if res[v]["eligible"]]
    chosen = None
    if elig:
        best = max(res[v]["score"] for v in elig)
        chosen = next(v for v in CANDS if v in elig and res[v]["score"] >= best - 0.005)
    json.dump({"candidates": res, "eligible": elig, "chosen": chosen}, open(outp, "w"), indent=1, sort_keys=True, default=float)
    print("ELIGIBLE:", elig, "CHOSEN:", chosen)
else:
    chosen, w1p, w2d, outp = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    w1, cfgs = load(w1p, w2d)
    held = [c for c in cfgs if c["pool"] in HELD]
    out = {"chosen": chosen}
    for tag, cs in (("all", cfgs), ("heldout", held)):
        out[tag] = {"n_configs": len(cs), "W2": {}, "W3": w3_stats(cs, chosen)}
        for n in NS:
            a, b = w2_stats(cs, chosen, n), w2_stats(cs, "V0", n)
            out[tag]["W2"][n] = {"chosen": a, "V0": b, "gain": a["power"] - b["power"], "gain_ci": cluster_ci(cs, chosen, n)}
    a1, b1 = w1_stats(w1, chosen), w1_stats(w1, "V0")
    out["W1"] = {"chosen": a1, "V0": b1}
    g = {n: out["all"]["W2"][n]["gain"] for n in NS}; gh = {n: out["heldout"]["W2"][n]["gain"] for n in NS}
    A = (not a1["fails"]) and all(out["all"]["W2"][n]["chosen"]["false_go"] <= DELTA and out["all"]["W2"][n]["chosen"]["configs_failing"] == 0
                                  and out["all"]["W2"][n]["chosen"]["wrong_nogo"] <= DELTA for n in NS)
    B = sum(x >= 0.04 for x in g.values()) >= 2 and all(x >= 0 for x in g.values()) and sum(x > 0 for x in gh.values()) >= 2
    Cw1 = a1["gain"] >= -0.01
    D = out["all"]["W3"]["v0_go_lost"] <= 0.03 * out["all"]["W3"]["v0_go"] and out["all"]["W3"]["flips"] == 0
    out["criteria"] = {"A_validity": bool(A), "B_power": bool(B), "C_W1_not_worse": bool(Cw1), "D_compat": bool(D)}
    out["ADOPT"] = bool(A and B and Cw1 and D)
    json.dump(out, open(outp, "w"), indent=1, sort_keys=True, default=float)
    print("gains all", {n: round(g[n], 4) for n in NS}, "heldout", {n: round(gh[n], 4) for n in NS}, "W1 gain", round(a1["gain"], 4))
    print("W3", out["all"]["W3"]); print("criteria", out["criteria"], "ADOPT:", out["ADOPT"])
