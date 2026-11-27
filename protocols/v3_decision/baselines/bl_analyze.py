"""Analysis for #170 (written before the results were read; readings in DECISION_bl_before_run.md).
python bl_analyze.py W1.json W2_DIR SUMMARY.json"""
import json
import random
import sys
from collections import defaultdict

from scipy.stats import binom

import bl_methods as M

w1p, w2d, outp = sys.argv[1], sys.argv[2], sys.argv[3]
DELTA = M.DELTA


def fails(k, n):                       # exact one-sided binomial: H0 rate <= delta
    return bool(k > DELTA * n and binom.sf(k - 1, n, DELTA) < 0.01)


# ------------------------------------------------------------------ W1
w1 = json.load(open(w1p))
W1 = []
for c in w1["cells"]:
    row = {"scenario": c["scenario"], "n": c["n"], "claim_true": c["claim_true"]}
    for m in M.METHODS:
        r = c[m]; reps = c["reps"]
        row[m] = {"false_go": r["false_go"] / reps, "power": r["power"] / reps, "go": r["go"] / reps,
                  "fails_validity": fails(r["false_go"], reps)}
    row["caac"]["wrong_nogo"] = (c["caac"]["nogo"] / c["reps"]) if c["claim_true"] else None
    W1.append(row)
w1_fail = {m: [f"{r['scenario']}@{r['n']}" for r in W1 if r[m]["fails_validity"]] for m in M.METHODS}
w1_max = {m: max(r[m]["false_go"] for r in W1) for m in M.METHODS}

# ------------------------------------------------------------------ W2 / W3
import gzip, os
_p = f"{w2d}/bl_w2.jsonl"
cfgs = [json.loads(l) for l in (open(_p) if os.path.exists(_p) else gzip.open(_p + ".gz", "rt"))]
x0_bad = [c["config"] for c in cfgs if not c["x0_ok"]]
if x0_bad:
    sys.exit(f"STOP: X0 mismatches {len(x0_bad)} e.g. {x0_bad[:3]}")
NS = sorted({int(n) for c in cfgs for n in c["w2"]})
REPS = 200


def pooled(cs, n):
    acc = defaultdict(lambda: [0, 0])
    for c in cs:
        cell = c["w2"][str(n)]
        for m in M.METHODS:
            go, fg, pw = cell[m]
            acc[m + "|false_go"][0] += fg; acc[m + "|false_go"][1] += REPS
            if c["any_feasible"]:
                acc[m + "|power"][0] += pw; acc[m + "|power"][1] += REPS
                acc[m + "|not_go_when_feasible"][0] += REPS - go; acc[m + "|not_go_when_feasible"][1] += REPS
        if c["any_feasible"]:
            acc["caac|wrong_nogo"][0] += cell["caac_nogo"]; acc["caac|wrong_nogo"][1] += REPS
    return acc


W2 = {}
for n in NS:
    tot = pooled(cfgs, n)
    rng = random.Random(0)
    boots = defaultdict(list)
    for _ in range(2000):
        s = [cfgs[rng.randrange(len(cfgs))] for _ in cfgs]
        b = pooled(s, n)
        for k, (a, d) in b.items():
            if d:
                boots[k].append(a / d)
    res = {}
    for k, (a, d) in sorted(tot.items()):
        bs = sorted(boots[k])
        res[k] = {"num": a, "den": d, "rate": a / d if d else None,
                  "ci95_cluster": [bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]] if len(bs) > 20 else None}
    for m in M.METHODS:
        res[m + "|configs_failing_validity"] = sum(fails(c["w2"][str(n)][m][1], REPS) for c in cfgs)
        # descriptive (added before reading results): per-configuration rates, so a pooled rate diluted by
        # easy configurations cannot hide a bad one
        rates = [c["w2"][str(n)][m][1] / REPS for c in cfgs]
        res[m + "|max_config_false_go"] = max(rates)
        res[m + "|configs_false_go_above_delta"] = sum(r > DELTA for r in rates)
    W2[str(n)] = res

W3 = {m: {"go": sum(c["w3"][m] is not None for c in cfgs),
          "go_on_infeasible_policy": sum(c["w3_feasible"][m] is False for c in cfgs),
          "go_and_caac_go": sum(c["w3"][m] is not None and c["w3"]["caac"] is not None for c in cfgs)}
      for m in M.METHODS}

summary = {"design": "DECISION_bl_before_run.md", "W1": W1, "W1_cells_failing_validity": w1_fail,
           "W1_max_false_go": w1_max, "W2_configs": len(cfgs),
           "W2_configs_with_feasible_policy": sum(c["any_feasible"] for c in cfgs),
           "W2": W2, "W3": W3, "X0_mismatches": 0}
json.dump(summary, open(outp, "w"), indent=1, sort_keys=True)

print("W1 max false GO:", {m: round(v, 3) for m, v in w1_max.items()})
print("W1 cells failing validity:", {m: len(v) for m, v in w1_fail.items()})
for n in NS:
    r = W2[str(n)]
    print(f"W2 n={n}")
    for m in M.METHODS:
        fg, pw = r[m + "|false_go"], r.get(m + "|power")
        print(f"  {m:11s} falseGO {fg['rate']:.4f} {[round(x,4) for x in fg['ci95_cluster']]}  "
              f"power {pw['rate']:.4f}  cfg_fail {r[m + '|configs_failing_validity']}")
    print("  caac wrong NO-GO", round(r["caac|wrong_nogo"]["rate"], 4))
print("W3", W3)
