"""Pick the greedy_cot baseline cost for the P3 Gate 2 check.

Reads outputs/e2_pareto.json (written by run_e2_pareto.py) and prints, to
stdout, a SINGLE number: greedy_cot's cost at its strongest operating point
(highest accuracy, cheapest among ties). The full greedy_cot frontier is
printed to stderr for transparency. Falls back to 270.0 (the P1 n=5 figure)
if the file or key is missing, so the pipeline never breaks.
"""
import json
import sys

FALLBACK = 270.0
try:
    d = json.load(open("outputs/e2_pareto.json"))
    pts = d["baselines"]["greedy_cot"]
    if not pts:
        raise ValueError("empty greedy_cot frontier")
    print("greedy_cot frontier (fresh, matched n=40):", file=sys.stderr)
    for p in sorted(pts, key=lambda x: x["cost"]):
        print(f"  cost={p['cost']:.1f}  acc={p['accuracy']:.3f}  {p['label']}", file=sys.stderr)
    best = max(pts, key=lambda p: (p["accuracy"], -p["cost"]))
    print(f"  -> chosen baseline: cost={best['cost']:.1f} at acc={best['accuracy']:.3f}", file=sys.stderr)
    print(f"{best['cost']:.4f}")
except Exception as e:  # noqa: BLE001 - never break the pipeline on extraction
    print(f"WARN extract_baseline: {e}; falling back to {FALLBACK}", file=sys.stderr)
    print(f"{FALLBACK:.4f}")
