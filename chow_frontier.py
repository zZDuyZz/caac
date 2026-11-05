"""The Chow-frontier figure + table (CAAC_MASTER §10, the paper's headline).

Measures the recovery rate r on each (dataset, model) pool, then plots the two
coverage caps against the base error rate:

  serve/abstain only   cap = min{1, pi/(1-alpha)}            <- PriceCheck's set
  with correction      cap = min{1, (pi+(1-pi)r)/(1-alpha)}  <- CAAC's set

Both are CAPS attained by an oracle; neither is a claim about a realised policy.
Real operating points (our pools, and PriceCheck's reported regime) are marked.

  python chow_frontier.py --small outputs/a2/math500_q1.5b.jsonl.gz \
      --big outputs/a2/math500_q7b.jsonl.gz --label "MATH-500 Qwen 1.5B->7B" \
      --out figs/chow_frontier --alpha 0.10

Several --small/--big/--label triples may be passed; each contributes its own r
and its own marked operating point. --no-plot skips matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caac.pool.frontier import (cap_correct, cap_serve, cost_ratio_at, frontier_curve,
                                pi_after, recovery_rate)
from caac.pool.schema import load_pool

# PriceCheck (2609.37493) operates at pi_pop ~ 0.957 -- quoted for context only,
# not re-measured here. See CAAC_MASTER §10.
PRICECHECK_PI = 0.957
PRICECHECK_LABEL = "PriceCheck regime (pi~0.957)"


def analyse(small_path, big_path, label, alpha, k, profile):
    small = load_pool(small_path)
    big = load_pool(big_path) if big_path else None
    rr = recovery_rate(small, big, k=k)
    pi = rr.pi_greedy
    lo, hi = rr.r_ci
    return {
        "label": label, "small": str(small_path), "big": str(big_path or ""),
        "alpha": alpha, "k": k,
        "recovery": rr.as_dict(),
        "pi": pi, "pi_after": pi_after(pi, rr.r),
        "cap_serve": cap_serve(pi, alpha), "cap_correct": cap_correct(pi, alpha, rr.r),
        "cap_correct_lo": cap_correct(pi, alpha, lo), "cap_correct_hi": cap_correct(pi, alpha, hi),
        "cost_ratio": cost_ratio_at(small, big, k=k, profile=profile),
    }


def plot(results, alpha, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pis = [i / 200 for i in range(4, 197)]
    x = [1 - p for p in pis]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), height_ratios=[2.2, 1],
                                  sharex=True)
    ax.plot(x, [cap_serve(p, alpha) for p in pis], color="#333", lw=2.2,
            label="serve / abstain only  (PriceCheck's action set)")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for res, c in zip(results, colors):
        r = res["recovery"]["r"]
        cc = [cap_correct(p, alpha, r) for p in pis]
        ax.plot(x, cc, color=c, lw=2.0, ls="--",
                label=f"+ answer-correcting actions  (r={r:.2f}, {res['label']})")
        ax.fill_between(x, [cap_serve(p, alpha) for p in pis], cc, color=c, alpha=0.07)
        ax2.plot(x, [a - b for a, b in zip(cc, [cap_serve(p, alpha) for p in pis])],
                 color=c, lw=1.8, ls="--")
        xi = 1 - res["pi"]
        ax.plot([xi, xi], [res["cap_serve"], res["cap_correct"]], color=c, lw=2.4, alpha=0.85)
        ax.plot([xi], [res["cap_serve"]], "o", color=c, ms=7, mec="k", mew=0.6)
        ax.plot([xi], [res["cap_correct"]], "^", color=c, ms=7, mec="k", mew=0.6)
        ax.annotate(f"{res['label']}\n{res['cap_serve']:.2f} $\\rightarrow$ {res['cap_correct']:.2f}"
                    f"  (cost ×{res['cost_ratio'].get('escalate', float('nan')):.0f})",
                    xy=(xi, res["cap_correct"]), xytext=(10, 6 if res["cap_correct"] < 0.9 else -38),
                    textcoords="offset points", fontsize=8, color=c,
                    bbox=dict(fc="white", ec=c, lw=0.6, alpha=0.85, pad=1.6))
    for a_ in (ax, ax2):
        a_.axvline(1 - PRICECHECK_PI, color="#888", ls=":", lw=1.3)
        a_.grid(alpha=0.22)
    ax.annotate(PRICECHECK_LABEL, xy=(1 - PRICECHECK_PI, 0.50), xytext=(5, 0),
                textcoords="offset points", fontsize=8, color="#555", rotation=90,
                va="center")
    ax.set_ylabel(f"max certifiable coverage  (alpha={alpha:.2f})")
    ax.set_title("Chow frontier: answer-correcting actions are a precondition,\n"
                 "not an optimisation, once the base error rate is large", fontsize=11)
    ax.set_ylim(0, 1.04); ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    ax2.set_ylabel("lift from correction")
    ax2.set_xlabel("base error rate of the candidate answer   (1 - pi)")
    ax2.set_xlim(0, 0.96); ax2.set_ylim(bottom=0)
    ax2.annotate("the two action sets are\nindistinguishable here", xy=(0.02, 0.01),
                 xytext=(0.07, 0.12), fontsize=8, color="#555",
                 arrowprops=dict(arrowstyle="->", color="#888", lw=0.9))
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="append", required=True)
    ap.add_argument("--big", action="append", default=[])
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--profile", default="flops", choices=["flops", "tokens"])
    ap.add_argument("--out", default="figs/chow_frontier")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    bigs = a.big + [None] * (len(a.small) - len(a.big))
    labels = a.label + [Path(s).stem for s in a.small[len(a.label):]]
    results = [analyse(s, b, l, a.alpha, a.k, a.profile)
               for s, b, l in zip(a.small, bigs, labels)]

    print(f"\n=== CHOW FRONTIER (alpha={a.alpha:.2f}, k={a.k}, cost={a.profile}) ===")
    print(f"  {'pool':34} {'pi':>6} {'r':>14} {'pi_after':>9} {'cap serve':>10} {'cap corr':>9} {'lift':>7}")
    for r in results:
        rec = r["recovery"]
        lo, hi = rec["r_ci95"]
        print(f"  {r['label'][:34]:34} {r['pi']:6.3f} {rec['r']:.2f}[{lo:.2f},{hi:.2f}] "
              f"{r['pi_after']:9.3f} {r['cap_serve']:10.3f} {r['cap_correct']:9.3f} "
              f"{r['cap_correct'] - r['cap_serve']:+7.3f}")
        print(f"  {'':34} recovered {rec['n_recovered_any']}/{rec['n_greedy_wrong']} greedy-wrong "
              f"(sample_more {rec['by_action'].get('sample_more', 0)}, "
              f"escalate {rec['by_action'].get('escalate', 0)}, "
              f"both {rec['by_action'].get('both', 0)});  cost x"
              + ", x".join(f"{k}:{v:.1f}" for k, v in r["cost_ratio"].items() if k != "profile"))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"alpha": a.alpha, "k": a.k, "profile": a.profile, "results": results,
               "curve": frontier_curve(a.alpha, results[0]["recovery"]["r"],
                                       [i / 100 for i in range(4, 97)])}
    Path(str(out) + ".json").write_text(json.dumps(payload, indent=2))
    print(f"  -> {out}.json")
    if not a.no_plot:
        print(f"  -> {plot(results, a.alpha, str(out) + '.png')} (+ .pdf)")


if __name__ == "__main__":
    main()
