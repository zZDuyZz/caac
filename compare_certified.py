"""The head-to-head figure: certified coverage vs cost, CAAC against PriceCheck.

Reads the JSON that ``certify.py`` writes (one file per policy per dataset) and
draws, for each dataset, every certified operating point in the
cost / coverage plane, with the two Chow caps from ``pool.frontier`` as the
ceilings that explain the gap:

  * a serve/abstain-only policy cannot rise above ``pi/(1-alpha)`` however good
    its check is -- that line is PriceCheck's ceiling, not its shortcoming;
  * allowing answer-correcting actions raises the ceiling to
    ``(pi+(1-pi)r)/(1-alpha)``, and CAAC's certified point sits under THAT one.

Both axes are things the paper claims: coverage is certified at the same alpha
with the same protocol for every point, and cost is the measured mean over the
same held-out questions, so a point that is up and to the left is strictly
better -- more questions answered, less compute, same risk guarantee.

    python compare_certified.py \
        --panel "GSM8K (alpha=0.10)" --small outputs/a2/gsm8k_q1.5b.jsonl.gz \
            --big outputs/a2/gsm8k_q7b.jsonl.gz \
            --json figs/certify_gsm8k_pc_matched.json=PriceCheck \
            --json figs/certify_gsm8k_caac.json=CAAC \
        --out figs/certified_frontier
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caac.pool.frontier import cap_correct, cap_serve, recovery_rate
from caac.pool.schema import load_pool

METHOD_STYLE = {           # (colour, marker) -- PriceCheck and CAAC stay fixed
    "PriceCheck": ("#d62728", "o"),
    "CAAC": ("#1f77b4", "^"),
}
FALLBACK = ("#7f7f7f", "s")


def read_point(path: str, label: str) -> list[dict]:
    """Every certified selection in one certify.py JSON, as plottable points."""
    d = json.loads(Path(path).read_text())
    out = []
    for method, sel in (d.get("selected") or {}).items():
        if not sel:
            continue
        out.append({
            "label": label, "method": method, "alpha": d["alpha"],
            "policy": d["policy"], "actions": d.get("actions"),
            "knob": sel["lambda"], "coverage": sel["test_coverage"],
            "risk": sel["test_selective_risk"], "cost": sel["test_cost"],
            "violated": sel.get("test_risk_violated", False),
            "price": sel.get("price_of_guarantee_coverage"),
            "n_calib": d["n_calib"], "n_test": d["n_test"],
            "floor": d.get("certification_floor_coverage"),
        })
    return out


def panel_caps(small_path, big_path, alpha, k):
    small = load_pool(small_path)
    big = load_pool(big_path) if big_path else None
    rr = recovery_rate(small, big, k=k)
    return {"pi": rr.pi_greedy, "r": rr.r,
            "cap_serve": cap_serve(rr.pi_greedy, alpha),
            "cap_correct": cap_correct(rr.pi_greedy, alpha, rr.r)}


def plot(panels, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6),
                             squeeze=False)
    for ax, p in zip(axes[0], panels):
        caps = p["caps"]
        ax.axhline(caps["cap_serve"], color="#d62728", ls="--", lw=1.6,
                   label=f"Chow cap, serve/abstain only ({caps['cap_serve']:.2f})")
        ax.axhline(caps["cap_correct"], color="#1f77b4", ls="--", lw=1.6,
                   label=f"Chow cap, + correcting actions ({caps['cap_correct']:.2f})")
        ax.axhspan(0, caps["cap_serve"], color="#d62728", alpha=0.04)
        costs = [pt["cost"] for pt in p["points"]]
        ax.set_xscale("log")
        ax.set_xlim(min(costs) / 2.4, max(costs) * 5.5)
        mid = (min(costs) * max(costs)) ** 0.5
        seen: dict[str, int] = {}
        for pt in sorted(p["points"], key=lambda x: -x["coverage"]):
            col, mk = METHOD_STYLE.get(pt["label"], FALLBACK)
            solid = pt["method"] != "bonferroni"
            ax.plot([pt["cost"]], [pt["coverage"]], mk, color=col,
                    mfc=col if solid else "none", ms=11, mew=1.8, zorder=5)
            # ours hangs below its marker, the baselines stack to the right, so
            # the boxes cannot collide however close two points land
            rank = seen.get(pt["label"], 0)
            seen[pt["label"]] = rank + 1
            ours = pt["label"] == "CAAC"
            tag = (f"{pt['label']} ({'fixed-seq' if solid else 'Bonferroni'})\n"
                   f"coverage {pt['coverage']:.3f}, risk {pt['risk']:.3f}")
            if pt["violated"]:
                tag += "\n[risk violated on test]"
            ax.annotate(tag, xy=(pt["cost"], pt["coverage"]),
                        xytext=(0, -38 - 26 * rank) if ours else (12, 2 - 30 * rank),
                        textcoords="offset points", fontsize=8, color=col,
                        ha="center" if ours else "left",
                        bbox=dict(fc="white", ec=col, lw=0.5, alpha=0.9, pad=1.4))
            # the one claim a serve/abstain policy provably cannot match
            if pt["label"] == "CAAC" and pt["coverage"] > caps["cap_serve"] + 1e-9:
                ax.annotate("above the serve/abstain ceiling",
                            xy=(pt["cost"], pt["coverage"]),
                            xytext=(pt["cost"], min(1.0, pt["coverage"] + 0.13)),
                            fontsize=8, color=col, ha="center",
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        ax.set_xlabel("mean cost per question (FLOPs, held out)")
        ax.set_ylabel("certified coverage on held-out questions")
        ax.set_title(f"{p['title']}\n"
                     f"pi={caps['pi']:.3f}, r={caps['r']:.2f}, "
                     f"n_calib={p['points'][0]['n_calib']}, "
                     f"n_test={p['points'][0]['n_test']}", fontsize=10)
        ax.set_ylim(0, 1.04)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7.5, loc="lower left", framealpha=0.95)
    fig.suptitle("Same risk guarantee, same test questions, same protocol: "
                 "only the action set differs", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="panel title; --small/--big/--json that follow belong to it")
    ap.add_argument("--small", action="append", required=True)
    ap.add_argument("--big", action="append", default=[])
    ap.add_argument("--json", action="append", default=[],
                    help="path=Label ; repeat. Use '|' to separate panels")
    ap.add_argument("--alpha", action="append", type=float, default=[])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="figs/certified_frontier")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    # --json entries are grouped per panel by a literal "|" separator
    groups, cur = [], []
    for item in a.json:
        if item == "|":
            groups.append(cur)
            cur = []
        else:
            cur.append(item)
    groups.append(cur)
    if len(groups) != len(a.panel):
        raise SystemExit(f"{len(a.panel)} panels but {len(groups)} --json groups "
                         f"(separate groups with --json '|')")

    bigs = a.big + [None] * (len(a.small) - len(a.big))
    panels = []
    print("\n=== CERTIFIED OPERATING POINTS ===")
    for title, sp, bp, grp in zip(a.panel, a.small, bigs, groups):
        pts = []
        for item in grp:
            path, _, label = item.partition("=")
            pts.extend(read_point(path, label or Path(path).stem))
        if not pts:
            raise SystemExit(f"panel {title!r}: no certified point in {grp}")
        alpha = pts[0]["alpha"]
        caps = panel_caps(sp, bp, alpha, a.k)
        panels.append({"title": title, "caps": caps, "points": pts})
        print(f"\n  {title}   pi={caps['pi']:.3f} r={caps['r']:.2f} "
              f"cap_serve={caps['cap_serve']:.3f} cap_correct={caps['cap_correct']:.3f}")
        print(f"    {'method':12} {'corr':>6} {'knob':>6} {'coverage':>9} {'risk':>7} "
              f"{'cost':>10} {'price':>7}")
        for pt in sorted(pts, key=lambda x: -x["coverage"]):
            print(f"    {pt['label']:12} {pt['method'][:4]:>6} {pt['knob']:6.3f} "
                  f"{pt['coverage']:9.3f} {pt['risk']:7.3f} {pt['cost']:10.3g} "
                  f"{(pt['price'] if pt['price'] is not None else float('nan')):+7.3f}"
                  + ("   RISK VIOLATED ON TEST" if pt["violated"] else ""))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".json").write_text(json.dumps(panels, indent=2))
    print(f"\n  -> {a.out}.json")
    if not a.no_plot:
        print(f"  -> {plot(panels, str(a.out) + '.png')} (+ .pdf)")


if __name__ == "__main__":
    main()
