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

Writes ONE figure per panel (``<out>_<slug>.png`` and ``.pdf``).
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


def _slug(title: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in title]
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:40]


def _overlap(bb, taken) -> float:
    """Weighted overlap area. Obstacles carry a weight so a *soft* one (the
    shaded band: undesirable to cover, not forbidden) always loses to a hard
    one (another label, a marker) rather than pushing a label on top of text."""
    total = 0.0
    for o, w in taken:
        total += w * (max(0.0, min(bb.x1, o.x1) - max(bb.x0, o.x0))
                      * max(0.0, min(bb.y1, o.y1) - max(bb.y0, o.y0)))
    return total


def _place(ax, fig, x, y, text, colour, taken, pad=3.0):
    """Put a label near (x, y) in the first candidate slot that collides with
    nothing already drawn.

    Hand-tuned offsets break the moment the numbers change, and every one of
    these figures is going to be redrawn on the Session 2 pools. So the slots
    are tried in order of preference and accepted only when the rendered box
    overlaps neither a previous label nor the axes edge -- the layout repairs
    itself for whatever the next run produces.
    """
    slots = [(18, 12), (18, -42), (-18, 12), (-18, -42), (18, 58), (-18, 58),
             (18, -88), (-18, -88), (0, 66), (0, -96)]
    renderer = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer)

    def draw(dx, dy):
        return ax.annotate(
            text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=8.5, color=colour, zorder=6,
            ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
            va="bottom" if dy > 0 else "top",
            arrowprops=dict(arrowstyle="-", color=colour, lw=0.7,
                            shrinkA=2, shrinkB=6),
            bbox=dict(fc="white", ec=colour, lw=0.7, alpha=0.95,
                      boxstyle="round,pad=0.32"))

    best = None               # (penalty, dx, dy, bbox) -- the least-bad slot
    for dx, dy in slots:
        ann = draw(dx, dy)
        fig.canvas.draw()
        bb = ann.get_window_extent(renderer).expanded(1.0 + pad / 100, 1.0 + pad / 100)
        ann.remove()          # every trial is removed; only the winner is redrawn
        outside = (bb.x0 < axbb.x0 or bb.x1 > axbb.x1
                   or bb.y0 < axbb.y0 or bb.y1 > axbb.y1)
        penalty = _overlap(bb, taken) + (1e9 if outside else 0.0)
        if penalty == 0.0:
            best = (0.0, dx, dy, bb)
            break
        if best is None or penalty < best[0]:
            best = (penalty, dx, dy, bb)
    _, dx, dy, bb = best
    taken.append((bb, 1.0))
    return draw(dx, dy)


def plot_panel(p, out_png):
    """One dataset, one figure. Cost on a LINEAR axis in units of 1e12 FLOPs:
    a log axis spaced these few points so unevenly that its minor ticks ran
    into each other, and the quantity spans well under one decade anyway."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caps, pts = p["caps"], p["points"]
    costs = [pt["cost"] / 1e12 for pt in pts]
    lo, hi = min(costs), max(costs)
    span = max(hi - lo, 0.2 * hi)
    x0, x1 = max(0.0, lo - 0.45 * span), hi + 0.45 * span

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, 1.06)

    # the band only a correcting action can reach -- the figure's whole point
    ax.axhspan(caps["cap_serve"], caps["cap_correct"], color="#1f77b4", alpha=0.07,
               zorder=0)
    ax.axhline(caps["cap_serve"], color="#d62728", ls="--", lw=1.7, zorder=1)
    ax.axhline(caps["cap_correct"], color="#1f77b4", ls="--", lw=1.7, zorder=1)

    fig.canvas.draw()
    taken = []
    renderer = fig.canvas.get_renderer()

    # markers FIRST, so every piece of text afterwards has to route around them
    for pt in pts:
        col, mk = METHOD_STYLE.get(pt["label"], FALLBACK)
        solid = pt["method"] != "bonferroni"
        h, = ax.plot([pt["cost"] / 1e12], [pt["coverage"]], mk, color=col,
                     mfc=col if solid else "white", ms=13, mew=2.0, zorder=5)
        fig.canvas.draw()
        taken.append((h.get_window_extent(renderer).expanded(1.7, 1.7), 1.0))

    def line_text(yd, txt, colour, ha="left"):
        """A caption that must sit ON its own horizontal line: only the x
        position and the side of the line are free, so try those."""
        choices = [(xf, va) for va in ("bottom", "top")
                   for xf in (0.015, 0.34, 0.62, 0.985)]
        best = None
        for xf, va in choices:
            t = ax.text(xf, yd, txt, transform=ax.get_yaxis_transform(),
                        fontsize=8.5, color=colour, va=va, zorder=6,
                        ha="right" if xf > 0.9 else ha,
                        bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.6))
            fig.canvas.draw()
            bb = t.get_window_extent(renderer)
            t.remove()
            out = bb.x0 < axbb.x0 or bb.x1 > axbb.x1 or bb.y0 < axbb.y0 or bb.y1 > axbb.y1
            pen = _overlap(bb, taken) + (1e9 if out else 0.0)
            if pen == 0.0:
                best = (0.0, xf, va, bb)
                break
            if best is None or pen < best[0]:
                best = (pen, xf, va, bb)
        _, xf, va, bb = best
        taken.append((bb, 1.0))
        return ax.text(xf, yd, txt, transform=ax.get_yaxis_transform(), fontsize=8.5,
                       color=colour, va=va, ha="right" if xf > 0.9 else ha, zorder=6,
                       bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.6))

    axbb = ax.get_window_extent(renderer)
    line_text(caps["cap_serve"],
              f"Chow ceiling, serve/abstain only = {caps['cap_serve']:.3f}", "#d62728")
    line_text(caps["cap_correct"],
              f"Chow ceiling, + answer-correcting actions = {caps['cap_correct']:.3f}",
              "#1f77b4")
    if caps["cap_correct"] - caps["cap_serve"] > 0.06:
        line_text((caps["cap_serve"] + caps["cap_correct"]) / 2,
                  "reachable only with answer-correcting actions", "#1f77b4")

    # Keep point labels OUT of the shaded band. A caption floating inside the
    # "only reachable with correcting actions" region reads as if that point
    # were in it, which for a baseline would be exactly the wrong claim.
    bx0, by0 = ax.transData.transform((x0, caps["cap_serve"]))
    bx1, by1 = ax.transData.transform((x1, caps["cap_correct"]))
    if by1 > by0:
        from matplotlib.transforms import Bbox
        taken.append((Bbox.from_extents(bx0, by0, bx1, by1), 0.08))

    for pt in sorted(pts, key=lambda x: -x["coverage"]):
        col, _ = METHOD_STYLE.get(pt["label"], FALLBACK)
        solid = pt["method"] != "bonferroni"
        tag = (f"{pt['label']}  ({'fixed-sequence' if solid else 'Bonferroni'})\n"
               f"coverage {pt['coverage']:.3f}   risk {pt['risk']:.3f}\n"
               f"cost {pt['cost'] / 1e12:.2f}e12 FLOPs")
        if pt["violated"]:
            tag += "\nrisk VIOLATED on test"
        _place(ax, fig, pt["cost"] / 1e12, pt["coverage"], tag, col, taken)

    ax.set_xlabel("mean cost per question  (10$^{12}$ FLOPs, held out)"
                  "   \u2192 cheaper is better")
    ax.set_ylabel("certified coverage on held-out questions")
    ax.set_title(f"{p['title']}\n"
                 f"\u03c0={caps['pi']:.3f}   r={caps['r']:.2f}   "
                 f"n_calib={pts[0]['n_calib']}   n_test={pts[0]['n_test']}\n"
                 "same test questions, same LTT protocol \u2014 only the action set differs",
                 fontsize=9.5)
    ax.grid(alpha=0.2, zorder=0)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)
    return out_png


def plot(panels, out_stem):
    """One figure PER dataset -- two panels in one canvas left the cost axes on
    different scales side by side, which invites exactly the wrong comparison."""
    return [plot_panel(p, f"{out_stem}_{_slug(p['title'])}.png") for p in panels]


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
        for f in plot(panels, str(a.out)):
            print(f"  -> {f} (+ .pdf)")


if __name__ == "__main__":
    main()
