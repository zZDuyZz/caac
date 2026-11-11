"""Where does the compute go, and what could be improved? (CPU only.)

Runs five diagnostics on one small/big pool pair and prints, after each, what the
number means in plain words. Run it on TRAIN or CALIB questions only, before any
controller is changed; none of this is a result.

    python diagnose.py --small pool_qs.jsonl.gz --big pool_qb.jsonl.gz \
        --lams 1,1.5,4.5 --json diagnose.json

``--qids`` restricts to a list of query ids (one per line), e.g. the train split.
"""

from __future__ import annotations

import argparse
import json

from caac.pool.diagnostics import (action_costs, perfect_stop, recoverable_width,
                                   signal_auroc, truncation_kind)
from caac.pool.schema import load_pool, load_texts, text_sidecar_path

K_STEPS = (2, 4, 8, 16)


def _p(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--big", required=True)
    ap.add_argument("--lams", default="1,1.5,4.5", help="price of a big token in small tokens")
    ap.add_argument("--qids", default=None, help="file with one query id per line (default: all)")
    ap.add_argument("--texts", default=None, help="text sidecar (default: next to --small)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    small, big = load_pool(a.small), load_pool(a.big)
    qids = sorted(q for q in small if q in big)
    if a.qids:
        keep = {ln.strip() for ln in open(a.qids, encoding="utf-8") if ln.strip()}
        qids = [q for q in qids if q in keep]
    lams = [float(x) for x in a.lams.split(",") if x.strip()]
    out: dict = {"small": a.small, "big": a.big, "n": len(qids)}
    print(f"{len(qids)} questions\n")

    print("== D1  what each action costs (next to one greedy answer from the small model) ==")
    out["d1"] = []
    for lam in lams:
        r = action_costs(qids, small, big, lam)
        out["d1"].append(r)
        ratios = "  ".join(f"{k}={v:.1f}x" for k, v in r["ratio_to_greedy"].items() if k != "answer_now")
        print(f"  lam={lam:g}:  {ratios}")
    print("  Meaning: each number is how many greedy answers' worth of compute that action costs. "
          "Where escalation or 16 samples is many times the greedy answer, a controller saves "
          "mostly by NOT taking that step.\n")

    print("== D2  perfect-stop ceiling (what knowing in advance would save) ==")
    out["d2"] = []
    for lam in lams:
        r = perfect_stop(qids, small, big, lam)
        out["d2"].append(r)
        print(f"  lam={lam:g}:  right at once {_p(r['right_now'])}, repairable {_p(r['repairable'])}, "
              f"nothing fixes it {_p(r['unrecoverable'])}")
        print(f"           cost per question in greedy-answer units: perfect stop "
              f"{r['perfect_stop_cost'] / r['greedy_turn_cost']:.2f}x, "
              f"try everything {r['try_everything_cost'] / r['greedy_turn_cost']:.2f}x; "
              f"{_p(r['repair_waste_share'])} of the try-everything repair spending lands on hopeless questions")
    print("  Meaning: this is the room an early-stop rule has. A real controller recovers only part "
          "of it, because it cannot know in advance. The gap between this ceiling and what the "
          "policy achieves is the part worth improving. Compare 'right at once' with "
          "'nothing fixes it': the larger the first, the more serving early matters; the larger the "
          "second, the more abstaining early matters.\n")

    print("== D3  can two cheap samples tell the cases apart (AUROC: 0.5 chance, 1.0 perfect) ==")
    r = signal_auroc(qids, small, big)
    out["d3"] = r
    if "skipped" in r:
        print(f"  skipped: {r['skipped']}")
    for name, t in r["targets"].items():
        if "skipped" in t:
            print(f"  {name}: skipped ({t['skipped']})")
            continue
        print(f"  {name} ({t['positives']}/{t['n']} of the wrong answers):")
        for sname, v in t["signals"].items():
            print(f"     {sname:<46} {v['auroc']:.2f}  [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
    print("  Meaning: close to 0.5 means the signal says nothing and waste cannot be cut with it; "
          "above about 0.8 means a stop rule built on it can work. Well below 0.5 is also "
          "informative: the signal points the opposite way.\n")

    print("== D4  how wide is the cheap-to-repair region ==")
    r = recoverable_width(qids, small, big, lams, K_STEPS)
    out["d4"] = r
    for k, row in r.items():
        cells = "  ".join(f"lam={lam}: {_p(v['cheap_recoverable'])}" for lam, v in row.items())
        print(f"  {k:<5} cheap-recoverable share of wrong answers -> {cells}")
    print("  Meaning: the share of wrong answers that sampling fixes at no more than the big "
          "model's price. If it stays small at every budget, savings must come from abstaining and "
          "choosing the right step, not from sampling.\n")

    print("== D5  are the cut-off answers loops or just long ==")
    tp = a.texts or str(text_sidecar_path(a.small))
    texts = load_texts(tp)
    if not texts:
        print(f"  no text sidecar at {tp}; skipped\n")
        out["d5"] = {"skipped": "no text sidecar"}
    else:
        r = truncation_kind({q: small[q] for q in qids}, texts)
        out["d5"] = r
        print(f"  {r['n_truncated_with_text']} cut-off answers; {_p(r['loop_share'])} repeat themselves "
              f"(repeats in the last 60 words >= {r['threshold']})")
        print(f"  read these by hand before claiming anything: loops {r['loop_examples']}, "
              f"long {r['long_examples']}")
        print("  Meaning: loops can be stopped early at no cost to accuracy and save tokens for "
              "every method; long-but-progressing answers just need a larger cap.\n")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
