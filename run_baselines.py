"""The accuracy-vs-cost sweep over every always-answer baseline (§11).

These policies never abstain, so accuracy and cost are the only axes they have.
That is the point of running them here: they say how much accuracy plain
test-time scaling buys per FLOP, which is the curve CAAC has to beat before any
claim about risk control is interesting.

    python run_baselines.py --small outputs/a2/gsm8k_q1.5b.jsonl.gz \
        --big outputs/a2/gsm8k_q7b.jsonl.gz --json figs/baselines_gsm8k.json

Reported for each configuration: accuracy, mean cost under three profiles, and
the mean number of samples actually drawn. The ``rounds`` column is the one the
adaptive-sampling literature usually omits -- Adaptive-Consistency's draws are
sequential, so it wins on tokens and loses on latency, and both belong in the
table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caac.pool.adaptive import adaptive_consistency, deepconf_offline, esc
from caac.pool.ledger import cost_profiles
from caac.pool.policies import cascade_force_escalate, greedy_only, self_consistency
from caac.pool.replay import make_orders
from caac.pool.schema import load_pool

PROFILES = ("tokens", "flops", "rounds")


def configs(k: int):
    """(name, fn(small, big, order)) for every always-answer baseline."""
    out = [("greedy (small)", lambda s, b, o: greedy_only(s))]
    for kk in (2, 4, k):
        if kk <= k:
            out.append((f"self-consistency k={kk}",
                        lambda s, b, o, kk=kk: self_consistency(s, kk, o)))
    for t in (0.90, 0.95, 0.99):
        out.append((f"adaptive-consistency t={t}",
                    lambda s, b, o, t=t: adaptive_consistency(s, o, t, k)))
    for w in (2, 4):
        out.append((f"ESC window={w}", lambda s, b, o, w=w: esc(s, o, w, k)))
    for e in (0.25, 0.5):
        out.append((f"DeepConf offline eta={e}",
                    lambda s, b, o, e=e: deepconf_offline(s, o, k, e)))
    out.append(("greedy (big) = always escalate",
                lambda s, b, o: cascade_force_escalate(s, b) if b else greedy_only(s)))
    return out


def run(qids, small, big, fn, orders):
    acc = 0.0
    cost = {p: 0.0 for p in PROFILES}
    drawn = 0.0
    for q in qids:
        for o in orders:
            out = fn(small[q], (big or {}).get(q), o)
            acc += float(out.correct)
            c = cost_profiles(out.calls)
            for p in PROFILES:
                cost[p] += c[p]
            # count completions from the BILLED calls, not from policy metadata:
            # it is the same number for every policy and cannot drift from cost
            drawn += sum(len(call.decode_tokens) for call in out.calls)
    n = len(qids) * len(orders)
    return {"accuracy": acc / n, "mean_samples": drawn / n,
            **{f"cost_{p}": cost[p] / n for p in PROFILES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--big", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--perms", type=int, default=10)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    small = load_pool(a.small)
    big = load_pool(a.big) if a.big else None
    qids = sorted(small if big is None else set(small) & set(big))
    orders = make_orders(min(r.n_samples for r in small.values()), a.perms, seed=0)

    print(f"\n=== ALWAYS-ANSWER BASELINES ({Path(a.small).name}, n={len(qids)}, "
          f"k={a.k}, {a.perms} permutations) ===")
    print(f"  {'policy':34} {'acc':>6} {'samples':>8} {'tokens':>9} "
          f"{'flops':>10} {'rounds':>7}")
    rows = {}
    for name, fn in configs(a.k):
        r = run(qids, small, big, fn, orders)
        rows[name] = r
        print(f"  {name:34} {r['accuracy']:6.3f} {r['mean_samples']:8.2f} "
              f"{r['cost_tokens']:9.1f} {r['cost_flops']:10.3g} {r['cost_rounds']:7.2f}")

    base = rows["greedy (small)"]["cost_flops"]
    print(f"\n  cost relative to answering immediately (FLOPs):")
    for name, r in rows.items():
        print(f"    {name:34} x{r['cost_flops'] / base:5.2f}   "
              f"acc {r['accuracy'] - rows['greedy (small)']['accuracy']:+.3f}")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"small": a.small, "big": a.big, "n": len(qids), "k": a.k,
             "perms": a.perms, "rows": rows}, indent=2))
        print(f"  -> {a.json}")


if __name__ == "__main__":
    main()
