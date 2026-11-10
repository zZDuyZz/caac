"""Where do reasoning failures sit? (CPU only, no GPU.)

Prints, for one small/big pool pair, the four primitive rescue groups and the
three cost-dependent regions over the questions the small model got wrong, plus
the harm rate on the ones it got right and the mean net value of each action.

This is the measurement the allocation claim rests on, so it runs on its own
and before any policy: if the regions do not separate, no controller can
exploit them, and the paper has to say so.

    python recovery_table.py --small pool_qs.jsonl.gz --big pool_qb.jsonl.gz \
        --k 16 --lams 1,1.5,4.5 --by-level

``--lams`` prices a big-model token in small-model tokens: the region split is
the only thing that moves with it, which is the point of reporting several.
"""

from __future__ import annotations

import argparse
import json

from caac.pool.protocol import split_cost
from caac.pool.recovery import recovery_table
from caac.pool.schema import load_pool


def _price(lam: float, big_model: str | None):
    def fn(calls) -> float:
        s, b = split_cost(calls, big_model)
        return s + lam * b
    return fn


def _pct(d: dict) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in sorted(d.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--big", default=None)
    ap.add_argument("--k", type=int, default=16, help="sample budget the cheap repair may use")
    ap.add_argument("--lams", default="1,1.5,4.5", help="price of a big token in small tokens")
    ap.add_argument("--by-level", action="store_true",
                    help="split by MATH difficulty level when the pool records one")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    small = load_pool(a.small)
    big = load_pool(a.big) if a.big else None
    big_model = next(iter(big.values())).model if big else None
    lams = [float(x) for x in a.lams.split(",") if x.strip()]

    groups: list[tuple[str, list[str]]] = [("all", sorted(small))]
    if a.by_level:
        by: dict[str, list[str]] = {}
        for q, r in small.items():
            lv = (r.meta or {}).get("level") if hasattr(r, "meta") else None
            by.setdefault(f"level {lv}" if lv is not None else "level ?", []).append(q)
        if len(by) > 1:
            groups += [(name, sorted(qs)) for name, qs in sorted(by.items())]

    out = {"small": a.small, "big": a.big, "k": a.k, "slices": {}}
    for name, qids in groups:
        base = recovery_table(qids, small, big, k=a.k)
        print(f"\n=== {name}  n={base['n']}  wrong={base['n_greedy_wrong']} "
              f"right={base['n_greedy_right']}")
        print(f"  groups (of wrong)   {_pct(base['groups'])}")
        print(f"  harm   (of right)   {_pct(base['harm'])}")
        print(f"  mean delta (of all) {_pct(base['mean_delta'])}")
        print(f"  oracle - deployable {base['oracle_minus_deployable']:.3f}"
              "   (how much a perfect selector would add)")
        rec = {"base": base, "regions_by_lam": {}}
        for lam in lams:
            t = recovery_table(qids, small, big, k=a.k, price=_price(lam, big_model))
            print(f"  regions @ lam={lam:g}   {_pct(t['regions'])}")
            rec["regions_by_lam"][f"{lam:g}"] = t["regions"]
        out["slices"][name] = rec

    print("\nunrecoverable = not repaired by THIS action set, model pool and sample "
          "budget. Rescue uses the deployable majority vote, never 'some sample was right'.")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
