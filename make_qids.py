"""Pick a STRATIFIED random subset of question ids for A3 (train/calib pools).

Why: the MATH mirror is ordered by subject (rows start with algebra), so
``--n-questions 1000`` would take ~all algebra and bias calibration -> the LTT
guarantee would be about the wrong distribution. This script samples with a
fixed seed, proportionally to subject x level (MATH) or uniformly (GSM8K), and
writes one qid per line for ``build_pool.py --qids-file``.

Decided 2026-11-05 (CAAC_MASTER §13): A3 MATH = 1,000 from math_train +
1,000 from math_test_ex500 (contamination check: compare train vs test
accuracy; the test half is a clean calib fallback). GSM8K = 1,500 from train.

Usage:
  python make_qids.py --dataset math_train      --n 1000 --out qids/a3_math_train.txt
  python make_qids.py --dataset math_test_ex500 --n 1000 --out qids/a3_math_test_ex500.txt
  python make_qids.py --dataset gsm8k --split train --n 1500 --out qids/a3_gsm8k_train.txt
"""

from __future__ import annotations

import argparse
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path


def stratified_sample(items: list[dict], n: int, keys=("subject", "level"), seed: int = 0) -> list[dict]:
    """Proportional allocation over strata (largest remainder), random within stratum.

    Deterministic given (items order, n, seed). If no item carries the keys,
    falls back to a uniform random sample.
    """
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        strata[tuple(it.get(k) for k in keys)].append(it)
    if len(strata) == 1:
        return sorted(rng.sample(items, n), key=lambda it: items.index(it))
    N = len(items)
    quotas = {s: n * len(v) / N for s, v in strata.items()}
    alloc = {s: int(q) for s, q in quotas.items()}
    rest = n - sum(alloc.values())
    # largest remainder; ties broken by a seeded shuffle (not by dict order)
    order = list(strata)
    rng.shuffle(order)
    order.sort(key=lambda s: -(quotas[s] - alloc[s]))
    for s in order[:rest]:
        alloc[s] += 1
    picked = []
    for s in sorted(strata, key=str):  # fixed iteration order
        pool = strata[s]
        k = min(alloc[s], len(pool))
        picked.extend(rng.sample(pool, k))
    pos = {id(it): i for i, it in enumerate(items)}
    return sorted(picked, key=lambda it: pos[id(it)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    # Contingency for benchmark saturation: if the big model is near ceiling on MATH-500 there
    # is no headroom left for a cost controller to demonstrate, so we re-draw from the hard tail
    # instead of a proportional sample. Levels are MATH's own "Level 1".."Level 5" strings.
    ap.add_argument("--min-level", type=int, default=None,
                    help="keep only items whose MATH level is >= this (e.g. 4)")
    a = ap.parse_args()

    from caac.data.benchmarks import load_benchmark
    items = load_benchmark(a.dataset, split=a.split)
    if a.min_level is not None:
        def lvl(it):
            m = re.search(r"\d+", str(it.get("level") or ""))
            return int(m.group()) if m else -1
        items = [it for it in items if lvl(it) >= a.min_level]
        if not items:
            raise SystemExit(f"--min-level {a.min_level}: no items left; check the level field")
    picked = stratified_sample(items, a.n, seed=a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(it["id"] + "\n" for it in picked))

    def dist(xs, key):
        return dict(sorted(Counter(str(it.get(key)) for it in xs).items()))
    summary = {"dataset": a.dataset, "split": a.split, "seed": a.seed,
               "min_level": a.min_level,
               "n_population": len(items), "n_picked": len(picked),
               "subject_pop": dist(items, "subject"), "subject_picked": dist(picked, "subject"),
               "level_pop": dist(items, "level"), "level_picked": dist(picked, "level")}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("dataset", "n_population", "n_picked")}))
    print("  subject picked:", summary["subject_picked"])
    print("  level picked:  ", summary["level_picked"])
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
