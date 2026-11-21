"""Nested random subset of a qids file (2026-11-20, Day 2b budget).

  python subset_qids.py qids/a3_math_train.txt --n 400 --seed 0 --out qids/a3_math_train_400.txt

The qids files keep the dataset's file order (MATH is ordered by subject), so the
first N lines are NOT a random sample. Every qid gets a rank from a hash of
(seed, qid); the subset is the N lowest ranks, written back in file order. The
ranks do not depend on N, so the 400-question subset is contained in the 600 one:
growing the train set later (learning curve) only ADDS questions, and build_pool's
per-question resume generates just the new ones.
"""

from __future__ import annotations

import argparse
import hashlib


def rank(seed: int, qid: str) -> str:
    return hashlib.sha256(f"{seed}|{qid}".encode()).hexdigest()


def subset(qids: list[str], n: int, seed: int = 0) -> list[str]:
    if n > len(qids):
        raise SystemExit(f"asked for {n} of {len(qids)} qids")
    if len(set(qids)) != len(qids):
        raise SystemExit("duplicate qids in the input file")
    keep = set(sorted(qids, key=lambda q: rank(seed, q))[:n])
    return [q for q in qids if q in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qids")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qids = [l.strip() for l in open(a.qids) if l.strip()]
    out = subset(qids, a.n, a.seed)
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"{a.out}: {len(out)}/{len(qids)} qids (seed {a.seed})")


if __name__ == "__main__":
    main()
