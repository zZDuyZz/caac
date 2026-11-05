"""Reproducibility check: a regenerated pool vs the old pool (same model,
dataset, config, GPU type). Greedy should match ~exactly; samples should match
too (same per-question seeds), up to rare bf16 nondeterminism.

  python compare_pools.py OLD.jsonl.gz NEW.jsonl.gz
"""

from __future__ import annotations

import sys

from caac.pool.schema import load_pool


def compare(old, new):
    common = sorted(set(old) & set(new))
    g_ans = sum(old[q].greedy.answer == new[q].greedy.answer for q in common)
    g_tok = sum(old[q].greedy.decode_tokens == new[q].greedy.decode_tokens for q in common)
    s_tot = s_ans = 0
    for q in common:
        for a, b in zip(old[q].samples, new[q].samples):
            s_tot += 1
            s_ans += a.answer == b.answer and a.decode_tokens == b.decode_tokens
    return {"n_old": len(old), "n_new": len(new), "n_common": len(common),
            "greedy_answer_match": g_ans / max(len(common), 1),
            "greedy_len_match": g_tok / max(len(common), 1),
            "sample_match": s_ans / max(s_tot, 1)}


def main():
    old, new = load_pool(sys.argv[1]), load_pool(sys.argv[2])
    r = compare(old, new)
    print(f"[compare] {sys.argv[2]}  common={r['n_common']} (old {r['n_old']}, new {r['n_new']})  "
          f"greedy answer={r['greedy_answer_match']:.3f} length={r['greedy_len_match']:.3f}  "
          f"samples={r['sample_match']:.3f}")
    if r["greedy_answer_match"] < 0.98:
        print("  WARNING: greedy differs on >2% -- check config/GPU type/model revision before using")


if __name__ == "__main__":
    main()
