"""Reproducibility check: a regenerated pool vs the old pool (same model,
dataset, config, GPU type).

2026-11-20 (Day 2a): token-for-token identity is NOT what vLLM V1 delivers -- two runs
of identical requests matched 0.893 on greedy answers and 0.237 on samples (150 GSM8K
questions, H200). The figure that matters for the paper is whether ACCURACY moves:
the pools are random draws, and a certificate only needs the two runs to come from the
same distribution. So the check now also reports greedy and sample accuracy of both
runs on the common questions, with a paired 95% interval of the difference, and warns
only when that interval excludes 0. Answer/length match stay as descriptive figures.

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
    acc = accuracy_shift(old, new, common)
    return {"n_old": len(old), "n_new": len(new), "n_common": len(common),
            "greedy_answer_match": g_ans / max(len(common), 1),
            "greedy_len_match": g_tok / max(len(common), 1),
            "sample_match": s_ans / max(s_tot, 1), **acc}


def _paired(diffs):
    """Mean of per-question differences and its 95% normal interval (None when n < 2)."""
    n = len(diffs)
    if n < 2:
        return (sum(diffs) / n if n else 0.0), None
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    h = 1.96 * (var / n) ** 0.5
    return m, (m - h, m + h)


def accuracy_shift(old, new, common):
    """Greedy and mean-sample accuracy of both runs on the common questions, paired by question.

    Samples are compared on the first k draws both records hold, so a slice re-generated
    with fewer samples (arm b: 4) is matched against the same first 4 of the main pool."""
    g_old = [float(old[q].greedy.correct) for q in common]
    g_new = [float(new[q].greedy.correct) for q in common]
    s_old, s_new = [], []
    for q in common:
        k = min(len(old[q].samples), len(new[q].samples))
        if k:
            s_old.append(sum(c.correct for c in old[q].samples[:k]) / k)
            s_new.append(sum(c.correct for c in new[q].samples[:k]) / k)
    n = max(len(common), 1)
    g_d, g_ci = _paired([b - a for a, b in zip(g_old, g_new)])
    s_d, s_ci = _paired([b - a for a, b in zip(s_old, s_new)])
    return {"greedy_acc_old": sum(g_old) / n, "greedy_acc_new": sum(g_new) / n,
            "greedy_acc_diff": g_d, "greedy_acc_ci": g_ci,
            "sample_acc_old": (sum(s_old) / len(s_old)) if s_old else float("nan"),
            "sample_acc_new": (sum(s_new) / len(s_new)) if s_new else float("nan"),
            "sample_acc_diff": s_d, "sample_acc_ci": s_ci}


def shifted(r) -> bool:
    """True when a paired 95% interval of an accuracy difference excludes 0."""
    return any(ci is not None and (ci[0] > 0 or ci[1] < 0)
               for ci in (r["greedy_acc_ci"], r["sample_acc_ci"]))


def _fmt_ci(ci):
    return "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def main():
    old, new = load_pool(sys.argv[1]), load_pool(sys.argv[2])
    r = compare(old, new)
    print(f"[compare] {sys.argv[2]}  common={r['n_common']} (old {r['n_old']}, new {r['n_new']})  "
          f"greedy answer={r['greedy_answer_match']:.3f} length={r['greedy_len_match']:.3f}  "
          f"samples={r['sample_match']:.3f}")
    print(f"  accuracy greedy {r['greedy_acc_old']:.3f} -> {r['greedy_acc_new']:.3f} "
          f"(diff {r['greedy_acc_diff']:+.3f}, 95% CI {_fmt_ci(r['greedy_acc_ci'])})  "
          f"samples {r['sample_acc_old']:.3f} -> {r['sample_acc_new']:.3f} "
          f"(diff {r['sample_acc_diff']:+.3f}, 95% CI {_fmt_ci(r['sample_acc_ci'])})")
    if shifted(r):
        print("  WARNING: accuracy moved between the two runs (95% CI excludes 0) -- the runs may not "
              "share one distribution; check config/GPU type/model revision before pooling them")
    elif r["greedy_answer_match"] < 0.98:
        print("  note: outputs differ token by token (expected with vLLM V1); accuracy shows no shift")


if __name__ == "__main__":
    main()
