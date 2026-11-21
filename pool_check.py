"""Check one pool right after it is written (2026-11-20, lessons from Day 2a).

  python pool_check.py POOL.jsonl.gz [--label NAME] [--seconds S] [--times TSV]

Prints accuracy, truncation (greedy / samples), thinking-marker leaks and tokens
per completion, with WARN lines when a figure needs a look -- Day 2a learned the
9.2% greedy truncation of math500_qs and the </think> leaks only after all four
pools had finished. With --times it appends one row per pool (label, questions,
completions, decode tokens, seconds, tokens/s, seconds per completion) so the
next budget comes from measured figures. Never fatal: exit code is always 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

WARN_TRUNC = 0.05      # share of completions cut at the cap
WARN_LEAK = 0.01       # share of completions with a thinking marker


def check(path, label=None, seconds=None, times=None):
    from s2_summary import pool_stats
    from caac.pool.schema import load_pool
    s = pool_stats(path)
    pool = load_pool(path)
    n = s["n"]
    comps = sum(1 + len(r.samples) for r in pool.values())
    toks = sum(r.greedy.decode_tokens + sum(c.decode_tokens for c in r.samples) for r in pool.values())
    label = label or Path(path).name
    leak = s["think_leaks"]
    print(f"[pool_check] {label}: n={n} greedy acc={s['greedy_acc']:.3f} maj@{s['k']}={s['maj_k']:.3f} "
          f"cut greedy/samples={s['trunc_greedy']:.1%}/{s['trunc_samples']:.1%} "
          f"think leaks={leak}{'' if leak < 0 else f' ({leak / max(comps, 1):.1%})'} "
          f"tok/completion={toks / max(comps, 1):.0f}")
    if s["trunc_greedy"] > WARN_TRUNC or s["trunc_samples"] > WARN_TRUNC:
        print(f"  WARN {label}: more than {WARN_TRUNC:.0%} cut at the cap -- note it; the cap is locked for this TAG, do NOT change it mid-run")
    if leak > 0 and leak / max(comps, 1) > WARN_LEAK:
        print(f"  WARN {label}: thinking markers in {leak} completions -- grading rule is decided on TRAIN later")
    if seconds is not None and times:
        row = [label, n, comps, toks, int(seconds), f"{toks / max(seconds, 1):.0f}", f"{seconds / max(comps, 1):.4f}"]
        new = not Path(times).exists()
        with open(times, "a") as f:
            if new:
                f.write("label\tquestions\tcompletions\tdecode_tokens\tseconds\ttok_per_s\ts_per_completion\n")
            f.write("\t".join(map(str, row)) + "\n")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--label")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--times")
    a = ap.parse_args()
    try:
        check(a.pool, a.label, a.seconds, a.times)
    except Exception as e:  # noqa: BLE001  never stop a paid run over a report
        print(f"  WARN pool_check could not read {a.pool}: {e}")


if __name__ == "__main__":
    main()
