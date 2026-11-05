"""Re-grade an existing pool with Math-Verify (CPU only) -- CAAC_MASTER §6.3.

Why: the GPU build may have graded with the weaker ``answers_match`` (e.g. it
misses ``60\\%`` == ``60``). This re-grades every completion with the PRIMARY
grader (Math-Verify) and writes a NEW pool with ``correct`` updated and
``correct_legacy`` preserved, WITHOUT touching the GPU.

Two sources, best first:
  * TEXT sidecar (``<pool>.text.jsonl.gz``): re-extract the raw answer from the
    stored generation text (lets a better extractor help too). Preferred.
  * stored normalized answer only (no sidecar): LOSSY -- Math-Verify sees the
    already-normalized string, so some equivalences are already destroyed. A
    warning is printed and the result is a LOWER BOUND on what a full re-grade
    would recover.

Also reports two-way disagreement between the graders and prints sample cases
for the ~50-case hand check PLAN §4 asks for (watch Math-Verify's over-permissive
cases: dropped bases like ``2516_8`` ~ ``2516``, dropped units like ``12\\text{cm}``).

Usage:
  python regrade_pool.py pool.jsonl.gz                 # -> pool.regraded.jsonl.gz
  python regrade_pool.py pool.jsonl.gz --out out.jsonl.gz --show 30
  python regrade_pool.py pool.jsonl.gz --no-text       # force lossy mode
"""

from __future__ import annotations

import argparse
from pathlib import Path

from caac.pool.schema import (load_pool, write_records, text_sidecar_path,
                              load_texts)
from caac.eval.answer_match import extract_answer, normalize_answer, answers_match
from caac.eval.grade import math_verify_correct, mathverify_available


def _pred_for(completion, text, use_text):
    """String to hand Math-Verify: raw from text if available, else stored answer."""
    if use_text and text is not None:
        raw = extract_answer(text)
        if raw is not None:
            return raw
    return completion.answer  # already-normalized (lossy) fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--out", default=None, help="default: <pool>.regraded.jsonl.gz")
    ap.add_argument("--no-text", action="store_true", help="ignore the text sidecar (lossy)")
    ap.add_argument("--show", type=int, default=20, help="sample disagreement cases to print")
    args = ap.parse_args()

    if not mathverify_available():
        raise SystemExit("math-verify not importable: `pip install math-verify`")

    pool_path = Path(args.pool)
    out_path = Path(args.out) if args.out else pool_path.with_name(
        pool_path.name.replace(".jsonl.gz", "").replace(".jsonl", "") + ".regraded.jsonl.gz")

    recs = load_pool(pool_path)
    sc = text_sidecar_path(pool_path)
    texts = {} if args.no_text else load_texts(sc)
    use_text = bool(texts)
    print(f"[{pool_path.name}] records={len(recs)}  "
          f"text sidecar: {'YES ('+str(sc.name)+')' if use_text else 'NONE -> LOSSY regrade'}")

    n_g = len(recs)
    base_correct = sum(1 for r in recs.values() if r.greedy.correct)
    base_legacy = sum(1 for r in recs.values() if r.greedy.correct_legacy)
    flips_to_true, flips_to_false = [], []
    disagree = 0  # new(MV) vs legacy on greedy

    for qid, r in recs.items():
        t = texts.get(qid, {})
        gtext = t.get("greedy_text")
        stexts = t.get("sample_texts", [])

        def regrade(c, text):
            pred = _pred_for(c, text, use_text)
            mv = math_verify_correct(r.gold, pred)
            if mv is None:
                return  # keep as-is
            if c.correct_legacy is None:  # backfill legacy if missing
                ans = c.answer
                c.correct_legacy = bool(ans is not None and answers_match(ans, r.gold))
            c.correct = mv

        before = r.greedy.correct
        regrade(r.greedy, gtext)
        if r.greedy.correct and not before:
            flips_to_true.append((qid, r.gold, r.greedy.answer))
        elif before and not r.greedy.correct:
            flips_to_false.append((qid, r.gold, r.greedy.answer))
        if bool(r.greedy.correct) != bool(r.greedy.correct_legacy):
            disagree += 1

        for c, text in zip(r.samples, stexts + [None] * (len(r.samples) - len(stexts))):
            regrade(c, text)

    new_correct = sum(1 for r in recs.values() if r.greedy.correct)
    write_records(out_path, list(recs.values()))

    print(f"  greedy acc: legacy={base_legacy/n_g:.3f}  old-correct={base_correct/n_g:.3f}  "
          f"-> Math-Verify={new_correct/n_g:.3f}")
    print(f"  greedy flips: ->correct={len(flips_to_true)}  ->wrong={len(flips_to_false)}  "
          f"| MV-vs-legacy disagreement (greedy)={disagree}/{n_g}")
    print(f"  wrote {out_path}")
    if args.show:
        print(f"  --- up to {args.show} greedy cases MV now marks CORRECT (hand-check over-permissive): ---")
        for qid, gold, pred in flips_to_true[:args.show]:
            print(f"    {qid}: gold={gold!r:30} pred={pred!r}")
        if flips_to_false:
            print(f"  --- greedy cases MV now marks WRONG (was correct): ---")
            for qid, gold, pred in flips_to_false[:args.show]:
                print(f"    {qid}: gold={gold!r:30} pred={pred!r}")


if __name__ == "__main__":
    main()
