"""Automatic smoke gate for Session 2 (replaces the manual 30-s eyeball pause).

  python smoke_check.py /workspace/pools_v2/smoke_gsm8k.jsonl.gz [--min-correct 3]

Exit code 0 = OK; non-zero = STOP (the run script aborts before spending GPU hours).
Checks: pool/text/token sidecars aligned by query_id; per-token arrays as long as
decode_tokens; conf_prefix32 and answer_raw present; the generated text really
contains the raw answer; at least --min-correct of the greedy answers graded
correct by Math-Verify (a dead grader or broken prompt shows up as ~0).
Always prints the table (gold / raw / MV / legacy / conf32) so it lands in the log.
"""

from __future__ import annotations

import argparse
import sys

from caac.pool.schema import load_pool, load_texts, text_sidecar_path, token_sidecar_path


def check(pool_path, min_correct=0):
    pool = load_pool(pool_path)
    texts = load_texts(text_sidecar_path(pool_path))
    toks = load_texts(token_sidecar_path(pool_path))   # same one-JSON-per-line format
    errs = []
    if not pool:
        return ["empty pool"], []
    if set(pool) != set(texts) or set(pool) != set(toks):
        errs.append(f"sidecar mismatch: pool={len(pool)} text={len(texts)} tok={len(toks)}")
    rows, n_ok = [], 0
    for q, r in pool.items():
        g = r.greedy
        t = toks.get(q)
        if g.conf_prefix32 is None:
            errs.append(f"{q}: conf_prefix32 missing")
        if g.answer_raw is None and g.finish_reason != "length":
            errs.append(f"{q}: answer_raw missing on a finished greedy")
        if t is not None:
            if len(t["greedy"]["lp"]) != g.decode_tokens:
                errs.append(f"{q}: greedy token sidecar length {len(t['greedy']['lp'])} != decode_tokens {g.decode_tokens}")
            if len(t["samples"]) != len(r.samples):
                errs.append(f"{q}: token sidecar has {len(t['samples'])} samples, pool {len(r.samples)}")
        tx = texts.get(q)
        if tx is not None and g.answer_raw is not None and g.answer_raw not in tx["greedy_text"]:
            errs.append(f"{q}: raw answer not found inside the saved greedy text")
        n_ok += bool(g.correct)
        rows.append((q, r.gold, g.answer_raw, g.correct, g.correct_legacy, g.conf_prefix32,
                     sum(s.correct for s in r.samples), len(r.samples), g.finish_reason))
    if n_ok < min_correct:
        errs.append(f"only {n_ok}/{len(pool)} greedy answers graded correct (< {min_correct}): grader/prompt suspect")
    return errs, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--min-correct", type=int, default=0)
    a = ap.parse_args()
    errs, rows = check(a.pool, a.min_correct)
    for q, gold, raw, mv, lg, c32, k, n, fin in rows:
        c = f"{c32:.3f}" if c32 is not None else "None"
        print(f"  {q:14} gold={gold!r:30} raw={raw!r:30} MV={mv} legacy={lg} conf32={c} "
              f"samples={k}/{n} finish={fin}")
    if errs:
        print("SMOKE FAIL:")
        for e in errs[:20]:
            print("  -", e)
        sys.exit(1)
    print("SMOKE OK:", a.pool)


if __name__ == "__main__":
    main()
