"""One-file summary of Session 2 -- send THIS to the assistant instead of the log.

  python s2_summary.py --pools /workspace/pools_v2 --out outputs/s2/s2_summary.md

Per pool: size, greedy accuracy, maj@k, truncation, grader disagreement (MV vs
answers_match), mean decode tokens, token-sidecar coverage, the engine/version
recorded in its manifest, a thinking-leak count, the determinism check
(re-generated slices vs the main pool, same model and seeds, in the same and in
a different batch composition). Missing files are reported, not fatal.

2026-11-06: the old "reproducibility vs A2" section is gone -- A2 is Qwen2.5 and
Session 2 now builds Qwen3.5 pools, so a per-question comparison would compare
two different models. ``--old`` is accepted and ignored for old command lines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caac.pool.estimators import maj_at_k_exact
from caac.pool.schema import GOLD, load_pool, text_sidecar_path, token_sidecar_path

TEST_POOLS = ["math500_qs", "math500_qb", "gsm8k_qs", "gsm8k_qb"]
THINK_MARKERS = ("<think>", "</think>", "<|think|>")


def thinking_leaks(path) -> int:
    """Completions whose saved text contains a thinking marker (protocol says OFF)."""
    from caac.pool.schema import load_texts
    tp = text_sidecar_path(path)
    if not tp.exists():
        return -1
    n = 0
    for rec in load_texts(tp).values():
        for t in [rec.get("greedy_text") or ""] + list(rec.get("sample_texts") or []):
            n += any(m in (t or "") for m in THINK_MARKERS)
    return n


def manifest_engine(path) -> str:
    mp = Path(str(path) + ".manifest.json")
    if not mp.exists():
        return "?"
    m = json.loads(mp.read_text())
    v = m.get("versions") or {}
    return f"{m.get('engine', 'v0?')} vllm={v.get('vllm')} lp={m.get('logprobs_mode', '-')}"


def pool_stats(path):
    pool = load_pool(path)
    n = len(pool)
    recs = list(pool.values())
    g_acc = sum(r.greedy.correct for r in recs) / n
    g_leg = sum(bool(r.greedy.correct_legacy) for r in recs) / n
    dis = sum(bool(r.greedy.correct) != bool(r.greedy.correct_legacy) for r in recs) / n
    ns = min(r.n_samples for r in recs)
    k = min(ns, 16)
    maj = (sum(maj_at_k_exact(r.canonical_sample_answers(), GOLD, k) for r in recs) / n) if ns >= 1 else float("nan")
    tr_g = sum(r.greedy.finish_reason == "length" for r in recs) / n
    tot = sum(len(r.samples) for r in recs) or 1
    tr_s = sum(c.finish_reason == "length" for r in recs for c in r.samples) / tot
    dec = sum(r.greedy.decode_tokens for r in recs) / n
    c32 = sum(r.greedy.conf_prefix32 is not None for r in recs) / n
    return {"n": n, "greedy_acc": g_acc, "greedy_acc_legacy": g_leg, "grader_disagree": dis,
            "k": k, "maj_k": maj, "trunc_greedy": tr_g, "trunc_samples": tr_s,
            "mean_decode_greedy": dec, "conf32_coverage": c32,
            "has_text": text_sidecar_path(path).exists(), "has_tok": token_sidecar_path(path).exists(),
            "think_leaks": thinking_leaks(path), "engine": manifest_engine(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--old", default=None, help="ignored (kept for old command lines)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = Path(a.pools)
    lines = ["# Session 2 summary", ""]
    lines += ["| pool | n | greedy acc (MV) | acc legacy | MV≠legacy | maj@k | trunc g/s | tok/greedy | conf32 | text | tok | think leaks | engine |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    names = sorted(p.name[:-len(".jsonl.gz")] for p in P.glob("*.jsonl.gz")
                   if not any(t in p.name for t in (".text.", ".tok.", ".regraded."))
                   and not p.name.startswith(("smoke_", "det_")))
    for nm in names:
        try:
            s = pool_stats(P / f"{nm}.jsonl.gz")
        except Exception as e:
            lines.append(f"| {nm} | ERROR {e} |"); continue
        lines.append(f"| {nm} | {s['n']} | {s['greedy_acc']:.3f} | {s['greedy_acc_legacy']:.3f} | {s['grader_disagree']:.3f} | "
                     f"{s['maj_k']:.3f} (k={s['k']}) | {s['trunc_greedy']:.1%}/{s['trunc_samples']:.1%} | "
                     f"{s['mean_decode_greedy']:.0f} | {s['conf32_coverage']:.0%} | {'Y' if s['has_text'] else 'N'} | {'Y' if s['has_tok'] else 'N'} | "
                     f"{s['think_leaks']}{' **CHECK**' if s['think_leaks'] > 0 else ''} | {s['engine']} |")
    missing = [nm for nm in TEST_POOLS if not (P / f"{nm}.jsonl.gz").exists()]
    if missing:
        lines += ["", f"Missing test pools: {', '.join(missing)}"]
    lines += ["", "## Determinism (re-generated GSM8K slices vs the main pool, same seeds)", ""]
    det_main = P / "gsm8k_qs.jsonl.gz"
    for nm, label, want in (("det_same_gsm8k_qs", "same batch composition", 0.98),
                            ("det_gsm8k_qs", "different batch composition", None)):
        det_new = P / f"{nm}.jsonl.gz"
        if not (det_main.exists() and det_new.exists()):
            lines.append(f"- {label}: not run")
            continue
        from compare_pools import compare
        r = compare(load_pool(det_main), load_pool(det_new))
        # Only the same-batch arm must match: V1 is not batch-invariant by design.
        flag = ("  **<98%: INVESTIGATE (our pipeline, not vLLM)**"
                if want is not None and r["greedy_answer_match"] < want else "")
        lines.append(f"- {label}: common={r['n_common']}  greedy={r['greedy_answer_match']:.3f}  "
                     f"length={r['greedy_len_match']:.3f}  samples={r['sample_match']:.3f}{flag}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
