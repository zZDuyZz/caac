"""One-file summary of Session 2 -- send THIS to the assistant instead of the log.

  python s2_summary.py --pools /workspace/pools_v2 --old outputs/a2 --out outputs/s2/s2_summary.md

Per pool: size, greedy accuracy, maj@k, truncation, grader disagreement (MV vs
answers_match), mean decode tokens, token-sidecar coverage, and (for the 4 test
pools) reproducibility vs the old A2 pool. Plus the go/no-go verdicts if present.
Missing files are reported, not fatal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caac.pool.estimators import maj_at_k_exact
from caac.pool.schema import GOLD, load_pool, text_sidecar_path, token_sidecar_path

TEST_POOLS = ["math500_q1.5b", "math500_q7b", "gsm8k_q1.5b", "gsm8k_q7b"]


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
            "has_text": text_sidecar_path(path).exists(), "has_tok": token_sidecar_path(path).exists()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--old", default="outputs/a2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P, O = Path(a.pools), Path(a.old)
    lines = ["# Session 2 summary", ""]
    lines += ["| pool | n | greedy acc (MV) | acc legacy | MV≠legacy | maj@k | trunc g/s | tok/greedy | conf32 | text | tok |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    names = sorted(p.name[:-len(".jsonl.gz")] for p in P.glob("*.jsonl.gz")
                   if not any(t in p.name for t in (".text.", ".tok.", ".regraded.")) and not p.name.startswith("smoke_"))
    for nm in names:
        try:
            s = pool_stats(P / f"{nm}.jsonl.gz")
        except Exception as e:
            lines.append(f"| {nm} | ERROR {e} |"); continue
        lines.append(f"| {nm} | {s['n']} | {s['greedy_acc']:.3f} | {s['greedy_acc_legacy']:.3f} | {s['grader_disagree']:.3f} | "
                     f"{s['maj_k']:.3f} (k={s['k']}) | {s['trunc_greedy']:.1%}/{s['trunc_samples']:.1%} | "
                     f"{s['mean_decode_greedy']:.0f} | {s['conf32_coverage']:.0%} | {'Y' if s['has_text'] else 'N'} | {'Y' if s['has_tok'] else 'N'} |")
    lines += ["", "## Reproducibility vs A2 (greedy answer / length / samples match)", ""]
    from compare_pools import compare
    for nm in TEST_POOLS:
        new, old = P / f"{nm}.jsonl.gz", O / f"{nm}.jsonl.gz"
        if new.exists() and old.exists():
            r = compare(load_pool(old), load_pool(new))
            flag = "" if r["greedy_answer_match"] >= 0.98 else "  **<98%: INVESTIGATE**"
            lines.append(f"- {nm}: common={r['n_common']}  greedy={r['greedy_answer_match']:.3f}  "
                         f"length={r['greedy_len_match']:.3f}  samples={r['sample_match']:.3f}{flag}")
        else:
            lines.append(f"- {nm}: missing ({'new' if not new.exists() else 'old'})")
    lines += ["", "## Pre-registered go/no-go (tag prereg-gonogo-v1)", ""]
    for nm in ("gonogo_math500", "gonogo_gsm8k"):
        f = Path(a.out).parent / f"{nm}.json"
        if f.exists():
            j = json.loads(f.read_text())
            hb = j["criterion_b"]
            lines.append(f"- {nm}: n={j['n']}  A(sample)={j['criterion_a']['sample_more_diff']:+.3f}  "
                         f"A(escalate)={j['criterion_a']['escalate_diff']:+.3f}  headroom={hb['headroom']}  "
                         f"=> **{'GO' if j['GO'] else 'NO-GO'}** (A={j['pass_a']}, B={j['pass_b']})")
        else:
            lines.append(f"- {nm}: not found")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
