"""How many completions fit in a token cap?  (CAAC_MASTER §12.1, "cap probe")

Day 1 (2026-11-09) showed that Qwen3.5 writes long solutions even with thinking off:
at max_tokens 1024 (MATH) / 512 (GSM8K) 23-46% of the greedy completions were cut
before the final \\boxed{}, and a cut completion is graded wrong. Pool accuracy then
measures "finishes within the cap", not the model.  This script reads a pool that was
generated with a cap far above what is needed (the probe: --max-tokens 4096) and
reports, for a grid of caps, the share of completions that would have been cut::

    trunc(c) = P(decode_tokens > c)      (a completion cut at the probe cap counts as > c)

and the smallest grid cap with trunc(c) <= --target (default 2%).  It reads only the
pool .jsonl.gz (no GPU, no text sidecar) and prints a PASTE line for session2_run.sh.

    python cap_curve.py probe/math500_qs.jsonl.gz probe/gsm8k_qs.jsonl.gz ... [--target 0.02]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys

GRID = (256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096)


def trunc_curve(tokens: list[int], caps=GRID) -> dict[int, float]:
    """Share of completions longer than each cap."""
    n = len(tokens)
    return {c: (sum(1 for t in tokens if t > c) / n if n else float("nan")) for c in caps}


def smallest_cap(curve: dict[int, float], target: float) -> int | None:
    """Smallest grid cap whose truncation share is <= target (None: even the largest fails)."""
    for c in sorted(curve):
        if curve[c] <= target:
            return c
    return None


def read_tokens(path: str) -> tuple[list[int], list[int], int]:
    """(decode_tokens of every completion, prompt_tokens per question, probe cap)."""
    toks, prompts, probe_cap = [], [], 0
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r.get("prompt_tokens", 0))
            for c in [r["greedy"]] + list(r.get("samples", [])):
                if c.get("finish_reason") == "length":
                    # cut at the probe cap: it needs MORE than that, so it counts as cut
                    # at every cap on the grid, including the probe cap itself
                    probe_cap = max(probe_cap, c["decode_tokens"])
                    toks.append(c["decode_tokens"] + 1)
                else:
                    toks.append(c["decode_tokens"])
    return toks, prompts, probe_cap


def round_up(x: int, m: int = 512) -> int:
    return ((x + m - 1) // m) * m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--target", type=float, default=0.02)  # 2026-11-09: 5%->2%, 5% leaves enough
    # truncated-and-scored-wrong answers for a reviewer to object; the token cost of the
    # tighter tail is small next to the credibility it buys (see CAAC_MASTER 3.12).
    args = ap.parse_args()
    rec: dict[str, int] = {}
    max_prompt = 0
    for p in args.pools:
        if ".text." in p or ".tok." in p:
            continue
        toks, prompts, probe_cap = read_tokens(p)
        curve = trunc_curve(toks)
        cap = smallest_cap(curve, args.target)
        max_prompt = max(max_prompt, max(prompts) if prompts else 0)
        name = p.split("/")[-1]
        print(f"\n{name}: {len(toks)} completions, probe cap hit by {sum(1 for t in toks if t > probe_cap > 0)}")
        print("  cap   " + " ".join(f"{c:>5}" for c in GRID))
        print("  trunc " + " ".join(f"{curve[c]*100:4.1f}%" for c in GRID))
        print(f"  smallest cap with truncation <= {args.target:.0%}: {cap if cap else 'NONE up to ' + str(GRID[-1])}")
        key = "MT_G" if "gsm" in name else "MT_M"
        if cap:
            rec[key] = max(rec.get(key, 0), cap)
        else:
            rec[key] = max(rec.get(key, 0), 10**9)      # marks "not reached"
    print()
    if any(v >= 10**9 for v in rec.values()):
        print("WARNING: at least one pool still truncates above the target at the largest cap; "
              "raise the probe cap (--max-tokens) or accept a higher target, then re-run.")
        sys.exit(2)
    mt_m, mt_g = rec.get("MT_M", 1024), rec.get("MT_G", 512)
    # max_model_len is NOT derived here: the probe sees a few hundred prompts, while the run
    # must fit the longest prompt of every set (859 tokens on MATH-500). session2_run.sh asks
    # preflight_prompts.py for it once the caps are known.
    print(f"RECOMMEND: MT_M={mt_m} MT_G={mt_g}   (target {args.target:.0%}; "
          f"longest probe prompt {max_prompt} tokens; max_model_len comes from preflight)")


if __name__ == "__main__":
    main()
