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

It also prints what the probe already tells about the run ahead, all from TRAIN questions:
the length distribution and generation speed (to forecast GPU time before the long stages),
and each model's greedy accuracy with the share of questions the small model gets wrong and
the big one right -- the room an escalation can use. No test item is read.

    python cap_curve.py probe/math500_qs.jsonl.gz probe/gsm8k_qs.jsonl.gz ... [--target 0.02]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys

GRID = (256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
CUT = 10**9      # length of a completion cut at the probe cap: unknown, longer than any cap


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
                    # cut at the probe cap: it needs MORE than that, by an unknown amount, so
                    # it counts as cut at every cap (also caps above the probe cap)
                    probe_cap = max(probe_cap, c["decode_tokens"])
                    toks.append(CUT)
                else:
                    toks.append(c["decode_tokens"])
    return toks, prompts, probe_cap


def plateau_cap(curve: dict[int, float], slack: float = 0.01) -> int:
    """Smallest cap within ``slack`` of the truncation left at the largest cap.

    Completions that never terminate are cut at every cap, so when they exceed the target no
    cap reaches it; the curve then flattens and this is where it does.
    """
    floor = curve[max(curve)]
    return next(c for c in sorted(curve) if curve[c] <= floor + slack)


def mean_tokens_at(tokens: list[int], cap: int) -> float:
    return sum(min(t, cap) for t in tokens) / len(tokens) if tokens else float("nan")


def read_greedy(path: str) -> dict[str, tuple[bool, int]]:
    """query_id -> (greedy correct, greedy decode tokens; CUT if it hit the probe cap)."""
    out = {}
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            g = r["greedy"]
            out[r["query_id"]] = (bool(g.get("correct")),
                                  CUT if g.get("finish_reason") == "length" else g["decode_tokens"])
    return out


def rescue_table(small: dict, big: dict, cap: int) -> dict[str, float]:
    """Greedy outcomes of a model pair on shared questions, with answers longer than ``cap``
    counted wrong (that is what the run will do to them)."""
    qs = sorted(set(small) & set(big))
    ok = lambda d, q: d[q][0] and d[q][1] <= cap
    n = len(qs) or 1
    s = [ok(small, q) for q in qs]
    b = [ok(big, q) for q in qs]
    return {"n": len(qs), "acc_small": sum(s) / n, "acc_big": sum(b) / n,
            "small_wrong_big_right": sum((not x) and y for x, y in zip(s, b)) / n,
            "both_wrong": sum((not x) and (not y) for x, y in zip(s, b)) / n,
            "small_right_big_wrong": sum(x and (not y) for x, y in zip(s, b)) / n}


def throughput(path: str) -> float | None:
    try:
        with open(path + ".manifest.json") as f:
            return (json.load(f).get("throughput") or {}).get("tok_per_s")
    except (OSError, ValueError):
        return None


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
    plateau: dict[str, int] = {}
    max_prompt = 0
    pools = [p for p in args.pools if ".text." not in p and ".tok." not in p]
    stats = {}
    for p in pools:
        toks, prompts, probe_cap = read_tokens(p)
        # nothing is known about caps above the cap the pool was generated with
        grid = tuple(c for c in GRID if not probe_cap or c <= probe_cap) or (probe_cap,)
        curve = trunc_curve(toks, caps=grid)
        cap = smallest_cap(curve, args.target)
        max_prompt = max(max_prompt, max(prompts) if prompts else 0)
        name = p.split("/")[-1]
        fin = sorted(t for t in toks if t != CUT)
        q = lambda x: fin[min(len(fin) - 1, int(x * len(fin)))] if fin else 0
        print(f"\n{name}: {len(toks)} completions, probe cap hit by {sum(1 for t in toks if t == CUT)}")
        print("  cap   " + " ".join(f"{c:>5}" for c in grid))
        print("  trunc " + " ".join(f"{curve[c]*100:4.1f}%" for c in grid))
        print(f"  finished completions: median {q(.5)}, p90 {q(.9)}, p98 {q(.98)} tokens")
        print(f"  smallest cap with truncation <= {args.target:.0%}: {cap if cap else 'NONE up to ' + str(grid[-1])}")
        key = "MT_G" if "gsm" in name else "MT_M"
        plateau[key] = max(plateau.get(key, 0), plateau_cap(curve))
        if cap:
            rec[key] = max(rec.get(key, 0), cap)
        else:
            rec[key] = max(rec.get(key, 0), 10**9)      # marks "not reached"
        stats[p] = (name, key, toks)
    print()
    failed = any(v >= 10**9 for v in rec.values())
    chosen = plateau if failed else rec

    print("FORECAST INPUTS (mean decode tokens per completion at the cap below; measured speed):")
    for p, (name, key, toks) in stats.items():
        tps = throughput(p)
        print(f"  {name}: cap {chosen[key]} -> mean {mean_tokens_at(toks, chosen[key]):.0f} tokens"
              + (f", {tps:.0f} tok/s" if tps else ", speed not recorded"))

    print("\nEARLY DIAGNOSTICS (greedy, TRAIN questions only; answers longer than the cap count wrong):")
    for p in pools:
        if "_qs." not in p:
            continue
        pb = p.replace("_qs.", "_qb.")
        if pb not in pools:
            continue
        name, key, _ = stats[p]
        t = rescue_table(read_greedy(p), read_greedy(pb), chosen[key])
        print(f"  {name.split('_q')[0]}: n={t['n']}  small acc {t['acc_small']:.3f}  big acc {t['acc_big']:.3f}  "
              f"| small wrong & big right {t['small_wrong_big_right']:.3f}  both wrong {t['both_wrong']:.3f}  "
              f"small right & big wrong {t['small_right_big_wrong']:.3f}")
    print()
    if failed:
        print("WARNING: at least one pool still truncates above the target at the largest cap. "
              "If its curve is flat by then, those completions never terminate and no cap helps; "
              "the cap where each curve flattens is")
        print(f"PLATEAU: MT_M={plateau.get('MT_M', 1024)} MT_G={plateau.get('MT_G', 512)}   "
              "(NOT a recommendation: send this file before choosing)")
        sys.exit(2)
    mt_m, mt_g = rec.get("MT_M", 1024), rec.get("MT_G", 512)
    # max_model_len is NOT derived here: the probe sees a few hundred prompts, while the run
    # must fit the longest prompt of every set (859 tokens on MATH-500). session2_run.sh asks
    # preflight_prompts.py for it once the caps are known.
    print(f"RECOMMEND: MT_M={mt_m} MT_G={mt_g}   (target {args.target:.0%}; "
          f"longest probe prompt {max_prompt} tokens; max_model_len comes from preflight)")


if __name__ == "__main__":
    main()
