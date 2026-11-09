"""Check every prompt of a run against its token cap before any GPU time is spent.

A prompt whose length plus its cap exceeds max_model_len is either rejected by vLLM or cut
at max_model_len and scored as a truncation, hours into a run. This tokenizes every prompt
of every set the run will generate (seconds, CPU only) and either fails at once or prints
the smallest max_model_len that fits them all.

    python preflight_prompts.py --model M1 --model M2 \\
        --set math500:test::2560 --set gsm8k:train:qids/a3_gsm8k_train.txt:1024 [--max-model-len N]

--set is dataset:split:qids_file:cap (qids_file may be empty). Without --max-model-len it prints
only the needed value on the last line, for use as MML=$(python preflight_prompts.py ...).
"""
from __future__ import annotations

import argparse
import sys

MARGIN = 64      # headroom for template/tokenizer differences between HF and vLLM
ROUND = 256


def needed_mml(lengths_and_caps: list[tuple[int, int]], margin: int = MARGIN, mult: int = ROUND) -> int:
    worst = max(n + cap for n, cap in lengths_and_caps)
    return ((worst + margin + mult - 1) // mult) * mult


def violations(items: list[tuple[str, int, int]], mml: int) -> list[tuple[str, int, int]]:
    return [(q, n, cap) for q, n, cap in items if n + cap + MARGIN > mml]


def parse_set(s: str) -> tuple[str, str, str | None, int]:
    ds, split, qf, cap = s.split(":")
    return ds, split, (qf or None), int(cap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--set", action="append", required=True, type=parse_set)
    ap.add_argument("--max-model-len", type=int, default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from caac.data.benchmarks import load_benchmark
    from caac.data.prompts import build_chat_prompt

    rows: list[tuple[str, int, int]] = []
    for m in a.model:
        tok = AutoTokenizer.from_pretrained(m)
        for ds, split, qf, cap in a.set:
            items = load_benchmark(ds, split=split)
            if qf:
                keep = {l.strip() for l in open(qf) if l.strip()}
                items = [it for it in items if it["id"] in keep]
            lens = [(it["id"], len(tok(build_chat_prompt(tok, it["question"], thinking=False)).input_ids), cap)
                    for it in items]
            rows += lens
            print(f"  {m} {ds}/{split}: {len(lens)} prompts, longest {max(n for _, n, _ in lens)} tokens, cap {cap}",
                  file=sys.stderr)
    need = needed_mml([(n, c) for _, n, c in rows])
    if a.max_model_len is None:
        print(need)
        return
    bad = violations(rows, a.max_model_len)
    if bad:
        print(f"FAIL: {len(bad)} prompts do not fit max_model_len={a.max_model_len} "
              f"(needs {need}); e.g. {bad[:3]}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: all {len(rows)} prompts fit max_model_len={a.max_model_len} (needs {need})", file=sys.stderr)


if __name__ == "__main__":
    main()
