"""Build the sample pool on GPU -- the ONE GPU artifact of PLAN_V2 v2.1.

For each question: 1 greedy (temperature=0) + n sampled completions from one
model, with prefill/decode token counts, finish_reason, per-completion seed and
a confidence feature. Writes one jsonl.gz per (model, dataset, split), in
RESUMABLE batches so a killed GPU rental loses nothing.

Seeding (PLAN_V2 section 3 -- this is the fix for the whole seed saga):
  vLLM 0.7.3 splits n>1 into n requests with seeds s, s+1, ..., s+n-1. So each
  question's seed base is spaced by >= n (here * 1024) and derived from a stable
  sha256 of (model|dataset|qid) -- NOT Python hash(), which is per-process salted
  and would make the pool unreproducible. Greedy is a separate temperature=0 call.

Confidence semantics: vLLM V0 returns logprobs AFTER temperature, so for the
sampled completions ``confidence`` is post-temperature (flagged in the record).
Use the GREEDY confidence as the controller's calibrated signal.

Run ONLY after sanity_check_seed.py confirms samples differ within one run.

Usage:
  python build_pool.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset math500 \
      --n-samples 32 --max-tokens 1024 --out pools/math500_qwen1.5b.jsonl.gz
  python build_pool.py --model Qwen/Qwen2.5-7B-Instruct --dataset math500 \
      --n-samples 8  --max-tokens 1024 --out pools/math500_qwen7b.jsonl.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def stable_seed(model: str, dataset: str, qid: str, spacing: int = 1024) -> int:
    h = hashlib.sha256(f"{model}|{dataset}|{qid}".encode()).hexdigest()[:8]
    return (int(h, 16) % (2**31 // spacing)) * spacing


def mean_token_prob(logprobs: list[float]) -> float:
    if not logprobs:
        return 0.0
    return sum(math.exp(lp) for lp in logprobs) / len(logprobs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, help="gsm8k | math500 | arc_challenge | ...")
    p.add_argument("--split", default="test")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--n-questions", type=int, default=None, help="cap for pilot")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--top-logprobs", type=int, default=5, help="top-k logprobs to store")
    p.add_argument("--batch-size", type=int, default=150, help="questions per generate() call")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=2048)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # project graders/loaders (patched versions must be in the repo)
    from caac.data.benchmarks import load_benchmark
    from caac.eval.answer_match import extract_answer, normalize_answer, answers_match

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    problems = load_benchmark(args.dataset, split=args.split, n=args.n_questions)
    tok = AutoTokenizer.from_pretrained(args.model)

    def build_prompt(q: str) -> str:
        msgs = [{"role": "user",
                 "content": q + "\n\nPlease reason step by step, and put your "
                                "final answer within \\boxed{}."}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enable_prefix_caching=True,
              seed=0)  # engine seed harmless; per-request seeds below drive diversity

    # --- resume: which qids already have a complete record on disk ---
    done = set()
    if outp.exists():
        import gzip
        op = gzip.open(outp, "rt") if str(outp).endswith(".gz") else open(outp)
        with op as f:
            for line in f:
                try:
                    done.add(json.loads(line)["query_id"])
                except Exception:
                    pass
    todo = [pr for pr in problems if pr["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go -> {outp}", flush=True)

    manifest = {
        "model": args.model, "dataset": args.dataset, "split": args.split,
        "n_samples": args.n_samples, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "top_logprobs": args.top_logprobs,
        "vllm_use_v1": os.environ.get("VLLM_USE_V1", "0"),
    }
    Path(str(outp) + ".manifest.json").write_text(json.dumps(manifest, indent=2))

    import gzip
    def append(records: list[str]):
        op = gzip.open(outp, "at") if str(outp).endswith(".gz") else open(outp, "a")
        with op as f:
            for r in records:
                f.write(r + "\n")

    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        prompts = [build_prompt(pr["question"]) for pr in batch]
        prefill = [len(tok(pr_text).input_ids) for pr_text in prompts]

        # greedy (temperature=0): one request each, seed ignored on greedy path
        greedy_sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                   logprobs=args.top_logprobs)
        greedy_out = llm.generate(prompts, greedy_sp)

        # samples: n per prompt in one request; vLLM gives per-child seeds s..s+n-1
        sample_outs = []
        for pr, prompt in zip(batch, prompts):
            base = stable_seed(args.model, args.dataset, pr["id"])
            sp = SamplingParams(n=args.n_samples, temperature=args.temperature,
                                max_tokens=args.max_tokens, logprobs=args.top_logprobs,
                                seed=base)
            sample_outs.append(llm.generate([prompt], sp)[0])

        records = []
        for pr, pf, g, so in zip(batch, prefill, greedy_out, sample_outs):
            def completion(o, seed=None):
                txt = o.text
                raw = extract_answer(txt)
                ans = normalize_answer(raw) if raw is not None else None
                lps = [lp_dict[tid].logprob
                       for tid, lp_dict in zip(o.token_ids, o.logprobs or [])
                       if o.logprobs and tid in lp_dict] if o.logprobs else []
                return {
                    "answer": ans,
                    "correct": bool(ans is not None and answers_match(ans, pr["answer"])),
                    "decode_tokens": len(o.token_ids),
                    "finish_reason": o.finish_reason or "stop",
                    "seed": seed,
                    "confidence": mean_token_prob(lps),
                    "mean_logprob": (sum(lps) / len(lps)) if lps else None,
                }
            rec = {
                "query_id": pr["id"], "dataset": args.dataset, "model": args.model,
                "gold": pr["answer"], "prompt_tokens": pf,
                "greedy": completion(g.outputs[0]),
                "samples": [completion(c, seed=stable_seed(args.model, args.dataset, pr["id"]) + i)
                            for i, c in enumerate(so.outputs)],
                "temperature": args.temperature,
                "confidence_semantics": "vllm_v0_post_temperature",
                "schema_version": "1.0",
            }
            records.append(json.dumps(rec, ensure_ascii=False))

        append(records)
        print(f"wrote {start + len(batch)}/{len(todo)}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
