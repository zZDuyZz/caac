"""Build the sample pool on GPU -- the ONE GPU artifact of PLAN_V2 v2.1.

For each question: 1 greedy (temperature=0) + n sampled completions from one
model, with prefill/decode token counts, finish_reason, per-completion seed and
a confidence feature. Writes one jsonl.gz per (model, dataset, split), in
RESUMABLE batches so a killed GPU rental loses nothing.

Grading (CAAC_MASTER §6.3): the PRIMARY grader is Math-Verify; ``answers_match``
is kept as the secondary verdict in ``correct_legacy``. Both are computed here so
the pool's numbers are authoritative right away. The raw generated TEXT is also
written to a sidecar ``<pool>.text.jsonl.gz`` so any grader / better extractor
can re-grade on CPU later without re-running the GPU (see regrade_pool.py).

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
  # new pool, faster config (DO NOT change config for an existing pool):
  python build_pool.py ... --max-num-seqs 512 --batch-size 300
  # re-run only specific questions (e.g. the ones that were truncated):
  python build_pool.py ... --qids-file truncated.txt --max-tokens 2048 --max-model-len 3072
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


def completion_dict(answer_raw, answer_norm, gold, decode_tokens, finish_reason,
                    seed, lps, grader, answers_match):
    """Assemble ONE completion record (pure; no vLLM types) and grade it.

    answer_raw  : extracted-but-not-normalized answer (best for Math-Verify)
    answer_norm : normalized answer (what we store + feed answers_match)
    Returns the dict stored in the pool.
    """
    from caac.eval.grade import grade
    legacy = bool(answer_norm is not None and answers_match(answer_norm, gold))
    primary = grade(gold, answer_raw if answer_raw is not None else answer_norm,
                    legacy_correct=legacy, mode=grader)
    return {
        "answer": answer_norm,
        "correct": bool(primary),
        "correct_legacy": legacy,
        "decode_tokens": decode_tokens,
        "finish_reason": finish_reason or "stop",
        "seed": seed,
        "confidence": mean_token_prob(lps),
        "mean_logprob": (sum(lps) / len(lps)) if lps else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, help="gsm8k | math500 | math_train | arc_challenge | ...")
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
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="vLLM max_num_seqs; None = vLLM default (256). ONLY for a NEW pool.")
    p.add_argument("--qids-file", default=None,
                   help="txt file, one qid per line; only build those questions")
    p.add_argument("--grader", default="mathverify", choices=["mathverify", "legacy"],
                   help="primary grader for 'correct' (answers_match always kept in correct_legacy)")
    p.add_argument("--no-text", action="store_true",
                   help="do NOT write the <pool>.text.jsonl.gz sidecar (not recommended)")
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # project graders/loaders (patched versions must be in the repo)
    from caac.data.benchmarks import load_benchmark
    from caac.eval.answer_match import extract_answer, normalize_answer, answers_match
    from caac.eval.grade import mathverify_available
    from caac.pool.schema import text_sidecar_path, append_texts

    if args.grader == "mathverify" and not mathverify_available():
        print("WARNING: --grader mathverify but math-verify not importable; "
              "'correct' will fall back to answers_match per item.", flush=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    sidecar = None if args.no_text else text_sidecar_path(outp)

    problems = load_benchmark(args.dataset, split=args.split, n=args.n_questions)
    if args.qids_file:
        keep = {l.strip() for l in open(args.qids_file) if l.strip()}
        problems = [pr for pr in problems if pr["id"] in keep]
        print(f"qids-file: giữ {len(problems)}/{len(keep)} câu", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    def build_prompt(q: str) -> str:
        msgs = [{"role": "user",
                 "content": q + "\n\nPlease reason step by step, and put your "
                                "final answer within \\boxed{}."}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    llm_kw = dict(model=args.model, gpu_memory_utilization=args.gpu_mem,
                  max_model_len=args.max_model_len, enable_prefix_caching=True,
                  seed=0)  # engine seed harmless; per-request seeds below drive diversity
    if args.max_num_seqs:
        llm_kw["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**llm_kw)

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
        "max_num_seqs": args.max_num_seqs, "max_model_len": args.max_model_len,
        "grader": args.grader, "has_text_sidecar": not args.no_text,
        "batch_size": args.batch_size,
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

        greedy_sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                   logprobs=args.top_logprobs)
        greedy_out = llm.generate(prompts, greedy_sp)

        sps = [SamplingParams(n=args.n_samples, temperature=args.temperature,
                              max_tokens=args.max_tokens, logprobs=args.top_logprobs,
                              seed=stable_seed(args.model, args.dataset, pr["id"]))
               for pr in batch]
        sample_outs = llm.generate(prompts, sps)

        records = []
        texts = []
        for pr, pf, g, so in zip(batch, prefill, greedy_out, sample_outs):
            def one(o, seed=None):
                txt = o.text
                raw = extract_answer(txt)
                ans = normalize_answer(raw) if raw is not None else None
                lps = [lp_dict[tid].logprob
                       for tid, lp_dict in zip(o.token_ids, o.logprobs or [])
                       if o.logprobs and tid in lp_dict] if o.logprobs else []
                d = completion_dict(raw, ans, pr["answer"], len(o.token_ids),
                                    o.finish_reason, seed, lps, args.grader, answers_match)
                return d, txt
            g_d, g_txt = one(g.outputs[0])
            s_pairs = [one(c, seed=stable_seed(args.model, args.dataset, pr["id"]) + i)
                       for i, c in enumerate(so.outputs)]
            rec = {
                "query_id": pr["id"], "dataset": args.dataset, "model": args.model,
                "gold": pr["answer"], "prompt_tokens": pf,
                "greedy": g_d,
                "samples": [d for d, _ in s_pairs],
                "temperature": args.temperature,
                "confidence_semantics": "vllm_v0_post_temperature",
                "schema_version": "1.0",
            }
            records.append(json.dumps(rec, ensure_ascii=False))
            if sidecar is not None:
                texts.append({"query_id": pr["id"], "greedy_text": g_txt,
                              "sample_texts": [t for _, t in s_pairs]})

        append(records)
        if sidecar is not None:
            append_texts(sidecar, texts)
        print(f"wrote {start + len(batch)}/{len(todo)}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
