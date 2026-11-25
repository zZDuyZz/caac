"""Convert LLMRouterBench and Monkey Business (downloaded to your disk) into CAAC-Pool v1.

Runs on YOUR machine with the standard library only (Python >= 3.9). It READS the downloads
and WRITES small pools; it never changes the downloads and sends nothing anywhere. No model
text is copied into the pools (only ids, correctness, token counts, costs, answer keys).

  python scripts\\convert_public.py D:\\data D:\\data\\pools          (both sources)
  python scripts\\convert_public.py D:\\data D:\\data\\pools --only lrb
  python scripts\\convert_public.py D:\\data D:\\data\\pools --only monkey

Output: one folder per (source, benchmark) with rows.jsonl.gz + meta.json, and convert_log.txt.

LLMRouterBench (D:\\data\\lrb\\bench-release\\<bench>\\[<split>\\]<model>\\<file>.json):
  qid = "<bench>/<split>:<index>";  draw 0 only (k = 1);  correct = (score == 1.0)
  out_tokens = completion_tokens;  in_tokens = prompt_tokens;  cost_usd = cost (from the source;
  for the 7-9B models the source does not say how cost was computed: label it estimated)
  finish = "unknown" (the source has no finish reason);  caps[model] = smallest power of two >=
  the largest completion_tokens of that model on that benchmark (a STRUCTURAL bound computed from
  token counts only, never from correctness; the source does not state max_tokens)
  The same index must be the same question for every model: a question whose text hash differs
  between models is dropped (counted in meta.notes). The "openrouter" folder is a router, not a
  model, and is skipped.
Monkey Business (D:\\data\\monkey\\<TASK>_<MODEL>.json, 10,000 samples per problem):
  qid = "<TASK>:<orig_dset_split>:<orig_dset_idx>";  draws 0..63 = the first 64 samples
  correct = is_corrects[d] (graded by the source);  answer_key = final answer extracted from the
  sample text (GSM8K: "#### x"; MATH: "The final answer is x"), compared as exact strings;
  none for CodeContests / MiniF2F
  out_tokens = min(cap, ceil(characters / 4)) -- ESTIMATED (no token counts in the source);
  in_tokens = ceil(prompt characters / 4) -- ESTIMATED;  caps = the source's max length
  (GSM8K / MATH 512, CodeContests 1024, MiniF2F 200);  finish = "unknown"
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

FORMAT, VERSION = "caac-pool", "1"
FIELDS = ("qid", "model", "draw", "correct", "answer_key", "out_tokens", "in_tokens", "finish",
          "is_greedy", "gpu_s", "cost_usd", "split", "flags", "group")
MONKEY_DRAWS = 64
MONKEY_CAPS = {"GSM8K": 512, "MATH": 512, "CodeContests": 1024, "MiniF2F-MATH": 200}
JUDGE_BENCH = {"hle", "simpleqa", "arenahard", "arenahard_coding", "arenahard_creative_writing",
               "arenahard_math"}
SKIP_MODELS = {"openrouter"}


# ---------------------------------------------------------------- writing (Pool v1)
def row(qid, model, draw, correct, **kw):
    d = {"qid": qid, "model": model, "draw": draw, "correct": bool(correct)}
    for k in FIELDS[4:]:
        v = kw.get(k)
        if v is not None and v != []:
            d[k] = v
    return d


def write_pool(out_dir, rows, meta):
    os.makedirs(out_dir, exist_ok=True)
    rows.sort(key=lambda r: (r["model"], r["qid"], r["draw"]))
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    with open(os.path.join(out_dir, "rows.jsonl.gz"), "wb") as f, \
            gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
        g.write(data)
    meta = dict(meta, format=FORMAT, version=VERSION)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pow2_at_least(x):
    return 1 if x <= 1 else 1 << (int(x) - 1).bit_length()


# ---------------------------------------------------------------- LLMRouterBench
def lrb_files(root):
    base = os.path.join(root, "lrb", "bench-release")
    out = []
    for d, _, fs in os.walk(base):
        for f in fs:
            if not f.endswith(".json"):
                continue
            rel = os.path.relpath(os.path.join(d, f), base).split(os.sep)
            if len(rel) == 4:
                bench, split, model = rel[0], rel[1], rel[2]
            elif len(rel) == 3:
                bench, split, model = rel[0], "-", rel[1]
            else:
                continue
            out.append((bench, split, model, os.path.join(d, f)))
    return sorted(out)


def convert_lrb(root, out_root, log):
    groups = defaultdict(list)
    for bench, split, model, path in lrb_files(root):
        if model in SKIP_MODELS:
            log(f"skip {bench}/{split}/{model}: router, not a model")
            continue
        groups[(bench, split)].append((model, path))
    made = []
    for (bench, split), items in sorted(groups.items()):
        t0 = time.time()
        rows, qhash, files = [], defaultdict(dict), []
        stats = defaultdict(int)
        maxtok = defaultdict(int)
        seen_model = set()
        for model, path in items:
            if model in seen_model:
                stats["duplicate_model_file"] += 1
                log(f"  {bench}/{split}: second file for {model} skipped: {path}")
                continue
            seen_model.add(model)
            with open(path, encoding="utf-8-sig") as f:
                obj = json.load(f)
            if obj.get("demo"):
                stats["demo_file_skipped"] += 1
                continue
            files.append({"file": os.path.basename(path), "sha256": sha_file(path)})
            seen_idx = set()
            for r in obj.get("records", []):
                idx = r.get("index")
                if idx in seen_idx:
                    stats["duplicate_index"] += 1
                    continue
                seen_idx.add(idx)
                qid = f"{bench}/{split}:{idx}"
                qhash[qid][model] = hashlib.sha256(str(r.get("origin_query", "")).encode("utf-8")).hexdigest()[:16]
                sc = r.get("score")
                if sc is None:
                    stats["score_missing"] += 1
                    continue
                sc = float(sc)
                if sc not in (0.0, 1.0):
                    stats["score_nonbinary"] += 1
                ct, pt, cost = r.get("completion_tokens"), r.get("prompt_tokens"), r.get("cost")
                if ct is None:
                    stats["completion_tokens_missing"] += 1
                else:
                    maxtok[model] = max(maxtok[model], int(ct))
                rows.append(row(qid, model, 0, sc == 1.0,
                                out_tokens=None if ct is None else int(ct),
                                in_tokens=None if pt is None else int(pt),
                                finish="unknown",
                                cost_usd=None if cost is None else float(cost)))
        bad_q = {q for q, h in qhash.items() if len(set(h.values())) > 1}
        if bad_q:
            rows = [r for r in rows if r["qid"] not in bad_q]
        stats["questions_dropped_text_mismatch"] = len(bad_q)
        caps = {m: pow2_at_least(v) for m, v in sorted(maxtok.items())}
        judge = bench in JUDGE_BENCH
        meta = {"source": {"name": "LLMRouterBench (NPULH/LLMRouterBench, bench-release)",
                           "benchmark": f"{bench}/{split}",
                           "url": "https://huggingface.co/datasets/NPULH/LLMRouterBench",
                           "version_or_sha": hashlib.sha256("".join(sorted(f["sha256"] for f in files)).encode()).hexdigest()[:16],
                           "files": files},
                "grader": {"kind": "llm-judge" if judge else "other",
                           "name": "LLMRouterBench evaluator (by the source)" + (" with an LLM judge" if judge else "")},
                "answer_equivalence": "none", "answer_keys": False,
                "caps": caps,
                "cost_label": "cost_usd from the source: API models priced by the source's table; 7-9B models: method not stated (estimated)",
                "caps_rule": "smallest power of two >= the largest completion_tokens of the model on this benchmark (structural, from token counts only)",
                "notes": f"k=1; finish unknown; stats: {dict(sorted(stats.items()))}",
                "adapter": {"name": "convert_public.lrb", "draw0": "the single generation"}}
        name = f"lrb_{bench}_{split}".replace("-", "_") if split != "-" else f"lrb_{bench}"
        write_pool(os.path.join(out_root, name), rows, meta)
        made.append(name)
        log(f"{name}: models={len(seen_model)} rows={len(rows)} stats={dict(sorted(stats.items()))} ({time.time() - t0:.0f}s)")
    return made


# ---------------------------------------------------------------- Monkey Business
def iter_json_array(path, chunk=16 << 20):
    """Yield the objects of a top-level JSON array without loading the whole file."""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8-sig") as f:
        buf, pos, started = "", 0, False
        eof = False
        while True:
            # skip whitespace and separators
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf) or eof:
                    break
                more = f.read(chunk)
                if not more:
                    eof = True
                buf, pos = buf[pos:] + more, 0
            if pos >= len(buf):
                return
            if not started:
                if buf[pos] != "[":
                    raise ValueError(f"{path}: not a JSON array")
                started, pos = True, pos + 1
                continue
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                more = f.read(chunk)
                if not more:
                    eof = True
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            buf, pos = buf[end:], 0


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
MATH_RE = re.compile(r"[Tt]he final answer is (.*?)(?:\.\s*I hope it is correct|\s*$)", re.S)


def answer_key(task, text):
    if task == "GSM8K":
        m = GSM_RE.findall(text)
        if not m:
            return None
        return m[-1].replace(",", "").rstrip(".") or None
    if task == "MATH":
        m = MATH_RE.findall(text)
        if not m:
            return None
        a = m[-1].strip().strip("$").strip()
        return re.sub(r"\s+", "", a) or None
    return None


def est_tokens(text, cap=None):
    t = math.ceil(len(text) / 4)
    return min(cap, t) if cap else t


def convert_monkey(root, out_root, log, draws=MONKEY_DRAWS):
    base = os.path.join(root, "monkey")
    by_task = defaultdict(list)
    for f in sorted(os.listdir(base)):
        if f.endswith(".json") and "_" in f:
            task, model = f[:-5].split("_", 1)
            by_task[task].append((model, os.path.join(base, f)))
    made = []
    for task, items in sorted(by_task.items()):
        cap = MONKEY_CAPS.get(task)
        rows, files, stats = [], [], defaultdict(int)
        for model, path in items:
            t0 = time.time()
            files.append({"file": os.path.basename(path), "size": os.path.getsize(path)})
            n = 0
            for prob in iter_json_array(path):
                n += 1
                qid = f"{task}:{prob.get('orig_dset_split')}:{prob.get('orig_dset_idx')}"
                samples, corrects = prob["samples"], prob["is_corrects"]
                if len(samples) < draws:
                    stats["problems_with_fewer_draws"] += 1
                pin = math.ceil(len(prob.get("prompt", "")) / 4)
                for d in range(min(draws, len(samples))):
                    txt = samples[d]
                    rows.append(row(qid, model, d, bool(corrects[d]),
                                    answer_key=answer_key(task, txt),
                                    out_tokens=est_tokens(txt, cap), in_tokens=pin,
                                    finish="unknown", is_greedy=False))
            log(f"  monkey {task}/{model}: problems={n} ({time.time() - t0:.0f}s)")
        keys = task in ("GSM8K", "MATH")
        grader = {"GSM8K": ("exact", "lm-eval GSM8K '#### x' string match (by the source)"),
                  "MATH": ("math-verify", "lm-eval minerva_math sympy equivalence (by the source)"),
                  "CodeContests": ("unit-tests", "AlphaCode-style tests (by the source)"),
                  "MiniF2F-MATH": ("other", "lean-dojo proof check, 10 s/tactic timeout (by the source)")}[task]
        meta = {"source": {"name": "ScalingIntelligence/monkey_business", "benchmark": task,
                           "url": "https://huggingface.co/datasets/ScalingIntelligence/monkey_business",
                           "version_or_sha": "sizes:" + ",".join(str(f["size"]) for f in files),
                           "files": files},
                "grader": {"kind": grader[0], "name": grader[1]},
                "answer_equivalence": "exact" if keys else "none", "answer_keys": keys,
                "caps": {m: cap for m, _ in items},
                "token_label": "out_tokens and in_tokens ESTIMATED as ceil(characters/4), out_tokens clipped at the cap",
                "notes": f"first {draws} of 10,000 samples per problem (temperature 0.6 / 0.5); "
                         f"answer keys compared as exact strings (votes may split equivalent forms); stats: {dict(stats)}",
                "adapter": {"name": "convert_public.monkey", "draw0": "first sample (not greedy)"}}
        name = f"monkey_{task}".replace("-", "_")
        write_pool(os.path.join(out_root, name), rows, meta)
        made.append(name)
        log(f"{name}: models={len(items)} rows={len(rows)}")
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("out_root")
    ap.add_argument("--only", choices=["lrb", "monkey"])
    a = ap.parse_args(argv)
    os.makedirs(a.out_root, exist_ok=True)
    logf = open(os.path.join(a.out_root, "convert_log.txt"), "a", encoding="utf-8", newline="\n")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"convert_public {time.strftime('%Y-%m-%d %H:%M:%S')} python={sys.version.split()[0]} root={a.data_root}")
    if a.only in (None, "lrb"):
        convert_lrb(a.data_root, a.out_root, log)
    if a.only in (None, "monkey"):
        convert_monkey(a.data_root, a.out_root, log)
    log("done")
    logf.close()


if __name__ == "__main__":
    main()
