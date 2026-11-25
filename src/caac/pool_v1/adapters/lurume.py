"""Adapter: Lurume/llm-routing-response-bank -> CAAC-Pool v1 (one pool per benchmark).

Input: the response metadata WITHOUT text (lurume_meta.jsonl.gz, made by caac_probe.py from
responses_eval.jsonl) and scores_eval.jsonl[.gz]. Mapping (one response = question x model):
  qid          = task_id;  model = model_id;  draw = 0 (sample_index is 0 everywhere: k = 1)
  correct      = quality == 1.0 (quality is 0/1; graded by the SOURCE: HELM exact-answer parser
                 for aime / gpqa / mmlu_pro, lcb_runner unit tests for livecodebench)
  out_tokens   = usage.output (includes reasoning tokens);  in_tokens = usage.input
  cost_usd     = official_usd (billed price reported by the source: MEASURED by the source)
  finish       = finish_reason; "length" = cut by the cap (-> no answer, escalated by the replay);
                 a missing finish_reason becomes "unknown"; content_filter / engine_overloaded
                 are kept: the source graded them, so they count as served answers (conservative
                 for GO)
  caps[model]  = the largest of (requested_max_tokens, observed usage.output) of that model in
                 that benchmark: the source reports some outputs ABOVE the requested maximum
                 (deepseek-v4-flash, up to 515,083 > 384,000), and a cap must bound every
                 observed cost; the count of such rows is written to meta.notes
No answer keys (gpqa has no text; the others are not needed with k = 1).
Rows without a score, or a score without a response, are dropped and counted in meta.notes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from caac.pool_v1.io import write_rows
from caac.pool_v1.schema import FORMAT, VERSION, Row

GRADER = {"livecodebench": ("unit-tests", "lcb_runner codegen_metrics (by the source)")}
DEFAULT_GRADER = ("exact", "HELM exact-answer parser (by the source)")


def _open(path):
    p = str(path)
    return gzip.open(p, "rt", encoding="utf-8-sig") if p.endswith(".gz") else open(p, encoding="utf-8-sig")


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(meta_path, scores_path):
    resp, score = {}, {}
    with _open(meta_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                resp[(r["task_id"], r["model_id"])] = r
    with _open(scores_path) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                score[(s["task_id"], s["model_id"])] = s
    return resp, score


def to_rows(resp: dict, score: dict):
    """Return {dataset: [Row]}, {dataset: caps}, counts of dropped cells."""
    rows, caps = defaultdict(list), defaultdict(dict)
    dropped = {"score_without_response": 0, "response_without_score": 0, "output_above_requested_max": 0}
    for key in sorted(set(resp) | set(score)):
        r, s = resp.get(key), score.get(key)
        if r is None:
            dropped["score_without_response"] += 1
            continue
        if s is None:
            dropped["response_without_score"] += 1
            continue
        q = s["quality"]
        if q not in (0, 1, 0.0, 1.0):
            raise ValueError(f"non-binary quality {q!r} for {key}")
        if str(r.get("sample_index", "0")) != "0":
            raise ValueError(f"sample_index {r.get('sample_index')!r} for {key}: adapter assumes k = 1")
        u = r["usage"]
        ds, m = r["dataset"], r["model_id"]
        rows[ds].append(Row(qid=r["task_id"], model=m, draw=0, correct=bool(q == 1),
                            out_tokens=int(u["output"]), in_tokens=int(u["input"]),
                            finish=r.get("finish_reason") or "unknown",
                            cost_usd=float(r["official_usd"]), is_greedy=False))
        cap = int(r["requested_max_tokens"])
        if int(u["output"]) > cap:
            dropped["output_above_requested_max"] += 1     # counted, kept (not dropped)
        caps[ds][m] = max(caps[ds].get(m, 0), cap, int(u["output"]))
    return rows, caps, dropped


def convert(meta_path, scores_path, out_root, datasets=None) -> list[Path]:
    resp, score = load(meta_path, scores_path)
    rows, caps, dropped = to_rows(resp, score)
    out = []
    files = [{"file": Path(meta_path).name, "sha256": _sha(meta_path)},
             {"file": Path(scores_path).name, "sha256": _sha(scores_path)}]
    for ds in sorted(rows):
        if datasets and ds not in datasets:
            continue
        d = Path(out_root) / f"lurume_{ds}"
        d.mkdir(parents=True, exist_ok=True)
        rr = sorted(rows[ds], key=lambda r: (r.model, r.qid, r.draw))
        write_rows(rr, d / "rows.jsonl.gz")
        kind, name = GRADER.get(ds, DEFAULT_GRADER)
        meta = {"format": FORMAT, "version": VERSION,
                "source": {"name": "Lurume/llm-routing-response-bank", "benchmark": ds,
                           "url": "https://huggingface.co/datasets/Lurume/llm-routing-response-bank",
                           "version_or_sha": ",".join(f["sha256"][:16] for f in files),
                           "files": files},
                "grader": {"kind": kind, "name": name},
                "answer_equivalence": "none", "answer_keys": False,
                "caps": caps[ds],
                "notes": (f"k=1; cost_usd = official_usd (billed, reported by the source); "
                          f"counts (all benchmarks; output_above_requested_max rows are KEPT and the cap "
                          f"raised to cover them): {dropped}"),
                "adapter": {"name": "lurume", "draw0": "sample_index 0"}}
        (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out.append(d)
    return out


if __name__ == "__main__":
    import sys
    for p in convert(sys.argv[1], sys.argv[2], sys.argv[3]):
        print(p)
