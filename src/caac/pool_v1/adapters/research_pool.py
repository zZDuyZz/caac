"""Adapter: the project's own research pools (``PoolRecord`` jsonl.gz from build_pool.py) -> CAAC-Pool v1.

Mapping (one PoolRecord = one question x one model):
  draw 0        = the greedy completion (is_greedy=True); draws 1..k = the samples in order
  correct       = Completion.correct (primary grader, Math-Verify)
  answer_key    = the label-free answer class used by the replay's votes
                  (PoolRecord.greedy_label / canonical_sample_answers); None = no vote
                  (unparsable, or a sample cut by the cap) -> meta.answer_keys = true
  out_tokens    = decode_tokens;  in_tokens = prompt_tokens (the same prompt for every draw;
                  how often prefill is billed is a CONFIG decision, not a data one)
  finish        = finish_reason
  split         = given by the caller (the file's role in the protocol)
Nothing is invented: GPU seconds and prices are not in a PoolRecord, so they stay empty.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from caac.pool.schema import PoolRecord, load_pool
from caac.pool_v1.io import write_rows
from caac.pool_v1.schema import FORMAT, VERSION, Row


def records_to_rows(records: dict[str, PoolRecord], split: str | None = None,
                    model_name: str | None = None) -> list[Row]:
    rows: list[Row] = []
    for qid in sorted(records):
        rec = records[qid]
        model = model_name or rec.model
        comps = [rec.greedy] + list(rec.samples)
        keys = [rec.greedy_label()] + list(rec.canonical_sample_answers())
        for d, (c, key) in enumerate(zip(comps, keys)):
            rows.append(Row(qid=qid, model=model, draw=d, correct=bool(c.correct),
                            answer_key=key, out_tokens=int(c.decode_tokens),
                            in_tokens=int(rec.prompt_tokens), finish=c.finish_reason,
                            is_greedy=(d == 0), split=split))
    return rows


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(pools: list[tuple[str, str]], out_dir: str | Path, *, source_name: str,
            source_url: str = "", caps: dict | None = None, notes: str = "") -> Path:
    """pools = [(path to PoolRecord jsonl.gz, split)]. Writes rows.jsonl.gz + meta.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    files = []
    for path, split in pools:
        recs = load_pool(path)
        rows += records_to_rows(recs, split=split)
        files.append({"file": Path(path).name, "sha256": sha256(path), "split": split,
                      "records": len(recs)})
    rows.sort(key=lambda r: (r.model, r.qid, r.draw))
    write_rows(rows, out / "rows.jsonl.gz")
    meta = {"format": FORMAT, "version": VERSION,
            "source": {"name": source_name, "url": source_url,
                       "version_or_sha": ",".join(f["sha256"][:16] for f in files), "files": files},
            "grader": {"kind": "math-verify", "name": "math-verify 0.9.0 (primary); answers_match kept as legacy"},
            "answer_equivalence": "math", "answer_keys": True,
            "caps": caps or {}, "notes": notes,
            "adapter": {"name": "research_pool", "draw0": "greedy"}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return out
