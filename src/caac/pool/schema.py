"""Sample-pool record schema + jsonl IO.

The pool is the single GPU artifact of PLAN_V2: for each question we store a
greedy completion plus n sampled completions from a model, with token counts
and per-token confidence. EVERY downstream experiment (self-consistency,
cascades, PriceCheck-style serve/abstain, the CAAC controller, LTT, the Chow
bound) reads only this file -- no further GPU.

Cost convention baked into the schema (see PLAN_V2 section 7):
  * ``prompt_tokens`` (prefill) is stored ONCE per (query, model) record,
    because all of that model's completions for the question share the prompt.
  * ``decode_tokens`` is stored per completion.
So a policy that draws k samples from one model pays prefill once + k decodes;
the ledger (ledger.py) enforces exactly that.

One jsonl file per (model, dataset, split). One line = one PoolRecord.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = "1.0"
GOLD = "__GOLD__"  # canonical label for any answer graded correct



@dataclass
class Completion:
    """One greedy or sampled completion."""

    answer: str | None          # extracted + normalized answer, None if unparsable
    correct: bool               # graded against gold by the PRIMARY grader (Math-Verify)
    decode_tokens: int          # generated tokens (NOT counting prefill)
    finish_reason: str          # "stop" | "length" | other; "length" => truncated
    correct_legacy: bool | None = None  # secondary grader (answers_match); None if not run
    seed: int | None = None     # the per-completion seed actually used
    confidence: float | None = None  # mean token prob; RAW for greedy (temp=0),
                                     # post-temperature for samples (flag below)
    mean_logprob: float | None = None
    answer_raw: str | None = None    # extracted but NOT normalized (keeps commas, bases,
                                     # \\frac); added 2026-11 -- None in older pools
    conf_prefix32: float | None = None  # mean token prob over the FIRST 32 generated
                                        # tokens = the PRE-REGISTERED difficulty proxy
                                        # (CAAC_MASTER §8.4); None in older pools


@dataclass
class PoolRecord:
    """All completions of ONE question from ONE model."""

    query_id: str
    dataset: str
    model: str
    gold: str
    prompt_tokens: int                 # prefill, shared by greedy + all samples
    greedy: Completion
    samples: list[Completion] = field(default_factory=list)
    temperature: float = 0.7
    confidence_semantics: str = "vllm_v0_post_temperature"  # see PLAN_V2 section 3
    schema_version: str = SCHEMA_VERSION
    meta: dict = field(default_factory=dict)

    # -- convenience views used all over the replay code --

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def sample_answers(self) -> list[str | None]:
        return [s.answer for s in self.samples]

    def canonical_sample_answers(self) -> list[str | None]:
        """Sample answers with every CORRECT one mapped to the single label GOLD.

        Voting must not compare normalized answers to the raw gold string: MATH
        golds are LaTeX (e.g. \\frac{1}{2}) and normalization rarely reproduces
        them byte-for-byte, so string equality would score correct answers as
        wrong. ``Completion.correct`` is already graded by the project's
        equivalence matcher, so we key the vote on it. Wrong answers keep their
        own labels (equivalent wrong forms stay split: a conservative choice).
        """
        return [GOLD if c.correct else c.answer for c in self.samples]

    def sample_correct(self) -> list[bool]:
        return [s.correct for s in self.samples]

    def n_correct_samples(self) -> int:
        return sum(s.correct for s in self.samples)


# --------------------------------------------------------------------------
# jsonl(.gz) IO -- atomic write per batch so a killed GPU rental loses nothing
# --------------------------------------------------------------------------


def _open(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def record_to_json(rec: PoolRecord) -> str:
    return json.dumps(asdict(rec), ensure_ascii=False)


def record_from_dict(d: dict) -> PoolRecord:
    d = dict(d)
    d["greedy"] = Completion(**d["greedy"])
    d["samples"] = [Completion(**s) for s in d.get("samples", [])]
    d.pop("schema_version", None)  # tolerated but re-set from default
    sv = d.pop("_schema_version", None)
    rec = PoolRecord(**{k: v for k, v in d.items() if k != "schema_version"})
    if sv:
        rec.schema_version = sv
    return rec


def write_records(path: str | Path, records: list[PoolRecord]) -> None:
    """Atomic write: to .tmp then os.replace, so a partial file never wins."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    # choose gzip by the FINAL name (".gz"), not the ".tmp" scratch name
    opener = (lambda: gzip.open(tmp, "wt", encoding="utf-8")) if str(path).endswith(".gz") \
        else (lambda: open(tmp, "w", encoding="utf-8"))
    with opener() as f:
        for rec in records:
            f.write(record_to_json(rec) + "\n")
    import os
    os.replace(tmp, path)


def read_records(path: str | Path) -> Iterator[PoolRecord]:
    path = Path(path)
    with _open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield record_from_dict(json.loads(line))


def load_pool(path: str | Path) -> dict[str, PoolRecord]:
    """Load a pool file keyed by query_id (one model, one dataset split)."""
    return {rec.query_id: rec for rec in read_records(path)}


# --------------------------------------------------------------------------
# text sidecar -- the raw generated text, stored NEXT TO the pool (not in it,
# not in git) so any grader (Math-Verify, a better extractor) can re-grade on
# CPU without re-running the GPU. One line per query_id:
#   {"query_id": ..., "greedy_text": ..., "sample_texts": [...]}
# --------------------------------------------------------------------------


def _sidecar_path(pool_path: str | Path, tag: str) -> Path:
    p = Path(pool_path)
    name = p.name
    for suf in (".jsonl.gz", ".jsonl", ".gz"):
        if name.endswith(suf):
            return p.with_name(name[: -len(suf)] + "." + tag + suf)
    return p.with_name(name + "." + tag)


def text_sidecar_path(pool_path: str | Path) -> Path:
    """`foo.jsonl.gz` -> `foo.text.jsonl.gz` (keeps the same compression)."""
    return _sidecar_path(pool_path, "text")


def token_sidecar_path(pool_path: str | Path) -> Path:
    """`foo.jsonl.gz` -> `foo.tok.jsonl.gz`: per-token signals, one line per query:
      {"query_id": ..., "greedy": {"lp": [...], "tc": [...], "ent": [...]},
       "samples": [{"lp": [...], "tc": [...]}, ...]}
    lp  = logprob of the CHOSEN token; tc = DeepConf token confidence
    (-mean of the top-k logprobs); ent = entropy of the renormalised top-k
    distribution (greedy only). Samples are post-temperature (vLLM V0).
    Not in git (large); keep next to the pool."""
    return _sidecar_path(pool_path, "tok")


def append_texts(path: str | Path, items: list[dict]) -> None:
    """Append text records (resumable, like the pool's own append)."""
    path = Path(path)
    if str(path).endswith(".gz"):
        f = gzip.open(path, "at", encoding="utf-8")
    else:
        f = open(path, "a", encoding="utf-8")
    with f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_texts(path: str | Path) -> dict[str, dict]:
    """Load a text sidecar keyed by query_id. Returns {} if the file is absent."""
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with _open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["query_id"]] = d
    return out
