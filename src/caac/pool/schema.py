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


@dataclass
class Completion:
    """One greedy or sampled completion."""

    answer: str | None          # extracted + normalized answer, None if unparsable
    correct: bool               # graded against gold (by the STRONGEST grader)
    decode_tokens: int          # generated tokens (NOT counting prefill)
    finish_reason: str          # "stop" | "length" | other; "length" => truncated
    seed: int | None = None     # the per-completion seed actually used
    confidence: float | None = None  # mean token prob; RAW for greedy (temp=0),
                                     # post-temperature for samples (flag below)
    mean_logprob: float | None = None


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _open(tmp, "w") as f:
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
