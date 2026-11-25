"""Row and meta definitions of CAAC-Pool v1 (pure Python, no dependencies).

Row (one generation of one model on one question)
-------------------------------------------------
required
  qid        str   question id, the same string for every model
  model      str   model id; small/big roles are chosen in the config, NOT here,
                   so one source with many models gives many pairs
  draw       int   0, 1, 2, ...; draw 0 is the answer the system would serve first
  correct    0/1   graded by the grader named in meta.json
optional (each one unlocks analyses, see the package docstring)
  answer_key str   canonical answer; equal strings = the same vote. The adapter
                   canonicalizes. Needed for SAMPLE MORE (level L2).
  out_tokens int   generated tokens (no prompt)
  in_tokens  int   prompt tokens (prefill); missing => 0 and the report says so
  finish     str   "stop" | "length" | other; "length" = cut by the cap
  is_greedy  bool  draw produced by greedy decoding
  gpu_s      float GPU seconds of this generation (measured)
  cost_usd   float money cost of this generation (measured or from a price list
                   named in meta.json)
  split      str   "train" | "calib" | "test"; missing => the tool splits by a
                   salted hash of qid and records it
  flags      list[str]  data-quality marks set by the adapter, e.g. "think_marker"
  group      str   slice label of the QUESTION (domain, source, difficulty ...); the report
                   lists risk per group on the tested half (descriptive, not certified)

meta.json
---------
required: format="caac-pool", version="1", source{name, url, version_or_sha},
          grader{kind, name}, answer_equivalence in {"math","exact","none"}
optional: caps{model: max out tokens}, prices{...}, notes,
          answer_keys (bool): true = the adapter gives answer_key to EVERY generation
          that has an extractable answer, so a missing key means "no answer" (not a
          vote: truncated or unparsable). false/absent = keys are optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FORMAT = "caac-pool"
VERSION = "1"

REQUIRED = ("qid", "model", "draw", "correct")
OPTIONAL = ("answer_key", "out_tokens", "in_tokens", "finish", "is_greedy",
            "gpu_s", "cost_usd", "split", "flags", "group")
SPLITS = ("train", "calib", "test")
GRADER_KINDS = ("math-verify", "exact", "unit-tests", "llm-judge", "other")
EQUIVALENCE = ("math", "exact", "none")


@dataclass
class Row:
    qid: str
    model: str
    draw: int
    correct: bool
    answer_key: str | None = None
    out_tokens: int | None = None
    in_tokens: int | None = None
    finish: str | None = None
    is_greedy: bool | None = None
    gpu_s: float | None = None
    cost_usd: float | None = None
    split: str | None = None
    flags: list[str] = field(default_factory=list)
    group: str | None = None


@dataclass
class Meta:
    source: dict
    grader: dict
    answer_equivalence: str
    caps: dict = field(default_factory=dict)
    prices: dict = field(default_factory=dict)
    notes: str = ""
    answer_keys: bool = False
    format: str = FORMAT
    version: str = VERSION
