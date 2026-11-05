"""Benchmark loaders.

Return a list of {'id', 'question', 'answer'} dicts. The synthetic loader is for
CPU smoke tests; the real loaders read from the HuggingFace datasets library.

CAAC_MASTER §6.2: train/calib splits must be DISJOINT from the test pools.
MATH-500 is a subset of the MATH test set, so the MATH train/calib loaders
exclude any problem whose text matches a MATH-500 problem (matched on collapsed
whitespace, robust across dataset mirrors).

NOT YET VERIFIED against a live run on the GPU box: the exact HF dataset name and
column names for full MATH and ARC. Run ``inspect_math.py`` first; the loaders
auto-detect common column layouts and accept an explicit ``source``/``config``.
"""

from __future__ import annotations

import re

__all__ = ["load_benchmark", "synthetic_benchmark", "load_gsm8k", "load_math500",
           "load_math", "load_arc_challenge"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _norm_problem(text: str) -> str:
    """Collapse whitespace so the same problem matches across mirrors."""
    return re.sub(r"\s+", " ", (text or "").strip())


def _extract_boxed(solution: str) -> str | None:
    r"""Pull the answer out of the LAST \boxed{...} in a solution (brace-aware)."""
    if not solution:
        return None
    idx = solution.rfind("\\boxed")
    if idx == -1:
        return None
    i = solution.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(solution):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[i + 1:j]
        j += 1
    return None


def _row_problem(row: dict) -> str | None:
    for k in ("problem", "question", "Problem", "query"):
        if row.get(k):
            return row[k]
    return None


def _row_answer(row: dict) -> str | None:
    for k in ("answer", "Answer", "final_answer", "solution_answer"):
        if row.get(k) not in (None, ""):
            return str(row[k])
    # fall back to extracting from a full solution
    for k in ("solution", "Solution", "output", "response"):
        if row.get(k):
            a = _extract_boxed(row[k])
            if a is not None:
                return a
    return None


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

def synthetic_benchmark(n: int = 20, seed: int = 0):
    """Deterministic toy problems for smoke-testing on CPU."""
    import numpy as np

    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        a, b = int(rng.integers(1, 50)), int(rng.integers(1, 50))
        items.append({"id": f"syn_{i}", "question": f"{a} + {b} = ?", "answer": str(a + b)})
    return items


def load_gsm8k(n: int | None = None, split: str = "test"):
    """Load GSM8K. Answer is the final number after '####' in the solution."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    items = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        # id must be unique across splits (train pool must not collide with test)
        qid = f"gsm8k_{i}" if split == "test" else f"gsm8k_{split}_{i}"
        items.append({"id": qid, "question": row["question"], "answer": gold})
    return items


def load_math500(n: int | None = None, split: str = "test"):
    """Load MATH-500 (Lightman et al. subset, HuggingFaceH4 mirror)."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    items = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        items.append({"id": f"math500_{i}", "question": row["problem"], "answer": row["answer"]})
    return items


def _math500_problem_set():
    """Collapsed-whitespace problem texts of MATH-500 (for disjointness)."""
    return {_norm_problem(it["question"]) for it in load_math500()}


# HF dataset names to try for the FULL MATH set (verify with inspect_math.py).
_MATH_SOURCES = [
    ("nlile/hendrycks-MATH-benchmark", None),
    ("EleutherAI/hendrycks_math", "all"),
    ("qwedsacf/competition_math", None),
]


def load_math(n: int | None = None, split: str = "train", exclude_math500: bool = True,
              source: str | None = None, config: str | None = None):
    """Load the FULL MATH set (train, or 'test minus MATH-500').

    split="train"            -> MATH train (7,500)
    split="test"             -> MATH test (5,000); with exclude_math500=True this
                                is the 4,500 'test minus MATH-500' calib pool.
    Pass source/config explicitly once inspect_math.py confirms what's cached.
    ids are 'math_{split}_{running}' so they never collide with 'math500_*'.
    """
    from datasets import load_dataset

    sources = [(source, config)] if source else _MATH_SOURCES
    ds = None
    last_err = None
    for src, cfg in sources:
        try:
            ds = load_dataset(src, cfg, split=split) if cfg else load_dataset(src, split=split)
            break
        except Exception as e:  # try the next mirror
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(f"could not load a MATH source {sources}; last error: {last_err}. "
                           f"Run inspect_math.py and pass --source/--config.")

    exclude = _math500_problem_set() if exclude_math500 else set()
    items, running = [], 0
    for row in ds:
        q = _row_problem(row)
        a = _row_answer(row)
        if q is None or a is None:
            continue
        if _norm_problem(q) in exclude:
            continue
        items.append({"id": f"math_{split}_{running}", "question": q, "answer": a})
        running += 1
        if n is not None and running >= n:
            break
    return items


def load_arc_challenge(n: int | None = None, split: str = "test"):
    """ARC-Challenge (multiple choice). Answer is the correct option LETTER.

    Prompt convention (CAAC_MASTER §6.1) ends with 'The best answer is: X'; the
    question here embeds the labelled choices so the model can produce a letter.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    items = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        choices = row["choices"]
        labels = list(choices["label"])
        texts = list(choices["text"])
        body = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        q = f"{row['question']}\n{body}"
        items.append({"id": f"arc_{split}_{i}", "question": q, "answer": row["answerKey"]})
    return items


def load_benchmark(name: str, split: str = "test", n: int | None = None, **kwargs):
    """Dispatch to a named benchmark."""
    if name == "synthetic":
        return synthetic_benchmark(n=n or 20, **kwargs)
    if name == "gsm8k":
        return load_gsm8k(n=n, split=split)
    if name == "math500":
        return load_math500(n=n, split=split)
    if name == "math_train":
        return load_math(n=n, split="train", exclude_math500=True, **kwargs)
    if name == "math_test_ex500":   # MATH test minus the MATH-500 subset
        return load_math(n=n, split="test", exclude_math500=True, **kwargs)
    if name in ("arc_challenge", "arc"):
        return load_arc_challenge(n=n, split=split)
    raise NotImplementedError(
        f"loader for '{name}' not wired yet. Available: synthetic, gsm8k, math500, "
        "math_train, math_test_ex500, arc_challenge."
    )
