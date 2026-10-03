"""Benchmark loaders.

Return a list of {'id', 'question', 'answer'} dicts. The synthetic loader is for
CPU smoke tests; the real loaders read from the HuggingFace datasets library.
"""

from __future__ import annotations

__all__ = ["load_benchmark", "synthetic_benchmark", "load_gsm8k", "load_math500"]


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
    """Load GSM8K. Answer is the final number after '####' in the solution.

    Returns dicts with the true gold answer, so accuracy is meaningful.
    """
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    items = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        # GSM8K gold answer is after the '####' marker.
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"id": f"gsm8k_{i}", "question": row["question"], "answer": gold})
    return items


def load_math500(n: int | None = None, split: str = "test"):
    """Load MATH-500 (Lightman et al. subset, via the HuggingFaceH4 mirror).

    Returns dicts with the gold final answer (already extracted by the
    dataset, unlike raw Hendrycks MATH where it's embedded in a \\boxed{}
    inside the solution).

    CAVEAT (added 2026-09-24, not yet verified against a live run): MATH
    answers are often not plain integers -- fractions ("\\frac{1}{2}"),
    expressions, intervals, etc. `caac.eval.answer_match.answers_match` was
    built and tuned against GSM8K-style bare numbers. If accuracy on
    MATH-500 comes out suspiciously low (e.g. near 0 even for the 7B model
    on "level 1" problems), check whether answers_match needs a MATH-aware
    normalization pass (strip \\boxed{}, LaTeX spacing, equivalent fraction
    forms) before trusting go_no_go_experiment.py's numbers for the MATH
    subset. GSM8K-only numbers from that script are unaffected by this.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    items = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        items.append({"id": f"math500_{i}", "question": row["problem"], "answer": row["answer"]})
    return items


def load_benchmark(name: str, split: str = "test", n: int | None = None, **kwargs):
    """Dispatch to a named benchmark."""
    if name == "synthetic":
        return synthetic_benchmark(n=n or 20, **kwargs)
    if name == "gsm8k":
        return load_gsm8k(n=n, split=split)
    if name == "math500":
        return load_math500(n=n, split=split)
    raise NotImplementedError(
        f"loader for '{name}' not wired yet. Available: synthetic, gsm8k, math500. "
        "(aime, gpqa, livecodebench, bbh added later.)"
    )
