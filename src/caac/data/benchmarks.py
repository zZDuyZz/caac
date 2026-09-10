"""Benchmark loaders.

Return a list of {'id', 'question', 'answer'} dicts. The synthetic loader is for
CPU smoke tests; the real loaders read from the HuggingFace datasets library.
"""

from __future__ import annotations

__all__ = ["load_benchmark", "synthetic_benchmark", "load_gsm8k"]


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


def load_benchmark(name: str, split: str = "test", n: int | None = None, **kwargs):
    """Dispatch to a named benchmark."""
    if name == "synthetic":
        return synthetic_benchmark(n=n or 20, **kwargs)
    if name == "gsm8k":
        return load_gsm8k(n=n, split=split)
    raise NotImplementedError(
        f"loader for '{name}' not wired yet. Available: synthetic, gsm8k. "
        "(math500, aime, gpqa, livecodebench, bbh added later.)"
    )
