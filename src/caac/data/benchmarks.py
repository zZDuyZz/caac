"""Benchmark loaders.

Thin wrappers returning a list of {'id', 'question', 'answer'} dicts. The real
loaders (GSM8K, MATH-500, AIME, GPQA, LiveCodeBench, BBH) read from disk or the
datasets library; a synthetic loader is provided so the pipeline runs on CPU
with no downloads.
"""

from __future__ import annotations

__all__ = ["load_benchmark", "synthetic_benchmark"]


def synthetic_benchmark(n: int = 20, seed: int = 0):
    """Deterministic toy problems for smoke-testing the pipeline end-to-end."""
    import numpy as np

    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        a, b = int(rng.integers(1, 50)), int(rng.integers(1, 50))
        items.append({"id": f"syn_{i}", "question": f"{a} + {b} = ?", "answer": str(a + b)})
    return items


def load_benchmark(name: str, split: str = "test", **kwargs):
    """Real loaders deferred to when datasets are available."""
    if name == "synthetic":
        return synthetic_benchmark(**kwargs)
    raise NotImplementedError(
        f"loader for '{name}' not wired yet; use 'synthetic' on CPU. "
        "Real loaders (gsm8k, math500, aime, gpqa, livecodebench, bbh) added in P1."
    )
