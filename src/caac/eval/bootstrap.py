"""Question-level bootstrap intervals for the numbers a selection reports.

The unit resampled is the QUESTION, never a sample or a draw order: two answers
to one question are not independent, and test tables are already averaged over
draw orders inside each row. Every interval is a plain percentile interval.

Two uses:
  * ``bootstrap_ci``   -- coverage, selective risk, accuracy and cost of one
                          configuration on one table, each with an interval.
  * ``paired_diff_ci`` -- method A minus method B on the SAME questions, both
                          resampled with the same indices. A ranking is only
                          claimed when this interval excludes zero; the
                          unpaired intervals overlap far more often than the
                          paired difference crosses zero, so comparing them by
                          eye would understate what the data show.

These intervals describe sampling noise of the reported test numbers. They are
not the risk certificate (that is the exact binomial test on calibration) and
must never be read as one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["METRICS", "per_question", "bootstrap_ci", "paired_diff_ci"]

METRICS = ("coverage", "selective_risk", "accuracy", "cost")


def per_question(table, j: int, lam: float, rows=None) -> dict[str, np.ndarray]:
    """The three per-question arrays every metric is a ratio of sums of."""
    r = np.arange(len(table.qids)) if rows is None else np.asarray(rows)
    return {"served": np.asarray(table.served[r, j], float),
            "wrong": np.asarray(table.wrong[r, j], float),
            "cost": np.asarray(table.cost(lam)[r, j], float)}


def _metrics(served: np.ndarray, wrong: np.ndarray, cost: np.ndarray, n: int) -> dict:
    """Metrics from per-resample SUMS (arrays over resamples)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = np.where(served > 0, wrong / np.where(served > 0, served, 1.0), np.nan)
    return {"coverage": served / n, "selective_risk": risk,
            "accuracy": (served - wrong) / n, "cost": cost / n}


def _resample_sums(arr: dict, idx: np.ndarray) -> dict:
    return {k: v[idx].sum(axis=1) for k, v in arr.items()}


def _interval(x: np.ndarray, level: float) -> tuple[float, float] | None:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    lo, hi = np.percentile(x, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def _indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("need at least one question")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(table, j: int, lam: float, rows=None, n_boot: int = 1000,
                 level: float = 0.95, seed: int = 0) -> dict:
    """Percentile interval for each metric of configuration ``j``.

    Returns {metric: {"point": x, "lo": a, "hi": b}}. A resample that serves
    nothing has no selective risk; it is left out of that metric's interval and
    the share of such resamples is reported as ``risk_undefined``.
    """
    arr = per_question(table, j, lam, rows)
    n = len(arr["served"])
    sums = _resample_sums(arr, _indices(n, n_boot, seed))
    boot = _metrics(sums["served"], sums["wrong"], sums["cost"], n)
    point = _metrics(np.array([arr["served"].sum()]), np.array([arr["wrong"].sum()]),
                     np.array([arr["cost"].sum()]), n)
    out = {}
    for m in METRICS:
        iv = _interval(boot[m], level)
        p = float(point[m][0])
        out[m] = {"point": p if np.isfinite(p) else None,
                  "lo": iv[0] if iv else None, "hi": iv[1] if iv else None}
    out["risk_undefined"] = float(np.mean(~np.isfinite(boot["selective_risk"])))
    out["n_boot"], out["level"], out["n"] = n_boot, level, n
    return out


def paired_diff_ci(table_a, ja: int, table_b, jb: int, lam: float, n_boot: int = 1000,
                   level: float = 0.95, seed: int = 0) -> dict:
    """Interval for (A - B) on each metric, resampling the same questions for both.

    The two tables must list the same questions in the same order (two families
    replayed on one test pool do). ``excludes_zero`` is the only ranking claim
    the paper may make from it.
    """
    if list(table_a.qids) != list(table_b.qids):
        raise ValueError("paired comparison needs the same questions in the same order")
    a, b = per_question(table_a, ja, lam), per_question(table_b, jb, lam)
    n = len(a["served"])
    idx = _indices(n, n_boot, seed)
    sa, sb = _resample_sums(a, idx), _resample_sums(b, idx)
    ma = _metrics(sa["served"], sa["wrong"], sa["cost"], n)
    mb = _metrics(sb["served"], sb["wrong"], sb["cost"], n)
    pa = _metrics(*(np.array([a[k].sum()]) for k in ("served", "wrong", "cost")), n)
    pb = _metrics(*(np.array([b[k].sum()]) for k in ("served", "wrong", "cost")), n)
    out = {}
    for m in METRICS:
        d = ma[m] - mb[m]
        iv = _interval(d, level)
        p = float(pa[m][0] - pb[m][0])
        out[m] = {"point": p if np.isfinite(p) else None,
                  "lo": iv[0] if iv else None, "hi": iv[1] if iv else None,
                  "excludes_zero": bool(iv is not None and (iv[0] > 0 or iv[1] < 0))}
    out["n_boot"], out["level"], out["n"] = n_boot, level, n
    return out
