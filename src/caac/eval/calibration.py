"""Calibration metrics for the correctness estimator.

These implement the measurements behind RQ1/H1 -- the go/no-go gate. If no
signal reaches usable calibration at any scale, the VOC rule has nothing
trustworthy to compute with. Pure numpy, runs anywhere.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "expected_calibration_error",
    "brier_score",
    "selective_risk_curve",
    "area_under_risk_coverage",
    "calibration_report",
]


def _validate(probs, labels):
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if probs.shape != labels.shape:
        raise ValueError(f"shape mismatch: {probs.shape} vs {labels.shape}")
    if probs.size == 0:
        raise ValueError("empty input")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probs must lie in [0, 1]")
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("labels must be binary 0/1")
    return probs, labels


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """Equal-width binned ECE. 0 is perfect; the proposal's gate is < 0.10."""
    probs, labels = _validate(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    ece, n = 0.0, probs.size
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c:
            ece += (c / n) * abs(probs[m].mean() - labels[m].mean())
    return float(ece)


def brier_score(probs, labels) -> float:
    """Mean squared error of the probability forecast. Lower is better."""
    probs, labels = _validate(probs, labels)
    return float(np.mean((probs - labels) ** 2))


def selective_risk_curve(probs, labels):
    """Risk-coverage curve by abstaining on the least confident items.

    Answers 'if the controller trusts only its top-k% beliefs, how often is it
    wrong?' -- exactly what a STOP decision relies on. Returns (coverage, risk).
    """
    probs, labels = _validate(probs, labels)
    order = np.argsort(-probs, kind="stable")
    errors = 1.0 - labels[order]
    n = probs.size
    coverage = np.arange(1, n + 1, dtype=np.float64) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    return coverage, risk


def area_under_risk_coverage(probs, labels) -> float:
    """AURC: area under the risk-coverage curve. Lower is better.

    Ranking-sensitive rather than magnitude-sensitive: a signal can be badly
    calibrated yet useful for ordering. Reporting both ECE and AURC separates
    'wrong scale' from 'wrong ordering'.
    """
    coverage, risk = selective_risk_curve(probs, labels)
    return float(np.trapezoid(risk, coverage))


def calibration_report(probs, labels, n_bins: int = 15) -> dict:
    """All headline calibration numbers for one (signal, model) pair."""
    probs, labels = _validate(probs, labels)
    return {
        "n": int(probs.size),
        "base_rate": float(labels.mean()),
        "mean_confidence": float(probs.mean()),
        "ece": expected_calibration_error(probs, labels, n_bins),
        "brier": brier_score(probs, labels),
        "aurc": area_under_risk_coverage(probs, labels),
    }
