"""Answer extraction and matching for verifiable-answer benchmarks.

Deliberately conservative: normalisation handles common formatting noise
(boxed answers, commas, fractions) and nothing more. Aggressive matching
inflates accuracy for every method equally but corrupts the correctness labels
the estimators train on -- the part that matters here.
"""

from __future__ import annotations

import re
from fractions import Fraction

__all__ = ["extract_answer", "normalize_answer", "answers_match"]

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_HASH = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*(?:/\d+)?")


def extract_answer(text):
    """Pull the final answer: \\boxed{} or #### if present, else last number."""
    if not text:
        return None
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    hashed = _HASH.findall(text)
    if hashed:
        return hashed[-1].strip()
    nums = _NUMBER.findall(text)
    return nums[-1].strip() if nums else None


def normalize_answer(answer):
    """Canonical form; collapses numerically-equal forms (1/2 == 0.5)."""
    if answer is None:
        return None
    text = answer.strip().rstrip(".").replace("$", "").replace(",", "")
    text = text.replace("\\!", "").replace("\\,", "").replace(" ", "")
    if not text:
        return None
    try:
        v = Fraction(text)
        return str(v.numerator) if v.denominator == 1 else f"{float(v):.6g}"
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return f"{float(text):.6g}"
    except ValueError:
        return text.lower()


def answers_match(predicted, reference):
    """True if the two answers agree after normalisation."""
    if predicted is None or reference is None:
        return False
    return normalize_answer(predicted) == normalize_answer(reference)
