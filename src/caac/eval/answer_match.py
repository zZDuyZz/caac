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

_BOXED_START = re.compile(r"\\boxed\{")
_HASH = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*(?:/\d+)?")
# MATH-style LaTeX fraction, e.g. \frac{1}{2} or \dfrac{-3}{4} -> "-3/4".
# Added 2026-09-24 for MATH-500 support: normalize_answer previously only
# understood plain "a/b", not LaTeX \frac{a}{b}, which is how MATH-500 gold
# answers and model \boxed{} outputs commonly express fractions.
_LATEX_FRAC = re.compile(r"\\d?frac\{(-?\d+)\}\{(-?\d+)\}")


def _find_boxed(text):
    """All \\boxed{...} contents, handling nested braces (e.g. \\frac{1}{2}).

    Fixed 2026-09-24: the old implementation was a flat regex
    `\\boxed\{([^}]*)\}` that stopped at the FIRST closing brace, so
    `\\boxed{\\frac{1}{2}}` was mis-extracted as `\\frac{1` -- silently wrong
    for the large fraction/exponent-heavy share of MATH-500 answers. This
    does real brace-depth counting instead.
    """
    results = []
    for m in _BOXED_START.finditer(text):
        depth = 1
        start = m.end()
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start:i - 1])
    return results


def extract_answer(text):
    """Pull the final answer: \\boxed{} or #### if present, else last number."""
    if not text:
        return None
    boxed = _find_boxed(text)
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
    frac_match = _LATEX_FRAC.fullmatch(text)
    if frac_match:
        text = f"{frac_match.group(1)}/{frac_match.group(2)}"
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
