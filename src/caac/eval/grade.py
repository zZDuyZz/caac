"""Shared grading helpers.

PLAN §4: the PRIMARY grader is Hugging Face Math-Verify
(``verify(parse(gold), parse(pred))`` -- gold first); ``answers_match`` is the
secondary/legacy grader. Both ``build_pool.py`` and ``regrade_pool.py`` import
from here so the grading logic lives in exactly one place.

Design notes / caveats (see CAAC_MASTER §3.3, §6.3):
  * Math-Verify handles LaTeX equivalence (``11\\sqrt2`` == ``11\\sqrt{2}``,
    ``\\frac65`` == ``1.2``) and strips trivial wrappers (``90^\\circ`` vs ``90``,
    ``x=5`` vs ``5``, ``60\\%`` vs ``60``), which ``answers_match`` misses.
  * Math-Verify is sometimes TOO permissive (``2516_8`` ~ ``2516`` drops the
    base; ``12\\text{cm}`` ~ ``12`` drops the unit). So we always keep the legacy
    verdict too and REPORT disagreements for a ~50-case hand check; we never
    silently trust one grader.
  * We wrap both sides in ``\\boxed{...}`` before parsing because the stored/raw
    answers are bare answer strings, not full solutions.
"""

from __future__ import annotations

import os

# Math-Verify's per-call timeout uses signal.alarm on Linux, but on Windows it
# SPAWNS A PROCESS PER CALL (slow + floods errors under Python 3.14). On Windows we
# therefore disable the timeout (None). Linux (the GPU box) keeps the default 5s.
_NO_TIMEOUT = os.name == "nt"


def _mv_fns():
    """Return (parse, verify) from math_verify, or None if unavailable."""
    try:
        from math_verify import parse, verify  # type: ignore
        return parse, verify
    except Exception:
        return None


def math_verify_correct(gold: str, pred: str | None) -> bool | None:
    """True/False via Math-Verify. None if pred is None-safe-False? No:

    Returns False if pred is None/empty. Returns None ONLY when Math-Verify is
    not installed or raised (so the caller can fall back to the legacy verdict).
    """
    if pred is None or str(pred).strip() == "":
        return False
    fns = _mv_fns()
    if fns is None:
        return None
    parse, verify = fns
    try:
        if _NO_TIMEOUT:
            g = parse("\\boxed{" + str(gold) + "}", parsing_timeout=None)
            p = parse("\\boxed{" + str(pred) + "}", parsing_timeout=None)
            return bool(verify(g, p, timeout_seconds=None))
        g = parse("\\boxed{" + str(gold) + "}")
        p = parse("\\boxed{" + str(pred) + "}")
        return bool(verify(g, p))
    except Exception:
        return None


def grade(gold: str, pred: str | None, legacy_correct: bool,
          mode: str = "mathverify") -> bool:
    """Primary correctness flag.

    mode="mathverify" (default): use Math-Verify; if it is unavailable or errors
      on this item, fall back to the legacy verdict (never crash a GPU run).
    mode="legacy": just return the legacy verdict.
    """
    if mode == "legacy":
        return bool(legacy_correct)
    mv = math_verify_correct(gold, pred)
    if mv is None:
        return bool(legacy_correct)
    return mv


def mathverify_available() -> bool:
    return _mv_fns() is not None
