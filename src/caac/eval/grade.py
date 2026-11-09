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
if _NO_TIMEOUT:
    import logging
    # the "Timeout is disabled ..." notice is expected here; keep logs readable
    logging.getLogger("math_verify").setLevel(logging.ERROR)


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


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def answer_clusters(answers: list[str | None]) -> list[int | None]:
    """Group extracted answers into equivalence classes WITHOUT the gold label.

    Voting and agreement checks must work the way a deployed system would: by
    comparing answers with each other. Two answers are the same when their
    normalized forms match or Math-Verify accepts them in both directions.
    Two plain numbers that normalize differently are different, so most pairs
    never reach Math-Verify. Each new answer is compared with the first member
    of every existing class; None (unparsable) gets no class.
    """
    from caac.eval.answer_match import normalize_answer

    fns = _mv_fns()
    parsed: dict[str, object] = {}

    def parse_once(a: str):
        if a not in parsed:
            try:
                kw = {"parsing_timeout": None} if _NO_TIMEOUT else {}
                parsed[a] = fns[0]("\\boxed{" + a + "}", **kw)
            except Exception:
                parsed[a] = None
        return parsed[a]

    def mv_same(a: str, b: str) -> bool:
        if fns is None:
            return False
        pa, pb = parse_once(a), parse_once(b)
        if not pa or not pb:
            return False
        kw = {"timeout_seconds": None} if _NO_TIMEOUT else {}
        try:
            return bool(fns[1](pa, pb, **kw)) and bool(fns[1](pb, pa, **kw))
        except Exception:
            return False

    reps: list[tuple[str, str]] = []
    out: list[int | None] = []
    for a in answers:
        n = normalize_answer(a) if a is not None else None
        if n is None:
            out.append(None)
            continue
        cid = None
        for j, (rn, ra) in enumerate(reps):
            if rn == n or (not (_is_number(rn) and _is_number(n)) and mv_same(str(a), ra)):
                cid = j
                break
        if cid is None:
            reps.append((n, str(a)))
            cid = len(reps) - 1
        out.append(cid)
    return out
