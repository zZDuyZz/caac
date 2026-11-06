"""Benchmark loaders.

Return a list of {'id', 'question', 'answer'} dicts. The synthetic loader is for
CPU smoke tests; the real loaders read from the HuggingFace datasets library.

CAAC_MASTER §6.2: train/calib splits must be DISJOINT from the test pools.
MATH-500 is a subset of the MATH test set, so the MATH train/calib loaders
exclude any problem whose text matches a MATH-500 problem (matched on collapsed
whitespace, robust across dataset mirrors).

MATH/ARC source names and columns were VERIFIED on 5 Nov 2026 (one-off inspection
script, kept in git history at tag pre-cleanup-2026-11-06)
(see the comment above ``_NLILE``).
"""

from __future__ import annotations

import re

__all__ = ["load_benchmark", "synthetic_benchmark", "load_gsm8k", "load_math500",
           "load_gsm_plus", "load_gsm_symbolic", "load_aime",
           "GSM_PLUS_SKIP_PERTURBATIONS",
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

    ds = load_dataset("openai/gsm8k", "main", split=split)
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


# VERIFIED by inspecting the dataset (5 Nov 2026):
#  * nlile/hendrycks-MATH-benchmark: split "train" = 12,000 rows = the PRM800K split
#    (original MATH train 7,500 + original MATH test minus MATH-500 = 4,500); split
#    "test" = 500 rows = MATH-500 itself. Columns: problem, solution, answer, subject,
#    level, unique_id ("train/algebra/1.json" or "test/algebra/1.json" = ORIGIN).
#  * EleutherAI/hendrycks_math has NO "all" config (7 per-subject configs) -> dropped.
#  * qwedsacf/competition_math: only "train" (12,500 = full MATH, no answer column) ->
#    generic fallback (answer extracted from the solution's \\boxed).
_NLILE = "nlile/hendrycks-MATH-benchmark"
_MATH_FALLBACKS = [("qwedsacf/competition_math", None)]


def _uid_to_id(uid: str) -> str:
    """'train/algebra/1.json' -> 'math_train_algebra_1' (stable across mirrors/order)."""
    return "math_" + re.sub(r"\.json$", "", uid).replace("/", "_")


def _load_math_nlile(split, exclude_math500):
    """Use unique_id origin: split='train' -> 7,500; split='test' -> 4,500 (test minus
    MATH-500). NB: the mirror's own 'test' split is MATH-500, so we never read it."""
    from datasets import load_dataset

    ds = load_dataset(_NLILE, split="train")
    prefix = "train/" if split == "train" else "test/"
    exclude = _math500_problem_set() if exclude_math500 else set()
    items = []
    for row in ds:
        uid = row.get("unique_id") or ""
        if not uid.startswith(prefix):
            continue
        q, a = _row_problem(row), _row_answer(row)
        if q is None or a is None:
            continue
        if _norm_problem(q) in exclude:
            continue
        items.append({"id": _uid_to_id(uid), "question": q, "answer": a,
                      # extra keys for stratified subsampling (make_qids.py);
                      # build_pool only reads id/question/answer
                      "subject": row.get("subject"), "level": row.get("level")})
    return items


def load_math(n: int | None = None, split: str = "train", exclude_math500: bool = True,
              source: str | None = None, config: str | None = None):
    """Load the FULL MATH set (train, or 'test minus MATH-500').

    split="train" -> MATH train (7,500)
    split="test"  -> MATH test (5,000); with exclude_math500=True this is the
                     4,500 'test minus MATH-500' calibration pool.
    Default source = nlile mirror (verified). ids are 'math_{train|test}_{subject}_{k}'
    (from the dataset's unique_id) so they never collide with 'math500_*'.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if source is None or source == _NLILE:
        # NO silent fallback: a different mirror has different ids/order, so the
        # qids files and pools would silently stop matching. Fail loudly instead;
        # pass source=... explicitly to use another mirror on purpose.
        items = _load_math_nlile(split, exclude_math500)
        if not items:
            raise RuntimeError(f"{_NLILE} returned 0 items for split={split!r}")
        return items[:n] if n is not None else items
    # explicit other mirror (ids by running index -- NOT compatible with nlile qids)
    from datasets import load_dataset

    sources = [(source, config)] if source else _MATH_FALLBACKS
    ds, last_err = None, None
    for src, cfg in sources:
        try:
            ds = load_dataset(src, cfg, split=split) if cfg else load_dataset(src, split=split)
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"could not load a MATH source; last error: {last_err}")
    exclude = _math500_problem_set() if exclude_math500 else set()
    items, running = [], 0
    for row in ds:
        q, a = _row_problem(row), _row_answer(row)
        if q is None or a is None or _norm_problem(q) in exclude:
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



# --------------------------------------------------------------------------
# CLEAN SETS (B3): benchmarks published AFTER / independently of Qwen2.5's
# training data, used to test whether CAAC's gains survive contamination
# control (CAAC_MASTER section 16).
#
# Schemas were read from the dataset cards on 2026-11-05 and are asserted at
# load time rather than guessed: a silently renamed column would produce an
# all-wrong run that looks like a real result.
# --------------------------------------------------------------------------

#: rows whose "answer" is deliberately unanswerable -- they must be dropped,
#: not graded, or every model scores ~0 on them and the set looks impossible.
GSM_PLUS_SKIP_PERTURBATIONS = {"critical thinking"}


def _require_columns(row: dict, needed, dataset: str, repo: str):
    missing = [c for c in needed if c not in row]
    if missing:
        raise KeyError(
            f"{dataset}: {repo} is missing column(s) {missing}; got {sorted(row)}. "
            f"The dataset schema changed -- fix the loader, do not guess."
        )


def load_gsm_plus(n: int | None = None, split: str = "test",
                  keep_critical_thinking: bool = False):
    """GSM-Plus (Li et al., ACL 2024): GSM8K perturbed eight ways.

    repo ``qintongli/GSM-Plus``; splits ``test`` (full) and ``testmini``;
    columns ``question``, ``answer``, ``solution``, ``perturbation_type``.

    The "critical thinking" perturbation removes information needed to solve
    the problem, so the gold answer is a refusal rather than a number. Those
    rows are dropped by default (lighteval does the same); keeping them would
    silently punish every policy and, worse, reward abstention for the wrong
    reason -- which would corrupt a paper about selective prediction.
    """
    from datasets import load_dataset

    ds = load_dataset("qintongli/GSM-Plus", split=split)
    items, skipped = [], 0
    for i, row in enumerate(ds):
        if i == 0:
            _require_columns(row, ("question", "answer", "perturbation_type"),
                             "gsm_plus", "qintongli/GSM-Plus")
        ptype = str(row.get("perturbation_type", "")).strip().lower()
        if not keep_critical_thinking and ptype in GSM_PLUS_SKIP_PERTURBATIONS:
            skipped += 1
            continue
        if n is not None and len(items) >= n:
            break
        gold = str(row["answer"]).strip().replace(",", "")
        items.append({"id": f"gsmplus_{split}_{i}", "question": row["question"],
                      "answer": gold, "perturbation_type": row.get("perturbation_type")})
    if not items:
        raise RuntimeError("gsm_plus: loaded 0 items -- check split name")
    return items


def load_gsm_symbolic(n: int | None = None, split: str = "test",
                      config: str = "main"):
    """GSM-Symbolic (Mirzadeh et al., ICLR 2025): templated GSM8K variants.

    repo ``apple/GSM-Symbolic``; configs ``main``, ``p1``, ``p2``; split
    ``test``; columns ``id``, ``instance``, ``question``, ``answer``,
    ``original_id``, ``original_question``, ``original_answer``.

    ``p1``/``p2`` add one and two extra clauses, so difficulty rises with the
    config -- a built-in difficulty axis worth reporting separately rather than
    pooling. The answer carries GSM8K's ``#### x`` marker.
    """
    # validate before importing datasets, so a bad config fails the same way
    # with or without the optional dependency installed
    if config not in ("main", "p1", "p2"):
        raise ValueError(f"gsm_symbolic config must be main/p1/p2, got {config!r}")
    from datasets import load_dataset

    ds = load_dataset("apple/GSM-Symbolic", config, split=split)
    items = []
    for i, row in enumerate(ds):
        if i == 0:
            _require_columns(row, ("question", "answer"), "gsm_symbolic",
                             "apple/GSM-Symbolic")
        if n is not None and len(items) >= n:
            break
        gold = str(row["answer"]).split("####")[-1].strip().replace(",", "")
        items.append({"id": f"gsmsym_{config}_{row.get('id', i)}_{row.get('instance', 0)}",
                      "question": row["question"], "answer": gold,
                      "original_id": row.get("original_id")})
    if not items:
        raise RuntimeError(f"gsm_symbolic[{config}]: loaded 0 items")
    return items


def load_aime(n: int | None = None, split: str = "train", year: str = "2024"):
    """AIME 2024 (``HuggingFaceH4/aime_2024``): 30 problems, split ``train``.

    Columns: ``id``, ``problem``, ``solution``, ``answer``, ``url``, ``year``.

    WARNING, and it belongs in the paper rather than in a footnote: 30
    questions cannot support a risk guarantee. Learn-then-Test needs at least
    ceil(log delta / log(1-alpha)) = 22 SERVED questions before even a
    zero-error run clears delta (``ltt.min_served_certifiable``), so at
    alpha=delta=0.1 a 30-question set could only ever certify a policy with
    coverage above 0.73 -- and nothing at all under Bonferroni. AIME is
    therefore an accuracy/cost stress test here, never a certification set.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/aime_2024", split=split)
    items = []
    for i, row in enumerate(ds):
        if i == 0:
            _require_columns(row, ("problem", "answer"), "aime",
                             "HuggingFaceH4/aime_2024")
        if n is not None and len(items) >= n:
            break
        items.append({"id": f"aime{year}_{row.get('id', i)}",
                      "question": row["problem"],
                      "answer": str(row["answer"]).strip()})
    if not items:
        raise RuntimeError("aime: loaded 0 items")
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
    if name == "gsm_plus":
        return load_gsm_plus(n=n, split=split, **kwargs)
    if name in ("gsm_symbolic", "gsm_symbolic_main"):
        return load_gsm_symbolic(n=n, split=split, config="main")
    if name in ("gsm_symbolic_p1", "gsm_symbolic_p2"):
        return load_gsm_symbolic(n=n, split=split, config=name.rsplit("_", 1)[1])
    if name in ("aime", "aime_2024"):
        return load_aime(n=n, split="train" if split == "test" else split)
    raise NotImplementedError(
        f"loader for '{name}' not wired yet. Available: synthetic, gsm8k, math500, "
        "math_train, math_test_ex500, arc_challenge, gsm_plus, gsm_symbolic[_p1|_p2], aime."
    )
