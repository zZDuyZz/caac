"""Tests for the clean-set loaders (B3).

No network: ``datasets.load_dataset`` is replaced by fixtures whose columns
match the dataset cards read on 2026-11-05. That is deliberate -- the point of
these tests is to pin the SCHEMA CONTRACT, so that when a dataset silently
renames a column the loader raises instead of producing a plausible-looking
all-wrong run.
"""

from __future__ import annotations

import pytest

from caac.data import benchmarks as B

GSM_PLUS_ROWS = [
    {"question": "Janet has 3 apples, buys 5 more. How many?", "answer": "8",
     "solution": "3+5=8", "perturbation_type": "numerical substitution"},
    {"question": "A train leaves at noon. What colour is it?", "answer": "unanswerable",
     "solution": "n/a", "perturbation_type": "critical thinking"},
    {"question": "Tom had 10, gave away 4. Left?", "answer": "6",
     "solution": "10-4=6", "perturbation_type": "distractor insertion"},
]

GSM_SYMBOLIC_ROWS = [
    {"id": 0, "instance": 1, "question": "Sam has 12 marbles, loses 5. How many?",
     "answer": "Sam has 12-5 = 7 marbles.\n#### 7",
     "original_id": 11, "original_question": "...", "original_answer": "#### 7"},
    {"id": 1, "instance": 0, "question": "Ann reads 4 pages a day for 6 days.",
     "answer": "4*6 = 24\n#### 24", "original_id": 12,
     "original_question": "...", "original_answer": "#### 24"},
]

AIME_ROWS = [
    {"id": 0, "problem": "Find the number of ordered pairs...", "answer": "204",
     "solution": "...", "url": "https://...", "year": "2024"},
    {"id": 1, "problem": "Let ABC be a triangle...", "answer": "80",
     "solution": "...", "url": "https://...", "year": "2024"},
]


@pytest.fixture
def fake_hub(monkeypatch):
    """Patch datasets.load_dataset with a table keyed by (repo, config)."""
    import types

    table = {}

    def load_dataset(repo, *args, **kwargs):
        config = args[0] if args and isinstance(args[0], str) else None
        key = (repo, config)
        if key not in table:
            raise AssertionError(f"unexpected load_dataset({repo!r}, {config!r})")
        return table[key]

    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    monkeypatch.setitem(__import__("sys").modules, "datasets", mod)
    return table


# --------------------------------------------------------------------------- #
# GSM-Plus
# --------------------------------------------------------------------------- #

def test_gsm_plus_drops_unanswerable_critical_thinking_rows(fake_hub):
    """These rows have no numeric gold answer. Grading them would punish every
    policy and reward abstention for a reason that is not risk."""
    fake_hub[("qintongli/GSM-Plus", None)] = GSM_PLUS_ROWS
    items = B.load_gsm_plus()
    assert len(items) == 2
    assert all(it["perturbation_type"] != "critical thinking" for it in items)
    assert [it["answer"] for it in items] == ["8", "6"]


def test_gsm_plus_can_keep_them_explicitly(fake_hub):
    fake_hub[("qintongli/GSM-Plus", None)] = GSM_PLUS_ROWS
    assert len(B.load_gsm_plus(keep_critical_thinking=True)) == 3


def test_gsm_plus_ids_are_unique_and_split_tagged(fake_hub):
    fake_hub[("qintongli/GSM-Plus", None)] = GSM_PLUS_ROWS
    ids = [it["id"] for it in B.load_gsm_plus(split="testmini")]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("gsmplus_testmini_") for i in ids)


def test_gsm_plus_strips_thousands_separators(fake_hub):
    rows = [{"question": "q", "answer": "1,234", "perturbation_type": "numerical"}]
    fake_hub[("qintongli/GSM-Plus", None)] = rows
    assert B.load_gsm_plus()[0]["answer"] == "1234"


def test_gsm_plus_raises_on_a_renamed_column(fake_hub):
    fake_hub[("qintongli/GSM-Plus", None)] = [
        {"query": "q", "answer": "1", "perturbation_type": "numerical"}]
    with pytest.raises(KeyError, match="question"):
        B.load_gsm_plus()


def test_gsm_plus_raises_when_everything_is_filtered_out(fake_hub):
    fake_hub[("qintongli/GSM-Plus", None)] = [GSM_PLUS_ROWS[1]]
    with pytest.raises(RuntimeError):
        B.load_gsm_plus()


# --------------------------------------------------------------------------- #
# GSM-Symbolic
# --------------------------------------------------------------------------- #

def test_gsm_symbolic_parses_the_gsm8k_answer_marker(fake_hub):
    fake_hub[("apple/GSM-Symbolic", "main")] = GSM_SYMBOLIC_ROWS
    items = B.load_gsm_symbolic()
    assert [it["answer"] for it in items] == ["7", "24"]


def test_gsm_symbolic_ids_encode_template_and_instance(fake_hub):
    """One template yields many instances; ids must not collide, or the pool
    will silently overwrite questions."""
    fake_hub[("apple/GSM-Symbolic", "main")] = GSM_SYMBOLIC_ROWS + [
        {**GSM_SYMBOLIC_ROWS[0], "instance": 2}]
    ids = [it["id"] for it in B.load_gsm_symbolic()]
    assert len(set(ids)) == 3


def test_gsm_symbolic_config_is_part_of_the_id(fake_hub):
    fake_hub[("apple/GSM-Symbolic", "p1")] = GSM_SYMBOLIC_ROWS
    assert all(it["id"].startswith("gsmsym_p1_") for it in B.load_gsm_symbolic(config="p1"))


def test_gsm_symbolic_rejects_an_unknown_config():
    with pytest.raises(ValueError):
        B.load_gsm_symbolic(config="p3")


def test_gsm_symbolic_raises_on_a_renamed_column(fake_hub):
    fake_hub[("apple/GSM-Symbolic", "main")] = [{"id": 0, "instance": 0, "q": "x"}]
    with pytest.raises(KeyError):
        B.load_gsm_symbolic()


# --------------------------------------------------------------------------- #
# AIME
# --------------------------------------------------------------------------- #

def test_aime_reads_problem_and_answer(fake_hub):
    fake_hub[("HuggingFaceH4/aime_2024", None)] = AIME_ROWS
    items = B.load_aime()
    assert items[0]["question"].startswith("Find the number")
    assert [it["answer"] for it in items] == ["204", "80"]


def test_aime_is_too_small_to_certify_anything(fake_hub):
    """Not a style point: with 30 questions the certification floor (section 3.8)
    rules out every operating point below 73% coverage, and Bonferroni is out
    entirely. The loader's docstring says so; this keeps it true."""
    from caac.pool.ltt import certification_floor, min_served_certifiable
    assert min_served_certifiable(0.1, 0.1) == 22
    assert certification_floor(30, 0.1, 0.1) > 0.70
    assert certification_floor(30, 0.1, 0.1 / 21) == 1.0


def test_aime_raises_on_a_renamed_column(fake_hub):
    fake_hub[("HuggingFaceH4/aime_2024", None)] = [{"id": 0, "question": "q", "answer": "1"}]
    with pytest.raises(KeyError, match="problem"):
        B.load_aime()


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

def test_load_benchmark_dispatches_every_clean_set(fake_hub):
    fake_hub[("qintongli/GSM-Plus", None)] = GSM_PLUS_ROWS
    fake_hub[("apple/GSM-Symbolic", "main")] = GSM_SYMBOLIC_ROWS
    fake_hub[("apple/GSM-Symbolic", "p1")] = GSM_SYMBOLIC_ROWS
    fake_hub[("apple/GSM-Symbolic", "p2")] = GSM_SYMBOLIC_ROWS
    fake_hub[("HuggingFaceH4/aime_2024", None)] = AIME_ROWS
    for name in ("gsm_plus", "gsm_symbolic", "gsm_symbolic_p1",
                 "gsm_symbolic_p2", "aime"):
        items = B.load_benchmark(name)
        assert items and {"id", "question", "answer"} <= set(items[0])


def test_unknown_benchmark_lists_the_clean_sets():
    with pytest.raises(NotImplementedError, match="gsm_plus"):
        B.load_benchmark("not_a_dataset")


def test_n_limits_the_number_of_items(fake_hub):
    fake_hub[("qintongli/GSM-Plus", None)] = GSM_PLUS_ROWS
    assert len(B.load_gsm_plus(n=1)) == 1
