"""MATH loader tests with a mock `datasets` (no network)."""
import sys, types
import pytest


def _install_mock(monkeypatch):
    m500 = [{"problem": f"P500 {i}", "answer": str(i)} for i in range(3)]
    rows = (
        [{"problem": f"tr {i}", "solution": "", "answer": f"a{i}", "unique_id": f"train/algebra/{i}.json"} for i in range(5)]
        + [{"problem": f"te {i}", "solution": "", "answer": f"b{i}", "unique_id": f"test/geometry/{i}.json"} for i in range(4)]
        # a MATH-500 problem sitting in the mirror's train split by mistake -> must be excluded
        + [{"problem": "P500   1", "solution": "", "answer": "1", "unique_id": "test/algebra/999.json"}]
    )
    def load_dataset(name, cfg=None, split=None, **kw):
        if name == "HuggingFaceH4/MATH-500":
            return m500
        if name == "nlile/hendrycks-MATH-benchmark":
            assert split == "train"          # never read the mirror's 'test' (= MATH-500)
            return rows
        raise RuntimeError("unknown " + name)
    mod = types.ModuleType("datasets"); mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_math_train_uses_unique_id_origin(monkeypatch):
    _install_mock(monkeypatch)
    from caac.data.benchmarks import load_benchmark
    items = load_benchmark("math_train")
    assert [it["id"] for it in items] == [f"math_train_algebra_{i}" for i in range(5)]
    assert items[0]["answer"] == "a0"


def test_math_test_excludes_math500(monkeypatch):
    _install_mock(monkeypatch)
    from caac.data.benchmarks import load_benchmark
    items = load_benchmark("math_test_ex500")
    assert len(items) == 4                      # the leaked MATH-500 problem is dropped
    assert all(it["id"].startswith("math_test_geometry_") for it in items)
    assert not any(it["id"].startswith("math500_") for it in items)


def test_n_truncates(monkeypatch):
    _install_mock(monkeypatch)
    from caac.data.benchmarks import load_benchmark
    assert len(load_benchmark("math_train", n=2)) == 2
