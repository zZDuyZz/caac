"""scripts/convert_public.py (LLMRouterBench + Monkey -> Pool v1) and the Lurume adapter, on fake data."""

import gzip
import importlib.util
import json
from pathlib import Path

from caac.pool_v1.io import load_pool_v1
from caac.pool_v1.validate import validate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("convert_public", ROOT / "scripts" / "convert_public.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def _lrb(tmp):
    base = tmp / "lrb" / "bench-release"
    for model, shift, cost in (("small-7b", 0, 1e-5), ("big-api", 1, 1e-3), ("openrouter", 0, 0.0)):
        recs = []
        for i in range(1, 121):
            q = f"question {i}" if not (model == "big-api" and i == 7) else "a different question"
            recs.append({"index": i, "origin_query": q, "prompt": "p", "prompt_tokens": 50,
                         "completion_tokens": 100 + i * (1 + shift), "cost": cost * i,
                         "score": float((i + shift) % 5 != 0), "prediction": "x", "ground_truth": "x",
                         "raw_output": "long text " * 5})
        d = base / "gpqa" / "test" / model
        d.mkdir(parents=True)
        (d / f"gpqa-test-{model}.json").write_text(json.dumps({"model_name": model, "demo": False, "records": recs}),
                                                    encoding="utf-8")
    d = base / "arenahard" / "small-7b"          # no split level, judge benchmark, a non-binary score
    d.mkdir(parents=True)
    (d / "a.json").write_text(json.dumps({"records": [{"index": 1, "origin_query": "q", "prompt_tokens": 5,
                                                       "completion_tokens": 9, "cost": 0.1, "score": 0.5}]}),
                              encoding="utf-8")


def _monkey(tmp):
    d = tmp / "monkey"
    d.mkdir()
    for model, good in (("Llama-3-8B-Instruct", 0.7), ("Llama-3-70B-Instruct", 0.9)):
        probs = []
        for i in range(5):
            samples, corr = [], []
            for s in range(70):
                ok = (s * 7 + i) % 10 < good * 10
                ans = "42" if ok else str(s % 3)
                samples.append(f" step one, so x = {ans}.\n#### {ans}" if s % 2 else f" so {ans}\n#### 1,0{ans}")
                corr.append(ok)
            probs.append({"question": f"Q{i}", "gt_answer": "#### 42", "prompt": "Question: ..." * 10,
                          "samples": samples, "is_corrects": corr, "orig_dset_split": "test", "orig_dset_idx": 100 + i})
        (d / f"GSM8K_{model}.json").write_text(json.dumps(probs, indent=2), encoding="utf-8")
    (d / "MATH_Pythia-70M.json").write_text(json.dumps([{"question": "q", "prompt": "p", "samples":
        ["Final Answer: The final answer is $\\frac{1}{2}$. I hope it is correct."] * 3, "is_corrects": [True, False, True],
        "orig_dset_split": "test", "orig_dset_idx": 1}]), encoding="utf-8")


def test_streaming_reader_matches_json_load(tmp_path):
    objs = [{"a": i, "s": "x" * (i * 37), "l": list(range(i))} for i in range(40)]
    p = tmp_path / "arr.json"
    p.write_text(json.dumps(objs, indent=2), encoding="utf-8")
    assert list(cp.iter_json_array(str(p), chunk=64)) == objs        # tiny chunks force re-reads
    p.write_text("[]", encoding="utf-8")
    assert list(cp.iter_json_array(str(p))) == []


def test_answer_keys():
    assert cp.answer_key("GSM8K", "so 2+2\n#### 1,234.") == "1234"
    assert cp.answer_key("GSM8K", "no answer") is None
    assert cp.answer_key("MATH", "Final Answer: The final answer is $\\frac{1}{2}$. I hope it is correct.") == "\\frac{1}{2}"
    assert cp.answer_key("CodeContests", "print(1)") is None


def test_convert_lrb_and_monkey(tmp_path):
    _lrb(tmp_path)
    _monkey(tmp_path)
    out = tmp_path / "pools"
    cp.main([str(tmp_path), str(out)])
    rows, rerr, meta, merr = load_pool_v1(out / "lrb_gpqa_test")
    assert not rerr and not merr
    assert {r.model for r in rows} == {"small-7b", "big-api"}        # router folder skipped
    assert all(r.qid != "gpqa/test:7" for r in rows)                  # text mismatch -> dropped
    assert "questions_dropped_text_mismatch': 1" in meta.notes
    assert meta.caps == {"big-api": 512, "small-7b": 256}             # 100+2*120=340 -> 512; 220 -> 256
    rep = validate(rows, meta)
    assert rep.ok
    rows, _, meta, _ = load_pool_v1(out / "lrb_arenahard")
    assert meta.grader["kind"] == "llm-judge" and "score_nonbinary': 1" in meta.notes
    rows, rerr, meta, merr = load_pool_v1(out / "monkey_GSM8K")
    assert not rerr and not merr and meta.answer_keys and meta.answer_equivalence == "exact"
    assert len(rows) == 2 * 5 * 64 and max(r.draw for r in rows) == 63
    assert all(r.out_tokens <= 512 for r in rows) and {r.answer_key for r in rows} >= {"42"}
    assert validate(rows, meta).ok
    rows, _, meta, _ = load_pool_v1(out / "monkey_MATH")
    assert rows[0].answer_key == "\\frac{1}{2}" and len(rows) == 3
    # deterministic bytes
    b1 = (out / "monkey_GSM8K" / "rows.jsonl.gz").read_bytes()
    cp.main([str(tmp_path), str(tmp_path / "pools2"), "--only", "monkey"])
    assert (tmp_path / "pools2" / "monkey_GSM8K" / "rows.jsonl.gz").read_bytes() == b1


def test_lurume_adapter(tmp_path):
    from caac.pool_v1.adapters.lurume import convert
    meta = tmp_path / "m.jsonl.gz"
    scores = tmp_path / "s.jsonl.gz"
    with gzip.open(meta, "wt", encoding="utf-8") as fm, gzip.open(scores, "wt", encoding="utf-8") as fs:
        for i in range(30):
            for m, cap in (("a-flash", 1000), ("a-pro", 2000)):
                fin = "length" if i == 3 else (None if i == 4 else "stop")
                fm.write(json.dumps({"task_id": f"ds:{i}", "dataset": "ds", "model_id": m, "sample_index": "0",
                                     "finish_reason": fin, "usage": {"input": 10, "output": 100 + i},
                                     "official_usd": "0.001", "requested_max_tokens": str(cap)}) + "\n")
                fs.write(json.dumps({"task_id": f"ds:{i}", "dataset": "ds", "model_id": m, "quality": float(i % 4 != 0)}) + "\n")
        fs.write(json.dumps({"task_id": "ds:99", "dataset": "ds", "model_id": "a-pro", "quality": 1.0}) + "\n")
    (d,) = convert(meta, scores, tmp_path / "out")
    rows, rerr, m, merr = load_pool_v1(d)
    assert not rerr and not merr and len(rows) == 60
    assert m.caps == {"a-flash": 1000, "a-pro": 2000} and "score_without_response': 1" in m.notes
    assert {r.finish for r in rows} == {"stop", "length", "unknown"}
    assert validate(rows, m).ok
