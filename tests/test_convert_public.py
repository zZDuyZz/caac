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


def test_monkey_real_token_counts(tmp_path):
    import pytest
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {"[UNK]": 0, "step": 1, "one": 2, "so": 3, "x": 4, "=": 5, "42": 6, "####": 7, "Question": 8}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tkp = tmp_path / "llama3.json"
    tk.save(str(tkp))
    _monkey(tmp_path)
    out = tmp_path / "pools"
    cp.main([str(tmp_path), str(out), "--only", "monkey", "--tok", f"Llama-3={tkp}"])
    rows, rerr, meta, merr = load_pool_v1(out / "monkey_GSM8K")
    assert not rerr and not merr and validate(rows, meta).ok
    meta_json = json.loads((out / "monkey_GSM8K" / "meta.json").read_text(encoding="utf-8"))
    assert set(meta_json["token_source"]) == {"Llama-3-8B-Instruct", "Llama-3-70B-Instruct"}
    assert all(v.startswith("measured:") for v in meta_json["token_source"].values())
    r = [x for x in rows if x.model == "Llama-3-8B-Instruct" and x.qid.endswith(":100") and x.draw == 1][0]
    assert r.out_tokens == 8                           # " step one, so x = 42.\n#### 42" under this word-level vocab
    assert r.in_tokens == 11                           # "Question: ...Question: ..." (x10) -> 11 whitespace tokens
    # Pythia has no --tok: stays estimated, in a pool that mixes both
    rows, _, _, _ = load_pool_v1(out / "monkey_MATH")
    meta_math = json.loads((out / "monkey_MATH" / "meta.json").read_text(encoding="utf-8"))
    assert meta_math["token_source"]["Pythia-70M"].startswith("estimated")
    # deterministic bytes with a tokenizer too
    cp.main([str(tmp_path), str(tmp_path / "pools2"), "--only", "monkey", "--tok", f"Llama-3={tkp}"])
    assert (tmp_path / "pools2" / "monkey_GSM8K" / "rows.jsonl.gz").read_bytes() == \
        (out / "monkey_GSM8K" / "rows.jsonl.gz").read_bytes()


def test_tok_option_errors(tmp_path, capsys):
    import pytest
    with pytest.raises(SystemExit):
        cp.main([str(tmp_path), str(tmp_path / "o"), "--tok", "Llama-3"])
    with pytest.raises(SystemExit):
        cp.main([str(tmp_path), str(tmp_path / "o"), "--tok", f"Llama-3={tmp_path / 'nope.json'}"])
