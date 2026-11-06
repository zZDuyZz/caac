import pytest

from caac.pool.schema import (Completion, PoolRecord, append_texts, text_sidecar_path,
                              token_sidecar_path, write_records)
import smoke_check as S


def _write(tmp_path, ok=True, tok_len_bug=False, n=3):
    pool = tmp_path / "s.jsonl.gz"
    recs, texts, toks = [], [], []
    for i in range(n):
        g = Completion(answer="42", correct=ok, decode_tokens=5, finish_reason="stop",
                       answer_raw="42", conf_prefix32=0.8)
        recs.append(PoolRecord(f"q{i}", "d", "m", "42", 10, g, [g, g]))
        texts.append({"query_id": f"q{i}", "greedy_text": "so \\boxed{42}", "sample_texts": ["a", "b"]})
        L = 4 if tok_len_bug else 5
        toks.append({"query_id": f"q{i}", "greedy": {"lp": [0.0] * L, "tc": [0.0] * L, "ent": [0.0] * L},
                     "samples": [{"lp": [0.0], "tc": [0.0]}] * 2})
    write_records(pool, recs)
    append_texts(text_sidecar_path(pool), texts)
    append_texts(token_sidecar_path(pool), toks)
    return pool


def test_ok(tmp_path):
    errs, rows = S.check(_write(tmp_path), min_correct=3)
    assert errs == [] and len(rows) == 3


def test_dead_grader_detected(tmp_path):
    errs, _ = S.check(_write(tmp_path, ok=False), min_correct=2)
    assert any("grader/prompt suspect" in e for e in errs)


def test_token_sidecar_length_bug(tmp_path):
    errs, _ = S.check(_write(tmp_path, tok_len_bug=True))
    assert any("token sidecar length" in e for e in errs)


def test_missing_sidecar(tmp_path):
    pool = _write(tmp_path)
    token_sidecar_path(pool).unlink()
    errs, _ = S.check(pool)
    assert any("sidecar mismatch" in e for e in errs)


def test_thinking_leak_is_a_failure(tmp_path):
    """Every pool is built with thinking OFF; a <think> in the output means the
    template switch did not take and the token budget went to reasoning text."""
    pool = _write(tmp_path)
    tp = text_sidecar_path(pool)
    tp.unlink()
    append_texts(tp, [{"query_id": f"q{i}", "greedy_text": "<think>hmm</think> so \\boxed{42}",
                       "sample_texts": ["a", "b"]} for i in range(3)])
    errs, _ = S.check(pool)
    assert any("thinking marker" in e for e in errs)
