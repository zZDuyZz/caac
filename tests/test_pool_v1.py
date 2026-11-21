"""CAAC-Pool v1: reading, capability detection and integrity checks."""

import gzip
import json

import pytest

from caac.pool_v1 import Meta, Row, levels, load_pool_v1, read_rows, validate, write_rows
from caac.pool_v1.validate import certification_floor


def meta(**kw):
    d = dict(source={"name": "unit", "url": "https://example.invalid", "version_or_sha": "0"},
             grader={"kind": "exact", "name": "eq"}, answer_equivalence="exact")
    d.update(kw)
    return Meta(**d)


def pool(n_q=60, k_small=4, k_big=2, tokens=True, keys=True, finish=True, acc_every=3):
    rows = []
    for i in range(n_q):
        for m, k, base in (("small", k_small, 100), ("big", k_big, 300)):
            for d in range(k):
                rows.append(Row(qid=f"q{i}", model=m, draw=d, correct=(i + d) % acc_every != 0,
                                answer_key=(str(i % 5) if keys else None),
                                out_tokens=(base + i if tokens else None),
                                finish=("stop" if finish else None), split="calib"))
    return rows


def codes(rep, sev=None):
    return sorted({i.code for i in rep.issues if sev is None or i.severity == sev})


def test_floor_matches_hand_values():
    assert certification_floor(0.10, 0.10) == 22
    assert certification_floor(0.05, 0.10) == 45
    assert certification_floor(0.20, 0.10) == 11


def test_capabilities_and_levels():
    rep = validate(pool(), meta(), pair=("small", "big"))
    assert rep.ok, rep.errors()
    assert rep.capabilities["small"] == ["risk", "audit", "early_stop", "sample_more", "truncation_check"]
    assert rep.level["small"] == "L3"
    assert rep.pair["level"] == "L3" and rep.pair["common_questions"] == 60


def test_k1_source_has_audit_and_early_stop_but_no_voting():
    rep = validate(pool(k_small=1, k_big=1, finish=False), meta())
    assert rep.capabilities["small"] == ["risk", "audit", "early_stop"]
    assert rep.level["small"] == "L1"


def test_no_tokens_is_risk_only():
    rep = validate(pool(tokens=False), meta())
    assert rep.level["small"] == "L0"
    assert "audit" not in rep.capabilities["small"]


def test_equivalence_none_disables_voting():
    rep = validate(pool(), meta(answer_equivalence="none"))
    assert "sample_more" not in rep.capabilities["small"]


def test_gpu_seconds_reach_l4():
    rows = pool()
    for r in rows:
        r.gpu_s = 0.5
    assert validate(rows, meta()).level["small"] == "L4"


def test_inferred_truncation_from_cap():
    rows = pool(finish=False)
    rep = validate(rows, meta(caps={"small": 120}))
    assert "truncation_check" in rep.capabilities["small"]
    assert "I_INFERRED" in codes(rep) and "W_CAP_HIT" in codes(rep)


def test_duplicate_and_leak_are_errors():
    rows = pool(n_q=50)
    rows.append(Row(qid="q0", model="small", draw=0, correct=True, split="test"))
    rep = validate(rows, meta())
    assert not rep.ok
    assert {"E_DUP", "E_LEAK"} <= set(codes(rep, "error"))


def test_uneven_gap_missing_pair_warnings():
    rows = [r for r in pool(n_q=60) if not (r.qid == "q1" and r.model == "small" and r.draw == 2)]
    rows = [r for r in rows if not (r.qid == "q2" and r.model == "big")]
    rep = validate(rows, meta(), pair=("small", "big"))
    assert rep.ok
    assert {"W_UNEVEN", "W_DRAW_GAP", "W_PAIR_MISSING"} <= set(codes(rep, "warning"))


def test_small_n_extreme_accuracy_heavy_tail():
    rows = pool(n_q=10)                              # 10 questions
    for r in rows:
        r.correct = True                             # accuracy 1.0
    for r in rows:
        if r.model == "small" and r.draw == 0 and r.qid == "q0":
            r.out_tokens = 10 ** 6
    for r in rows:
        if r.model == "small" and r.qid in ("q1", "q2"):
            r.out_tokens = 50_000
    rep = validate(rows, meta())
    w = set(codes(rep, "warning"))
    assert {"W_N_FLOOR", "W_ACC_EXTREME", "W_HEAVY_TAIL"} <= w


def test_greedy_rules():
    rows = pool(n_q=50)
    for r in rows:
        if r.model == "small":
            r.is_greedy = r.draw == 0
    assert validate(rows, meta()).ok
    rows[1].is_greedy = True        # small q0 draw 1
    assert "E_GREEDY" in codes(validate(rows, meta()), "error")


def test_meta_errors_and_judge_warning():
    rep = validate(pool(), meta(grader={"kind": "llm-judge", "name": "x"}, answer_equivalence="fuzzy"))
    assert "E_META" in codes(rep, "error") and "W_JUDGE" in codes(rep, "warning")
    assert "E_META" in codes(validate(pool(), None), "error")


def test_flags_and_unknown_fields(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"qid": "a", "model": "s", "draw": 0, "correct": 1, "flags": ["think_marker"],
                             "weird": 3}) + "\n" + "{bad json\n" +
                 json.dumps({"qid": "b", "model": "s", "draw": 0}) + "\n")
    rows, errs = read_rows(p)
    assert len(rows) == 1 and [ln for ln, _ in errs] == [2, 3]
    rep = validate(rows, meta(), row_errors=errs)
    assert "E_PARSE" in codes(rep, "error")
    assert {"W_FLAG", "W_UNKNOWN_FIELD"} <= set(codes(rep, "warning"))


def test_csv_jsonl_gz_round_trip_and_byte_stability(tmp_path):
    rows = pool(n_q=5)
    g1, g2 = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    write_rows(rows, g1)
    write_rows(rows, g2)
    assert g1.read_bytes() == g2.read_bytes()            # mtime=0 -> identical bytes
    back, errs = read_rows(g1)
    assert not errs and back == rows
    c = tmp_path / "rows.csv"
    c.write_text("qid,model,draw,correct,out_tokens,split,flags\n"
                 "q0,s,0,1,10,calib,\nq0,s,1,0,12,calib,think_marker;loop\n")
    back, errs = read_rows(c)
    assert not errs and back[1].flags == ["think_marker", "loop"] and back[0].correct is True


def test_load_directory(tmp_path):
    write_rows(pool(n_q=3), tmp_path / "rows.jsonl")
    (tmp_path / "meta.json").write_text(json.dumps({
        "format": "caac-pool", "version": "1",
        "source": {"name": "u", "url": "x", "version_or_sha": "y"},
        "grader": {"kind": "exact", "name": "eq"}, "answer_equivalence": "exact"}))
    rows, rerr, m, merr = load_pool_v1(tmp_path)
    assert len(rows) == 18 and not rerr and not merr and m.answer_equivalence == "exact"
    rep = validate(rows, m, row_errors=rerr, meta_errors=merr)
    assert rep.ok


def test_report_dict_is_deterministic():
    a = json.dumps(validate(pool(), meta(), pair=("small", "big")).to_dict(), sort_keys=True)
    b = json.dumps(validate(list(reversed(pool())), meta(), pair=("small", "big")).to_dict(), sort_keys=True)
    assert a == b
