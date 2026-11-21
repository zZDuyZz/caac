"""research_pool adapter: PoolRecord -> CAAC-Pool v1 keeps every field it maps and invents none."""

from caac.pool.schema import Completion, PoolRecord, write_records
from caac.pool_v1 import load_pool_v1, validate
from caac.pool_v1.adapters.research_pool import convert, records_to_rows


def comp(ans, ok, tok, fin="stop"):
    return Completion(answer=ans, correct=ok, decode_tokens=tok, finish_reason=fin, answer_raw=ans)


def recs(model):
    return {
        "q1": PoolRecord(query_id="q1", dataset="d", model=model, gold="4", prompt_tokens=30,
                         greedy=comp("4", True, 50),
                         samples=[comp("4", True, 60), comp("5", False, 70), comp("4", True, 999, "length")]),
        "q2": PoolRecord(query_id="q2", dataset="d", model=model, gold="7", prompt_tokens=31,
                         greedy=comp(None, False, 80, "length"),
                         samples=[comp("7", True, 40), comp("7", True, 41), comp("8", False, 42)]),
    }


def test_rows_mirror_records():
    r = recs("small")
    rows = records_to_rows(r, split="calib")
    assert len(rows) == 8
    by = {(x.qid, x.draw): x for x in rows}
    assert by[("q1", 0)].is_greedy and not by[("q1", 1)].is_greedy
    assert [by[("q1", d)].out_tokens for d in range(4)] == [50, 60, 70, 999]
    assert by[("q1", 3)].finish == "length" and by[("q1", 3)].answer_key is None    # cut sample: no vote
    assert by[("q2", 0)].answer_key is None                                         # unparsable greedy
    assert by[("q1", 1)].answer_key == by[("q1", 0)].answer_key != by[("q1", 2)].answer_key
    assert all(x.in_tokens == r[x.qid].prompt_tokens and x.split == "calib" for x in rows)
    assert [x.correct for x in rows if x.qid == "q2"] == [False, True, True, False]
    assert all(x.gpu_s is None and x.cost_usd is None for x in rows)                # nothing invented


def test_convert_writes_valid_pool(tmp_path):
    ps, pb = tmp_path / "s.jsonl.gz", tmp_path / "b.jsonl.gz"
    write_records(ps, list(recs("small").values()))
    write_records(pb, list(recs("big").values()))
    out = convert([(str(ps), "calib"), (str(pb), "calib")], tmp_path / "v1", source_name="unit",
                  source_url="https://example.invalid", caps={"small": 999, "big": 999})
    rows, rerr, meta, merr = load_pool_v1(out)
    assert not rerr and not merr and meta.answer_keys and meta.answer_equivalence == "math"
    rep = validate(rows, meta, pair=("small", "big"))
    assert rep.ok
    assert rep.pair["capabilities"] == ["risk", "audit", "early_stop", "sample_more", "truncation_check"]
    msgs = [i.message for i in rep.issues if i.code == "W_CORRECT_NO_KEY"]
    assert msgs and "cut by the cap" in msgs[0]
    a = (out / "rows.jsonl.gz").read_bytes()
    convert([(str(ps), "calib"), (str(pb), "calib")], tmp_path / "v1", source_name="unit",
            source_url="https://example.invalid", caps={"small": 999, "big": 999})
    assert (out / "rows.jsonl.gz").read_bytes() == a                                # same input -> same bytes
