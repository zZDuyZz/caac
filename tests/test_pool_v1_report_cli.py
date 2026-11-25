"""Report R1-R5, byte stability, CLI, protocol lock, and the betting cost test."""

import json
import random
import shutil
from pathlib import Path

import pytest

from caac.pool_v1.audit import AuditInput, PolicyData, _betting_p, audit
from caac.pool_v1.cli import main
from caac.pool_v1.config import config_from_dict
from caac.pool_v1.io import write_rows
from caac.pool_v1.lock import check_config_against_lock, lock_hash
from caac.pool_v1.report import clopper_pearson, dumps, make_report, to_markdown
from caac.pool_v1.schema import Meta, Row

ROOT = Path(__file__).resolve().parents[1]
S, B = "org/small", "org/big"


def synth(n, p_small=0.97, p_big=0.95, seed=0, cap=1000):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        q = f"q{i:05d}"
        ok = rng.random() < p_small
        rows.append(Row(q, S, 0, ok, answer_key="a" if ok else "b", out_tokens=rng.randint(50, 300),
                        in_tokens=40, finish="stop", is_greedy=True, split="test"))
        rows.append(Row(q, B, 0, rng.random() < p_big, answer_key="a", out_tokens=rng.randint(50, 300),
                        in_tokens=40, finish="stop", is_greedy=True, split="test"))
    meta = Meta(source={"name": "synthetic", "url": "local", "version_or_sha": "1"},
                grader={"kind": "exact", "name": "synthetic"}, answer_equivalence="exact",
                caps={S: cap, B: cap}, answer_keys=True)
    return rows, meta


CFG = {"small": S, "big": B, "cost": {"lambda": 8.0},
       "policies": {"k": [1], "cut": [None, 250], "on_unsure": ["escalate"]}}


def test_report_go_on_clear_case_and_sections():
    rows, meta = synth(3000)
    rep = make_report(rows, meta, config_from_dict(CFG))
    assert rep["R1_verdict"]["verdict"] == "GO"
    assert set(rep) >= {"R1_verdict", "R2_estimates", "R3_cost_measures", "R4_break_even", "R5_data_quality"}
    r2 = rep["R2_estimates"]
    lo, hi = r2["risk_interval"]
    assert lo <= r2["risk"] <= hi and 0 < r2["coverage"] <= 1
    assert rep["R4_break_even"]["break_even_lambda"] < 8.0          # GO at lambda 8 -> break-even below 8
    assert "Verdict: GO" in to_markdown(rep)


def test_report_is_byte_stable_and_order_free():
    rows, meta = synth(600, seed=4)
    c = config_from_dict(CFG)
    a = dumps(make_report(rows, meta, c))
    b = dumps(make_report(list(reversed(rows)), meta, c))
    assert a == b


def test_report_stops_on_pool_errors():
    rows, meta = synth(50)
    rows.append(rows[0])                                    # duplicate (qid, model, draw)
    rep = make_report(rows, meta, config_from_dict(CFG))
    assert rep["R1_verdict"]["verdict"] == "NOT RUN" and not rep["R5_data_quality"]["ok"]


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 22, 0.80)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.1 ** (1 / 22), rel=1e-6)
    lo, hi = clopper_pearson(5, 100, 0.95)
    assert 0.016 < lo < 0.017 and 0.112 < hi < 0.114


def test_betting_is_valid_and_more_powerful_with_small_variance():
    rng = random.Random(1)
    x_null = [rng.uniform(-1, 1) for _ in range(2000)]       # mean 0: H0 true
    assert _betting_p(x_null, -1, 1) > 0.05
    x_alt = [rng.uniform(-0.05, 0.25) for _ in range(500)]    # mean 0.1, small spread in [-1, 1]
    assert _betting_p(x_alt, -1, 1) < 1e-6
    inp = AuditInput([f"q{i}" for i in range(4)], [1, 1, 1, 1], [PolicyData("p", [True] * 4, [False] * 4, [0.1] * 4)], 1)
    assert audit(inp, cost_test="betting").verdict == "INCONCLUSIVE"
    with pytest.raises(ValueError):
        audit(inp, cost_test="hoeffding")


def test_cli_end_to_end_and_lock(tmp_path, capsys):
    rows, meta = synth(400, seed=2)
    pool = tmp_path / "pool"
    pool.mkdir()
    write_rows(rows, pool / "rows.jsonl.gz")
    (pool / "meta.json").write_text(json.dumps({"format": "caac-pool", "version": "1", "source": meta.source,
                                                "grader": meta.grader, "answer_equivalence": "exact",
                                                "answer_keys": True, "caps": meta.caps}))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(CFG))
    lock = ROOT / "protocols" / "audit_lock_v1.json"
    assert main(["validate", str(pool)]) == 0
    out = tmp_path / "r.json"
    assert main(["audit", str(pool), "--config", str(cfg), "--out", str(out), "--md", str(tmp_path / "r.md"),
                 "--lock", str(lock)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["input"]["lock"] == lock_hash(lock) and rep["input"]["lock_status"] in ("draft", "frozen")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(CFG, alpha=0.2)))
    assert main(["audit", str(pool), "--config", str(bad), "--lock", str(lock)]) == 2
    assert "alpha" in capsys.readouterr().err


def test_lock_rejects_cut_not_in_protocol_and_missing_no_cut():
    lock = ROOT / "protocols" / "audit_lock_v1.json"
    c = config_from_dict(dict(CFG, policies={"k": [1], "cut": [None, 333]}))
    assert any("cut 333" in p for p in check_config_against_lock(c, lock, caps={S: 1000}))
    c = config_from_dict(dict(CFG, policies={"k": [1], "cut": [250]}))
    assert any("must include null" in p for p in check_config_against_lock(c, lock, caps={S: 1000}))
    c = config_from_dict(dict(CFG, policies={"k": [1], "cut": [None, 250], "on_unsure": ["abstain"]}))
    assert any("on_unsure" in p for p in check_config_against_lock(c, lock, caps={S: 1000}))


def test_example_files_run(tmp_path):
    ex = ROOT / "examples"
    pool = tmp_path / "mypool"
    pool.mkdir()
    shutil.copy(ex / "pool_v1_template.csv", pool / "rows.csv")
    shutil.copy(ex / "meta_example.json", pool / "meta.json")
    assert main(["audit", str(pool), "--config", str(ex / "config_example.json"),
                 "--out", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["R1_verdict"]["verdict"] == "INCONCLUSIVE"
