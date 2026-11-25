"""Lock v2 (amended after v1, 25/11/2026): cut budgets, cost measures, caps covering observed
outputs, the R4 planning estimate. Synthetic data only."""

import gzip
import json
import random
from pathlib import Path

from caac.pool_v1.config import config_from_dict
from caac.pool_v1.crossmap import max_unit_price, measures_for, policy_grid, run
from caac.pool_v1.io import load_pool_v1, write_rows
from caac.pool_v1.lock import allowed_cuts, check_config_against_lock, load_lock
from caac.pool_v1.report import make_report
from caac.pool_v1.schema import Row
from caac.pool_v1.validate import validate

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "protocols" / "audit_lock_v1.json"
V2 = ROOT / "protocols" / "audit_lock_v2.json"


def test_v2_keeps_every_threshold_of_v1():
    a, b = load_lock(V1), load_lock(V2)
    assert b["status"] == "frozen" and b["version"] == "2" and a["status"] == "frozen"
    assert a["audit"] == b["audit"] and a["sensitivity"] == b["sensitivity"]
    assert a["pairs"]["primary"] == b["pairs"]["primary"]
    assert a["benchmarks"]["exclude_from_primary"] == b["benchmarks"]["exclude_from_primary"]
    assert all(x["seen_v1_results"] for x in b["amendments"]) and len(b["amendments"]) == 4


def test_cut_budgets_below_the_cap_only():
    pol = load_lock(V2)["policies"]
    assert allowed_cuts(pol, 384000) >= {512, 32768, 96000, 192000}
    assert allowed_cuts(pol, 512) == {128, 256}                      # no budget >= the cap
    assert allowed_cuts(load_lock(V1)["policies"], 384000) == {96000, 192000}   # v1 unchanged


def _rows(n=300, seed=0, over_cap=False):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        for m, p_ok, tok, price in (("Qwen3-8B", 0.7, 3000, 2e-7), ("qwen3-235b-a22b-2507", 0.9, 2000, 1e-6)):
            t = rng.randint(tok // 4, tok * 2)
            if over_cap and m == "Qwen3-8B" and i == 0:
                t = 20000
            ok = rng.random() < p_ok and t < 4000
            out.append(Row(f"q{i:04d}", m, 0, ok, out_tokens=t, in_tokens=80, finish="unknown",
                           cost_usd=(t + 80) * price))
    return out


def _meta(caps):
    from caac.pool_v1.schema import Meta
    return Meta(format="caac-pool", version="1", source={"name": "syn", "url": "x", "version_or_sha": "x"},
                grader={"kind": "exact", "name": "syn"}, answer_equivalence="none", answer_keys=False, caps=caps)


def test_lrb_measures_and_grid_under_v2():
    L2 = load_lock(V2)
    rows, meta = _rows(), _meta({"Qwen3-8B": 16384, "qwen3-235b-a22b-2507": 16384})
    pair = {"small": "Qwen3-8B", "big": "qwen3-235b-a22b-2507", "tier": "primary"}
    names = [m["name"] for m in measures_for("lrb_x", rows, meta, pair, L2)]
    assert names == ["tokens_params", "tokens_lambda1"]               # usd skipped: locally run small model
    lam = [m for m in measures_for("lrb_x", rows, meta, pair, L2) if m["name"] == "tokens_params"][0]["cost"]["lambda"]
    assert abs(lam - 22 / 8.2) < 1e-9                                  # activated parameters
    g = policy_grid(rows, meta, pair, L2)
    assert g["k"] == [1] and g["cut"] == [None, 512, 1024, 2048, 4096, 8192]
    cfg = config_from_dict({"small": pair["small"], "big": pair["big"], **{k: L2["audit"][k] for k in
                            ("alpha", "delta", "s", "salt", "cost_test")},
                            "cost": {"measure": "tokens", "lambda": lam}, "policies": g})
    assert check_config_against_lock(cfg, V2, caps=meta.caps) == []
    assert check_config_against_lock(cfg, V1, caps=meta.caps)        # the same grid is outside v1


def test_price_counts_zero_output_rows():
    rows = [Row("a", "m", 0, True, out_tokens=0, in_tokens=100, cost_usd=1.0),
            Row("b", "m", 0, True, out_tokens=100, in_tokens=100, cost_usd=0.2)]
    assert max_unit_price(rows, "m") == 0.01


def test_validator_flags_output_above_cap():
    rows = _rows(over_cap=True)
    rep = validate(rows, _meta({"Qwen3-8B": 16384, "qwen3-235b-a22b-2507": 16384}))
    assert any(i.code == "W_OUT_OVER_CAP" and i.model == "Qwen3-8B" for i in rep.issues)


def test_lurume_cap_covers_observed_output(tmp_path):
    from caac.pool_v1.adapters.lurume import convert
    meta, scores = tmp_path / "m.jsonl.gz", tmp_path / "s.jsonl.gz"
    with gzip.open(meta, "wt", encoding="utf-8") as fm, gzip.open(scores, "wt", encoding="utf-8") as fs:
        for i in range(10):
            out = 5000 if i == 0 else 100
            fm.write(json.dumps({"task_id": f"d:{i}", "dataset": "d", "model_id": "x", "sample_index": "0",
                                 "finish_reason": "stop", "usage": {"input": 5, "output": out},
                                 "official_usd": "0.001", "requested_max_tokens": "1000"}) + "\n")
            fs.write(json.dumps({"task_id": f"d:{i}", "dataset": "d", "model_id": "x", "quality": 1.0}) + "\n")
    (d,) = convert(meta, scores, tmp_path / "o")
    rows, _, m, _ = load_pool_v1(d)
    assert m.caps == {"x": 5000} and "output_above_requested_max': 1" in m.notes
    assert not any(i.code == "W_OUT_OVER_CAP" for i in validate(rows, m).issues)


def test_r4_planning_estimate_when_nothing_passes_half_a():
    rows, meta = _rows(seed=3), _meta({"Qwen3-8B": 16384, "qwen3-235b-a22b-2507": 16384})
    L2 = load_lock(V2)
    pair = {"small": "Qwen3-8B", "big": "qwen3-235b-a22b-2507"}
    cfg = config_from_dict({**pair, **{k: L2["audit"][k] for k in ("alpha", "delta", "s", "salt", "cost_test")},
                            "cost": {"measure": "tokens", "lambda": 22 / 8.2},
                            "policies": policy_grid(rows, meta, pair, L2)})
    rep = make_report(rows, meta, cfg)
    r4 = rep["R4_break_even"]
    if rep["R1_verdict"].get("chosen_on_half_a") is None:
        c = r4["certification"]
        assert c["basis"].startswith("lowest-risk") and c["reachable"] in (True, False)
        assert (c["questions_needed"] is None) == (c["reachable"] is False)
    # risk counts the FINAL answer (small or escalated big): with both models mostly wrong every
    # candidate is far above alpha -> unreachable
    bad = [Row(r.qid, r.model, 0, (int(r.qid[1:]) % 3 == 0), out_tokens=r.out_tokens,
               in_tokens=r.in_tokens, finish=r.finish, cost_usd=r.cost_usd) for r in rows]
    c = make_report(bad, meta, cfg)["R4_break_even"]["certification"]
    assert c["reachable"] is False and c["questions_needed"] is None


def test_v2_map_runs(tmp_path):
    P = tmp_path / "pools" / "lrb_syn"
    P.mkdir(parents=True)
    write_rows(_rows(), P / "rows.jsonl.gz")
    (P / "meta.json").write_text(json.dumps({"format": "caac-pool", "version": "1",
                                             "source": {"name": "syn", "url": "x", "version_or_sha": "x"},
                                             "grader": {"kind": "exact", "name": "syn"}, "answer_equivalence": "none",
                                             "answer_keys": False,
                                             "caps": {"Qwen3-8B": 16384, "qwen3-235b-a22b-2507": 16384}}), encoding="utf-8")
    s = run(tmp_path / "pools", V2, tmp_path / "map", log=lambda *_: None)
    lines = [json.loads(l) for l in (tmp_path / "map" / "map.jsonl").read_text(encoding="utf-8").splitlines()]
    assert s["pre_registered"] and {x["measure"] for x in lines} == {"tokens_params", "tokens_lambda1"}
    assert all("reachable" in x for x in lines)
