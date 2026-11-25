"""Cross-source map on SYNTHETIC pools (the real sources are audited only after the lock is frozen)."""

import json
import random
from pathlib import Path

from caac.pool_v1.crossmap import list_configs, naive_go, pairs_for, run
from caac.pool_v1.io import load_pool_v1, write_rows
from caac.pool_v1.schema import Row

LOCK = Path(__file__).resolve().parents[1] / "protocols" / "audit_lock_v1.json"


def _pool(d: Path, models, n, seed, judge=False, draws=1, keys=False):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        q = f"q{i:05d}"
        for m, (p_ok, tok, usd_per_tok, cap) in models.items():
            for k in range(draws):
                ok = rng.random() < p_ok
                t = rng.randint(tok // 2, tok * 3 // 2)
                rows.append(Row(q, m, k, ok, answer_key=("a" if ok else str(rng.randint(0, 3))) if keys else None,
                                out_tokens=t, in_tokens=50, finish="stop", cost_usd=(t + 50) * usd_per_tok))
    d.mkdir(parents=True)
    write_rows(rows, d / "rows.jsonl.gz")
    meta = {"format": "caac-pool", "version": "1", "source": {"name": "syn", "url": "x", "version_or_sha": "x"},
            "grader": {"kind": "llm-judge" if judge else "exact", "name": "syn"},
            "answer_equivalence": "exact" if keys else "none", "answer_keys": keys,
            "caps": {m: v[3] for m, v in models.items()}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_map_runs_and_counts_criteria(tmp_path):
    P = tmp_path / "pools"
    M = {"deepseek-v4-flash": (0.95, 200, 1e-6, 1024), "deepseek-v4-pro": (0.96, 400, 4e-6, 1024),
         "other": (0.80, 300, 2e-6, 1024)}
    _pool(P / "lurume_mmlu_pro", M, 600, 1)
    _pool(P / "lurume_tiny", M, 50, 2)                                     # < n_min: no pair
    _pool(P / "lrb_arenahard", {"Qwen3-8B": (0.9, 200, 1e-6, 1024), "qwen3-235b-a22b-2507": (0.9, 300, 3e-6, 1024)},
          300, 3, judge=True)
    _pool(P / "monkey_GSM8K", {"Llama-3-8B-Instruct": (0.92, 150, 0, 512), "Llama-3-70B-Instruct": (0.97, 150, 0, 512)},
          200, 4, draws=4, keys=True)
    s = run(P, LOCK, tmp_path / "map", log=lambda *_: None)
    lines = [json.loads(l) for l in (tmp_path / "map" / "map.jsonl").read_text(encoding="utf-8").splitlines()]
    assert s["pre_registered"] is False and all(x["lock_status"] == "draft" for x in lines)
    prim = [x for x in lines if x["tier"] == "primary"]
    assert {(x["pool"], x["small"], x["measure"]) for x in prim} >= {
        ("lurume_mmlu_pro", "deepseek-v4-flash", "usd"), ("lurume_mmlu_pro", "deepseek-v4-flash", "tokens_lambda1"),
        ("monkey_GSM8K", "Llama-3-8B-Instruct", "tokens_params")}
    assert not any(x["pool"] == "lurume_tiny" for x in lines)
    assert all(x["tier"] == "excluded" for x in lines if x["pool"] == "lrb_arenahard")
    sec = [x for x in lines if x["tier"] == "secondary" and x["pool"] == "lurume_mmlu_pro"]
    assert {(x["small"], x["big"]) for x in sec} == {("deepseek-v4-flash", "other"), ("other", "deepseek-v4-pro")}
    for x in lines:
        assert x["verdict"] in ("GO", "NO-GO", "INCONCLUSIVE") and (tmp_path / "map" / "reports" / x["report"]).exists()
    assert set(s) >= {"primary", "secondary", "excluded", "lock_hash"}
    assert len(list_configs(P, LOCK)) == len(lines)          # the dry run lists exactly what is audited
    # same inputs -> same map bytes
    a = (tmp_path / "map" / "map.jsonl").read_bytes()
    run(P, LOCK, tmp_path / "map2", log=lambda *_: None)
    assert (tmp_path / "map2" / "map.jsonl").read_bytes() == a


def test_naive_rule_is_the_point_estimate():
    from caac.pool_v1.audit import AuditInput, PolicyData

    class RR:
        pass
    rr = RR()
    rr.audit_input = AuditInput(qids=["a", "b", "c", "d"], big_cost=[10, 10, 10, 10], c_max=20,
                                policies=[PolicyData("p", [True] * 4, [False, False, False, False], [5, 5, 5, 5]),
                                          PolicyData("q", [True] * 4, [True, False, False, False], [1, 1, 1, 1])])
    assert naive_go(rr, 0.10, 0.20) == (True, "p")           # q is cheaper but has risk 25%
    assert naive_go(rr, 0.10, 0.60) == (False, None)


def test_primary_pairs_come_from_the_lock_by_name(tmp_path):
    P = tmp_path / "p"
    _pool(P / "lurume_x", {"glm-5-3-flash": (0.9, 100, 5e-6, 512), "glm-5.3": (0.9, 100, 1e-6, 512)}, 120, 5)
    rows, _, meta, _ = load_pool_v1(P / "lurume_x")
    L = json.loads(LOCK.read_text(encoding="utf-8"))
    pr = pairs_for("lurume_x", rows, L)
    # declared pair kept even though the "flash" model is dearer here (names, not outcomes or prices)
    assert {"small": "glm-5-3-flash", "big": "glm-5.3", "tier": "primary", "n_common": 120} in pr
