"""overhead.py: fit time, bundle size, cold decision time, GPU reference."""
import json
import sys
from pathlib import Path

from caac.pool.protocol import Family


def _run(monkeypatch, p, out, extra=()):
    import overhead
    monkeypatch.setattr(sys, "argv", [
        "overhead.py", "--train-small", p["train_small"], "--train-big", p["train_big"],
        "--pool-small", p["calib_small"], "--pool-big", p["calib_big"], "--n", "60",
        "--json", str(out), *extra])
    overhead.main()
    return json.loads(Path(out).read_text())


def test_reports_every_part_and_compares_with_gpu_time_when_throughput_is_known(tmp_path, monkeypatch):
    from test_protocol import _write_parts
    p = _write_parts(tmp_path, n=300)
    for side, tps in (("small", 6000.0), ("big", 4100.0)):
        Path(p[f"calib_{side}"] + ".manifest.json").write_text(json.dumps(
            {"throughput": {"decode_tokens": 1, "gen_seconds": 1, "tok_per_s": tps}}))
    r = _run(monkeypatch, p, tmp_path / "o.json")
    assert r["n_questions"] == 60 and r["fit_seconds"] > 0
    assert r["bundle_bytes"]["stage1"] > 0 and r["bundle_bytes_total"] >= r["bundle_bytes"]["stage1"]
    assert 0 < r["decision_ms"]["median"] <= r["decision_ms"]["p95"]
    assert r["decision_ms"]["upper_bound"] is True
    assert r["gpu_ms_per_question_amortised"] > 0 and r["overhead_share_of_gpu_time"] > 0
    assert r["throughput_tok_per_s"] == {"small": 6000.0, "big": 4100.0}
    assert 0 < r["probe_share_of_cost"] < 1
    costs = {v["cost"] for v in r["feature_cost"].values()}
    assert costs == {"free", "extra"}


def test_without_manifests_the_gpu_comparison_is_left_out(tmp_path, monkeypatch):
    from test_protocol import _write_parts
    p = _write_parts(tmp_path, n=300)
    r = _run(monkeypatch, p, tmp_path / "o.json", ["--family", "caac_staged_value", "--rho", "0.03"])
    assert r["gpu_ms_per_question_amortised"] is None and r["overhead_share_of_gpu_time"] is None
    assert r["family"] == "caac_staged_value"


def test_pick_config_finds_the_nearest_grid_point():
    from overhead import pick_config
    fam = Family("f", [(0.0, 0.9), (0.8, 0.9), (0.8, 0.5)], make=lambda c: None)
    assert pick_config(fam, 0.8, 0.9) == 1
    assert pick_config(fam, 0.75, 0.55) == 2
