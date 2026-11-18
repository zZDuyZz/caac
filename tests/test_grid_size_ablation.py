"""Grid-size ablation (grid_size_ablation.py) and the fine-grid CAAC arm."""
import json
import sys

import pytest

from caac.pool import families as F


def test_default_size_is_the_paper_grid():
    from grid_size_ablation import grids
    rhos, taus = grids(len(F.RHO_GRID), len(F.TAU_GRID))
    assert rhos == tuple(F.RHO_GRID) and taus == list(F.TAU_GRID)


def test_grids_keep_both_special_rho_ends_and_run_conservative_first():
    from grid_size_ablation import grids
    for n_rho, n_tau in [(2, 2), (3, 8), (11, 32), (21, 64)]:
        rhos, taus = grids(n_rho, n_tau)
        assert len(rhos) == n_rho and len(taus) == n_tau
        assert rhos[0] == 0.0 and rhos[-1] == 1.01          # buy freely ... buy nothing
        assert list(rhos) == sorted(rhos) and len(set(rhos)) == n_rho
        assert taus == sorted(taus, reverse=True) and taus[0] == 0.99 and taus[-1] == 0.5
    with pytest.raises(ValueError):
        grids(1, 8)


def test_fine_grid_arm_is_larger_and_off_by_default():
    import certify
    assert len(certify.RHO_GRID_FINE) * len(certify.TAU_GRID_FINE) > len(F.RHO_GRID) * len(F.TAU_GRID)
    assert certify.TAU_GRID_FINE[0] > certify.TAU_GRID_FINE[-1]
    assert "caac_staged_biggrid" in certify.CAAC


def test_driver_end_to_end(tmp_path, monkeypatch):
    import grid_size_ablation
    from test_protocol import _write_parts
    p = _write_parts(tmp_path, n=450)
    out = tmp_path / "g.json"
    monkeypatch.setattr(sys, "argv", [
        "grid_size_ablation.py", "--train-small", p["train_small"], "--train-big", p["train_big"],
        "--pool-small", p["calib_small"], "--pool-big", p["calib_big"], "--sizes", "3x4,6x16",
        "--splits", "20", "--alpha", "0.2", "--json", str(out)])
    grid_size_ablation.main()
    res = json.loads(out.read_text())
    assert [(r["n_rho"], r["n_tau"]) for r in res["sizes"]] == [(3, 4), (6, 16)]
    for r in res["sizes"]:
        assert r["n_configs"] == r["n_rho"] * r["n_tau"] and r["n_after_prune"] <= r["n_configs"]
        if r.get("certify"):
            assert r["certify"]["splits"] == 20 and "mean_n_certified" in r["certify"]


def test_certify_runs_the_fine_grid_arm(tmp_path, monkeypatch):
    import certify
    from test_protocol import _write_parts
    p = _write_parts(tmp_path, n=300)
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "certify.py", "--train-small", p["train_small"], "--train-big", p["train_big"],
        "--calib-small", p["calib_small"], "--calib-big", p["calib_big"],
        "--test-small", p["test_small"], "--test-big", p["test_big"],
        "--families", "caac_staged_biggrid", "--lams", "5", "--coverages", "0.5", "--perms", "2",
        "--alpha", "0.2", "--json", str(out)])
    certify.main()
    res = json.loads(out.read_text())
    assert "5" in res["families"]["caac_staged_biggrid"]["by_lam"]
