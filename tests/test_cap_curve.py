import gzip
import json
import subprocess
import sys

from cap_curve import GRID, round_up, smallest_cap, trunc_curve


def test_trunc_curve_counts_longer_than_cap():
    c = trunc_curve([100, 600, 1100, 4096], caps=(512, 1024, 4096))
    assert c == {512: 0.75, 1024: 0.5, 4096: 0.0}


def test_smallest_cap_and_none():
    curve = {512: 0.4, 1024: 0.06, 2048: 0.04}
    assert smallest_cap(curve, 0.05) == 2048
    assert smallest_cap(curve, 0.01) is None
    assert smallest_cap(curve, 0.5) == 512


def test_round_up():
    assert round_up(2049) == 2560 and round_up(2048) == 2048


def _pool(path, lengths):
    with gzip.open(path, "wt") as f:
        for i, L in enumerate(lengths):
            comp = {"decode_tokens": L, "finish_reason": "length" if L >= 4096 else "stop"}
            f.write(json.dumps({"query_id": f"q{i}", "prompt_tokens": 80,
                                "greedy": comp, "samples": [comp]}) + "\n")


def test_cli_recommends_cap_and_model_len(tmp_path):
    m, g = tmp_path / "math500_qs.jsonl.gz", tmp_path / "gsm8k_qs.jsonl.gz"
    _pool(m, [300] * 90 + [1500] * 10)          # 10% above 1024 -> needs 1536
    _pool(g, [200] * 100)                       # fits in 256
    out = subprocess.run([sys.executable, "cap_curve.py", str(m), str(g)],
                         capture_output=True, text=True, check=True).stdout
    assert "RECOMMEND: MT_M=1536 MT_G=256 " in out and "MML=" not in out, out


def test_cli_fails_when_largest_cap_still_truncates(tmp_path):
    m = tmp_path / "math500_qs.jsonl.gz"
    _pool(m, [4096] * 20 + [300] * 80)          # 20% cut even at 4096
    r = subprocess.run([sys.executable, "cap_curve.py", str(m)], capture_output=True, text=True)
    assert r.returncode == 2 and "WARNING" in r.stdout


def test_plateau_is_where_the_curve_flattens():
    from cap_curve import plateau_cap
    # 5% never terminate: no cap reaches 2%, the curve is flat from 1024 on
    assert plateau_cap({512: 0.40, 1024: 0.055, 2048: 0.05, 4096: 0.05}) == 1024
    assert plateau_cap({512: 0.40, 1024: 0.20, 2048: 0.08, 4096: 0.0}) == 4096


def test_mean_tokens_at_a_cap():
    from cap_curve import mean_tokens_at
    assert mean_tokens_at([100, 300, 5000], 1000) == (100 + 300 + 1000) / 3


def test_rescue_table_counts_long_answers_as_wrong():
    from cap_curve import rescue_table
    small = {"a": (True, 100), "b": (False, 100), "c": (True, 3000), "d": (False, 50)}
    big = {"a": (True, 100), "b": (True, 100), "c": (True, 200), "d": (False, 50)}
    t = rescue_table(small, big, cap=1024)
    assert t["n"] == 4 and t["acc_small"] == 0.25 and t["acc_big"] == 0.75
    assert t["small_wrong_big_right"] == 0.5 and t["both_wrong"] == 0.25


def _graded_pool(path, rows, tps=None):
    with gzip.open(path, "wt") as f:
        for i, (L, ok) in enumerate(rows):
            comp = {"decode_tokens": L, "finish_reason": "length" if L >= 4096 else "stop", "correct": ok}
            f.write(json.dumps({"query_id": f"q{i}", "prompt_tokens": 80, "greedy": comp, "samples": [comp]}) + "\n")
    if tps:
        with open(str(path) + ".manifest.json", "w") as f:
            json.dump({"throughput": {"tok_per_s": tps}}, f)


def test_cli_prints_forecast_inputs_and_early_diagnostics(tmp_path):
    s, b = tmp_path / "math_train_qs.jsonl.gz", tmp_path / "math_train_qb.jsonl.gz"
    _graded_pool(s, [(300, True)] * 60 + [(300, False)] * 40, tps=6000.0)
    _graded_pool(b, [(300, True)] * 90 + [(300, False)] * 10)
    out = subprocess.run([sys.executable, "cap_curve.py", str(s), str(b)],
                         capture_output=True, text=True, check=True).stdout
    assert "6000 tok/s" in out and "speed not recorded" in out
    assert "small acc 0.600  big acc 0.900" in out and "small wrong & big right 0.300" in out


def test_cli_names_the_plateau_when_no_cap_reaches_the_target(tmp_path):
    m = tmp_path / "math500_qs.jsonl.gz"
    _pool(m, [4096] * 20 + [300] * 80)
    r = subprocess.run([sys.executable, "cap_curve.py", str(m)], capture_output=True, text=True)
    assert r.returncode == 2 and "PLATEAU: MT_M=384" in r.stdout and "RECOMMEND" not in r.stdout


def test_a_pool_cut_below_the_grid_top_says_nothing_about_larger_caps(tmp_path):
    """Completions cut at 1024 need more than 1024 by an unknown amount: caps above it must
    not be reported as free of truncation."""
    m = tmp_path / "math500_qs.jsonl.gz"
    with gzip.open(m, "wt") as f:
        for i in range(100):
            cut = i < 50
            comp = {"decode_tokens": 1024 if cut else 300, "finish_reason": "length" if cut else "stop"}
            f.write(json.dumps({"query_id": f"q{i}", "prompt_tokens": 80, "greedy": comp, "samples": []}) + "\n")
    r = subprocess.run([sys.executable, "cap_curve.py", str(m)], capture_output=True, text=True)
    assert r.returncode == 2 and "NONE up to 1024" in r.stdout and "1536" not in r.stdout
