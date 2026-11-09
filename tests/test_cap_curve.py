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
