import pytest

from caac.pool.schema import Completion, PoolRecord
import gonogo as G

S, B = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"


def _c(ok, dec=100, conf32=None):
    return Completion(answer="x" if ok else "w", correct=ok, decode_tokens=dec,
                      finish_reason="stop", conf_prefix32=conf32, confidence=0.9)


def _pools(n=90):
    """Easy third: everyone right. Mid: greedy wrong, samples 6/8 right, 7B right.
    Hard third: small always wrong, 7B right half the time."""
    small, big = {}, {}
    for i in range(n):
        q = f"q{i}"
        band = i % 3                      # 0 hard, 1 mid, 2 easy
        conf = (0.3, 0.6, 0.9)[band] + i * 1e-4
        if band == 2:
            g, samp, g7 = True, [True] * 8, True
        elif band == 1:
            g, samp, g7 = False, [True] * 6 + [False] * 2, True
        else:
            g, samp, g7 = False, [False] * 8, (i % 2 == 0)
        small[q] = PoolRecord(q, "d", S, "x", 50, _c(g, conf32=conf), [_c(s) for s in samp])
        big[q] = PoolRecord(q, "d", B, "x", 50, _c(g7), [])
    return small, big


def test_terciles_and_criteria():
    small, big = _pools()
    rows, fb = G.build_rows(small, big)
    assert not fb and len(rows) == 90
    res = G.evaluate(rows)
    hard, mid, easy = res["terciles"]
    assert hard["n"] == mid["n"] == easy["n"] == 30
    assert easy["escalate"] == pytest.approx(0.0)
    assert hard["escalate"] == pytest.approx(0.5, abs=0.04)
    assert res["criterion_a"]["escalate_diff"] >= 0.10 and res["pass_a"]
    # mid tercile: maj@8 with 6/8 correct = 1.0, maj@1 = 0.75
    assert mid["sample_more"] == pytest.approx(0.25)


def test_headroom_oracle_beats_fixed():
    small, big = _pools()
    rows, _ = G.build_rows(small, big)
    hb = G.headroom(rows)
    # oracle: easy -> A0, mid -> A1 (cheaper than 7B in FLOPs), hard -> A2 only where it helps
    assert hb["C_oracle"] < hb["C_baseline"]
    assert hb["headroom"] > 0.15


def test_refuses_without_preregistered_proxy():
    small, big = _pools(9)
    for r in small.values():
        r.greedy.conf_prefix32 = None
    with pytest.raises(SystemExit):
        G.build_rows(small, big)
    rows, fb = G.build_rows(small, big, allow_fallback=True)
    assert fb


def test_hull_and_interpolation():
    hull = G._lower_hull([(1, 0.5), (2, 0.55), (3, 0.9), (2.5, 0.6)])
    assert hull == [(1, 0.5), (3, 0.9)]           # (2,.55),(2.5,.6) are under the chord
    assert G.cost_at_acc(hull, 0.7) == pytest.approx(2.0)
    assert G.cost_at_acc(hull, 0.95) == float("inf")
