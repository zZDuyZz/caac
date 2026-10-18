"""Go/no-go readout for the A1 pilot — CPU only, reuses the repo's own pool code.

Run from repo root (PYTHONPATH=src):
  python check_pilot.py --math15 ... --math7 ... --gsm15 ... --gsm7 ...

It does NOT invent thresholds. It prints:
  (1) accuracy vs the paper anchors (2607.28576) — a PROMPT/GRADER sanity check;
      if pilot is far off, the pool is mis-measured, fix before spending more GPU.
  (2) the Chow-bound story (PLAN_V2 section 8): pi_pop of serve/abstain-only vs
      after escalation, with the certified-coverage caps. This is what tells you
      answer-correcting actions have headroom.
The formal E0/E1 gates (oracle >=20% cost; ECE<0.10) come AFTER the controller
is trained on the A3 split — they are not computed from the pilot alone.
"""
from __future__ import annotations

import argparse

from caac.pool.schema import load_pool
from caac.pool.estimators import maj_at_k_exact, chow_coverage_bound

ALPHAS = (0.10, 0.15, 0.20)
# Anchors from 2607.28576 (llama.cpp Q8, not an exact match — for sanity only).
ANCHORS = {
    "gsm8k_1.5b_greedy": 0.701, "gsm8k_1.5b_maj16": 0.809,
    "math500_1.5b_greedy": 0.441, "math500_1.5b_maj16": 0.558,
    "math500_7b_greedy": 0.755,
}


def greedy_acc(pool):
    return sum(r.greedy.correct for r in pool.values()) / len(pool)


def trunc_rate(pool):
    g = sum(r.greedy.finish_reason == "length" for r in pool.values()) / len(pool)
    tot = sum(len(r.samples) for r in pool.values()) or 1
    s = sum(c.finish_reason == "length" for r in pool.values() for c in r.samples) / tot
    return g, s


def maj_acc(pool, k):
    k_used = min([k] + [r.n_samples for r in pool.values()])  # the k actually voted
    vals = [maj_at_k_exact(r.sample_answers(), r.gold, k_used) for r in pool.values()
            if r.n_samples >= 1]
    return (sum(vals) / len(vals)) if vals else float("nan"), k_used


def line(name, val, anchor=None):
    s = f"  {name:28} {val:6.3f}"
    if anchor is not None:
        d = val - anchor
        flag = "OK" if abs(d) <= 0.07 else "CHECK prompt/grader"
        s += f"   anchor {anchor:.3f}  Δ{d:+.3f}  [{flag}]"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--math15", required=True); p.add_argument("--math7", required=True)
    p.add_argument("--gsm15", required=True);  p.add_argument("--gsm7", required=True)
    a = p.parse_args()

    m15, m7 = load_pool(a.math15), load_pool(a.math7)
    g15, g7 = load_pool(a.gsm15), load_pool(a.gsm7)

    print("\n================ (1) ACCURACY vs ANCHORS (sanity) ================")
    print(f"  [sizes] MATH-500 1.5b={len(m15)} 7b={len(m7)} | GSM8K 1.5b={len(g15)} 7b={len(g7)}")
    gm15 = greedy_acc(m15); gm7 = greedy_acc(m7); gg15 = greedy_acc(g15); gg7 = greedy_acc(g7)
    mm, mmk = maj_acc(m15, 16); gmaj, gmajk = maj_acc(g15, 16)
    print(line("MATH-500 1.5B greedy", gm15, ANCHORS["math500_1.5b_greedy"]))
    print(line(f"MATH-500 1.5B maj@{mmk}", mm, ANCHORS["math500_1.5b_maj16"]))
    print(line("MATH-500 7B greedy", gm7, ANCHORS["math500_7b_greedy"]))
    print(line("GSM8K 1.5B greedy", gg15, ANCHORS["gsm8k_1.5b_greedy"]))
    print(line(f"GSM8K 1.5B maj@{gmajk}", gmaj, ANCHORS["gsm8k_1.5b_maj16"]))
    print(line("GSM8K 7B greedy", gg7))

    tg, ts = trunc_rate(m15)
    print(f"\n  MATH-500 1.5B truncated: greedy {tg:.1%}, samples {ts:.1%}"
          f"   {'[>15% -> cân nhắc max-tokens 1536]' if max(tg,ts) > 0.15 else '[ok]'}")

    print("\n================ (2) CHOW BOUND (headroom cho hành động sửa) ================")
    print("  pi_pop = tỉ lệ câu trả lời ỨNG VIÊN đúng. serve/abstain-only bị chặn ở")
    print("  min{1, pi_pop/(1-alpha)}. Escalate sang 7B nâng pi_pop -> nâng trần.\n")
    for tag, pi_small, pi_big in [("MATH-500", gm15, gm7), ("GSM8K", gg15, gg7)]:
        print(f"  [{tag}] pi_pop 1.5B-greedy={pi_small:.3f} | 7B-greedy={pi_big:.3f}")
        for al in ALPHAS:
            cs = chow_coverage_bound(pi_small, al); cb = chow_coverage_bound(pi_big, al)
            print(f"      alpha={al:.2f}:  serve/abstain cap={cs:.3f}   sau escalate cap={cb:.3f}"
                  f"   (+{cb-cs:.3f})")
    print("\n  Đọc: trần serve/abstain càng THẤP so với escalate -> câu chuyện paper càng")
    print("  mạnh (hành động sửa câu trả lời là điều kiện cần, không phải tối ưu thêm).")
    print("\n  LƯU Ý: đây là readout để quyết định có chạy tiếp A2 không. Gate E0/E1")
    print("  chính thức tính sau, khi controller đã huấn luyện trên split A3.\n")


if __name__ == "__main__":
    main()
