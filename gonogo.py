"""PRE-REGISTERED go/no-go (CAAC_MASTER §8.4) -- CPU only, reads two pools.

Thresholds are FIXED and must not be changed after seeing numbers:
  GO  iff  [A] for AT LEAST ONE action, marginal benefit(hardest tercile)
               - marginal benefit(easiest tercile) >= 0.10 (10 points)
     AND  [B] total headroom >= 15% relative vs the best fixed baseline.

Operational definitions (frozen 2026-11-05, BEFORE any pool with the proxy
existed; written into CAAC_MASTER §8.4):
  * Difficulty proxy = mean token prob of the FIRST 32 generated tokens of the
    SMALL model's greedy (``conf_prefix32``). Terciles by this proxy over the
    questions; LOW confidence = HARD.
  * Actions (benefit measured per tercile, on the same questions):
      sample-more = maj@8 - maj@1 of the small model's samples (exact
                    estimators over the pool; maj@1 = mean single-sample acc).
                    Secondary (reported, not used for the verdict): maj@8 - greedy.
      escalate    = acc(big greedy) - acc(small greedy).
  * Headroom [B] = cost saving at iso-accuracy:
        1 - C_oracle(A*) / C_baseline(A*)
    A*        = accuracy of the most accurate FIXED baseline;
    C_baseline= cost of reaching A* on the lower convex hull of the fixed
                baselines {small greedy, SC@8 small, big greedy, greedy+SC@8,
                greedy+escalate} (randomised mixing allowed);
    C_oracle  = min cost of a perfect-foresight per-question router over the
                CAAC action set {answer now (small greedy), greedy+sample-8,
                greedy+escalate, stage-0 escalate (probe prefill + big greedy)}
                to reach A* (LP relaxation = per-question concave hull).
    Cost profile: FLOPs (2*params*tokens; ledger.py). "sample more after
    greedy" re-pays prefill (pessimistic, CAAC_MASTER §9). Tokens reported too.
  * Verdict on MATH-500 (main axis). GSM8K is printed as the saturated control.
  * 95% bootstrap CIs (2,000 resamples of questions) are printed for
    information; the verdict uses the point estimates as pre-registered.

Usage:
  python gonogo.py --small pools/math500_q1.5b.jsonl.gz --big pools/math500_q7b.jsonl.gz
  python gonogo.py ... --json out/gonogo_math500.json
  # testing only (NOT the pre-registered proxy; refuses to print a verdict):
  python gonogo.py ... --allow-proxy-fallback
"""

from __future__ import annotations

import argparse
import json
import random

from caac.pool.estimators import maj_at_k_exact
from caac.pool.ledger import Call, cost_profiles
from caac.pool.schema import GOLD, load_pool

THR_TERCILE = 0.10
THR_HEADROOM = 0.15
K = 8


# --------------------------------------------------------------------------
# per-question table
# --------------------------------------------------------------------------

def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def build_rows(small, big, k=K, profile="flops", allow_fallback=False):
    qids = sorted(set(small) & set(big))
    rows, used_fallback = [], False
    for q in qids:
        s, b = small[q], big[q]
        proxy = s.greedy.conf_prefix32
        if proxy is None:
            if not allow_fallback:
                raise SystemExit(
                    f"{q}: no conf_prefix32 (pool built before 2026-11). The pre-registered "
                    "proxy is unavailable; rebuild the pool. (--allow-proxy-fallback = testing only)")
            proxy, used_fallback = s.greedy.confidence, True
        if s.n_samples < k:
            raise SystemExit(f"{q}: small model has {s.n_samples} samples < k={k}")
        cand = s.canonical_sample_answers()
        p1 = maj_at_k_exact(cand, GOLD, 1)
        p8 = maj_at_k_exact(cand, GOLD, k)
        mean_dec = _mean(c.decode_tokens for c in s.samples)
        g15 = Call(s.model, s.prompt_tokens, [s.greedy.decode_tokens], 0)
        samp = Call(s.model, s.prompt_tokens, [mean_dec] * k, 1)   # expected k-subset cost
        g7 = Call(b.model, b.prompt_tokens, [b.greedy.decode_tokens], 1)

        def c(calls):
            return cost_profiles(calls)[profile]

        rows.append({
            "qid": q, "proxy": proxy,
            "g15": float(s.greedy.correct), "g7": float(b.greedy.correct), "p1": p1, "p8": p8,
            # CAAC action set (greedy first)
            "A0": (c([g15]), float(s.greedy.correct)),
            "A1": (c([g15, samp]), p8),
            "A2": (c([g15, g7]), float(b.greedy.correct)),
            # stage-0 escalate: question-only probe (small prefill, no decode) + big greedy
            "E0": (c([Call(s.model, s.prompt_tokens, [], 0),
                      Call(b.model, b.prompt_tokens, [b.greedy.decode_tokens], 1)]),
                   float(b.greedy.correct)),
            # fixed baselines that do not start with small greedy
            "SC8": (c([Call(s.model, s.prompt_tokens, [mean_dec] * k, 0)]), p8),
            "G7": (c([Call(b.model, b.prompt_tokens, [b.greedy.decode_tokens], 0)]),
                   float(b.greedy.correct)),
        })
    return rows, used_fallback


# --------------------------------------------------------------------------
# [A] terciles
# --------------------------------------------------------------------------

def terciles(rows):
    """Split by proxy ascending: [hard, mid, easy] (low confidence = hard)."""
    srt = sorted(rows, key=lambda r: (r["proxy"], r["qid"]))
    n = len(srt)
    a, b = n // 3, (2 * n) // 3
    return srt[:a], srt[a:b], srt[b:]


def tercile_table(rows):
    out = []
    for name, part in zip(("hard", "mid", "easy"), terciles(rows)):
        out.append({
            "tercile": name, "n": len(part),
            "proxy_range": (min(r["proxy"] for r in part), max(r["proxy"] for r in part)),
            "acc_g15": _mean(r["g15"] for r in part),
            "sample_more": _mean(r["p8"] - r["p1"] for r in part),
            "sample_more_vs_greedy": _mean(r["p8"] - r["g15"] for r in part),
            "escalate": _mean(r["g7"] - r["g15"] for r in part),
        })
    return out


def criterion_a(table):
    hard, easy = table[0], table[-1]
    return {
        "sample_more_diff": hard["sample_more"] - easy["sample_more"],
        "escalate_diff": hard["escalate"] - easy["escalate"],
        "secondary_sample_more_vs_greedy_diff":
            hard["sample_more_vs_greedy"] - easy["sample_more_vs_greedy"],
    }


# --------------------------------------------------------------------------
# [B] headroom
# --------------------------------------------------------------------------

def _lower_hull(points):
    """Lower-right efficient hull of (cost, acc): min cost for each acc,
    convex (mixing two policies is allowed)."""
    pts = sorted(set(points))
    # keep only Pareto points (higher acc must cost more)
    pareto = []
    for c, a in pts:
        if not pareto or a > pareto[-1][1]:
            pareto.append((c, a))
    hull = []
    for p in pareto:  # convex hull, upper-left in acc-vs-cost => concave acc(cost)
        while len(hull) >= 2:
            (c1, a1), (c2, a2) = hull[-2], hull[-1]
            # remove hull[-1] if it lies below the chord hull[-2] -> p
            if (a2 - a1) * (p[0] - c1) <= (p[1] - a1) * (c2 - c1):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def cost_at_acc(hull, target):
    """Interpolated min cost to reach mean accuracy `target` on a hull."""
    if target <= hull[0][1]:
        return hull[0][0]
    for (c1, a1), (c2, a2) in zip(hull, hull[1:]):
        if a1 <= target <= a2:
            t = 0.0 if a2 == a1 else (target - a1) / (a2 - a1)
            return c1 + t * (c2 - c1)
    return float("inf")


def baseline_hull(rows):
    names = ("A0", "A1", "A2", "SC8", "G7")
    pts = {nm: (_mean(r[nm][0] for r in rows), _mean(r[nm][1] for r in rows)) for nm in names}
    return pts, _lower_hull(list(pts.values()))


ORACLE_ACTIONS = ("A0", "A1", "A2", "E0")


def oracle_cost_at_acc(rows, target):
    """Perfect-foresight router over ORACLE_ACTIONS, LP relaxation: start every
    question at its CHEAPEST action, then buy the steepest accuracy-per-cost
    upgrades (per-question concave hull) until mean accuracy reaches `target`."""
    n = len(rows)
    cost = acc = 0.0
    incs = []
    for r in rows:
        hull = _lower_hull([r[a] for a in ORACLE_ACTIONS])
        cost += hull[0][0]
        acc += hull[0][1]
        for (c1, a1), (c2, a2) in zip(hull, hull[1:]):
            incs.append(((a2 - a1) / max(c2 - c1, 1e-12), c2 - c1, a2 - a1))
    need = target * n - acc
    if need <= 0:
        return cost / n
    for slope, dc, da in sorted(incs, key=lambda x: -x[0]):
        if da >= need:
            return (cost + dc * need / da) / n
        cost += dc
        need -= da
        acc += da
    return float("inf")  # target unreachable by the oracle


def headroom(rows):
    pts, hull = baseline_hull(rows)
    a_star = max(a for _, a in pts.values())
    c_base = cost_at_acc(hull, a_star)
    c_or = oracle_cost_at_acc(rows, a_star)
    return {"A_star": a_star, "C_baseline": c_base, "C_oracle": c_or,
            "headroom": (1 - c_or / c_base) if c_base and c_or != float("inf") else None,
            "baselines": {k: {"cost": v[0], "acc": v[1]} for k, v in pts.items()}}


# --------------------------------------------------------------------------
# verdict + bootstrap
# --------------------------------------------------------------------------

def evaluate(rows):
    tab = tercile_table(rows)
    ca = criterion_a(tab)
    hb = headroom(rows)
    pass_a = max(ca["sample_more_diff"], ca["escalate_diff"]) >= THR_TERCILE
    pass_b = hb["headroom"] is not None and hb["headroom"] >= THR_HEADROOM
    return {"terciles": tab, "criterion_a": ca, "criterion_b": hb,
            "pass_a": pass_a, "pass_b": pass_b, "GO": pass_a and pass_b}


def bootstrap(rows, n_boot=2000, seed=0):
    rng = random.Random(seed)
    keys = ("sample_more_diff", "escalate_diff")
    acc = {k: [] for k in keys + ("headroom",)}
    n = len(rows)
    for _ in range(n_boot):
        rs = [rows[rng.randrange(n)] for _ in range(n)]
        ca = criterion_a(tercile_table(rs))
        for k in keys:
            acc[k].append(ca[k])
        h = headroom(rs)["headroom"]
        if h is not None:
            acc["headroom"].append(h)
    ci = {}
    for k, v in acc.items():
        v = sorted(v)
        ci[k] = (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (None, None)
    return ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--big", required=True)
    ap.add_argument("--profile", default="flops", choices=["flops", "tokens"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-proxy-fallback", action="store_true")
    a = ap.parse_args()

    small, big = load_pool(a.small), load_pool(a.big)
    rows, fb = build_rows(small, big, profile=a.profile, allow_fallback=a.allow_proxy_fallback)
    res = evaluate(rows)
    res["ci95"] = bootstrap(rows, a.n_boot) if a.n_boot else {}
    res["tokens_headroom"] = headroom(build_rows(small, big, profile="tokens",
                                                 allow_fallback=a.allow_proxy_fallback)[0])["headroom"]
    res.update({"n": len(rows), "profile": a.profile, "proxy_fallback": fb,
                "small": a.small, "big": a.big})

    print(f"\n=== PRE-REGISTERED GO/NO-GO  ({a.small} vs {a.big}; n={len(rows)}; cost={a.profile}) ===")
    if fb:
        print("  !!! PROXY FALLBACK (full-trace confidence) -- NOT the pre-registered proxy. TEST ONLY.")
    print(f"  {'tercile':7} {'n':>5} {'proxy range':>17} {'acc g15':>8} {'sample+':>8} {'(vs g)':>8} {'escal.':>8}")
    for t in res["terciles"]:
        lo, hi = t["proxy_range"]
        print(f"  {t['tercile']:7} {t['n']:5d} {lo:8.3f}-{hi:<8.3f} {t['acc_g15']:8.3f} "
              f"{t['sample_more']:+8.3f} {t['sample_more_vs_greedy']:+8.3f} {t['escalate']:+8.3f}")
    ca, ci = res["criterion_a"], res["ci95"]
    for k in ("sample_more_diff", "escalate_diff"):
        lo, hi = ci.get(k, (None, None))
        cis = f"  95%CI [{lo:+.3f}, {hi:+.3f}]" if lo is not None else ""
        print(f"  [A] {k:18} hard-easy = {ca[k]:+.3f}  (need >= {THR_TERCILE:.2f}){cis}")
    hb = res["criterion_b"]
    print("  [B] fixed baselines (cost, acc):",
          ", ".join(f"{k}=({v['cost']:.3g}, {v['acc']:.3f})" for k, v in hb["baselines"].items()))
    lo, hi = ci.get("headroom", (None, None))
    hs = f"{hb['headroom']:.3f}" if hb["headroom"] is not None else "n/a"
    cis = f"  95%CI [{lo:.3f}, {hi:.3f}]" if lo is not None else ""
    print(f"  [B] A*={hb['A_star']:.3f}  C_base={hb['C_baseline']:.4g}  C_oracle={hb['C_oracle']:.4g}"
          f"  headroom={hs} (need >= {THR_HEADROOM:.2f}){cis}   [tokens profile: "
          f"{res['tokens_headroom'] if res['tokens_headroom'] is None else round(res['tokens_headroom'], 3)}]")
    if fb:
        print("  VERDICT: (withheld -- proxy fallback)")
    else:
        print(f"  VERDICT: A={'PASS' if res['pass_a'] else 'fail'}  B={'PASS' if res['pass_b'] else 'fail'}"
              f"  ==> {'GO' if res['GO'] else 'NO-GO'}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  -> {a.json}")


if __name__ == "__main__":
    main()
