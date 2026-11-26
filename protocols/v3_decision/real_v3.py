"""Run the CANDIDATE design (chosen by simulation, see DECISION_before_real.md) ONCE on the real
public pools, next to v2, same configurations, same lock thresholds (alpha, delta, s, Pi, pairs, measures).

Candidate ("v3 candidate"):
  GO     no split; for every pi in Pi on ALL questions: exact binomial 'R > alpha' and betting
         'saving < s', each at delta/|Pi| (Bonferroni over Pi; intersection-union per pi).
         GO if some pi passes; report the cheapest passing pi.   [U1]
  range  d = (1-s)B - C bounded by the STRUCTURAL range [-(a_max + s b_max), (1-s) b_max]
         (an escalated question costs its small part plus exactly its big-only cost).   [U2]
  NO-GO  unchanged rule (every pi rejected by either leg at delta/2), with the structural range.
  gap    questions needed for GO at power .8, both legs, same tests (planning; never a verdict).
Usage: python real_v3.py POOLS LOCK OUT.jsonl [only_pools_csv]"""
import json, sys, os, math, time
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pchk6", "src"))
from caac.pool_v1.crossmap import (load_lock, load_pool_v1, included, pairs_for, common_questions,
                                   policy_grid, measures_for, max_unit_price)
from caac.pool_v1.config import config_from_dict
from caac.pool_v1.replay import replay
from caac.pool_v1.audit import (audit, _betting_p, _order, p_risk_below, p_risk_above)
from gap import gap


def parts(sub, meta, cfg, ms, inp, rr):
    """(a_max, b_max) with a_max + b_max = c_max; checked against every observed cost."""
    big = cfg.big
    if ms["cost"]["measure"] == "usd":
        caps = meta.caps
        pb = max_unit_price(sub, big)
        in_b = max((r.in_tokens or 0) for r in sub if r.model == big)
        cap_b = max(caps[big], max((r.out_tokens or 0) for r in sub if r.model == big))
        b_max = (cap_b + in_b) * pb
    else:
        c = cfg.cost
        in_b = max((r.in_tokens or 0) for r in sub if r.model == big and r.qid in set(inp.qids))
        b_max = meta.caps[big] * c.price_out_big + in_b * c.price_in_big
    a_max = inp.c_max - b_max
    assert a_max > 0 and b_max > 0, (a_max, b_max)
    assert max(inp.big_cost) <= b_max * (1 + 1e-9), "big cost above its structural bound"
    for p in inp.policies:
        sm = rr.serve_small[p.name]          # True = the small model's answer is served (no escalation)
        small = [p.cost[i] - (0.0 if sm[i] else inp.big_cost[i]) for i in range(len(inp.qids))]
        assert min(small) >= -1e-9 and max(small) <= a_max * (1 + 1e-9), "small part outside its bound"
    return a_max, b_max


def v3(inp, a_max, b_max, alpha, delta, s, salt):
    n = len(inp.qids); m = len(inp.policies)
    lo, hi = -(a_max + s * b_max), (1 - s) * b_max
    order = _order(inp.qids, salt)
    rows, go_best, all_rej = [], None, True
    for p in inp.policies:
        sv, e = sum(p.served), sum(p.error)
        d = [(1 - s) * inp.big_cost[i] - p.cost[i] for i in range(n)]
        assert min(d) >= lo - 1e-9 * abs(lo) and max(d) <= hi + 1e-9 * abs(hi)
        pr_go = p_risk_below(e, sv, alpha)
        ps_go = _betting_p(d, lo, hi, order) if (sv and e / sv <= alpha) else 1.0
        pr_ng = p_risk_above(e, sv, alpha)
        ps_ng = _betting_p([-v for v in d], -hi, -lo, order)
        passed = pr_go <= delta / m and ps_go <= delta / m
        rej = pr_ng <= delta / 2 or ps_ng <= delta / 2
        all_rej &= rej
        mc = sum(p.cost) / n
        rows.append({"policy": p.name, "p_risk": pr_go, "p_saving": ps_go, "go_pass": passed, "nogo_rej": rej,
                     "risk": e / sv if sv else None, "saving": 1 - mc / (sum(inp.big_cost) / n), "coverage": sv / n})
        if passed and (go_best is None or mc < go_best[0]):
            go_best = (mc, p.name)
    if go_best and all_rej:
        verdict = "INCONCLUSIVE"
    elif go_best:
        verdict = "GO"
    elif all_rej:
        verdict = "NO-GO"
    else:
        verdict = "INCONCLUSIVE"
    # evidence gap (planning): min over candidates pointing the right way
    g_best = None
    if verdict == "INCONCLUSIVE":
        for p in inp.policies:
            sv, e = sum(p.served), sum(p.error)
            d = [(1 - s) * inp.big_cost[i] - p.cost[i] for i in range(n)]
            if not sv or e / sv >= alpha or sum(d) <= 0:
                continue
            g = gap(e, sv, n, d, lo, alpha, delta / m, 0.8, split_factor=1)
            if g["reachable"] and (g_best is None or g["questions_needed"] < g_best["questions_needed"]):
                g_best = dict(g, policy=p.name)
    return {"verdict": verdict, "chosen": go_best[1] if go_best else None, "policies": rows,
            "gap80": g_best, "range": [lo, hi], "c_max_over_mean_big": inp.c_max / (sum(inp.big_cost) / n),
            "struct_lo_over_mean_big": -lo / (sum(inp.big_cost) / n)}


if __name__ == "__main__":
    pools, lockp, outp = sys.argv[1:4]
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    L = load_lock(lockp); a = L["audit"]
    out = open(outp, "w"); t0 = time.time()
    for dpath in sorted(p for p in Path(pools).iterdir() if (p / "meta.json").exists()):
        if only and dpath.name not in only:
            continue
        rows, rerr, meta, merr = load_pool_v1(dpath)
        if rerr or merr or meta is None:
            continue
        inc, why = included(dpath.name, L)
        if not inc:
            continue
        for pair in pairs_for(dpath.name, rows, L):
            if pair["tier"] not in ("primary", "secondary"):
                continue
            q = common_questions(rows, pair["small"], pair["big"])
            sub = [r for r in rows if r.model in (pair["small"], pair["big"]) and r.qid in q]
            grid = policy_grid(sub, meta, pair, L)
            for ms in measures_for(dpath.name, sub, meta, pair, L):
                cfg = config_from_dict({"small": pair["small"], "big": pair["big"], "alpha": a["alpha"], "delta": a["delta"],
                                        "s": a["s"], "salt": a["salt"], "cost_test": a["cost_test"], "cost": ms["cost"],
                                        "policies": grid})
                try:
                    rr = replay(sub, meta, cfg)
                except ValueError as e:
                    continue
                inp = rr.audit_input
                v2 = audit(inp, alpha=cfg.alpha, delta=cfg.delta, s=cfg.s, salt=cfg.salt, cost_test=cfg.cost_test)
                am, bm = parts(sub, meta, cfg, ms, inp, rr)
                r3 = v3(inp, am, bm, cfg.alpha, cfg.delta, cfg.s, cfg.salt)
                rec = {"pool": dpath.name, "small": pair["small"], "big": pair["big"], "tier": pair["tier"],
                       "measure": ms["name"], "n": len(inp.qids), "n_policies": len(inp.policies),
                       "v2": v2.verdict, "v2_chosen": v2.chosen, "v3": r3["verdict"], "v3_chosen": r3["chosen"],
                       "gap80": r3["gap80"], "c_max_over_mean_big": r3["c_max_over_mean_big"],
                       "struct_lo_over_mean_big": r3["struct_lo_over_mean_big"], "policies": r3["policies"]}
                out.write(json.dumps(rec) + "\n"); out.flush()
        print(dpath.name, "done", round(time.time() - t0), "s", flush=True)
    out.close()
