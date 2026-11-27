"""Certified saving (U-a): a lower bound on the saving of a policy that also passes the claim's risk leg.
SUPPLEMENTARY: it never changes a verdict. Design and criteria: protocols/v3_decision/ua/DECISION_ua_before_run.md.

For each candidate policy, at lvl = delta/|Pi| (the GO level of each leg):
  risk_pass  the GO risk leg of audit3 (relative: betting on eps - D; absolute: exact binomial vs alpha),
             with audit3's gate (mean D <= eps, resp. observed risk <= alpha);
  LCB        fixed sequence over S_GRID ascending: "saving < s" rejected by betting on d_s = (1-s)B - C in
             the structural range, at lvl; LCB = last s rejected before the first non-rejection.
S* = max LCB over policies with risk_pass. Statement "pi* meets the risk claim AND saves >= S*" is wrong
with probability <= delta (union over policies of {risk_pass and (risk claim false or LCB > saving)},
each <= lvl: a false risk claim passes with prob <= lvl; otherwise LCB > saving rejects the first TRUE
nested null of the sequence, prob <= lvl)."""
from __future__ import annotations

from caac.pool_v1.audit import (AuditInput, BET_CAP, REL_MODE, REL_MODES, REL_TEST, REL_TESTS, S_GRID, _betting_fixed_n_p,
                                _betting_p, _betting_p_cap, _check3, _check_relative, _order, p_risk_below,
                                policy_lo, rel_quality_go_p, rel_risk_p, structural_range, test_order)
from caac.pool_v1.audit import DEFAULT_SALT


def saving_lcb(inp: AuditInput, j: int, lvl: float, order: list[int], mode: str | None = None) -> float | None:
    """mode None: v3-8 cost test (global structural range, bet cap 0.5; absolute claim and legacy relative).
    mode 'fixed_n' / 'anytime': the GO cost test of that mode (per-policy range, v3-10 / v3-11)."""
    p, n = inp.policies[j], len(inp.qids)
    lcb = None
    for s in S_GRID:
        lo, hi = structural_range(inp.small_max, inp.big_max, s)
        d = [(1.0 - s) * inp.big_cost[i] - p.cost[i] for i in range(n)]
        if mode is None:
            ps = _betting_p(d, lo, hi, order)
        elif mode == "fixed_n":
            ps = _betting_fixed_n_p(d, policy_lo(inp, p, s), hi, lvl, order)
        else:
            ps = _betting_p_cap(d, policy_lo(inp, p, s), hi, order, BET_CAP)
        if ps <= lvl:
            lcb = s
        else:
            break
    return lcb


def certified_saving(inp: AuditInput, *, claim: str = "relative", alpha: float = 0.10, eps: float = 0.02,
                     delta: float = 0.10, salt: str = DEFAULT_SALT, rel_test: str | None = None,
                     rel_mode: str = REL_MODE) -> dict:
    _check3(inp)
    rel = claim == "relative"
    if rel:
        _check_relative(inp)
    elif claim != "absolute":
        raise ValueError("claim must be 'absolute' or 'relative'")
    n, m = len(inp.qids), len(inp.policies)
    lvl = delta / m
    legacy = rel_test in REL_TESTS
    mode = rel_mode if (rel and not legacy) else None
    if mode is not None and mode not in REL_MODES:
        raise ValueError(f"rel_mode must be one of {REL_MODES}")
    order = test_order(inp.qids, salt, mode == "anytime")
    mean_b = sum(inp.big_cost) / n
    per = []
    for j, p in enumerate(inp.policies):
        sv, e = sum(p.served), sum(p.error)
        if rel:
            D = [int(bool(p.error[i])) - int(bool(inp.big_error[i])) for i in range(n)]
            gate = sum(D) / n <= eps
            if legacy:
                pr = rel_risk_p(D, eps, order, rel_test)
            else:
                pr = rel_quality_go_p(D, eps, order, lvl, mode)
        else:
            pr = p_risk_below(e, sv, alpha)
            gate = bool(sv) and e / sv <= alpha
        rp = pr <= lvl and gate
        lcb = saving_lcb(inp, j, lvl, order, mode) if rp else None
        per.append({"policy": p.name, "risk_pass": rp, "p_risk_go": pr, "saving_lower": lcb,
                    "saving_hat": round(1 - (sum(p.cost) / n) / mean_b, 6) if mean_b else None})
    ok = [x for x in per if x["risk_pass"] and x["saving_lower"] is not None]
    best = max(ok, key=lambda x: (x["saving_lower"], -per.index(x))) if ok else None
    return {"claim": claim, "eps": eps if rel else None, **({"rel_mode": mode} if mode else {}), "alpha": None if rel else alpha, "delta": delta,
            "level_each_leg": lvl, "grid": [S_GRID[0], S_GRID[-1], 0.02],
            "certified_saving": best["saving_lower"] if best else None,
            "policy": best["policy"] if best else None,
            "saving_hat_of_policy": best["saving_hat"] if best else None,
            "n_risk_pass": sum(x["risk_pass"] for x in per), "policies": per,
            "note": "supplementary; never changes the verdict; S* < 0 means a certified cost INCREASE of at most -S*"}
