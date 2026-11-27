"""The CAAC report (R1-R5). Same input -> same bytes.

R1 verdict and its guarantee      R2 risk / saving on the tested half
R3 the verdict under every cost measure (measured vs estimated)
R4 break-even lambda and the certification cost (questions still needed)
R5 data quality (validator) and replay notes
"""

from __future__ import annotations

import copy
import json
import math

from caac.pool_v1.audit import audit, audit3, binom_cdf, certified_frontier, in_half_a, served_needed, structural_range
from caac.pool_v1.certsave import certified_saving
from caac.pool_v1.gap import gap as evidence_gap, gap_relative
from caac.pool_v1.action import action_from_report
from caac.pool_v1.config import Config
from caac.pool_v1.replay import replay
from caac.pool_v1.schema import Meta, Row
from caac.pool_v1.validate import certification_floor, validate

REPORT_VERSION = "1"


def _f(x, sig=6):
    """Round for a stable, readable file (None and non-finite kept explicit)."""
    if x is None:
        return None
    if isinstance(x, bool) or isinstance(x, int):
        return x
    if not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return float(f"{x:.{sig}g}")


def clopper_pearson(k: int, n: int, conf: float) -> tuple[float, float]:
    """Two-sided exact interval for a binomial proportion (bisection on the exact cdf)."""
    if n == 0:
        return 0.0, 1.0
    a = (1 - conf) / 2

    def solve(f, lo=0.0, hi=1.0):
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) >= a)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) <= a)
    return lower, upper


def _run_audit(inp, cfg: Config):
    if cfg.design == "v3":
        return audit3(inp, alpha=cfg.alpha, delta=cfg.delta, s=cfg.s, salt=cfg.salt,
                      claim=cfg.claim, eps=cfg.eps, rel_mode=cfg.rel_mode)
    return audit(inp, alpha=cfg.alpha, delta=cfg.delta, s=cfg.s, salt=cfg.salt, cost_test=cfg.cost_test)


def _guarantee(cfg: Config) -> str:
    if cfg.design == "v3" and cfg.claim == "relative":
        return (f"Each verdict is wrong with probability <= {cfg.delta}: GO means the chosen policy's error rate "
                f"exceeds the big model's alone (same questions, the big model's first answer) by at most "
                f"{cfg.eps} AND it saves >= {cfg.s:.0%} of the big-model-only cost, both tested on all questions at "
                f"level delta/|Pi| (Bonferroni over the candidate set); NO-GO means every candidate policy fails "
                f"one of the two claims. Assumes questions exchangeable with the deployment, costs bounded "
                f"(structural range), candidates fixed in advance, and the declared use ({cfg.rel_mode}: "
                + ("the result is read once, at this n" if cfg.rel_mode == "fixed_n" else
                   "betting only, valid at every look while questions arrive") + ") fixed before the data were read.")
    if cfg.design == "v3":
        return (f"Each verdict is wrong with probability <= {cfg.delta}: GO means the chosen policy has "
                f"risk (error rate among served answers) <= {cfg.alpha} AND saves >= {cfg.s:.0%} of the "
                f"big-model-only cost, both tested on all questions at level delta/|Pi| (Bonferroni over the "
                f"candidate set); NO-GO means every candidate policy fails one of the two claims. Assumes "
                f"questions exchangeable with the deployment, costs bounded (structural range), candidates "
                f"fixed in advance.")
    return (f"Each verdict is wrong with probability <= {cfg.delta}: GO means the chosen policy has "
            f"risk (error rate among served answers) <= {cfg.alpha} AND saves >= {cfg.s:.0%} of the "
            f"big-model-only cost; NO-GO means no candidate policy does both. Assumes questions "
            f"exchangeable with the deployment, costs bounded by c_max, candidates fixed in advance.")


def _measure_status(rows: list[Row], cfg: Config, measure: str, notes: list[str]) -> str:
    if measure == "tokens":
        return "measured tokens x configured prices"
    est = any(n.startswith(f"{measure}:") and "ESTIMATED" in n for n in notes)
    return f"measured per generation{' (early-stopped draws estimated)' if est else ''}"


def _with_measure(cfg: Config, measure: str, c_max, c_max_big=None):
    c = copy.deepcopy(cfg)
    c.cost.measure, c.cost.c_max = measure, c_max
    c.cost.c_max_big = c_max_big
    return c


def make_report(rows: list[Row], meta: Meta | None, cfg: Config, *, row_errors=(), meta_errors=(),
                input_info: dict | None = None) -> dict:
    rows = sorted(rows, key=lambda r: (r.qid, r.model, r.draw))
    val = validate(rows, meta, row_errors=row_errors, meta_errors=meta_errors,
                   alpha=cfg.alpha, delta=cfg.delta, pair=(cfg.small, cfg.big), design=cfg.design)
    rep = {"tool": {"name": "caac", "report_version": REPORT_VERSION},
           "input": input_info or {}, "config": cfg.to_dict()}
    issue_counts: dict[str, int] = {}
    for i in val.issues:
        issue_counts[i.code] = issue_counts.get(i.code, 0) + 1
    r5 = {"ok": val.ok, "level": {m: val.level.get(m) for m in (cfg.small, cfg.big)},
          "pair": val.pair, "issue_counts": dict(sorted(issue_counts.items())),
          "issues": [i.__dict__ for i in sorted(val.issues, key=lambda i: (i.severity, i.code, i.model, i.message))
                     if i.severity != "info"],
          "info": sorted({i.code for i in val.issues if i.severity == "info"})}
    if not val.ok:
        rep["R1_verdict"] = {"verdict": "NOT RUN", "reason": "the pool has errors (see R5)"}
        rep["R5_data_quality"] = r5
        if cfg.design == "v3":
            rep["R1_verdict"]["action"] = action_from_report(rep)
        return rep

    rr = replay(rows, meta, cfg)
    v3 = cfg.design == "v3"
    res = _run_audit(rr.audit_input, cfg)
    r5["replay_notes"] = list(rr.notes)
    mb = sum(rr.audit_input.big_cost) / len(rr.audit_input.big_cost)
    r5["cost_bound_ratio"] = _f(rr.audit_input.c_max / mb) if mb > 0 else "inf"
    if mb > 0 and rr.audit_input.c_max / mb > 20:
        r5["replay_notes"].append(
            f"cost bound c_max is {rr.audit_input.c_max / mb:.0f}x the mean big-model cost: the cost leg "
            "has little power at this sample size (a lower cap, or a per-question bound, would help)")
    if v3:
        rep["R1_verdict"] = {"verdict": res.verdict, "guarantee": _guarantee(cfg), "design": "v3",
                             "alpha": cfg.alpha, "delta": cfg.delta, "s": cfg.s, "salt": cfg.salt,
                             "cost_test": cfg.cost_test, "n_questions": res.n_questions,
                             "n_policies": len(rr.policies), "policies": [p.name for p in rr.policies],
                             "level_each_leg": _f(res.go["level_each_leg"]), "chosen": res.chosen,
                             "policies_passing": res.go["policies_passing"], "conflict": res.conflict}
        inp0 = rr.audit_input
        if inp0.big_error is not None and inp0.qids:
            rep["R1_verdict"]["big_only_error"] = _f(sum(bool(x) for x in inp0.big_error) / len(inp0.qids))
        if cfg.claim == "relative":
            rep["R1_verdict"].update({"claim": "relative", "eps": cfg.eps, "rel_test": res.go.get("rel_test"),
                                      "rel_mode": res.go.get("rel_mode")})
        cs = certified_saving(inp0, claim=cfg.claim or "absolute", alpha=cfg.alpha, eps=cfg.eps, delta=cfg.delta,
                              salt=cfg.salt, rel_mode=cfg.rel_mode)
        rep["R1_verdict"]["certified_saving"] = {
            "value": cs["certified_saving"], "policy": cs["policy"],
            "saving_hat_of_policy": _f(cs["saving_hat_of_policy"]), "n_risk_pass": cs["n_risk_pass"],
            "note": ("lower bound on the saving of a policy that also meets the risk claim, same guarantee "
                     "(wrong with probability <= delta); GO iff value >= s; supplementary, never changes the "
                     "verdict; a negative value bounds a cost increase")}
    else:
      rep["R1_verdict"] = {"verdict": res.verdict, "guarantee": _guarantee(cfg), "alpha": cfg.alpha,
                         "delta": cfg.delta, "s": cfg.s, "salt": cfg.salt, "cost_test": cfg.cost_test,
                         "n_questions": res.n_questions, "n_half_a": res.n_a, "n_half_b": res.n_b,
                         "n_policies": len(rr.policies), "policies": [p.name for p in rr.policies],
                         "chosen_on_half_a": res.chosen, "conflict": res.conflict}

    # ---- R2
    if v3:
        if res.chosen is not None:
            g = res.go
            lo, hi = clopper_pearson(g["errors"], g["served"], 1 - cfg.delta / len(rr.policies))
            rep["R2_estimates"] = {"policy": res.chosen, "tested": "all questions",
                                   "served": g["served"], "errors": g["errors"], "risk": _f(g["risk"]),
                                   "risk_interval": [_f(lo), _f(hi)],
                                   "interval_level": _f(1 - cfg.delta / len(rr.policies)),
                                   "coverage": _f(g["served"] / res.n_questions),
                                   "saving": _f(g["saving"]), "p_risk": _f(g["p_risk"]),
                                   "p_saving": _f(g["p_saving"])}
            if cfg.claim == "relative":
                rep["R2_estimates"]["delta_hat"] = _f(g.get("delta_hat"))
                rep["R2_estimates"]["big_only_error"] = _f(g.get("big_only_error"))
        else:
            rep["R2_estimates"] = {"policy": None, "closest_to_go": res.go.get("closest_policy")}
    elif res.chosen is not None:
        g = res.go
        lo, hi = clopper_pearson(g["errors_b"], g["served_b"], 1 - cfg.delta)
        rep["R2_estimates"] = {"policy": res.chosen, "tested_half": "B",
                               "served": g["served_b"], "errors": g["errors_b"], "risk": _f(g["risk_b"]),
                               "risk_interval": [_f(lo), _f(hi)], "interval_level": 1 - cfg.delta,
                               "coverage": _f(g["served_b"] / res.n_b) if res.n_b else None,
                               "saving": _f(g["saving_b"]), "p_risk": _f(g["p_risk"]),
                               "p_saving": _f(g["p_saving"])}
    else:
        rep["R2_estimates"] = {"policy": None, "reason": res.go.get("reason")}
    grp = _by_group(rows, cfg, rr, res, all_questions=v3)
    if grp is not None:
        rep["R2_estimates"]["by_group"] = grp
    allp = []
    for row, pd in zip(res.nogo["policies"], rr.audit_input.policies):
        x = {k: _f(v) if isinstance(v, float) else v for k, v in row.items()}
        if not v3:
            x["coverage"] = _f(sum(pd.served) / len(pd.served))     # share of questions answered
        allp.append(x)
    rep["R2_estimates"]["all_policies"] = allp
    if any(p.on_unsure == "abstain" for p in rr.policies):
        r5_note = ("ABSTAIN candidates present: saving can come from not answering; read 'coverage' "
                   "in R2 (the primary claim in the protocol lock uses ESCALATE only)")
    else:
        r5_note = None

    # ---- R3
    meas = [{"measure": cfg.cost.measure, "c_max": cfg.cost.c_max, "c_max_big": cfg.cost.c_max_big}] + list(cfg.cost.also)
    r3 = {}
    for mdef in meas:
        m = mdef["measure"]
        if m in r3:
            continue
        try:
            c2 = _with_measure(cfg, m, mdef.get("c_max"), mdef.get("c_max_big"))
            rr2 = replay(rows, meta, c2) if m != cfg.cost.measure else rr
            res2 = _run_audit(rr2.audit_input, cfg) if m != cfg.cost.measure else res
            r3[m] = {"verdict": res2.verdict, "chosen" if v3 else "chosen_on_half_a": res2.chosen,
                     "status": _measure_status(rows, cfg, m, rr2.notes), "c_max": _f(rr2.audit_input.c_max)}
        except ValueError as e:
            r3[m] = {"verdict": "NOT AVAILABLE", "reason": str(e)}
    verdicts = {v["verdict"] for v in r3.values() if v["verdict"] not in ("NOT AVAILABLE",)}
    rep["R3_cost_measures"] = {"by_measure": r3, "verdict_changes_with_measure": len(verdicts) > 1}

    # ---- R4
    r4 = {"lambda": _f(cfg.cost.lam)}
    name = res.chosen
    if name is not None:
        n = len(rr.qids)
        sbar = sum(rr.small_tokens[name]) / n
        ebar = sum(rr.big_tokens_escalated[name]) / n
        bbar = sum(rr.big_only_tokens) / n
        r4["mean_tokens"] = {"small_used": _f(sbar), "big_when_escalated": _f(ebar), "big_only": _f(bbar)}
        r4["break_even_lambda"] = _f(sbar / (bbar - ebar)) if bbar > ebar else "inf"
        need = (1 - cfg.s) * bbar - ebar
        r4["lambda_needed_for_saving_s"] = _f(sbar / need) if need > 0 else "inf"
        r4["note"] = ("on all audited questions, output tokens only (prompt tokens ignored); "
                      "small-first saves at a token price ratio lambda above break_even_lambda")
        if v3:
            r4["certification"] = _go_reached_v3(res, cfg, len(rr.policies))
    if name is not None and not v3:
        n = len(rr.qids)
        served, errors = res.go["served_b"], res.go["errors_b"]
        n_b = res.n_b
        rate = errors / served if served else None
        cert = {"risk_on_half_b": _f(rate), "served_share_half_b": _f(served / n_b) if n_b else None,
                "questions_available": n}
        sn = served_needed(rate, cfg.alpha, cfg.delta) if rate is not None else None
        if sn is None:
            cert["questions_needed"] = None
            cert["reachable"] = False
            cert["reason"] = ("risk on half B is >= alpha: more questions of the same kind will not "
                              "certify this policy")
        else:
            share = served / n_b if n_b else 0
            total = math.ceil(2 * sn / share) if share > 0 else None
            cert["served_needed_on_tested_half"] = sn
            cert["reachable"] = True
            cert["questions_needed"] = total
            cert["more_questions_needed"] = max(0, total - n) if total is not None else None
        cert["note"] = "ESTIMATE for the risk leg only, if the observed rates hold; the cost leg may need more"
        r4["certification"] = cert
    elif v3:
        if name is None:
            r4["reason"] = "no policy passed both legs on all questions"
            r4["certification"] = _evidence_gap(rr.audit_input, cfg)
    else:
        r4["reason"] = "no policy reached risk <= alpha on half A"
        # planning estimate (does not change the verdict): the candidate with the lowest risk on
        # ALL audited questions; if even its rate is >= alpha, more questions of the same kind
        # cannot give GO; otherwise the questions needed if that rate held (risk leg only)
        inp = rr.audit_input
        best = None
        for p in inp.policies:
            sv = sum(p.served)
            if sv:
                rate = sum(p.error) / sv
                if best is None or rate < best[0]:
                    best = (rate, p.name, sv)
        cert = {"basis": "lowest-risk candidate on all audited questions (no policy passed half A)",
                "questions_available": len(inp.qids)}
        if best is None:
            cert.update({"questions_needed": None, "reachable": False, "reason": "no candidate served any question"})
        else:
            rate, pname, sv = best
            cert.update({"policy": pname, "risk_all": _f(rate), "served_share": _f(sv / len(inp.qids))})
            sn = served_needed(rate, cfg.alpha, cfg.delta) if rate < cfg.alpha else None
            if sn is None:
                cert.update({"questions_needed": None, "reachable": False,
                             "reason": "the lowest observed risk is >= alpha: more questions of the same "
                                       "kind will not give GO with these candidates"})
            else:
                share = sv / len(inp.qids)
                total = math.ceil(2 * sn / share) if share > 0 else None
                cert.update({"served_needed_on_tested_half": sn, "questions_needed": total, "reachable": True,
                             "more_questions_needed": max(0, total - len(inp.qids)) if total else None})
        cert["note"] = "ESTIMATE for the risk leg only, if the observed rate holds; the cost leg may need more"
        r4["certification"] = cert
    if v3 and cfg.claim == "relative":
        m = len(rr.policies)
        floor = certification_floor(cfg.eps, cfg.delta / m) if cfg.eps > 0 else None
        r4["minimum_questions_zero_errors"] = floor
        r4["minimum_note"] = (f"even with zero harm (the policy never wrong where big-only is right), GO needs >= "
                              f"{floor} questions with the exact harm-count test (more with the betting test) at "
                              f"level delta/{m}" if floor else "eps = 0: GO needs the policy to be strictly better")
        r4["certified_frontier"] = None
    elif v3:
        m = len(rr.policies)
        floor = certification_floor(cfg.alpha, cfg.delta / m)
        r4["minimum_questions_zero_errors"] = floor
        r4["minimum_note"] = (f"even with zero errors, GO needs >= {floor} served answers on all questions "
                              f"(level delta/{m} per leg, Bonferroni over {m} candidates)")
        if (input_info or {}).get("sensitivity_cell"):
            r4["certified_frontier"] = None
        else:
            r4["certified_frontier"] = certified_frontier(rr.audit_input, delta=cfg.delta, salt=cfg.salt)
            r4["certified_frontier"]["note"] = ("supplementary: for each alpha, the largest saving s certified by some "
                                                "candidate, all bounds hold jointly at level 1-delta; the verdict above "
                                                "does not use it")
    else:
        floor = certification_floor(cfg.alpha, cfg.delta)
        r4["minimum_questions_zero_errors"] = 2 * floor
        r4["minimum_note"] = (f"even with zero errors, GO needs >= {floor} served answers on the tested half, "
                              f"i.e. about {2 * floor} questions if every question is served")
    rep["R4_break_even"] = r4
    if r5_note:
        r5["replay_notes"].append(r5_note)
    rep["R5_data_quality"] = r5
    if v3:
        rep["R1_verdict"]["action"] = action_from_report(rep)    # amendment v3-8: never changes the verdict
    return rep


def _evidence_gap(inp, cfg: Config) -> dict:
    """Design v3 planning output: questions still needed for BOTH legs (power 0.8, same tests), planned on
    the best-looking candidate (observed risk < alpha and mean d > 0). Never a verdict."""
    n, m = len(inp.qids), len(inp.policies)
    lo, _ = structural_range(inp.small_max, inp.big_max, cfg.s)
    if cfg.claim == "relative":
        return _evidence_gap_relative(inp, cfg, lo)
    best = None
    seen_any = False
    for p in inp.policies:
        sv, e = sum(p.served), sum(p.error)
        d = [(1 - cfg.s) * inp.big_cost[i] - p.cost[i] for i in range(n)]
        if not sv or e / sv >= cfg.alpha or sum(d) <= 0:
            continue
        seen_any = True
        g = evidence_gap(e, sv, n, d, lo, cfg.alpha, cfg.delta / m, 0.8)
        if g["reachable"] and (best is None or g["questions_needed"] < best["questions_needed"]):
            best = dict(g, policy=p.name)
    out = {"basis": "best-looking candidate (risk < alpha and mean d > 0), power 0.8, tests at level delta/|Pi|",
           "questions_available": n}
    if best is None:
        out.update(reachable=False, questions_needed=None,
                   reason=("no candidate has observed risk < alpha and a positive mean of d: more questions of "
                           "the same kind will not give GO" if not seen_any else
                           "no candidate with observed risk < alpha and mean d > 0 reaches power 0.8 within 200000 questions"))
    else:
        out.update(reachable=True, policy=best["policy"], questions_needed=best["questions_needed"],
                   more_questions_needed=max(0, best["questions_needed"] - n), blocking=best["blocking"],
                   m_risk_served=best["m_risk_served"], m_saving_questions=best["m_saving_questions"],
                   risk=_f(best["rate"]), coverage=_f(best["coverage"]), mean_d=_f(best["mean_d"]))
    out["note"] = "PLANNING estimate, plug-in from the observed data; it never changes the verdict"
    return out


def _evidence_gap_relative(inp, cfg: Config, lo: float) -> dict:
    """Relative claim: plan on the best-looking candidate (observed R_pi - R_big < eps and mean d > 0)."""
    from caac.pool_v1.audit import policy_lo
    n, m = len(inp.qids), len(inp.policies)
    best, seen_any = None, False
    for p in inp.policies:
        D = [int(bool(p.error[i])) - int(bool(inp.big_error[i])) for i in range(n)]
        d = [(1 - cfg.s) * inp.big_cost[i] - p.cost[i] for i in range(n)]
        if sum(D) / n >= cfg.eps or sum(d) <= 0:
            continue
        seen_any = True
        g = gap_relative(D, n, d, policy_lo(inp, p, cfg.s), cfg.eps, cfg.delta / m, 0.8, mode=cfg.rel_mode,
                         hi_d=(1.0 - cfg.s) * inp.big_max)
        if g["reachable"] and (best is None or g["questions_needed"] < best["questions_needed"]):
            best = dict(g, policy=p.name)
    out = {"basis": "best-looking candidate (R_pi - R_big < eps and mean d > 0), power 0.8, tests at level "
                    "delta/|Pi|", "questions_available": n, "claim": "relative", "eps": cfg.eps, "rel_mode": cfg.rel_mode}
    if best is None:
        out.update(reachable=False, questions_needed=None,
                   reason=("no candidate has an observed error within eps of big-only and a positive mean of d: "
                           "more questions of the same kind will not give GO" if not seen_any else
                           "no such candidate reaches power 0.8 within 200000 questions"))
    else:
        out.update(reachable=True, policy=best["policy"], questions_needed=best["questions_needed"],
                   more_questions_needed=max(0, best["questions_needed"] - n), blocking=best["blocking"],
                   m_risk_questions=best["m_risk_questions"], m_saving_questions=best["m_saving_questions"],
                   delta_hat=_f(best["delta_hat"]), harm_rate=_f(best["harm_rate"]), mean_d=_f(best["mean_d"]))
    out["note"] = "PLANNING estimate, plug-in from the observed data; it never changes the verdict"
    return out


def _go_reached_v3(res, cfg: Config, m: int) -> dict:
    """Design v3, verdict GO: both legs already passed on all questions; nothing more is needed."""
    g = res.go
    return {"basis": "GO reached: both legs passed on all questions at level delta/|Pi|",
            "policy": res.chosen, "questions_available": res.n_questions, "questions_needed": 0,
            "more_questions_needed": 0, "reachable": True, "risk": _f(g["risk"]),
            "coverage": _f(g["served"] / res.n_questions) if res.n_questions else None,
            "level_each_leg": _f(cfg.delta / m)}


def _by_group(rows: list[Row], cfg: Config, rr, res, all_questions: bool = False) -> dict | None:
    """Risk of the chosen policy per question group on the tested questions (half B in v2, all
    questions in v3). DESCRIPTIVE: the guarantee is for the whole pool; no multiplicity correction
    is applied to the groups."""
    if res.chosen is None:
        return None
    gmap = {r.qid: r.group for r in rows if r.model == cfg.small and r.draw == 0 and r.group}
    if not gmap:
        return None
    pd = next(p for p in rr.audit_input.policies if p.name == res.chosen)
    acc: dict[str, list[int]] = {}
    for q, sv, er in zip(rr.audit_input.qids, pd.served, pd.error):
        if not all_questions and in_half_a(q, cfg.salt):
            continue
        a = acc.setdefault(gmap.get(q, "(none)"), [0, 0, 0])
        a[0] += 1
        a[1] += 1 if sv else 0
        a[2] += 1 if er else 0
    out = {}
    for g in sorted(acc):
        n, sv, er = acc[g]
        row = {"questions": n, "served": sv, "errors": er}
        if sv:
            lo, hi = clopper_pearson(er, sv, 1 - cfg.delta)
            row.update({"risk": _f(er / sv), "risk_interval": [_f(lo), _f(hi)],
                        "lower_bound_above_alpha": lo > cfg.alpha})
        out[g] = row
    return {"policy": res.chosen, ("tested" if all_questions else "tested_half"): ("all questions" if all_questions else "B"),
            "interval_level": 1 - cfg.delta,
            "note": "descriptive only: the verdict covers the whole pool, not each group; "
                    "no multiplicity correction; small groups have wide intervals",
            "groups": out}


def dumps(rep: dict) -> str:
    return json.dumps(rep, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _certified_saving_line(c: dict, delta) -> str:
    if c.get("value") is None:
        return ("- certified saving: none (no candidate policy passes the risk leg at this sample size)")
    v = c["value"]
    what = (f"at least {v:.0%} of the big-model-only cost" if v > 0 else
            f"cost at most {-v:.0%} above big-only (no saving certified)" if v < 0 else "at least 0% (no loss)")
    return (f"- certified saving: {what} with policy {c['policy']} (estimate {c.get('saving_hat_of_policy')}); "
            f"wrong with probability <= {delta}")


def to_markdown(rep: dict) -> str:
    v = rep["R1_verdict"]
    L = ["# CAAC report", ""]
    if v["verdict"] == "NOT RUN":
        L += [f"**Verdict: NOT RUN** - {v['reason']}", ""]
        if "action" in v:
            L += ["## What to do", ""] + [f"- {t}" for t in v["action"]["text"]] + [""]
    else:
        L += [f"**Verdict: {v['verdict']}**" + (" (GO and NO-GO both fired; reported as INCONCLUSIVE)"
                                               if v.get("conflict") else ""), "", v["guarantee"], "",
              ] + ([f"- claim: relative, error within {v['eps']} of big-only (tests for declared use: "
                    f"{v.get('rel_mode')}; fixed_n = read once at this n, anytime = valid at every look); "
                    f"big-only error {v.get('big_only_error')}"] if v.get("claim") == "relative" else []) + (
                   [_certified_saving_line(v["certified_saving"], v.get("delta"))] if "certified_saving" in v else []) + (
                   [f"- questions: {v['n_questions']} (all questions test every policy; design v3)",
                    f"- candidate policies: {v['n_policies']}; each leg at level {v['level_each_leg']}; "
                    f"chosen (cheapest passing): {v['chosen']}",
                    f"- cost test: {v['cost_test']}", ""] if v.get("design") == "v3" else
                   [f"- questions: {v['n_questions']} (half A {v['n_half_a']} to choose, half B {v['n_half_b']} to test)",
                    f"- candidate policies: {v['n_policies']}; chosen on half A: {v['chosen_on_half_a']}",
                    f"- cost test: {v['cost_test']}; split salt: `{v['salt']}`", ""])
        if "action" in v:
            L += ["## What to do", ""] + [f"- {t}" for t in v["action"]["text"]] + [""]
        r2 = rep["R2_estimates"]
        if r2.get("policy"):
            lo, hi = r2["risk_interval"]
            L += ["## Risk and saving (" + ("all questions" if v.get("design") == "v3" else "tested half") + ")", "",
                  f"- served {r2['served']}, wrong {r2['errors']}: risk {r2['risk']} "
                  f"(interval {lo} to {hi} at level {r2['interval_level']})",
                  f"- answered {r2['coverage']} of the tested questions; saving vs big model only: {r2['saving']}", ""]
        bg = r2.get("by_group")
        if bg:
            L += ["## Risk by group (" + ("all questions" if v.get("design") == "v3" else "tested half")
                  + ", descriptive, not certified)", ""]
            for g, x in bg["groups"].items():
                if "risk" in x:
                    flag = " **lower bound above alpha**" if x["lower_bound_above_alpha"] else ""
                    L.append(f"- {g}: served {x['served']}/{x['questions']}, wrong {x['errors']}, risk {x['risk']} "
                             f"(interval {x['risk_interval'][0]} to {x['risk_interval'][1]}){flag}")
                else:
                    L.append(f"- {g}: no served answers ({x['questions']} questions)")
            L += ["", bg["note"], ""]
        r3 = rep["R3_cost_measures"]
        L += ["## Cost measures", ""]
        for m, x in sorted(r3["by_measure"].items()):
            L.append(f"- {m}: {x['verdict']}" + (f" ({x['status']})" if "status" in x else f" ({x.get('reason')})"))
        L += [f"- verdict changes with the measure: {'yes' if r3['verdict_changes_with_measure'] else 'no'}", ""]
        r4 = rep["R4_break_even"]
        L += ["## Break-even and certification cost", ""]
        if rep["R1_verdict"].get("design") == "v3" and "break_even_lambda" not in r4:
            c = r4.get("certification") or {}
            L += [f"- {r4.get('reason')}",
                  f"- evidence gap ({c.get('basis')}): questions needed {c.get('questions_needed')} "
                  f"(have {c.get('questions_available')})" + (f"; blocking leg: {c['blocking']}" if c.get("blocking") else "")
                  + (f"; {c['reason']}" if c.get("reason") else "")]
        elif "break_even_lambda" in r4:
            c = r4["certification"]
            L += [f"- break-even lambda: {r4['break_even_lambda']} (configured lambda {r4['lambda']})",
                  f"- lambda needed for the required saving: {r4['lambda_needed_for_saving_s']}",
                  f"- questions needed (estimate): {c.get('questions_needed')} "
                  f"(have {c['questions_available']})" + (f"; {c['reason']}" if c.get("reason") else ""), ""]
        else:
            L += [f"- {r4.get('reason')}"]
            c = r4.get("certification")
            if c:
                L += [f"- planning estimate ({c['basis']}): questions needed {c.get('questions_needed')} "
                      f"(have {c['questions_available']})" + (f"; {c['reason']}" if c.get("reason") else "")]
        L += [f"- {r4['minimum_note']}", ""]
    r5 = rep["R5_data_quality"]
    L += ["## Data quality", "", f"- pool ok: {r5['ok']}; levels: " +
          ", ".join(f"{m} {lv}" for m, lv in r5["level"].items())]
    if "cost_bound_ratio" in r5:
        L.append(f"- cost bound / mean big-model cost: {r5['cost_bound_ratio']}")
    for i in r5["issues"]:
        L.append(f"- {i['severity']} {i['code']} ({i['model']}): {i['message']}")
    for n in r5.get("replay_notes", []):
        L.append(f"- note: {n}")
    return "\n".join(L) + "\n"


def half_of(qid: str, salt: str) -> str:
    return "A" if in_half_a(qid, salt) else "B"
