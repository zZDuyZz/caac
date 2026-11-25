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

from caac.pool_v1.audit import audit, binom_cdf, in_half_a, served_needed
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


def _guarantee(cfg: Config) -> str:
    return (f"Each verdict is wrong with probability <= {cfg.delta}: GO means the chosen policy has "
            f"risk (error rate among served answers) <= {cfg.alpha} AND saves >= {cfg.s:.0%} of the "
            f"big-model-only cost; NO-GO means no candidate policy does both. Assumes questions "
            f"exchangeable with the deployment, costs bounded by c_max, candidates fixed in advance.")


def _measure_status(rows: list[Row], cfg: Config, measure: str, notes: list[str]) -> str:
    if measure == "tokens":
        return "measured tokens x configured prices"
    est = any(n.startswith(f"{measure}:") and "ESTIMATED" in n for n in notes)
    return f"measured per generation{' (early-stopped draws estimated)' if est else ''}"


def _with_measure(cfg: Config, measure: str, c_max):
    c = copy.deepcopy(cfg)
    c.cost.measure, c.cost.c_max = measure, c_max
    return c


def make_report(rows: list[Row], meta: Meta | None, cfg: Config, *, row_errors=(), meta_errors=(),
                input_info: dict | None = None) -> dict:
    rows = sorted(rows, key=lambda r: (r.qid, r.model, r.draw))
    val = validate(rows, meta, row_errors=row_errors, meta_errors=meta_errors,
                   alpha=cfg.alpha, delta=cfg.delta, pair=(cfg.small, cfg.big))
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
        return rep

    rr = replay(rows, meta, cfg)
    res = audit(rr.audit_input, alpha=cfg.alpha, delta=cfg.delta, s=cfg.s, salt=cfg.salt,
                cost_test=cfg.cost_test)
    r5["replay_notes"] = list(rr.notes)
    mb = sum(rr.audit_input.big_cost) / len(rr.audit_input.big_cost)
    r5["cost_bound_ratio"] = _f(rr.audit_input.c_max / mb) if mb > 0 else "inf"
    if mb > 0 and rr.audit_input.c_max / mb > 20:
        r5["replay_notes"].append(
            f"cost bound c_max is {rr.audit_input.c_max / mb:.0f}x the mean big-model cost: the cost leg "
            "has little power at this sample size (a lower cap, or a per-question bound, would help)")
    rep["R1_verdict"] = {"verdict": res.verdict, "guarantee": _guarantee(cfg), "alpha": cfg.alpha,
                         "delta": cfg.delta, "s": cfg.s, "salt": cfg.salt, "cost_test": cfg.cost_test,
                         "n_questions": res.n_questions, "n_half_a": res.n_a, "n_half_b": res.n_b,
                         "n_policies": len(rr.policies), "policies": [p.name for p in rr.policies],
                         "chosen_on_half_a": res.chosen, "conflict": res.conflict}

    # ---- R2
    if res.chosen is not None:
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
    allp = []
    for row, pd in zip(res.nogo["policies"], rr.audit_input.policies):
        x = {k: _f(v) if isinstance(v, float) else v for k, v in row.items()}
        x["coverage"] = _f(sum(pd.served) / len(pd.served))     # share of questions answered
        allp.append(x)
    rep["R2_estimates"]["all_policies"] = allp
    if any(p.on_unsure == "abstain" for p in rr.policies):
        r5_note = ("ABSTAIN candidates present: saving can come from not answering; read 'coverage' "
                   "in R2 (the primary claim in the protocol lock uses ESCALATE only)")
    else:
        r5_note = None

    # ---- R3
    meas = [{"measure": cfg.cost.measure, "c_max": cfg.cost.c_max}] + list(cfg.cost.also)
    r3 = {}
    for mdef in meas:
        m = mdef["measure"]
        if m in r3:
            continue
        try:
            c2 = _with_measure(cfg, m, mdef.get("c_max"))
            rr2 = replay(rows, meta, c2) if m != cfg.cost.measure else rr
            res2 = audit(rr2.audit_input, alpha=cfg.alpha, delta=cfg.delta, s=cfg.s, salt=cfg.salt,
                         cost_test=cfg.cost_test) if m != cfg.cost.measure else res
            r3[m] = {"verdict": res2.verdict, "chosen_on_half_a": res2.chosen,
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
        served, errors = res.go["served_b"], res.go["errors_b"]
        n_b = res.n_b
        rate = errors / served if served else None
        cert = {"risk_on_half_b": _f(rate), "served_share_half_b": _f(served / n_b) if n_b else None,
                "questions_available": n}
        sn = served_needed(rate, cfg.alpha, cfg.delta) if rate is not None else None
        if sn is None:
            cert["questions_needed"] = None
            cert["reason"] = ("risk on half B is >= alpha: more questions of the same kind will not "
                              "certify this policy")
        else:
            share = served / n_b if n_b else 0
            total = math.ceil(2 * sn / share) if share > 0 else None
            cert["served_needed_on_tested_half"] = sn
            cert["questions_needed"] = total
            cert["more_questions_needed"] = max(0, total - n) if total is not None else None
        cert["note"] = "ESTIMATE for the risk leg only, if the observed rates hold; the cost leg may need more"
        r4["certification"] = cert
    else:
        r4["reason"] = "no policy reached risk <= alpha on half A"
    floor = certification_floor(cfg.alpha, cfg.delta)
    r4["minimum_questions_zero_errors"] = 2 * floor
    r4["minimum_note"] = (f"even with zero errors, GO needs >= {floor} served answers on the tested half, "
                          f"i.e. about {2 * floor} questions if every question is served")
    rep["R4_break_even"] = r4
    if r5_note:
        r5["replay_notes"].append(r5_note)
    rep["R5_data_quality"] = r5
    return rep


def dumps(rep: dict) -> str:
    return json.dumps(rep, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def to_markdown(rep: dict) -> str:
    v = rep["R1_verdict"]
    L = ["# CAAC report", ""]
    if v["verdict"] == "NOT RUN":
        L += [f"**Verdict: NOT RUN** - {v['reason']}", ""]
    else:
        L += [f"**Verdict: {v['verdict']}**" + (" (GO and NO-GO both fired; reported as INCONCLUSIVE)"
                                               if v.get("conflict") else ""), "", v["guarantee"], "",
              f"- questions: {v['n_questions']} (half A {v['n_half_a']} to choose, half B {v['n_half_b']} to test)",
              f"- candidate policies: {v['n_policies']}; chosen on half A: {v['chosen_on_half_a']}",
              f"- cost test: {v['cost_test']}; split salt: `{v['salt']}`", ""]
        r2 = rep["R2_estimates"]
        if r2.get("policy"):
            lo, hi = r2["risk_interval"]
            L += ["## Risk and saving (tested half)", "",
                  f"- served {r2['served']}, wrong {r2['errors']}: risk {r2['risk']} "
                  f"(interval {lo} to {hi} at level {r2['interval_level']})",
                  f"- answered {r2['coverage']} of the tested questions; saving vs big model only: {r2['saving']}", ""]
        r3 = rep["R3_cost_measures"]
        L += ["## Cost measures", ""]
        for m, x in sorted(r3["by_measure"].items()):
            L.append(f"- {m}: {x['verdict']}" + (f" ({x['status']})" if "status" in x else f" ({x.get('reason')})"))
        L += [f"- verdict changes with the measure: {'yes' if r3['verdict_changes_with_measure'] else 'no'}", ""]
        r4 = rep["R4_break_even"]
        L += ["## Break-even and certification cost", ""]
        if "break_even_lambda" in r4:
            c = r4["certification"]
            L += [f"- break-even lambda: {r4['break_even_lambda']} (configured lambda {r4['lambda']})",
                  f"- lambda needed for the required saving: {r4['lambda_needed_for_saving_s']}",
                  f"- questions needed (estimate): {c.get('questions_needed')} "
                  f"(have {c['questions_available']})" + (f"; {c['reason']}" if c.get("reason") else ""), ""]
        else:
            L += [f"- {r4.get('reason')}"]
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
