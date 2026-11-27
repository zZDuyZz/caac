"""Action guidance for a design-v3 report (amendment v3-8).

A pure function of fields already in the report: no new test, no new p-value, no data access.
It never changes the verdict. Rule fixed in protocols/v3_decision/uc/DECISION_uc_before_run.md.
"""
from __future__ import annotations


def _pct(x) -> str:
    return f"{x:.0%}"


def _quality_ok(p: dict, claim: str, alpha, eps) -> bool:
    if claim == "relative":
        return p.get("delta_hat") is not None and p["delta_hat"] < eps
    return bool(p.get("served")) and p.get("risk") is not None and p["risk"] < alpha


def action_from_report(rep: dict) -> dict:
    v = rep["R1_verdict"]
    verdict = v["verdict"]
    if verdict == "NOT RUN":
        return {"kind": "fix_data", "text": ["Fix the errors listed under data quality, then rerun."]}
    claim = v.get("claim") or "absolute"
    alpha, eps, s, delta = v.get("alpha"), v.get("eps"), v.get("s"), v.get("delta")
    pols = rep["R2_estimates"].get("all_policies") or []
    cs = v.get("certified_saving") or {}
    r4 = rep.get("R4_break_even") or {}
    quality = (f"error within {eps} of the large model" if claim == "relative" else f"risk <= {alpha}")
    out: dict = {}
    text: list[str] = []

    if verdict == "GO":
        out["kind"] = "deploy"
        out["policy"] = v.get("chosen")
        out["budget_saving"] = cs.get("value")
        text.append(f"Deploy policy {v.get('chosen')} on questions like the audited ones.")
        if cs.get("value") is not None:
            text.append(f"Budget with the certified saving ({_pct(cs['value'])} of the big-model-only cost), "
                        "not with the point estimate.")
        flagged = []
        bg = rep["R2_estimates"].get("by_group")
        if bg and claim != "relative":
            flagged = sorted(g for g, x in bg["groups"].items() if x.get("lower_bound_above_alpha"))
        out["groups_to_check"] = flagged
        if flagged:
            text.append("Check these groups before routing them to the small model (descriptive, not "
                        "certified): " + ", ".join(flagged) + ".")
        mc = bool((rep.get("R3_cost_measures") or {}).get("verdict_changes_with_measure"))
        out["verdict_changes_with_measure"] = mc
        if mc:
            text.append("The verdict depends on the cost measure: budget only in the locked measure.")
        text.append("Re-audit on new data if the question mix or the models change.")
        out["text"] = text
        return out

    lvl = delta / 2 if delta is not None else None
    if verdict == "NO-GO":
        rq = rs = rb = 0
        for p in pols:
            q = p.get("p_risk_above") is not None and p["p_risk_above"] <= lvl
            sv = p.get("p_saving_below") is not None and p["p_saving_below"] <= lvl
            rb += q and sv
            rq += q and not sv
            rs += sv and not q
        out.update(kind="keep_big", rejected_by_quality=rq, rejected_by_saving=rs, rejected_by_both=rb)
        text.append("Keep the large model for this traffic: every candidate policy is rejected "
                    f"({rq} on quality only, {rs} on saving only, {rb} on both).")
        if rq + rb == 0:
            text.append(f"Quality is not the problem: the small model is too expensive to reach a saving of "
                        f"{_pct(s)}; a cheaper small model or cheaper escalation is a new audit.")
        elif rs + rb == 0:
            text.append(f"Saving is not the problem: no candidate meets {quality}; a stronger small model "
                        "is a new audit.")
        text.append("Trying other models or policies needs a new lock and new audit data.")
    elif v.get("conflict"):
        out["kind"] = "check_data"
        text.append("GO and NO-GO both fired (probability <= delta under the assumptions): check the data "
                    "and the independence of questions before using this result.")
    else:
        c = r4.get("certification") or {}
        out["kind"] = "collect_or_stop"
        floor = r4.get("minimum_questions_zero_errors")
        n = v.get("n_questions")
        out["below_floor"] = bool(floor is not None and n is not None and n < floor)
        if c.get("reachable"):
            out.update(type="reachable", blocking=c.get("blocking"), policy=c.get("policy"),
                       more_questions_needed=c.get("more_questions_needed"),
                       questions_needed=c.get("questions_needed"))
            text.append(f"Not decided yet. About {c.get('more_questions_needed')} more questions of the same kind "
                        f"({c.get('questions_needed')} in total) would likely give GO with {c.get('policy')} "
                        f"(planning estimate, power 0.8); the {c.get('blocking')} leg needs them.")
            text.append("Collect them, then run the whole audit once on all questions under the same lock.")
        else:
            qok = [p for p in pols if _quality_ok(p, claim, alpha, eps)]
            sok = [p for p in pols if p.get("saving") is not None and s is not None and p["saving"] > s]
            both = [p for p in qok if p in sok]
            blocking = ("risk" if not qok else "saving" if not sok else "trade_off" if not both else "power")
            out.update(type="unreachable", blocking=blocking)
            why = {"risk": f"no candidate meets {quality} even on the point estimate",
                   "saving": f"the candidates that meet {quality} save less than {_pct(s)}",
                   "trade_off": f"the candidates that meet {quality} are not the ones that save {_pct(s)}",
                   "power": "the margin is too thin to certify within 200000 questions"}[blocking]
            text.append(f"More questions of the same kind are not expected to give GO: {why}. In practice, "
                        "do not deploy these candidates; NO-GO is not certified either.")
        if out["below_floor"]:
            text.append(f"This pool has {n} questions, below the minimum {floor} at which even perfect data "
                        "could give GO.")
    if cs.get("value") is not None and cs["value"] > 0:
        out["smaller_saving_certified"] = {"value": cs["value"], "policy": cs.get("policy")}
        text.append(f"A smaller saving is already certified: at least {_pct(cs['value'])} with "
                    f"{cs.get('policy')} (same guarantee); it is not GO at the locked s = {s}.")
    out["text"] = text
    return out
