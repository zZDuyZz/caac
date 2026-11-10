"""Certify every method under one protocol and compare them at each price ratio.

    python certify.py --train-small T_s --train-big T_b --calib-small C_s --calib-big C_b \
        --test-small X_s --test-big X_b --alpha 0.10 --json out/math500.json

Data: the estimators are fitted on TRAIN, every family is certified on CALIB
(one draw order per question, exact binomial counts), and every number reported
comes from TEST (averaged over draw orders). The three parts share no question.
Without --calib-*, the train pool is split in two by a hash of the question id.

For each price ratio lam (one big-model token costs lam small-model tokens):
  * each family's certified configurations are found once; for the
    lam-independent baselines the certificate does not change with lam
  * selections: most coverage; and the cheapest configuration whose calibration
    coverage reaches each target in --coverages (the matched-coverage table)
  * the plug-in selector (no test) is reported beside each, to show what the
    guarantee buys; the oracle bounds everything from above
  * phase map: at each (lam, coverage target), the cheapest family on test, and
    the action mix CAAC chose there
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from caac.pool import families as F
from caac.pool.estimator import (EscalateGate, PerActionEstimator, RecoveryEstimator,
                                 SecondStageEstimator)
from caac.pool.ltt import min_served_certifiable
from caac.pool.protocol import (build_table, certified_frontier, certify, pick_max_coverage,
                                pick_min_cost, plugin_select, prune, summarize)
from caac.pool.replay import make_orders
from caac.pool.schema import load_pool

DEFAULT_LAMS = "0.5,1,1.5,2,3,5,8,12,20,35,60"
BASELINES = ("small_abstain", "big_abstain", "cascade_abstain", "sc_abstain", "sc_abstain_greedy", "esc_abstain",
             "adaptive_consistency_abstain", "pricecheck", "conformal_cascade",
             "conformal_cascade_abstain")
# "caac_staged_value" is the recoverability arm: it buys a correcting action
# only when the action's estimated NET value clears rho, so a question nothing
# can repair is abstained on before the big model is billed. "caac_staged" is
# the same policy gating on P(correct | action) -- the confidence-shaped arm
# the new one has to beat. Both run; neither is assumed to win.
CAAC = ("caac", "caac_staged", "caac_staged_value", "caac_staged_no_sample",
        "caac_staged_no_escalate")
# A net-value gate lives on the [-1, 1] scale, not on a probability.
VALUE_RHOS = (-1.0, 0.0, 0.02, 0.05, 0.1)


def hash_split(pool: dict, frac: float, seed: str = "caac-split-v2"):
    a, b = {}, {}
    for q, r in pool.items():
        h = int(hashlib.sha256(f"{seed}|{q}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (a if h < frac else b)[q] = r
    return a, b


def baseline_family(name: str, n_big: int):
    return {"small_abstain": F.small_abstain, "big_abstain": F.big_abstain,
            "cascade_abstain": F.cascade_abstain, "sc_abstain": F.sc_abstain,
            "sc_abstain_greedy": lambda: F.sc_abstain(greedy_vote=True),
            "esc_abstain": lambda: F.adaptive_abstain("esc"),
            "adaptive_consistency_abstain": lambda: F.adaptive_abstain("adaptive_consistency"),
            "pricecheck": lambda: F.pricecheck(n_big),
            "conformal_cascade": F.conformal_cascade,
            "conformal_cascade_abstain": lambda: F.conformal_cascade(abstain_last=True),
            }[name]()


def fit_estimators(tr_s, tr_b, k_steps):
    """Stage 1 (pre-action), stage 2 (after samples), stage 3 (after escalation),
    and the net-value estimator the allocator gates on -- all on TRAIN only."""
    q = sorted(tr_s)
    est = F.Memo(PerActionEstimator(k=8).fit(q, tr_s, tr_b))
    est2 = SecondStageEstimator(k=max(k_steps), ks=k_steps).fit(
        q, tr_s, make_orders(min(r.n_samples for r in tr_s.values()), 4, seed=11),
        stage1=lambda r: est(r))
    est3 = EscalateGate().fit(q, tr_s, tr_b, stage1=lambda r: est(r)) if tr_b else None
    rec = F.Memo(RecoveryEstimator(k=max(k_steps)).fit(q, tr_s, tr_b))
    return est, est2, est3, rec


def caac_family(name: str, ests, lam: float, big_model, k_steps):
    est, est2, est3, rec = ests
    full = ("answer_now", "sample_more", "escalate")
    if name == "caac":
        return F.caac(est, lam, big_model)
    acts = {"caac_staged": full, "caac_staged_value": full,
            "caac_staged_no_sample": ("answer_now", "escalate"),
            "caac_staged_no_escalate": ("answer_now", "sample_more")}[name]
    value_gate = name == "caac_staged_value"
    return F.caac_staged_family(
        est, est2, lam, big_model, actions=acts, k_steps=k_steps, name=name, est3=est3,
        rho_mode="value" if value_gate else "phat",
        recovery=rec if value_gate else None,
        rhos=VALUE_RHOS if value_gate else F.RHO_GRID)


def evaluate(fam, train, calib, test, a, orders, lams, coverages):
    """Certify one family on calib, report on test, for every lam in ``lams``."""
    t_train = build_table(fam, sorted(train[0]), *train)
    fam = prune(fam, t_train, len(calib[0]), a.alpha, a.delta)
    if not fam.configs:
        return {"n_configs": 0, "by_lam": {}}
    t_cal = build_table(fam, sorted(calib[0]), *calib)
    t_tst = build_table(fam, sorted(test[0]), *test, orders=orders)
    ok, _ = certify(t_cal, None, a.alpha, a.delta)
    plug = plugin_select(t_cal, None, a.alpha)
    out = {"n_configs": len(fam.configs), "n_certified": int(ok.sum()), "by_lam": {}}
    for lam in lams:
        rec = {}
        j = pick_max_coverage(t_cal, None, ok)
        rec["max_coverage"] = _pick(t_cal, t_tst, j, lam)
        jp = pick_max_coverage(t_cal, None, plug)
        rec["plugin_max_coverage"] = _pick(t_cal, t_tst, jp, lam)
        rec["min_cost"] = {f"{c:g}": _pick(t_cal, t_tst, pick_min_cost(t_cal, None, ok, lam, c), lam)
                           for c in coverages}
        rec["frontier"] = [_pick(t_cal, t_tst, j2, lam)
                           for j2 in certified_frontier(t_cal, None, ok, lam)]
        out["by_lam"][f"{lam:g}"] = rec
    return out


def _pick(t_cal, t_tst, j, lam):
    if j is None:
        return None
    test = summarize(t_tst, None, j, lam)
    test["actions"] = t_tst.actions[j]
    return {"config": _jsonable(t_cal.configs[j]), "calib": _strip(summarize(t_cal, None, j, lam)),
            "test": _strip(test)}


def _jsonable(x):
    return list(x) if isinstance(x, tuple) else x


def _strip(d):
    d = dict(d)
    d.pop("config", None)
    return d


def main():
    ap = argparse.ArgumentParser()
    for part in ("train", "calib", "test"):
        ap.add_argument(f"--{part}-small", required=part != "calib")
        ap.add_argument(f"--{part}-big", default=None)
    ap.add_argument("--calib-frac", type=float, default=0.5,
                    help="share of the train pool used for calibration when --calib-* is absent")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--lams", default=DEFAULT_LAMS)
    ap.add_argument("--coverages", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--families", default=",".join(BASELINES + CAAC))
    # 2 is in the ladder because at a realistic price ratio a 4-sample vote can
    # already cost as much as one escalation, and then the cheap repair has to
    # be cheaper than that to exist at all.
    ap.add_argument("--k-steps", default="2,4,8,16")
    ap.add_argument("--perms", type=int, default=10, help="draw orders averaged on test")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tr_s = load_pool(a.train_small)
    tr_b = load_pool(a.train_big) if a.train_big else None
    if a.calib_small:
        ca_s = load_pool(a.calib_small)
        ca_b = load_pool(a.calib_big) if a.calib_big else None
    else:
        ca_s, tr_s = hash_split(tr_s, a.calib_frac)
        ca_b = tr_b
    te_s = load_pool(a.test_small)
    te_b = load_pool(a.test_big) if a.test_big else None
    for nm, s, b in (("train", tr_s, tr_b), ("calib", ca_s, ca_b), ("test", te_s, te_b)):
        if b is not None:
            missing = set(s) - set(b)
            if missing:
                raise SystemExit(f"{nm}: {len(missing)} questions have no big-model record")
    if set(tr_s) & set(ca_s) or set(tr_s) & set(te_s) or set(ca_s) & set(te_s):
        raise SystemExit("train, calib and test must not share a question")

    lams = [float(x) for x in a.lams.split(",")]
    coverages = [float(x) for x in a.coverages.split(",")]
    k_steps = tuple(int(x) for x in a.k_steps.split(","))
    names = [x.strip() for x in a.families.split(",") if x.strip()]
    n_big = min((r.n_samples for r in (ca_b or {}).values()), default=0)
    big_model = next(iter(te_b.values())).model if te_b else None
    width = min(r.n_samples for r in te_s.values())
    orders = make_orders(width, a.perms, seed=1)

    ests = fit_estimators(tr_s, tr_b, k_steps) if any(n in CAAC for n in names) else None

    train, calib, test = (tr_s, tr_b), (ca_s, ca_b), (te_s, te_b)
    res = {"alpha": a.alpha, "delta": a.delta, "n_train": len(tr_s), "n_calib": len(ca_s),
           "n_test": len(te_s), "lams": lams, "coverages": coverages, "k_steps": list(k_steps),
           "n_min_fixed_sequence": min_served_certifiable(a.alpha, a.delta),
           "families": {}, "oracle": {}, "phase_map": {}}
    for nm in names:
        if nm in CAAC:
            res["families"][nm] = {"by_lam": {}}
            for lam in lams:
                fam = caac_family(nm, ests, lam, big_model, k_steps)
                r = evaluate(fam, train, calib, test, a, orders, [lam], coverages)
                res["families"][nm]["by_lam"].update(r["by_lam"])
                res["families"][nm].setdefault("n_certified", {})[f"{lam:g}"] = r.get("n_certified", 0)
        else:
            res["families"][nm] = evaluate(baseline_family(nm, n_big), train, calib, test, a,
                                           orders, lams, coverages)
        print(f"  done: {nm}", flush=True)
    for lam in lams:
        res["oracle"][f"{lam:g}"] = F.oracle(sorted(te_s), te_s, te_b, lam, k_steps)
        res["phase_map"][f"{lam:g}"] = phase_row(res, lam, coverages)

    report(res)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
        print(f"-> {a.json}")


def phase_row(res, lam, coverages):
    """Cheapest family on test at each coverage target, among certified picks."""
    row = {}
    for c in coverages:
        best = None
        for nm, fr in res["families"].items():
            pk = fr["by_lam"].get(f"{lam:g}", {}).get("min_cost", {}).get(f"{c:g}")
            if pk and (best is None or pk["test"]["cost"] < best[1]["test"]["cost"]):
                best = (nm, pk)
        caac = res["families"].get("caac_staged", {}).get("by_lam", {}).get(f"{lam:g}", {})
        cpk = caac.get("min_cost", {}).get(f"{c:g}")
        row[f"{c:g}"] = {"winner": best[0] if best else None,
                         "winner_cost": best[1]["test"]["cost"] if best else None,
                         "caac_staged_actions": cpk["test"]["actions"] if cpk else None}
    return row


def report(res):
    a, lams = res["alpha"], res["lams"]
    print(f"\n=== certified comparison: alpha={a} delta={res['delta']}  train/calib/test = "
          f"{res['n_train']}/{res['n_calib']}/{res['n_test']} ===")
    for lam in lams:
        L = f"{lam:g}"
        print(f"\n-- lam = {L}  (oracle: coverage {res['oracle'][L]['coverage']:.3f}, "
              f"cost {res['oracle'][L]['cost']:.0f})")
        print(f"  {'family':30s} {'cert':>4s} {'cov':>6s} {'risk':>6s} {'cost':>8s}   "
              + "  ".join(f"cost@{c:g}" for c in res["coverages"]))
        for nm, fr in res["families"].items():
            r = fr["by_lam"].get(L)
            if not r:
                print(f"  {nm:30s}  (nothing certifiable)")
                continue
            m = r["max_coverage"]
            ncert = fr["n_certified"][L] if isinstance(fr.get("n_certified"), dict) else fr.get("n_certified")
            head = (f"{m['test']['coverage']:6.3f} {m['test']['selective_risk']:6.3f} "
                    f"{m['test']['cost']:8.0f}") if m else f"{'-':>6s} {'-':>6s} {'-':>8s}"
            cells = "  ".join(f"{r['min_cost'][f'{c:g}']['test']['cost']:9.0f}"
                              if r["min_cost"].get(f"{c:g}") else f"{'-':>9s}"
                              for c in res["coverages"])
            flag = " !" if m and m["test"]["selective_risk"] > a else ""
            print(f"  {nm:30s} {ncert:4d} {head}   {cells}{flag}")
        winners = ", ".join(f"{c}:{v['winner']}" for c, v in res["phase_map"][L].items())
        print(f"  cheapest at each coverage: {winners}")


if __name__ == "__main__":
    main()
