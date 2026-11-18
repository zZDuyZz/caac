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
  * router / router_abstain (RouteLLM-style) need a router fitted on TRAIN; with
    --hidden-train/--hidden-calib/--hidden-test it reads the small model's
    question-only hidden states (PCA fitted on train), otherwise only the
    prompt length, which is a weak router and is reported as such.
  * --bootstrap B: a question-level percentile interval on every test number,
    and, at each (lam, coverage target), the PAIRED interval of every family
    minus --ref on the same test questions. A ranking is claimed only where the
    paired interval excludes zero (caac.eval.bootstrap).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import json
from dataclasses import dataclass
from pathlib import Path

from caac.eval.bootstrap import bootstrap_ci, paired_diff_ci
from caac.pool import families as F
from caac.pool.estimator import (EscalateGate, PerActionEstimator, RecoveryEstimator,
                                 SecondStageEstimator)
from caac.pool.costmodel import CostModel
from caac.pool.hidden import hidden_extras
from caac.pool.router import QuestionRouter
from caac.pool.ltt import min_served_certifiable
from caac.pool.state_value import StateValueEstimator
from caac.pool.protocol import (build_table, concat_tables, certified_frontier, certify, pick_max_coverage,
                                pick_min_cost, plugin_select, prune, summarize)
from caac.pool.replay import make_orders
from caac.pool.schema import load_pool

DEFAULT_LAMS = "0.5,1,1.5,2,3,5,8,12,20,35,60"
BASELINES = ("small_abstain", "big_abstain", "cascade_abstain", "sc_abstain", "sc_abstain_greedy", "esc_abstain",
             "adaptive_consistency_abstain", "pricecheck", "conformal_cascade",
             "conformal_cascade_abstain", "router", "router_abstain")
ROUTERS = ("router", "router_abstain")
# "caac_staged_value" is the recoverability arm: it buys a correcting action
# only when the action's estimated NET value clears rho, so a question nothing
# can repair is abstained on before the big model is billed. "caac_staged" is
# the same policy gating on P(correct | action) -- the confidence-shaped arm
# the new one has to beat. Both run; neither is assumed to win.
CAAC = ("caac", "caac_staged", "caac_staged_value", "caac_staged_delta", "caac_staged_state",
        "caac_staged_no_sample", "caac_staged_no_escalate", "caac_staged_biggrid")
# Gates on net value do not live on a probability scale. "delta" thresholds the
# estimated net gain in [-1, 1]; "value" thresholds net gain per greedy turn of
# extra cost. The last entry of each grid buys nothing.
DELTA_RHOS = (-1.0, 0.0, 0.02, 0.05, 0.1, 0.2, 1.01)
VALUE_RHOS = (-1.0, 0.0, 0.01, 0.03, 0.1, 0.3, 10.0)
# The "more flexible" arm of the simplification question (MASTER 12.7m B2):
# the same staged policy on a finer (rho, tau) grid -- 11 x 32 = 352
# configurations against 6 x 16 = 96. Fixed before any data, like every grid.
RHO_GRID_FINE = (0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01)
TAU_GRID_FINE = tuple(round(0.995 - i * (0.995 - 0.5) / 31, 4) for i in range(32))


def hash_split(pool: dict, frac: float, seed: str = "caac-split-v2"):
    a, b = {}, {}
    for q, r in pool.items():
        h = int(hashlib.sha256(f"{seed}|{q}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (a if h < frac else b)[q] = r
    return a, b


def baseline_family(name: str, n_big: int, n_small: int = 16, router=None):
    """``n_small`` / ``n_big`` are the draws every split holds, so a rule that
    counts draws is the same rule at calibration and at test time. ``router`` is
    the fitted QuestionRouter the two routing families need."""
    cc = dict(n_small=min(16, n_small), n_big=min(16, n_big))
    if name in ROUTERS:
        if router is None:
            raise SystemExit(f"{name} needs a router fitted on TRAIN (see fit_router)")
        return F.router(router, abstain=name == "router_abstain")
    return {"small_abstain": F.small_abstain, "big_abstain": F.big_abstain,
            "cascade_abstain": F.cascade_abstain, "sc_abstain": F.sc_abstain,
            "sc_abstain_greedy": lambda: F.sc_abstain(greedy_vote=True),
            "esc_abstain": lambda: F.adaptive_abstain("esc"),
            "adaptive_consistency_abstain": lambda: F.adaptive_abstain("adaptive_consistency"),
            "pricecheck": lambda: F.pricecheck(n_big),
            "conformal_cascade": lambda: F.conformal_cascade(**cc),
            "conformal_cascade_abstain": lambda: F.conformal_cascade(abstain_last=True, **cc),
            }[name]()


@dataclass
class Fitted:
    """Everything CAAC learns, fitted on TRAIN only."""
    stage1: object          # P(correct | action) from the greedy turn
    stage2: object          # P(vote correct | samples drawn)
    gate: object            # P(big answer correct | what is visible after escalating)
    recovery: object        # net gain of a step from the greedy turn alone
    state: object           # net gain of a step from every state
    cost: CostModel         # expected price of a step before it is paid for


def fit_estimators(tr_s, tr_b, k_steps):
    q = sorted(tr_s)
    est = F.Memo(PerActionEstimator(k=8).fit(q, tr_s, tr_b))
    width = min(r.n_samples for r in tr_s.values())
    est2 = SecondStageEstimator(k=max(k_steps), ks=k_steps).fit(
        q, tr_s, make_orders(width, 4, seed=11), stage1=lambda r: est(r))
    est3 = EscalateGate().fit(q, tr_s, tr_b, stage1=lambda r: est(r)) if tr_b else None
    rec = F.Memo(RecoveryEstimator(k=max(k_steps)).fit(q, tr_s, tr_b))
    state = StateValueEstimator(steps=k_steps).fit(q, tr_s, tr_b, make_orders(width, 3, seed=13),
                                                   stage1=lambda r: est(r))
    return Fitted(est, est2, est3, rec, state, CostModel().fit(tr_s, tr_b))


def fit_fold_estimators(tr_s, tr_b, k_steps, folds: int = 5, seed: int = 0):
    """K-fold estimators on TRAIN: [(held-out qids, estimators fit without them)].

    The training table decides which configurations are tested and in what order.
    Scored in-sample it is badly optimistic -- estimators fit on train look nearly
    perfect there (Day 1 MATH-500: risk 0.000 on train vs ~0.06 on calibration) --
    so every training question is scored by estimators that never saw it.
    """
    qids = sorted(tr_s)
    random.Random(seed).shuffle(qids)
    out = []
    for f in range(folds):
        held = set(qids[f::folds])
        fs = {q: r for q, r in tr_s.items() if q not in held}
        fb = {q: r for q, r in tr_b.items() if q not in held} if tr_b else None
        out.append((sorted(held), fit_estimators(fs, fb, k_steps)))
    return out


def fit_router(tr_s, tr_b, extras):
    if tr_b is None:
        raise SystemExit("the router needs the big model's training pool")
    return QuestionRouter(extras=extras or {}).fit(sorted(tr_s), tr_s, tr_b)


def fit_fold_routers(tr_s, tr_b, extras, folds: int = 5, seed: int = 0):
    """K-fold routers on TRAIN, with the same fold split as fit_fold_estimators."""
    qids = sorted(tr_s)
    random.Random(seed).shuffle(qids)
    out = []
    for f in range(folds):
        held = set(qids[f::folds])
        fs = {q: r for q, r in tr_s.items() if q not in held}
        fb = {q: r for q, r in tr_b.items() if q not in held}
        out.append((sorted(held), fit_router(fs, fb, extras)))
    return out


def router_extras(a, train_qids, need_qids):
    """Projected hidden states for the router, or {} (prompt length only)."""
    if not getattr(a, "hidden_train", None):
        print("WARNING: router without --hidden-*: it sees only the prompt length (a weak router)",
              flush=True)
        return {}
    others = [p for p in (getattr(a, "hidden_calib", None), getattr(a, "hidden_test", None),
                          getattr(a, "hidden_pool", None)) if p]
    ex, _ = hidden_extras([a.hidden_train], others, sorted(train_qids), dims=a.hidden_dims,
                          pooling=a.hidden_pooling, layer=a.hidden_layer)
    missing = [q for q in need_qids if q not in ex]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no hidden state (first: {missing[:3]})")
    return ex


def add_hidden_args(ap, parts=("train", "calib", "test")):
    for part in parts:
        ap.add_argument(f"--hidden-{part}", default=None,
                        help=f"extract_hidden.py .npz covering the {part} questions (router only)")
    ap.add_argument("--hidden-pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--hidden-layer", type=int, default=-1, help="index into the KEPT layers")
    ap.add_argument("--hidden-dims", type=int, default=32, help="PCA dimensions, fitted on train")


def train_table(nm, fam, train, fold_ests, lam, big_model, k_steps, fold_routers=None):
    """The table ``prune`` reads: cross-fitted for CAAC families and for the
    routers (both fit on train), plain for the other baselines (they fit nothing,
    so in-sample is already out-of-sample)."""
    tr_s, tr_b = train
    if nm in ROUTERS and fold_routers:
        return concat_tables([build_table(F.router(r, abstain=nm == "router_abstain"), held, tr_s, tr_b)
                              for held, r in fold_routers])
    if nm not in CAAC or not fold_ests:
        return build_table(fam, sorted(tr_s), tr_s, tr_b)
    return concat_tables([build_table(caac_family(nm, e, lam, big_model, k_steps), held, tr_s, tr_b)
                          for held, e in fold_ests])


def caac_family(name: str, ests, lam: float, big_model, k_steps):
    est, est2, est3, rec = ests.stage1, ests.stage2, ests.gate, ests.recovery
    full = ("answer_now", "sample_more", "escalate")
    if name == "caac":
        return F.caac(est, lam, big_model, cost_model=ests.cost)
    if name == "caac_staged_biggrid":
        return F.caac_staged_family(est, est2, lam, big_model, grid=list(TAU_GRID_FINE), actions=full,
                                    k_steps=k_steps, name=name, est3=est3, rhos=RHO_GRID_FINE,
                                    cost_model=ests.cost)
    acts = {"caac_staged": full, "caac_staged_value": full, "caac_staged_delta": full, "caac_staged_state": full,
            "caac_staged_no_sample": ("answer_now", "escalate"),
            "caac_staged_no_escalate": ("answer_now", "sample_more")}[name]
    mode = {"caac_staged_value": "value", "caac_staged_delta": "delta",
            "caac_staged_state": "state"}.get(name, "phat")
    rhos = {"value": VALUE_RHOS, "delta": DELTA_RHOS, "state": VALUE_RHOS, "phat": F.RHO_GRID}[mode]
    gate = {"phat": None, "state": ests.state}.get(mode, rec)
    return F.caac_staged_family(
        est, est2, lam, big_model, actions=acts, k_steps=k_steps, name=name, est3=est3,
        rho_mode=mode, recovery=gate, rhos=rhos, cost_model=ests.cost)


def evaluate(fam, train, calib, test, a, orders, lams, coverages, t_train=None, keep=None):
    """Certify one family on calib, report on test, for every lam in ``lams``.
    ``t_train`` (the table ``prune`` reads) defaults to an in-sample replay.
    ``keep`` (a dict) receives {(lam, coverage): (test table, pick)} for the
    paired bootstrap; it never enters the JSON."""
    if t_train is None:
        t_train = build_table(fam, sorted(train[0]), *train)
    fam = prune(fam, t_train, len(calib[0]), a.alpha, a.delta)
    if not fam.configs:
        return {"n_configs": 0, "by_lam": {}}
    t_cal = build_table(fam, sorted(calib[0]), *calib)
    t_tst = build_table(fam, sorted(test[0]), *test, orders=orders)
    ok, _ = certify(t_cal, None, a.alpha, a.delta)
    plug = plugin_select(t_cal, None, a.alpha)
    out = {"n_configs": len(fam.configs), "n_certified": int(ok.sum()), "by_lam": {}}
    nb = getattr(a, "bootstrap", 0)
    for lam in lams:
        rec = {}
        j = pick_max_coverage(t_cal, None, ok)
        rec["max_coverage"] = _pick(t_cal, t_tst, j, lam, nb)
        jp = pick_max_coverage(t_cal, None, plug)
        rec["plugin_max_coverage"] = _pick(t_cal, t_tst, jp, lam, nb)
        rec["min_cost"] = {}
        for c in coverages:
            jc = pick_min_cost(t_cal, None, ok, lam, c)
            rec["min_cost"][f"{c:g}"] = _pick(t_cal, t_tst, jc, lam, nb)
            if keep is not None and jc is not None:
                keep[(f"{lam:g}", f"{c:g}")] = (t_tst, jc)
        rec["frontier"] = [_pick(t_cal, t_tst, j2, lam)
                           for j2 in certified_frontier(t_cal, None, ok, lam)]
        out["by_lam"][f"{lam:g}"] = rec
    return out


def _pick(t_cal, t_tst, j, lam, n_boot=0):
    if j is None:
        return None
    test = summarize(t_tst, None, j, lam)
    test["actions"] = t_tst.actions[j]
    if n_boot:
        test["ci"] = bootstrap_ci(t_tst, j, lam, n_boot=n_boot, seed=0)
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
    # the fine-grid arm is opt-in: 352 configurations per lam is slow, and it answers
    # one ablation question rather than being a method of its own
    ap.add_argument("--families", default=",".join(BASELINES + tuple(n for n in CAAC if n != "caac_staged_biggrid")))
    # 2 is in the ladder because at a realistic price ratio a 4-sample vote can
    # already cost as much as one escalation, and then the cheap repair has to
    # be cheaper than that to exist at all.
    ap.add_argument("--k-steps", default="2,4,8,16")
    ap.add_argument("--perms", type=int, default=10, help="draw orders averaged on test")
    add_hidden_args(ap)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="question-level bootstrap resamples for test intervals (0 = off; 1000 for the paper)")
    ap.add_argument("--ref", default="caac_staged",
                    help="family every other family is compared with in the paired bootstrap")
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
    # draws available in EVERY split: train prunes, calib certifies, test reports
    n_big = min((r.n_samples for p in (tr_b, ca_b, te_b) for r in (p or {}).values()), default=0)
    n_small = min(r.n_samples for p in (tr_s, ca_s, te_s) for r in p.values())
    big_model = next(iter(te_b.values())).model if te_b else None
    width = min(r.n_samples for r in te_s.values())
    orders = make_orders(width, a.perms, seed=1)

    ests = fit_estimators(tr_s, tr_b, k_steps) if any(n in CAAC for n in names) else None
    fold_ests = fit_fold_estimators(tr_s, tr_b, k_steps) if ests else []
    router, fold_routers = None, []
    if any(n in ROUTERS for n in names):
        ex = router_extras(a, tr_s, list(tr_s) + list(ca_s) + list(te_s))
        router = fit_router(tr_s, tr_b, ex)
        fold_routers = fit_fold_routers(tr_s, tr_b, ex)

    train, calib, test = (tr_s, tr_b), (ca_s, ca_b), (te_s, te_b)
    res = {"alpha": a.alpha, "delta": a.delta, "n_train": len(tr_s), "n_calib": len(ca_s),
           "n_test": len(te_s), "lams": lams, "coverages": coverages, "k_steps": list(k_steps),
           "n_min_fixed_sequence": min_served_certifiable(a.alpha, a.delta),
           "families": {}, "oracle": {}, "phase_map": {}}
    kept = {}           # family -> {(lam, coverage): (test table, pick)}, for the paired bootstrap
    for nm in names:
        kept[nm] = {}
        if nm in CAAC:
            res["families"][nm] = {"by_lam": {}}
            for lam in lams:
                fam = caac_family(nm, ests, lam, big_model, k_steps)
                tt = train_table(nm, fam, train, fold_ests, lam, big_model, k_steps)
                r = evaluate(fam, train, calib, test, a, orders, [lam], coverages, t_train=tt,
                             keep=kept[nm])
                res["families"][nm]["by_lam"].update(r["by_lam"])
                res["families"][nm].setdefault("n_certified", {})[f"{lam:g}"] = r.get("n_certified", 0)
        else:
            fam = baseline_family(nm, n_big, n_small, router)
            tt = (train_table(nm, fam, train, [], None, big_model, k_steps, fold_routers)
                  if nm in ROUTERS else None)
            res["families"][nm] = evaluate(fam, train, calib, test, a, orders, lams, coverages,
                                           t_train=tt, keep=kept[nm])
        print(f"  done: {nm}", flush=True)
    for lam in lams:
        res["oracle"][f"{lam:g}"] = F.oracle(sorted(te_s), te_s, te_b, lam, k_steps)
        res["phase_map"][f"{lam:g}"] = phase_row(res, lam, coverages)
    if a.bootstrap:
        res["paired"] = paired_table(kept, a.ref, a.bootstrap)

    report(res)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
        print(f"-> {a.json}")


def paired_table(kept, ref, n_boot):
    """{lam: {coverage: {family: paired interval of family - ref}}} at matched
    coverage targets, on the same test questions."""
    out = {}
    if ref not in kept:
        return {"_note": f"reference family {ref!r} was not run"}
    for key, (t_ref, j_ref) in kept[ref].items():
        lam, cov = key
        for nm, picks in kept.items():
            if nm == ref or key not in picks:
                continue
            t, j = picks[key]
            out.setdefault(lam, {}).setdefault(cov, {})[nm] = paired_diff_ci(
                t, j, t_ref, j_ref, float(lam), n_boot=n_boot, seed=0)
    return out


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
        print(f"  {'family':30s} {'cert':>4s} {'cov':>6s} {'risk':>6s} {'cost':>8s} {'waste':>6s}   "
              + "  ".join(f"cost@{c:g}" for c in res["coverages"]))
        for nm, fr in res["families"].items():
            r = fr["by_lam"].get(L)
            if not r:
                print(f"  {nm:30s}  (nothing certifiable)")
                continue
            m = r["max_coverage"]
            ncert = fr["n_certified"][L] if isinstance(fr.get("n_certified"), dict) else fr.get("n_certified")
            head = (f"{m['test']['coverage']:6.3f} {m['test']['selective_risk']:6.3f} "
                    f"{m['test']['cost']:8.0f} {m['test']['waste_share']:6.2f}"
                    ) if m else f"{'-':>6s} {'-':>6s} {'-':>8s} {'-':>6s}"
            cells = "  ".join(f"{r['min_cost'][f'{c:g}']['test']['cost']:9.0f}"
                              if r["min_cost"].get(f"{c:g}") else f"{'-':>9s}"
                              for c in res["coverages"])
            flag = " !" if m and m["test"]["selective_risk"] > a else ""
            print(f"  {nm:30s} {ncert:4d} {head}   {cells}{flag}")
        winners = ", ".join(f"{c}:{v['winner']}" for c, v in res["phase_map"][L].items())
        print(f"  cheapest at each coverage: {winners}")
        for c, fams in res.get("paired", {}).get(L, {}).items():
            cells = []
            for nm, d in fams.items():
                cd = d["cost"]
                if cd["lo"] is None:
                    continue
                tag = ("cheaper" if cd["hi"] < 0 else "dearer" if cd["lo"] > 0 else "tie")
                cells.append(f"{nm} {cd['point']:+.0f} [{cd['lo']:+.0f},{cd['hi']:+.0f}] {tag}")
            if cells:
                print(f"  paired cost vs ref @cov {c}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
