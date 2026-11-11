"""Five measurements that say WHERE the money goes, before any policy is tuned.

Run on the v3 pools (CPU only) and read BEFORE changing a controller: each one
separates a different reason a controller could fail to save compute, so the fix
follows from the measurement instead of from a guess.

  D1 ``action_costs``       what each action costs next to the greedy turn
  D2 ``perfect_stop``       how much repair spending goes to questions nothing can fix
  D3 ``signal_auroc``       can two cheap samples tell the cases apart
  D4 ``recoverable_width``  how wide the "cheap to repair" region is, per sample budget
  D5 ``truncation_kind``    are the cut-off answers loops or just long

Nothing here touches the certification test. These are diagnostics for choosing
what to improve on TRAIN/CALIB data; the numbers are not results.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from caac.pool.estimator import LogisticHead, features
from caac.pool.families import lam_cost
from caac.pool.policies import agreement_with_greedy, calls_for_action
from caac.pool.recovery import recovery_table, sample_correct_prob
from caac.pool.voting import greedy_vote_correct

__all__ = ["action_costs", "perfect_stop", "signal_auroc", "recoverable_width",
           "truncation_kind", "repetition_score", "auroc"]

STEPS = (2, 4, 8, 16)


def _big_name(big: dict | None) -> str | None:
    return next(iter(big.values())).model if big else None


def _pairs(qids, small: dict, big: dict | None):
    return [q for q in qids if q in small and (big is None or q in big)]


# ---------------------------------------------------------------------------
# D1  what each action costs
# ---------------------------------------------------------------------------

def action_costs(qids, small: dict, big: dict | None, lam: float, ks=STEPS) -> dict:
    """Mean billed cost of every action, and its ratio to the greedy turn.

    Reading: if escalation costs many greedy turns, savings must come from NOT
    escalating (abstaining or sampling instead); if sampling k=16 costs more
    than escalating, sampling that far is never the cheap repair.
    """
    qids = _pairs(qids, small, big)
    price = lam_cost(lam, _big_name(big))
    acc = {"answer_now": 0.0, "escalate": 0.0, **{f"sample_{k}": 0.0 for k in ks}}
    for q in qids:
        s, b = small[q], (big or {}).get(q)
        order = list(range(s.n_samples))
        acc["answer_now"] += price(calls_for_action(s, b, "answer_now", 0, order))
        if b is not None:
            acc["escalate"] += price(calls_for_action(s, b, "escalate", 0, order))
        for k in ks:
            acc[f"sample_{k}"] += price(calls_for_action(s, b, "sample_more", k, order))
    n = max(1, len(qids))
    mean = {a: v / n for a, v in acc.items()}
    if big is None:
        mean.pop("escalate")
    g = mean["answer_now"] or 1.0
    return {"lam": lam, "n": len(qids), "mean_cost": mean,
            "ratio_to_greedy": {a: v / g for a, v in mean.items()}}


# ---------------------------------------------------------------------------
# D2  the perfect-stop ceiling
# ---------------------------------------------------------------------------

def perfect_stop(qids, small: dict, big: dict | None, lam: float, ks=STEPS) -> dict:
    """How much repair spending goes to questions that no action can fix.

    Reference: a controller that, on every question, pays for the greedy turn,
    all ``max(ks)`` samples AND the big model ("try everything"). A controller
    that knew in advance which questions are hopeless would skip all of that on
    them and, on the others, buy only the cheapest step that works.

    "Works" uses the deployable vote (P(vote right) >= 0.5), never "some sample
    was right". A question no action repairs is ``unrecoverable`` relative to
    THIS action set and sample budget, not impossible.
    """
    qids = _pairs(qids, small, big)
    price = lam_cost(lam, _big_name(big))
    n = max(1, len(qids))
    k_all = max(ks)
    tot_all = tot_perfect = tot_greedy = 0.0
    waste_all = rep_all = 0.0
    right_now = repairable = hopeless = 0
    for q in qids:
        s, b = small[q], (big or {}).get(q)
        order = list(range(s.n_samples))
        g = price(calls_for_action(s, b, "answer_now", 0, order))
        every = g
        every += price(calls_for_action(s, b, "sample_more", k_all, order)) - g
        if b is not None:
            every += price(calls_for_action(s, b, "escalate", 0, order)) - g
        tot_all += every
        tot_greedy += g
        rep_all += every - g
        if s.greedy.correct:
            right_now += 1
            tot_perfect += g
            continue
        opts = []
        for k in ks:
            if s.n_samples and sample_correct_prob(s, k) >= 0.5:
                opts.append(price(calls_for_action(s, b, "sample_more", k, order)))
        if b is not None and b.greedy.correct:
            opts.append(price(calls_for_action(s, b, "escalate", 0, order)))
        if opts:
            repairable += 1
            tot_perfect += min(opts)
        else:
            hopeless += 1
            tot_perfect += g                       # abstain at once
            waste_all += every - g
    return {
        "lam": lam, "n": len(qids),
        "right_now": right_now / n, "repairable": repairable / n, "unrecoverable": hopeless / n,
        "coverage_ceiling": (right_now + repairable) / n,
        "try_everything_cost": tot_all / n, "perfect_stop_cost": tot_perfect / n,
        "greedy_turn_cost": tot_greedy / n,
        "repair_waste_share": (waste_all / rep_all) if rep_all else 0.0,
        "saving_vs_try_everything": 1.0 - (tot_perfect / tot_all) if tot_all else 0.0,
    }


# ---------------------------------------------------------------------------
# D3  can two cheap samples tell the cases apart
# ---------------------------------------------------------------------------

def auroc(y, s) -> float:
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = s.argsort().argsort() + 1.0
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ci(y, s, B: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    out = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        a = auroc(y[i], s[i])
        if a == a:
            out.append(a)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (float("nan"),) * 2


def _cv_scores(X, y, folds: int = 5, seed: int = 0):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    out = np.zeros(n)
    for f in np.array_split(idx, folds):
        m = np.ones(n, bool)
        m[f] = False
        out[f] = LogisticHead(1.0).fit(X[m], y[m]).predict(X[f])
    return out


def signal_auroc(qids, small: dict, big: dict, k_draws: int = 2, n_orders: int = 30,
                 seed: int = 3, k_vote: int = 8, min_pos: int = 5) -> dict:
    """Among questions the small model got wrong, how well do signals separate
    "sampling repairs it" and "nothing repairs it"?

    Compared: greedy-only signals (free), then evidence after ``k_draws`` samples
    (cheap), then both in a cross-validated logistic head. 0.5 is chance, 1.0 is
    perfect, below 0.5 means the signal points the opposite way. The evidence and the
    label come from DISJOINT samples. Needs the greedy-wrong population to be large enough; otherwise the
    target is reported as skipped rather than estimated from a handful of cases.
    """
    qids = [q for q in _pairs(qids, small, big) if not small[q].greedy.correct]
    out = {"n_greedy_wrong": len(qids), "k_draws": k_draws, "targets": {}}
    if len(qids) < 2 * min_pos:
        out["skipped"] = "too few greedy-wrong questions"
        return out
    rng = random.Random(seed)
    Xg = np.array([features(small[q]) for q in qids])
    agree = np.zeros(len(qids))
    top = np.zeros(len(qids))
    smp = np.zeros(len(qids), bool)
    for j, q in enumerate(qids):
        s = small[q]
        canon = s.canonical_sample_answers()
        # The evidence is drawn from the first half of the samples and the label is the
        # vote over the held-out second half: the same sample must never be both the
        # signal and the answer key, or the signal looks better than it is.
        half = s.n_samples // 2
        held = list(range(half, s.n_samples))[:k_vote]
        smp[j] = bool(held) and greedy_vote_correct(s, held) >= 0.5
        a_, t_ = [], []
        for _ in range(n_orders):
            idx = rng.sample(range(half), min(k_draws, half)) if half else []
            a_.append(agreement_with_greedy(s, idx) if idx else 0.0)
            votes = [canon[i] for i in idx if canon[i] is not None]
            t_.append(Counter(votes).most_common(1)[0][1] / max(1, len(idx)) if votes else 0.0)
        agree[j], top[j] = float(np.mean(a_)), float(np.mean(t_))
    esc = np.array([bool(big[q].greedy.correct) for q in qids])
    targets = {"sampling_repairs": smp, "unrecoverable": ~(smp | esc)}
    sig = {"greedy features (cross-validated)": None,
           f"top_share after {k_draws} samples": top,
           f"agreement with greedy after {k_draws} samples": agree,
           "greedy + evidence (cross-validated)": None}
    Xp = np.column_stack([Xg, agree, top])
    for name, y in targets.items():
        if y.sum() < min_pos or (~y).sum() < min_pos:
            out["targets"][name] = {"skipped": f"{int(y.sum())}/{len(y)} positives"}
            continue
        rows = {}
        for sname, v in sig.items():
            if sname.startswith("greedy features"):
                v = _cv_scores(Xg, y)
            elif sname.startswith("greedy + evidence"):
                v = _cv_scores(Xp, y)
            lo, hi = _ci(y, v)
            rows[sname] = {"auroc": auroc(y, v), "ci95": [lo, hi]}
        out["targets"][name] = {"positives": int(y.sum()), "n": len(y), "signals": rows}
    return out


# ---------------------------------------------------------------------------
# D4  how wide is the cheap-to-repair region
# ---------------------------------------------------------------------------

def recoverable_width(qids, small: dict, big: dict, lams=(1.0, 1.5, 4.5), ks=STEPS) -> dict:
    """Share of the wrong answers that sampling repairs at no more than the
    big model's price, for each sample budget and price ratio.

    Reading: this region is where cheap repair can save money. If it is narrow
    at every budget and price, savings have to come from abstaining and from
    choosing the right step, not from sampling.
    """
    qids = _pairs(qids, small, big)
    big_model = _big_name(big)
    out = {}
    for k in ks:
        row = {}
        for lam in lams:
            t = recovery_table(qids, small, big, k=k, price=lam_cost(lam, big_model))
            row[f"{lam:g}"] = dict(t["regions"])
        out[f"k={k}"] = row
    return out


# ---------------------------------------------------------------------------
# D5  are the cut-off answers loops or just long
# ---------------------------------------------------------------------------

def repetition_score(text: str, n: int = 4, tail: int = 60) -> float:
    """Share of word n-grams in the LAST ``tail`` words that repeat an earlier one:
    ~0 for text that keeps moving, near 1 for a loop. A loop that starts late sits in
    the tail of a long answer, so a whole-text score would dilute it away. Texts with
    fewer than a few dozen words in the tail score 0."""
    w = text.split()[-tail:]
    total = len(w) - n + 1
    if total < 20:
        return 0.0
    return 1.0 - len(set(tuple(w[i:i + n]) for i in range(total))) / total


def truncation_kind(small: dict, texts: dict, threshold: float = 0.5, n_examples: int = 5) -> dict:
    """Of the greedy answers cut off by the token cap, how many are loops?

    The threshold is declared here, before looking at the data; read the listed
    examples by hand before claiming anything from the split. A loop would be
    fixed by stopping generation early (cheaper for everyone); long but
    progressing text is just a question that needs a bigger cap.
    """
    cut = [q for q, r in small.items() if r.greedy.finish_reason == "length" and q in texts]
    loops, longs = [], []
    for q in cut:
        sc = repetition_score(texts[q].get("greedy_text", ""))
        (loops if sc >= threshold else longs).append((q, sc))
    n = len(cut)
    return {"n_truncated_with_text": n, "threshold": threshold,
            "loop_share": (len(loops) / n) if n else 0.0,
            "loop_examples": [q for q, _ in sorted(loops, key=lambda x: -x[1])[:n_examples]],
            "long_examples": [q for q, _ in sorted(longs, key=lambda x: x[1])[:n_examples]]}
