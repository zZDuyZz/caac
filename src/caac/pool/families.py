"""Every method in the comparison, written as a Family for protocol.py.

All grids are fixed before any calibration label is read. Thresholds run from
conservative to liberal so a one-dimensional grid is one fixed-sequence path.

  small_abstain     small greedy, serve iff its confidence clears t
  big_abstain       big greedy only, serve iff its confidence clears t
  cascade_abstain   serve small if confident, else ask the big model and serve
                    it if confident, else abstain. Three published methods have
                    this shape and differ in the signal and the calibration,
                    not in the actions: Trust-or-Escalate (2407.18370, judges,
                    fixed-sequence testing), LEC (2512.01556, two thresholds
                    under an FDR constraint) and Fanconi & van der Schaar
                    (2506.11887, confidence then a human tier). Replayed on one
                    pool they collapse onto this family, which is why the
                    comparison varies the signal rather than adding three
                    near-identical policies.
  conformal_cascade Conformal Cascade (2607.25018): a tier commits when its
                    sampled answer set is a singleton; fixed sample count, and
                    no abstention unless asked for
  sc_abstain        k small samples, serve the vote iff its share clears t
  adaptive_abstain  ESC / Adaptive-Consistency, then the same vote-share gate
  pricecheck        PriceCheck's schedule family (2609.37493): re-solve checks
                    vote on the small model's greedy answer, which is the only
                    answer it can ever serve
  caac / caac_staged
                    ours: the cheapest action predicted safe enough; the staged
                    version re-decides after each paid step

Costs inside a policy (which action is cheapest) use ``lam_cost``: tokens of the
small model plus ``lam`` times tokens of the big one. The price ratio is a
deployment constant, so a CAAC family is built per ``lam`` -- known in advance,
never fitted to data.
"""

from __future__ import annotations

from collections import Counter

from caac.pool.ledger import Call
from caac.pool.policies import (PolicyOutcome, _greedy_call, _samples_call,
                                caac_certified, caac_staged)
from caac.pool.adaptive import adaptive_consistency, esc
from caac.pool.protocol import Family, split_cost
from caac.pool.schema import PoolRecord
from caac.pool.voting import serve_with_greedy

__all__ = ["CONF_GRID", "TAU_GRID", "SHARE_GRID", "lam_cost", "small_abstain", "big_abstain",
           "cascade_abstain", "sc_abstain", "adaptive_abstain", "conformal_cascade",
           "pricecheck", "caac", "caac_staged_family", "RHO_GRID", "oracle", "Memo"]

CONF_GRID = [0.999, 0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.0]
TAU_GRID = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.9, 0.88, 0.85, 0.8, 0.75, 0.7,
            0.6, 0.5]
SHARE_GRID = [1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.0]


def lam_cost(lam: float, big_model: str | None):
    def price(calls) -> float:
        s, b = split_cost(calls, big_model)
        return s + lam * b
    return price


def _conf(c) -> float:
    return c.confidence if c.confidence is not None else 0.0


def _abstain(calls, **meta) -> PolicyOutcome:
    return PolicyOutcome(False, None, False, calls, "abstain", dict(meta))


# --------------------------------------------------------------------------- #
# selective prediction and cascades
# --------------------------------------------------------------------------- #

def small_abstain(grid=CONF_GRID) -> Family:
    def make(t):
        def fn(s: PoolRecord, b, order):
            calls = [_greedy_call(s)]
            if _conf(s.greedy) >= t:
                return PolicyOutcome(True, s.greedy.answer, s.greedy.correct, calls, "answer_now")
            return _abstain(calls)
        return fn
    return Family("small_abstain", list(grid), make, uses_order=False)


def big_abstain(grid=CONF_GRID) -> Family:
    def make(t):
        def fn(s, b: PoolRecord, order):
            calls = [_greedy_call(b)]
            if _conf(b.greedy) >= t:
                return PolicyOutcome(True, b.greedy.answer, b.greedy.correct, calls, "escalate")
            return _abstain(calls)
        return fn
    return Family("big_abstain", list(grid), make, needs_big=True, uses_order=False)


def cascade_abstain(grid_small=CONF_GRID, grid_big=CONF_GRID) -> Family:
    """One testing path per small-model threshold, each running over the big
    model's threshold from conservative to liberal."""
    configs = [(t1, t2) for t1 in grid_small for t2 in grid_big]
    paths = [[i * len(grid_big) + j for j in range(len(grid_big))] for i in range(len(grid_small))]

    def make(cfg):
        t1, t2 = cfg

        def fn(s, b, order):
            if _conf(s.greedy) >= t1:
                return PolicyOutcome(True, s.greedy.answer, s.greedy.correct,
                                     [_greedy_call(s)], "answer_now")
            calls = [_greedy_call(s, 0), _greedy_call(b, 1)]
            if _conf(b.greedy) >= t2:
                return PolicyOutcome(True, b.greedy.answer, b.greedy.correct, calls, "escalate")
            return _abstain(calls)
        return fn
    return Family("cascade_abstain", configs, make, paths, needs_big=True, uses_order=False)


def _share(labels: list) -> tuple[str | None, float]:
    votes = [v for v in labels if v is not None]
    if not votes:
        return None, 0.0
    top, n = Counter(votes).most_common(1)[0]
    return top, n / len(labels)


def sc_abstain(ks=(1, 2, 4, 8, 16), grid=SHARE_GRID, greedy_vote: bool = False) -> Family:
    """Fixed-k self-consistency; serve the vote iff its share of the k draws
    clears t. One path per k. ``greedy_vote`` gives it CAAC's selector (greedy
    counts as a vote) so the comparison does not hinge on the voter."""
    configs = [(k, t) for k in ks for t in grid]
    paths = [[i * len(grid) + j for j in range(len(grid))] for i in range(len(ks))]

    def make(cfg):
        k, t = cfg

        def fn(s: PoolRecord, b, order):
            idxs = [i for i in order if i < s.n_samples][:k]
            calls = [_samples_call(s, idxs, 0)]
            canon = s.canonical_sample_answers()
            top, share = _share([canon[i] for i in idxs])
            if greedy_vote:
                # the greedy answer is a vote, so its turn is billed
                calls = [_greedy_call(s, 0), _samples_call(s, idxs, 1)]
                top = serve_with_greedy(s, idxs)
                share = (sum(canon[i] == top for i in idxs) + (s.greedy_label() == top)) / (k + 1)
            if len(idxs) == k and top is not None and share >= t:
                return PolicyOutcome(True, top, s.label_correct(top), calls, "sample_more")
            return _abstain(calls)
        return fn
    return Family("sc_abstain_greedy" if greedy_vote else "sc_abstain", configs, make, paths)


def conformal_cascade(n_small: int = 16, n_big: int = 16, grid=SHARE_GRID,
                      abstain_last: bool = False) -> Family:
    """Conformal Cascade (2607.25018): a tier is trusted when its answer set is
    a singleton.

    Each tier draws a FIXED number of samples and scores an answer class by
    1 - its frequency; the prediction set keeps the classes scoring at or below
    the calibrated threshold, which here is the grid over the frequency cut.
    One class in the set means the tier commits; otherwise the query moves up.

    Two differences from CAAC are the comparison, so they are kept exactly:
    the sample count never adapts, and the last tier has no abstention -- it
    answers whatever it has. That caps how small an alpha the family can be
    certified at, which is the point of putting it on the same axes.
    ``abstain_last`` enables the natural extension (abstain when the big tier
    is not a singleton either) so the family also has a risk-coverage curve;
    the faithful version is the default.
    """
    def make(cut):
        def fn(s: PoolRecord, b, order):
            si = [i for i in order if i < s.n_samples][:n_small]
            calls = [_samples_call(s, si, 0)]
            canon = s.canonical_sample_answers()
            top, share = _share([canon[i] for i in si])
            if top is not None and share >= cut:
                return PolicyOutcome(True, top, s.label_correct(top), calls, "sample_more",
                                     {"tier": "small", "share": share})
            bi = [i for i in (order or []) if i < b.n_samples][:n_big]
            # the big tier falls back to its greedy turn when the pool holds no
            # samples for it, so the baseline is runnable on either pool shape
            calls.append(_samples_call(b, bi, 1) if bi else _greedy_call(b, 1))
            if bi:
                btop, bshare = _share([b.canonical_sample_answers()[i] for i in bi])
            else:
                btop, bshare = b.greedy_label(), 1.0
            if abstain_last and not (btop is not None and bshare >= cut):
                return _abstain(calls, tier="big", share=bshare)
            return PolicyOutcome(True, btop, b.label_correct(btop), calls, "escalate",
                                 {"tier": "big", "share": bshare})
        return fn
    name = "conformal_cascade_abstain" if abstain_last else "conformal_cascade"
    return Family(name, list(grid), make, needs_big=True)


def adaptive_abstain(kind: str = "esc", grid=SHARE_GRID, max_k: int = 16) -> Family:
    """ESC or Adaptive-Consistency decides how many to draw; the vote is served
    iff its share of the drawn samples clears t. Neither method can abstain on
    its own, so this gate is what puts them under the same certificate."""
    run = {"esc": lambda s, o: esc(s, o, window=4, max_k=max_k),
           "adaptive_consistency": lambda s, o: adaptive_consistency(s, o, max_k=max_k)}[kind]

    def make(t):
        def fn(s: PoolRecord, b, order):
            out = run(s, order)
            drawn = [i for i in order if i < s.n_samples][:out.meta.get("n_drawn", 0)]
            canon = s.canonical_sample_answers()
            share = (sum(canon[i] == out.answer for i in drawn) / len(drawn)) if drawn else 0.0
            if out.answer is not None and share >= t:
                return out
            return _abstain(out.calls)
        return fn
    return Family(f"{kind}_abstain", list(grid), make)


# --------------------------------------------------------------------------- #
# PriceCheck: checks vote on the small greedy answer; it is the only answer served
# --------------------------------------------------------------------------- #

def _big_agrees(s: PoolRecord, b: PoolRecord) -> list[bool]:
    """Does each big-model re-solve agree with the small greedy answer?
    Answers are compared with each other (label-free), cached per pair."""
    cache = s.__dict__.setdefault("_big_agree", {})
    key = (b.model, b.query_id)
    if key not in cache:
        from caac.eval.grade import answer_clusters
        g = s.greedy
        cand = (g.answer_raw or g.answer) if g.answer is not None else None
        raw = [(c.answer_raw or c.answer) if c.answer is not None else None for c in b.samples]
        ids = answer_clusters([cand] + raw)
        cache[key] = [ids[0] is not None and i == ids[0] for i in ids[1:]]
    return cache[key]


def _small_agrees(s: PoolRecord, idxs: list[int]) -> list[bool]:
    g = s.greedy_label()
    canon = s.canonical_sample_answers()
    return [g is not None and canon[i] == g for i in idxs]


def _serve_candidate(s: PoolRecord, calls, **meta) -> PolicyOutcome:
    return PolicyOutcome(True, s.greedy.answer, s.greedy.correct, calls, "answer_now", dict(meta))


def _race(s: PoolRecord, b: PoolRecord, k: int, j: int, calls: list, rnd: int) -> PolicyOutcome:
    """Draw big re-solves one at a time: serve at k agreements, abstain at j
    disagreements or when the pool runs out. Each draw is its own round."""
    agree = _big_agrees(s, b)
    a = d = 0
    for i, ok in enumerate(agree):
        calls.append(Call(b.model, b.prompt_tokens, [b.samples[i].decode_tokens], rnd))
        rnd += 1
        a, d = a + ok, d + (not ok)
        if a >= k:
            return _serve_candidate(s, calls, race=(k, j), draws=i + 1)
        if d >= j:
            break
    return _abstain(calls, race=(k, j))


def pricecheck(n_big: int = 8) -> Family:
    """The deployed-family analogue of PriceCheck: unanimity and threshold rules
    on small re-solves, unanimity on big re-solves, adaptive races on big
    re-solves, and small-first cascades into a race. Tested as unordered rules
    (Bonferroni over the family), as in the paper. Rules needing more big draws
    than the calibration pool holds are left out, so every rule is the same at
    calibration and at test time."""
    configs = [("unanimous_small", n) for n in (2, 3, 4, 8)]
    configs += [("threshold_small", m, 8) for m in (5, 6, 7)]
    configs += [("unanimous_big", n) for n in (2, 3, 4) if n <= n_big]
    races = [(k, j) for k in (2, 3, 4) for j in (1, 2, 3) if k + j - 1 <= n_big]
    configs += [("race", k, j) for k, j in races]
    configs += [("small_first", k, j) for k, j in races]

    def make(cfg):
        kind = cfg[0]

        def fn(s: PoolRecord, b: PoolRecord, order):
            calls = [_greedy_call(s, 0)]
            if kind in ("unanimous_small", "threshold_small"):
                n = cfg[1] if kind == "unanimous_small" else cfg[2]
                m = n if kind == "unanimous_small" else cfg[1]
                idxs = [i for i in order if i < s.n_samples][:n]
                calls.append(_samples_call(s, idxs, 1))
                if len(idxs) == n and sum(_small_agrees(s, idxs)) >= m:
                    return _serve_candidate(s, calls)
                return _abstain(calls)
            if kind == "unanimous_big":
                n = cfg[1]
                idxs = list(range(min(n, b.n_samples)))
                calls.append(_samples_call(b, idxs, 1))
                if len(idxs) == n and all(_big_agrees(s, b)[:n]):
                    return _serve_candidate(s, calls)
                return _abstain(calls)
            if kind == "race":
                return _race(s, b, cfg[1], cfg[2], calls, 1)
            # small_first: three cheap votes settle the clear cases
            idxs = [i for i in order if i < s.n_samples][:3]
            calls.append(_samples_call(s, idxs, 1))
            m = sum(_small_agrees(s, idxs))
            if len(idxs) == 3 and m == 3:
                return _serve_candidate(s, calls, decided_by="small")
            if m == 0:
                return _abstain(calls, decided_by="small")
            return _race(s, b, cfg[1], cfg[2], calls, 2)
        return fn
    paths = [[i] for i in range(len(configs))]
    return Family("pricecheck", configs, make, paths, needs_big=True)


# --------------------------------------------------------------------------- #
# CAAC
# --------------------------------------------------------------------------- #

class Memo:
    """Per-record cache for an estimator, so a grid replay scores each question once."""

    def __init__(self, fn):
        self.fn, self.cache = fn, {}

    def __call__(self, s: PoolRecord, b: PoolRecord | None = None):
        key = (s.model, s.query_id)
        if key not in self.cache:
            self.cache[key] = self.fn(s, b)
        return self.cache[key]


def caac(est, lam: float, big_model: str | None, grid=TAU_GRID,
         actions=("answer_now", "sample_more", "escalate"), k: int = 8,
         name: str = "caac") -> Family:
    price = lam_cost(lam, big_model)
    scorer = est if isinstance(est, Memo) else Memo(est)

    def make(tau):
        def fn(s, b, order):
            return caac_certified(s, b, scorer, tau=tau, order=order, k=k,
                                  actions=actions, cost_fn=price)
        return fn
    return Family(name, list(grid), make, needs_big="escalate" in actions)


# 1.01 buys nothing, so the family contains plain answer-or-abstain as a special case
RHO_GRID = (0.0, 0.3, 0.6, 0.8, 0.9, 1.01)


def caac_staged_family(est, est2, lam: float, big_model: str | None, grid=TAU_GRID,
                       actions=("answer_now", "sample_more", "escalate"),
                       k_steps=(2, 4, 8, 16), name: str = "caac_staged", est3=None,
                       rhos=RHO_GRID, rho_mode: str = "phat", recovery=None) -> Family:
    """Configurations (rho, tau): one fixed-sequence path over tau per rho, so
    delta is split over the few rho values only.

    ``rho_mode`` selects what the rho gate means (see ``policies.RHO_MODES``).
    "delta" and "value" need ``recovery``, a RecoveryEstimator, and are the arm
    that abstains before buying compute that cannot repair the answer; "phat"
    is the confidence-shaped arm it is compared against. ``rhos`` is read on
    the gate's own scale, so a net-value gate wants a grid around zero.
    """
    price = lam_cost(lam, big_model)
    scorer = est if isinstance(est, Memo) else Memo(est)
    rec = recovery if recovery is None or isinstance(recovery, Memo) else Memo(recovery)
    gated = est3 is not None or (rho_mode != "phat" and rec is not None)
    rhos = tuple(rhos) if gated else (None,)
    configs = [(r, t) for r in rhos for t in grid]
    paths = [[i * len(grid) + j for j in range(len(grid))] for i in range(len(rhos))]

    def make(cfg):
        rho, tau = cfg

        def fn(s, b, order):
            return caac_staged(s, b, scorer, est2, tau=tau, order=order, k=max(k_steps),
                               actions=actions, k_steps=tuple(k_steps), cost_fn=price,
                               escalate_gate=est3, rho=rho, rho_mode=rho_mode,
                               recovery=rec)
        return fn
    return Family(name, configs, make, paths, needs_big="escalate" in actions)


# --------------------------------------------------------------------------- #
# oracle: an upper bound, not a method (it reads the labels)
# --------------------------------------------------------------------------- #

def oracle(qids, small, big, lam: float, k_steps=(2, 4, 8, 16)) -> dict:
    """Cheapest action that could have been right, else abstain (paying the
    greedy turn). Sampling counts as right when ANY usable draw among the first
    k is right -- a perfect selector -- so coverage is a ceiling for every
    method, whatever its voter. Cost is for the generation order and is a
    reference, not a bound on a permuted replay."""
    served = cost = 0.0
    for q in qids:
        s, b = small[q], (big or {}).get(q)
        price = lam_cost(lam, b.model if b is not None else None)
        g = [_greedy_call(s, 0)]
        opts = []
        if s.greedy.correct:
            opts.append(price(g))
        for k in k_steps:
            idxs = list(range(min(k, s.n_samples)))
            canon = s.canonical_sample_answers()
            if any(s.label_correct(canon[i]) for i in idxs):
                opts.append(price(g + [_samples_call(s, idxs, 1)]))
        if b is not None and b.greedy.correct:
            opts.append(price(g + [_greedy_call(b, 1)]))
        elif b is not None and any(b.label_correct(a) for a in b.canonical_sample_answers()):
            # a cascade that votes over big-model draws can be right where its greedy is not
            opts.append(price(g + [_samples_call(b, list(range(b.n_samples)), 1)]))
        if opts:
            served += 1
            cost += min(opts)
        else:
            cost += price(g)
    n = max(1, len(qids))
    return {"coverage": served / n, "selective_risk": 0.0, "cost": cost / n}
