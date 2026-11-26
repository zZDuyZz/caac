"""T2: shared simulation bench (known truth) for the CAAC audit and its candidate upgrades.

Uses the SAME test functions as the frozen v2 code (caac.pool_v1.audit): exact binomial for the
risk leg, test by betting for the saving leg. Nothing here changes the lock.

Data model per question i (matches replay.py, on_unsure = escalate):
  small cost  a_i  (bounded, possibly heavy tailed), big cost b_i (same family)
  policy k = (c_k, q_k, R_k): the small model answers a share q_k; the rest is escalated and
              answered by the big model (error R_BIG); every question is served;
              small-answer error r_s chosen so that the system error is exactly R_k;
              cost = c_k * a_i  (+ b_i if escalated)
  truth:      R_k (system error),  saving_k = 1 - (c_k E[a] + (1 - q_k) E[b]) / E[b]
Claim of interest at (alpha, s): exists k with R_k <= alpha and saving_k >= s.

Variants compared (all GO procedures; NO-GO is the v2 one for all):
  v2       split A/B, pick the cheapest policy with risk <= alpha on A, test it on B, each leg at delta
  bonf     no split: every policy tested on ALL questions, each leg at delta/|Pi|
  xfit     v2 run twice (A->B and B->A), each at delta/2; GO if either passes
  top3     split A/B, keep the 3 cheapest policies with risk <= alpha on A, test each on B at delta/3
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pchk6", "src"))
from caac.pool_v1.audit import (_betting_p, _order, in_half_a, p_risk_above,  # noqa: E402
                                p_risk_below, DEFAULT_SALT)

ALPHA, DELTA, S = 0.10, 0.10, 0.20
R_BIG = 0.03     # error rate of the big model on escalated questions


# ----------------------------------------------------------------------------- data
class Truth:
    """A family of policies. cost_kind: 'uniform' (light tail) or 'lognorm' (heavy tail, clipped)."""

    def __init__(self, pols, cost_kind="uniform", ratio=30.0, mean_a=1.0, mean_b=3.0):
        self.pols = pols                      # list of (c_k, q_k, r_k)
        self.kind, self.ratio = cost_kind, ratio
        self.mean_a_param, self.mean_b_param = mean_a, mean_b
        # c_max: every policy cost <= max(c_k) * amax + bmax
        if cost_kind == "lognorm":
            self.amax, self.bmax = ratio * mean_a, ratio * mean_b
        else:
            self.amax, self.bmax = 2.0 * mean_a, 2.0 * mean_b
        rng = random.Random(12345)
        A = [self._draw(rng, mean_a, self.amax) for _ in range(400_000)]
        Bv = [self._draw(rng, mean_b, self.bmax) for _ in range(400_000)]
        self.Ea, self.Eb = sum(A) / len(A), sum(Bv) / len(Bv)
        self.c_max = max(c for c, _, _ in pols) * self.amax + self.bmax

    def _draw(self, rng, mean, cap):
        if self.kind == "uniform":
            return rng.uniform(0, 2 * mean)
        return min(cap, rng.lognormvariate(math.log(mean) - 0.8, 1.265))   # sigma^2/2 = .8

    def risk(self, k):
        return self.pols[k][2]

    def saving(self, k):
        c, q, _ = self.pols[k]
        return 1.0 - (c * self.Ea + (1 - q) * self.Eb) / self.Eb

    def claim(self, alpha=ALPHA, s=S):
        return any(self.risk(k) <= alpha + 1e-12 and self.saving(k) >= s - 1e-12 for k in range(len(self.pols)))

    def sample(self, n, rng, prefix="q"):
        qids = [f"{prefix}{rng.getrandbits(48):012x}{i}" for i in range(n)]
        a = [self._draw(rng, self.mean_a_param, self.amax) for _ in range(n)]
        b = [self._draw(rng, self.mean_b_param, self.bmax) for _ in range(n)]
        pols = []
        for c, q, R in self.pols:
            # as in replay.py with on_unsure = escalate: every question is served; the small model
            # answers a share q (error r_s), the rest go to the big model (error R_BIG).
            r_s = (R - (1 - q) * R_BIG) / q
            assert 0 <= r_s <= 1, (c, q, R)
            small = [rng.random() < q for _ in range(n)]
            served = [True] * n
            err = [(rng.random() < r_s) if small[i] else (rng.random() < R_BIG) for i in range(n)]
            cost = [c * a[i] + (0.0 if small[i] else b[i]) for i in range(n)]
            pols.append((served, err, cost))
        return {"qids": qids, "big": b, "pols": pols, "c_max": self.c_max,
                "a_part_max": max(c for c, _, _ in self.pols) * self.amax, "b_part_max": self.bmax}


# ----------------------------------------------------------------------------- legs
def d_range(data, s, structural):
    """Range of d = (1-s)B - C. Global (v2): [-c_max, (1-s) c_max]. Structural (U2): an escalated
    question costs small part + exactly the big-only cost of that question, so
    d in [-(a_max + s b_max), (1-s) b_max]   (a_max: small-part bound, b_max: big-part bound)."""
    if not structural:
        return -data["c_max"], (1 - s) * data["c_max"]
    return -(data["a_part_max"] + s * data["b_part_max"]), (1 - s) * data["b_part_max"]


STRUCT = False   # set by the runner


def legs(data, k, idx, alpha=ALPHA, s=S, salt=DEFAULT_SALT):
    """(p_risk, p_saving) for policy k on question subset idx; p small = evidence for the claim."""
    served, err, cost = data["pols"][k]
    sv = sum(served[i] for i in idx)
    e = sum(err[i] for i in idx)
    d = [(1 - s) * data["big"][i] - cost[i] for i in idx]
    lo, hi = d_range(data, s, STRUCT)
    assert min(d) >= lo - 1e-9 and max(d) <= hi + 1e-9
    pr = p_risk_below(e, sv, alpha)
    ps = _betting_p(d, lo, hi, _order([data["qids"][i] for i in idx], salt))
    return pr, ps


def mean_cost(data, k, idx):
    c = data["pols"][k][2]
    return sum(c[i] for i in idx) / len(idx)


def emp_risk(data, k, idx):
    served, err, _ = data["pols"][k]
    sv = sum(served[i] for i in idx)
    return (sum(err[i] for i in idx) / sv) if sv else None


def nogo(data, alpha=ALPHA, s=S, delta=DELTA, salt=DEFAULT_SALT):
    n = len(data["qids"]); idx = list(range(n))
    lo, hi = d_range(data, s, STRUCT)
    order = _order(data["qids"], salt)
    for k, (served, err, cost) in enumerate(data["pols"]):
        sv, e = sum(served), sum(err)
        d = [(1 - s) * data["big"][i] - cost[i] for i in idx]
        pr = p_risk_above(e, sv, alpha)
        ps = _betting_p([-v for v in d], -hi, -lo, order)
        if not (pr <= delta / 2 or ps <= delta / 2):
            return False
    return True


def split(data, salt=DEFAULT_SALT):
    A = [i for i, q in enumerate(data["qids"]) if in_half_a(q, salt)]
    B = [i for i, q in enumerate(data["qids"]) if not in_half_a(q, salt)]
    return A, B


def candidates_on(data, idx, alpha):
    out = []
    for k in range(len(data["pols"])):
        r = emp_risk(data, k, idx)
        if r is not None and r <= alpha:
            out.append((mean_cost(data, k, idx), k))
    return sorted(out)


# ----------------------------------------------------------------------------- GO variants
def go_v2(data, alpha=ALPHA, s=S, delta=DELTA):
    A, B = split(data)
    c = candidates_on(data, A, alpha)
    if not c or len(B) < 2:
        return None
    k = c[0][1]
    pr, ps = legs(data, k, B, alpha, s)
    return k if (pr <= delta and ps <= delta) else None


def go_bonf(data, alpha=ALPHA, s=S, delta=DELTA):
    idx = list(range(len(data["qids"])))
    m = len(data["pols"])
    best = None
    for k in range(m):
        r = emp_risk(data, k, idx)
        if r is None or r > alpha:          # cannot pass the binomial leg; skip (saves time, same result)
            continue
        pr, ps = legs(data, k, idx, alpha, s)
        if pr <= delta / m and ps <= delta / m:
            mc = mean_cost(data, k, idx)
            if best is None or mc < best[0]:
                best = (mc, k)
    return None if best is None else best[1]


def go_xfit(data, alpha=ALPHA, s=S, delta=DELTA):
    A, B = split(data)
    for sel, test in ((A, B), (B, A)):
        c = candidates_on(data, sel, alpha)
        if c and len(test) >= 2:
            k = c[0][1]
            pr, ps = legs(data, k, test, alpha, s)
            if pr <= delta / 2 and ps <= delta / 2:
                return k
    return None


def go_topm(data, alpha=ALPHA, s=S, delta=DELTA, m=3):
    A, B = split(data)
    c = candidates_on(data, A, alpha)[:m]
    if not c or len(B) < 2:
        return None
    mm = len(c)
    for _, k in c:
        pr, ps = legs(data, k, B, alpha, s)
        if pr <= delta / mm and ps <= delta / mm:
            return k
    return None


VARIANTS = {"v2": go_v2, "bonf": go_bonf, "xfit": go_xfit, "top3": go_topm}


def verdict(data, go_fn, alpha=ALPHA, s=S, delta=DELTA):
    g = go_fn(data, alpha, s, delta)
    ng = nogo(data, alpha, s, delta)
    if g is not None and ng:
        return "INCONCLUSIVE"
    return "GO" if g is not None else ("NO-GO" if ng else "INCONCLUSIVE")
