"""CAAC-Audit: a certified verdict on "is a small-model-first policy worth deploying here?"

Claim of interest, for a finite set of candidate policies Pi fixed BEFORE the audit data
is read:   exists pi in Pi with  R_pi <= alpha  and  saving_pi >= s,
where R_pi = P(error | served) is the selective risk and
saving_pi = 1 - E[C_pi] / E[B], B = cost of answering with the big model only.
The saving condition is used in linear form, E[d_pi] >= 0 with d_pi = (1-s) B - C_pi,
so no ratio of estimates appears.

Verdicts (each wrong with probability <= delta, under exchangeable questions and costs
bounded in [0, c_max]):
  GO           split the questions into A/B by a salted hash of qid; choose pi_hat on A
               only (cheapest policy whose empirical risk on A is <= alpha); on B reject
               "R > alpha" (exact binomial) AND "E[d] < 0" (empirical Bernstein), EACH at
               level delta. An intersection-union test: when the claim is false for
               pi_hat at least one null is true, so P(GO) <= delta.
  NO-GO        on all questions, for EVERY pi reject {R <= alpha and E[d] >= 0} -- by the
               binomial test of "R <= alpha" OR the Bernstein test of "E[d] >= 0", each at
               delta/2 (a union of two rejections). If some pi* satisfies the claim, NO-GO
               needs rejecting a true null for pi*, so P(NO-GO) <= delta. No multiplicity
               correction over Pi is needed.
  INCONCLUSIVE otherwise, and also when both GO and NO-GO fire (flagged "conflict";
               this happens with probability <= delta).
Only the standard library is used, so the numbers can be checked by hand.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

DEFAULT_SALT = "caac-audit-v1"


# ----------------------------------------------------------------------------- tests
def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Bin(n, p). Exact sum in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    logs = [_log_binom_pmf(i, n, p) for i in range(k + 1)]
    m = max(logs)
    if m == -math.inf:
        return 0.0
    return min(1.0, math.exp(m) * sum(math.exp(x - m) for x in logs))


def p_risk_below(errors: int, served: int, alpha: float) -> float:
    """p-value against H0: R > alpha (small = evidence R <= alpha). P(Bin(served, alpha) <= errors)."""
    return 1.0 if served == 0 else binom_cdf(errors, served, alpha)


def p_risk_above(errors: int, served: int, alpha: float) -> float:
    """p-value against H0: R <= alpha (small = evidence R > alpha). P(Bin(served, alpha) >= errors)."""
    return 1.0 if served == 0 else 1.0 - binom_cdf(errors - 1, served, alpha)


def _bernstein_slack(var: float, n: int, width: float, d: float) -> float:
    """Maurer & Pontil (2009, Thm 4): with prob >= 1-d, E[X] <= mean + slack (one side)."""
    L = math.log(2.0 / d)
    return math.sqrt(2.0 * var * L / n) + 7.0 * width * L / (3.0 * (n - 1))


def p_mean_nonneg(x: list[float], lo: float, hi: float) -> float:
    """p-value against H0: E[x] < 0 (small = evidence E[x] >= 0), x in [lo, hi].
    = smallest d such that the lower bound mean - slack(d) is >= 0."""
    return _bern_p([-v for v in x], -hi, -lo)


def p_mean_neg(x: list[float], lo: float, hi: float) -> float:
    """p-value against H0: E[x] >= 0 (small = evidence E[x] < 0), x in [lo, hi]."""
    return _bern_p(x, lo, hi)


def _bern_p(x: list[float], lo: float, hi: float) -> float:
    """Smallest d in (0,1] with mean(x) + slack(d) < 0 (evidence E[x] < 0); 1.0 if none."""
    n = len(x)
    if n < 2:
        return 1.0
    mean = sum(x) / n
    if mean >= 0:
        return 1.0
    var = sum((v - mean) ** 2 for v in x) / (n - 1)
    w = hi - lo
    ok = lambda d: mean + _bernstein_slack(var, n, w, d) < 0
    if not ok(1.0):
        return 1.0
    lo_d, hi_d = 1e-300, 1.0
    for _ in range(200):
        mid = math.sqrt(lo_d * hi_d)
        if ok(mid):
            hi_d = mid
        else:
            lo_d = mid
        if hi_d / lo_d < 1 + 1e-9:
            break
    return hi_d


# ----------------------------------------------------------------------------- split
def in_half_a(qid: str, salt: str = DEFAULT_SALT) -> bool:
    """Fixed, data-independent A/B assignment: first byte of sha256(salt|qid) is even."""
    return hashlib.sha256(f"{salt}|{qid}".encode()).digest()[0] % 2 == 0


# ----------------------------------------------------------------------------- audit
@dataclass
class PolicyData:
    """Per-question outcome of one candidate policy, aligned with AuditInput.qids."""
    name: str
    served: list[bool]
    error: list[bool]        # True only if served and wrong
    cost: list[float]


@dataclass
class AuditInput:
    qids: list[str]
    big_cost: list[float]    # cost of answering every question with the big model only
    policies: list[PolicyData]
    c_max: float             # upper bound of every cost (e.g. cap x price per token)


@dataclass
class AuditResult:
    verdict: str                          # GO | NO-GO | INCONCLUSIVE
    alpha: float
    delta: float
    s: float
    n_questions: int
    n_a: int
    n_b: int
    chosen: str | None = None             # pi_hat chosen on A (GO path)
    go: dict = field(default_factory=dict)
    nogo: dict = field(default_factory=dict)
    conflict: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("verdict", "alpha", "delta", "s", "n_questions", "n_a",
                                              "n_b", "chosen", "go", "nogo", "conflict", "notes")}


def _check(inp: AuditInput) -> None:
    n = len(inp.qids)
    if len(set(inp.qids)) != n:
        raise ValueError("duplicate qids")
    if len(inp.big_cost) != n:
        raise ValueError("big_cost length differs from qids")
    if not inp.policies:
        raise ValueError("no candidate policies")
    for p in inp.policies:
        if not (len(p.served) == len(p.error) == len(p.cost) == n):
            raise ValueError(f"policy {p.name}: arrays differ in length from qids")
        if any(e and not s for e, s in zip(p.error, p.served)):
            raise ValueError(f"policy {p.name}: an error on a question it did not serve")
    for v in list(inp.big_cost) + [c for p in inp.policies for c in p.cost]:
        if not (0.0 <= v <= inp.c_max) or math.isnan(v):
            raise ValueError(f"cost {v} outside [0, c_max={inp.c_max}]: costs must be bounded "
                             "(cut costs only DOWNWARD for B and never for C, or raise c_max)")


def _stats(p: PolicyData, idx: list[int], big: list[float], s: float):
    served = sum(p.served[i] for i in idx)
    errors = sum(p.error[i] for i in idx)
    d = [(1.0 - s) * big[i] - p.cost[i] for i in idx]
    mean_c = sum(p.cost[i] for i in idx) / len(idx) if idx else float("nan")
    mean_b = sum(big[i] for i in idx) / len(idx) if idx else float("nan")
    return served, errors, d, mean_c, mean_b


def audit(inp: AuditInput, *, alpha: float = 0.10, delta: float = 0.10, s: float = 0.20,
          salt: str = DEFAULT_SALT) -> AuditResult:
    _check(inp)
    n = len(inp.qids)
    lo, hi = -inp.c_max, (1.0 - s) * inp.c_max          # range of d = (1-s)B - C
    A = [i for i, q in enumerate(inp.qids) if in_half_a(q, salt)]
    B = [i for i, q in enumerate(inp.qids) if not in_half_a(q, salt)]
    res = AuditResult("INCONCLUSIVE", alpha, delta, s, n, len(A), len(B))

    # ---- GO
    cands = []
    for j, p in enumerate(inp.policies):
        served, errors, _, mean_c, _ = _stats(p, A, inp.big_cost, s)
        if served and errors / served <= alpha:
            cands.append((mean_c, j))
    go = False
    if cands and len(B) >= 2:
        j = min(cands)[1]
        p = inp.policies[j]
        served, errors, d, mean_c, mean_b = _stats(p, B, inp.big_cost, s)
        pr = p_risk_below(errors, served, alpha)
        ps = p_mean_nonneg(d, lo, hi)
        go = pr <= delta and ps <= delta
        res.chosen = p.name
        res.go = {"policy": p.name, "served_b": served, "errors_b": errors,
                  "risk_b": round(errors / served, 6) if served else None,
                  "saving_b": round(1 - mean_c / mean_b, 6) if mean_b else None,
                  "p_risk": pr, "p_saving": ps, "level_each_leg": delta, "passed": go}
    else:
        res.go = {"policy": None, "passed": False,
                  "reason": "no policy with empirical risk <= alpha on half A" if not cands else "half B too small"}

    # ---- NO-GO
    rows = []
    all_rejected = True
    for p in inp.policies:
        served, errors, d, mean_c, mean_b = _stats(p, list(range(n)), inp.big_cost, s)
        pr = p_risk_above(errors, served, alpha)
        ps = p_mean_neg(d, lo, hi)
        rej = pr <= delta / 2 or ps <= delta / 2
        all_rejected &= rej
        rows.append({"policy": p.name, "served": served, "errors": errors,
                     "risk": round(errors / served, 6) if served else None,
                     "saving": round(1 - mean_c / mean_b, 6) if mean_b else None,
                     "p_risk_above": pr, "p_saving_below": ps, "rejected": rej})
    res.nogo = {"level_each_leg": delta / 2, "policies": rows, "passed": all_rejected}

    if go and all_rejected:
        res.verdict, res.conflict = "INCONCLUSIVE", True
        res.notes.append("GO and NO-GO both fired (probability <= delta): reported as INCONCLUSIVE")
    elif go:
        res.verdict = "GO"
    elif all_rejected:
        res.verdict = "NO-GO"
    return res


def served_needed(rate: float, alpha: float, delta: float, n_max: int = 200_000) -> int | None:
    """Served answers needed on the tested half for the risk leg to pass IF the error rate
    stays at `rate` (an ESTIMATE, not a guarantee). None if rate >= alpha or n_max exceeded."""
    if rate >= alpha:
        return None
    lo, hi = 1, 1
    while hi <= n_max and binom_cdf(math.floor(rate * hi), hi, alpha) > delta:
        lo, hi = hi, hi * 2
    if hi > n_max:
        return None
    while lo < hi:                      # first n that passes; cdf is not monotone in n, so
        mid = (lo + hi) // 2            # this is the first pass found by bisection on a
        if binom_cdf(math.floor(rate * mid), mid, alpha) <= delta:   # doubling bracket
            hi = mid
        else:
            lo = mid + 1
    return hi
