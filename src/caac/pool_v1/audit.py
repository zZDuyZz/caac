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


# ------------------------------------------------- alternative cost test: betting
def _betting_p(x: list[float], lo: float, hi: float, order: list[int] | None = None) -> float:
    """p-value against H0: E[x] <= 0 (small = evidence E[x] > 0), x in [lo, hi], lo < 0 < hi.

    Test by betting (Waudby-Smith & Ramdas, JRSS-B 2024): wealth K_t = prod(1 + l_i x_i)
    with a PREDICTABLE bet l_i in [0, 0.5/|lo|] (depends on x_1..x_{i-1} only), so
    1 + l_i x_i >= 0.5 > 0. Under H0 (i.i.d., mean <= 0) K is a nonnegative
    supermartingale; by Ville's inequality P(max_t K_t >= 1/d) <= d, so p = 1/max K.
    The order is fixed and data-independent (the caller passes a hash order); the bound
    uses only the range, not a variance estimate, and adapts to small variance."""
    n = len(x)
    if n < 2 or lo >= 0 or hi <= 0:
        return 1.0
    idx = order if order is not None else list(range(n))
    lmax = 0.5 / (-lo)
    logk, best = 0.0, 0.0
    s1 = s2 = 0.0
    m = 0
    for i in idx:
        mu = s1 / m if m else 0.0
        var = (s2 / m - mu * mu) if m else 0.0
        lam = 0.0 if mu <= 0 else min(lmax, mu / (var + mu * mu + 1e-12))
        logk += math.log1p(lam * x[i])
        best = max(best, logk)
        s1 += x[i]; s2 += x[i] * x[i]; m += 1
    return min(1.0, math.exp(-best))


def _order(qids: list[str], salt: str) -> list[int]:
    """A fixed, data-independent order of the questions (hash of qid)."""
    return sorted(range(len(qids)), key=lambda i: hashlib.sha256(f"order|{salt}|{qids[i]}".encode()).hexdigest())


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
    small_part: list[float] | None = None   # v3: cost of the small model's draws (cost - small_part is 0 or the big-only cost)


@dataclass
class AuditInput:
    qids: list[str]
    big_cost: list[float]    # cost of answering every question with the big model only
    policies: list[PolicyData]
    c_max: float             # upper bound of every cost (e.g. cap x price per token)
    small_max: float | None = None   # v3: bound of the small part (c_max = small_max + big_max)
    big_max: float | None = None     # v3: bound of the big-only cost of one question
    big_error: list[bool] | None = None  # relative claim: big model's draw 0 wrong (the big-only baseline)


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
          salt: str = DEFAULT_SALT, cost_test: str = "bernstein") -> AuditResult:
    """cost_test: "bernstein" (empirical Bernstein, default) or "betting" (test by betting).
    Both are valid at the stated level; which one is used must be fixed in advance."""
    _check(inp)
    if cost_test not in ("bernstein", "betting"):
        raise ValueError("cost_test must be 'bernstein' or 'betting'")
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
        if cost_test == "betting":   # evidence E[d] > 0 (boundary E[d] = 0 counted as the null)
            ps = _betting_p(d, lo, hi, _order([inp.qids[i] for i in B], salt))
        else:
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
        if cost_test == "betting":   # evidence E[d] < 0: bet on -d
            ps = _betting_p([-v for v in d], -hi, -lo, _order(inp.qids, salt))
        else:
            ps = p_mean_neg(d, lo, hi)
        rej = pr <= delta / 2 or ps <= delta / 2
        all_rejected &= rej
        rows.append({"policy": p.name, "served": served, "errors": errors,
                     "risk": round(errors / served, 6) if served else None,
                     "saving": round(1 - mean_c / mean_b, 6) if mean_b else None,
                     "p_risk_above": pr, "p_saving_below": ps, "rejected": rej})
    res.nogo = {"level_each_leg": delta / 2, "policies": rows, "passed": all_rejected}
    res.notes.append(f"cost test: {cost_test}")

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


# =============================================================================== design v3
# Amendment v3 (see protocols/audit_lock_v3.json). Same claim, same candidate set, same tests;
# three changes: (U1) no A/B split: every policy is tested on ALL questions, each leg at delta/|Pi|
# (Bonferroni over Pi, i.e. Learn-then-Test with two risks per policy: the selective risk and the
# mean of d); (U2) d lives in a STRUCTURAL range: an escalated question costs its small part plus
# exactly its big-only cost, so d is in [-(a_max + s b_max), (1-s) b_max] instead of
# [-c_max, (1-s) c_max]; (R4) planning gives the questions needed for BOTH legs (caac.pool_v1.gap).

def structural_range(small_max: float, big_max: float, s: float) -> tuple[float, float]:
    """Range of d = (1-s)B - C when C = (small part a in [0, small_max]) + (B if escalated) and
    B in [0, big_max]: served by the small model d = (1-s)B - a; escalated d = -sB - a."""
    if not (s < 1.0):
        raise ValueError("s must be < 1")
    return -(small_max + max(s, 0.0) * big_max), (1.0 - s) * big_max


def _check3(inp: AuditInput) -> None:
    _check(inp)
    if inp.small_max is None or inp.big_max is None or not (inp.small_max > 0 and inp.big_max > 0):
        raise ValueError("design v3 needs the split of the cost bound (small_max, big_max)")
    tol = 1e-9
    for i, b in enumerate(inp.big_cost):
        if b > inp.big_max * (1 + tol):
            raise ValueError(f"big-only cost {b} above big_max {inp.big_max}")
    for p in inp.policies:
        if p.small_part is None or len(p.small_part) != len(inp.qids):
            raise ValueError(f"policy {p.name}: design v3 needs small_part, one value per question")
        for i, (c, a) in enumerate(zip(p.cost, p.small_part)):
            if a < -tol or a > inp.small_max * (1 + tol):
                raise ValueError(f"policy {p.name}: small part {a} outside [0, {inp.small_max}]")
            diff = c - a
            if abs(diff) > tol * max(1.0, abs(c)) and abs(diff - inp.big_cost[i]) > tol * max(1.0, abs(c)):
                raise ValueError(f"policy {p.name}: cost is not small part (+ big-only cost) at question {i}")


REL_TESTS = ("bet", "harm", "mix")
REL_TEST = "bet"          # GO risk leg of the relative claim; FIXED by the simulation rule (DECISION rev. 2)


def rel_risk_p(D: list[int], eps: float, order: list[int], test: str) -> float:
    """p-value against H0: R_pi - R_big > eps (small = evidence for the relative claim).
      bet   test by betting on x = eps - D, x in [eps - 1, eps + 1]
      harm  exact one-sided binomial on the harm count #{D = +1} vs eps (valid: R_pi - R_big <= P(D = +1))
      mix   min(1, 2 min(p_bet, p_harm))"""
    if test not in REL_TESTS:
        raise ValueError(f"relative test must be one of {REL_TESTS}")
    n = len(D)
    pb = ph = 1.0
    if test in ("bet", "mix"):
        pb = _betting_p([eps - v for v in D], eps - 1.0, eps + 1.0, order)
    if test in ("harm", "mix"):
        ph = p_risk_below(sum(1 for v in D if v > 0), n, eps) if eps > 0 else 1.0
    if test == "bet":
        return pb
    if test == "harm":
        return ph
    return min(1.0, 2.0 * min(pb, ph))


RELATIVE_NOTE = ("relative claim: R_pi - R_big <= eps (paired with the big model's draw 0, the answer an "
                 "escalated question receives) and saving >= s")


def _check_relative(inp: AuditInput) -> None:
    """The relative claim pairs every question with the big-only baseline: every question must be served,
    and an escalated question must carry exactly the big model's correctness (D_i = 0)."""
    if inp.big_error is None or len(inp.big_error) != len(inp.qids):
        raise ValueError("relative claim needs big_error, one value per question")
    tol = 1e-9
    for p in inp.policies:
        if not all(p.served):
            raise ValueError(f"policy {p.name}: relative claim needs every question served (on_unsure = escalate)")
        for i in range(len(inp.qids)):
            b = inp.big_cost[i]
            if b > 0 and abs((p.cost[i] - p.small_part[i]) - b) <= tol * max(1.0, abs(p.cost[i])):
                if bool(p.error[i]) != bool(inp.big_error[i]):
                    raise ValueError(f"policy {p.name}: escalated question {i} does not carry the big-only answer")


def audit3(inp: AuditInput, *, alpha: float = 0.10, delta: float = 0.10, s: float = 0.20,
           salt: str = DEFAULT_SALT, claim: str = "absolute", eps: float = 0.02,
           rel_test: str | None = None) -> AuditResult:
    """Design v3. Guarantee: each verdict is wrong with probability <= delta.
      GO       some policy passes BOTH legs on all questions, each at delta/|Pi|: exact binomial
               rejecting "R > alpha" and betting rejecting "E[d] < 0". By a union bound over the
               policies that violate the claim, P(some violating policy passes) <= delta; GO reports
               the cheapest policy among those that pass (all of them hold jointly).
      NO-GO    for EVERY policy, "R <= alpha" is rejected by the binomial test OR "E[d] >= 0" by
               betting, each at delta/2 (unchanged from v2; no correction over Pi needed).
      else INCONCLUSIVE (also when both fire: flagged conflict).
    claim = "relative": the risk leg becomes R_pi - R_big <= eps, tested by betting on
      x_i = eps - D_i with D_i = e_pi,i - e_big,i in {-1, 0, 1} (range [eps - 1, eps + 1]), at the same
      levels; NO-GO rejects "R_pi - R_big <= eps" by betting on D_i - eps. Same guarantee."""
    if claim not in ("absolute", "relative"):
        raise ValueError("claim must be 'absolute' or 'relative'")
    _check3(inp)
    rel = claim == "relative"
    if rel:
        if not (0.0 <= eps < 1.0):
            raise ValueError("eps must be in [0, 1)")
        _check_relative(inp)
    n, m = len(inp.qids), len(inp.policies)
    lo, hi = structural_range(inp.small_max, inp.big_max, s)
    order = _order(inp.qids, salt)
    res = AuditResult("INCONCLUSIVE", alpha, delta, s, n, 0, n)
    lvl = delta / m
    rows, passing, all_rej = [], [], True
    mean_b = sum(inp.big_cost) / n
    eb = sum(bool(x) for x in inp.big_error) if rel else None
    for j, p in enumerate(inp.policies):
        sv, e = sum(p.served), sum(p.error)
        d = [(1.0 - s) * inp.big_cost[i] - p.cost[i] for i in range(n)]
        if rel:
            D = [int(bool(p.error[i])) - int(bool(inp.big_error[i])) for i in range(n)]
            dh = sum(D) / n
            pr_go = rel_risk_p(D, eps, order, rel_test or REL_TEST)
            ps_go = _betting_p(d, lo, hi, order) if dh <= eps else 1.0
            pr_ng = _betting_p([v - eps for v in D], -1.0 - eps, 1.0 - eps, order)
        else:
            pr_go = p_risk_below(e, sv, alpha)
            ps_go = _betting_p(d, lo, hi, order) if (sv and e / sv <= alpha) else 1.0
            pr_ng = p_risk_above(e, sv, alpha)
        ps_ng = _betting_p([-v for v in d], -hi, -lo, order)
        ok = pr_go <= lvl and ps_go <= lvl
        rej = pr_ng <= delta / 2 or ps_ng <= delta / 2
        all_rej &= rej
        mc = sum(p.cost) / n
        rows.append({"policy": p.name, "served": sv, "errors": e,
                     "risk": round(e / sv, 6) if sv else None,
                     "saving": round(1 - mc / mean_b, 6) if mean_b else None,
                     "coverage": round(sv / n, 6),
                     "p_risk_go": pr_go, "p_saving_go": ps_go, "go_pass": ok,
                     "p_risk_above": pr_ng, "p_saving_below": ps_ng, "rejected": rej})
        if rel:
            rows[-1]["delta_hat"] = round(dh, 6)
            rows[-1]["big_errors"] = eb
        if ok:
            passing.append((mc, j))
    go = bool(passing)
    res.go = {"level_each_leg": lvl, "n_policies": m, "passed": go,
              "policies_passing": [rows[j]["policy"] for _, j in sorted(passing)]}
    if go:
        _, j = min(passing)
        res.chosen = inp.policies[j].name
        res.go.update({"policy": res.chosen, **{k: rows[j][k] for k in ("served", "errors", "risk", "saving")},
                       "p_risk": rows[j]["p_risk_go"], "p_saving": rows[j]["p_saving_go"]})
    else:
        res.go["policy"] = None
        # closest to GO (descriptive): smallest larger-of-the-two go p-values
        cl = min(range(m), key=lambda j: max(rows[j]["p_risk_go"], rows[j]["p_saving_go"]))
        res.go["closest_policy"] = rows[cl]["policy"]
    res.nogo = {"level_each_leg": delta / 2, "policies": rows, "passed": all_rej}
    res.notes.append("design v3: no split; Bonferroni over the candidate set; structural cost range")
    if rel:
        res.notes.append(RELATIVE_NOTE + f" (eps = {eps})")
        res.go["claim"], res.go["eps"], res.go["rel_test"] = "relative", eps, rel_test or REL_TEST
        res.go["big_only_error"] = round(eb / n, 6)
        if go:
            res.go["delta_hat"] = rows[j]["delta_hat"]
    if go and all_rej:
        res.verdict, res.conflict = "INCONCLUSIVE", True
        res.notes.append("GO and NO-GO both fired (probability <= delta): reported as INCONCLUSIVE")
    elif go:
        res.verdict = "GO"
    elif all_rej:
        res.verdict = "NO-GO"
    return res


# ----------------------------------------------------------- certified frontier (supplementary)
S_GRID = tuple(round(0.02 * i, 2) for i in range(-25, 50))      # s from -0.50 to 0.96, fixed in advance


def risk_upper_bound(errors: int, served: int, level: float) -> float:
    """One-sided exact upper bound: smallest r with P(Bin(served, r) <= errors) <= level."""
    if served == 0:
        return 1.0
    lo, hi = errors / served, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(errors, served, mid) <= level:
            hi = mid
        else:
            lo = mid
    return hi


def certified_frontier(inp: AuditInput, *, delta: float = 0.10, salt: str = DEFAULT_SALT,
                       alphas=(0.05, 0.10, 0.20)) -> dict:
    """Supplementary report. For every policy: an upper bound on its risk and a lower bound on its
    saving, each at level delta/(2|Pi|); all 2|Pi| bounds hold jointly with probability >= 1 - delta.
    The saving bound is the largest s of the fixed grid whose test (betting on d_s, structural range,
    ascending s, stop at the first non-rejection: nested nulls need no correction) rejects
    'saving < s'. Hence every (alpha, s) with some policy having UCB_risk <= alpha and
    LCB_saving >= s is certified at once. frontier[alpha] = largest certified s (None if none)."""
    _check3(inp)
    n, m = len(inp.qids), len(inp.policies)
    lvl = delta / (2 * m)
    order = _order(inp.qids, salt)
    per = []
    for p in inp.policies:
        sv, e = sum(p.served), sum(p.error)
        u = risk_upper_bound(e, sv, lvl)
        lcb = None
        if u < 0.5:                        # a policy whose risk bound is >= .5 never matters below
            for s in S_GRID:
                lo = -(inp.small_max + max(s, 0.0) * inp.big_max)
                hi = (1.0 - s) * inp.big_max
                d = [(1.0 - s) * inp.big_cost[i] - p.cost[i] for i in range(n)]
                if _betting_p(d, lo, hi, order) <= lvl:
                    lcb = s
                else:
                    break
        per.append({"policy": p.name, "risk_upper": round(u, 6), "saving_lower": lcb})
    fr = {}
    for a in alphas:
        vals = [x["saving_lower"] for x in per if x["risk_upper"] <= a and x["saving_lower"] is not None]
        fr[str(a)] = max(vals) if vals else None
    return {"level_each_bound": lvl, "n_bounds": 2 * m, "frontier": fr, "policies": per}
