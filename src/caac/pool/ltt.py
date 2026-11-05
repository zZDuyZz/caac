"""Learn-then-Test certification of selective risk (PLAN_V2 sections 6, 8).

Guarantee (Angelopoulos et al. 2021, "Learn then Test"):
    P( sup_{lambda in certified set} R(lambda) <= alpha ) >= 1 - delta,
with R(lambda) the selective error (share of SERVED answers that are wrong),
over an i.i.d./exchangeable calibration set drawn separately from the data
used to train the estimators.

Per configuration lambda, the exact one-sided binomial p-value for
H0: R(lambda) > alpha is F(e; n_served, alpha) where e = wrong served, and
n_served = served. Certified iff n_served>0 and p <= delta/|G| (Bonferroni),
or via fixed-sequence testing which spends no multiplicity and usually
certifies more (this is the PriceCheck finding we exploit).

This module is pure stats -- a "policy at a given lambda" is summarized as
(n_served, n_wrong_served, coverage, mean_cost); we don't re-run the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, log

__all__ = [
    "binom_cdf",
    "pvalue_selective_risk",
    "min_served_certifiable",
    "certification_floor",
    "certify_bonferroni",
    "certify_fixed_sequence",
    "LambdaResult",
]


def binom_cdf(e: int, n: int, p: float) -> float:
    """P(X <= e) for X ~ Binomial(n, p). Exact; n is small (calibration size)."""
    if n == 0:
        return 1.0
    p = min(max(p, 0.0), 1.0)
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, e + 1))


def min_served_certifiable(alpha: float, delta: float) -> int:
    """Fewest served questions that CAN ever be certified, even with 0 errors.

    A flawless run still only reaches p = (1-alpha)^n, so no configuration whose
    served count is below ``ceil(log delta / log(1-alpha))`` is certifiable at
    any risk level -- the data cannot get there. At alpha=delta=0.1 that is 22
    questions; under Bonferroni, with delta/|G| in place of delta, 21 grid
    points push it to 58.

    This is why an aggressive, very-low-coverage operating point is not
    conservative but simply uncertifiable, and why calibration-set size shows up
    in the results as a floor on coverage rather than as noise.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0,1)")
    return int(ceil(log(delta) / log(1.0 - alpha)))


def certification_floor(n_calib: int, alpha: float, delta: float) -> float:
    """Minimum coverage any certifiable configuration must have on ``n_calib``."""
    if n_calib <= 0:
        return 1.0
    return min(1.0, min_served_certifiable(alpha, delta) / n_calib)


def pvalue_selective_risk(n_served: int, n_wrong: int, alpha: float) -> float:
    """One-sided p-value for H0: true selective risk > alpha.

    Small p => we can reject H0 => the configuration is safe at level alpha.
    """
    if n_served == 0:
        return 1.0  # serving nothing certifies nothing
    return binom_cdf(n_wrong, n_served, alpha)


@dataclass
class LambdaResult:
    """One configuration's calibration summary."""

    lam: object                  # the configuration (threshold tuple, name, ...)
    n_served: int
    n_wrong: int
    coverage: float              # served / total calibration items
    mean_cost: float             # under the chosen cost profile
    pvalue: float = 1.0
    certified: bool = False


def certify_bonferroni(results: list[LambdaResult], alpha: float, delta: float) -> list[LambdaResult]:
    """Flat Bonferroni over the whole family |G| = len(results)."""
    g = len(results)
    thresh = delta / g if g else delta
    for r in results:
        r.pvalue = pvalue_selective_risk(r.n_served, r.n_wrong, alpha)
        r.certified = r.n_served > 0 and r.pvalue <= thresh
    return results


def certify_fixed_sequence(ordered: list[LambdaResult], alpha: float, delta: float) -> list[LambdaResult]:
    """Fixed-sequence testing: test in the given order at level delta, stop at
    the first failure. Spends no multiplicity, so it usually certifies more --
    but the ORDER must be fixed in advance (e.g. conservative lambda first).
    """
    passed = True
    for r in ordered:
        r.pvalue = pvalue_selective_risk(r.n_served, r.n_wrong, alpha)
        if passed and r.n_served > 0 and r.pvalue <= delta:
            r.certified = True
        else:
            r.certified = False
            passed = False  # stop certifying after the first failure
    return ordered


def select_max_coverage(results: list[LambdaResult]) -> LambdaResult | None:
    """Among certified configs, the one serving the most (PriceCheck default)."""
    certified = [r for r in results if r.certified]
    return max(certified, key=lambda r: r.coverage) if certified else None


def select_min_cost(results: list[LambdaResult], min_coverage: float = 0.0) -> LambdaResult | None:
    """Among certified configs above a coverage floor, the cheapest."""
    cands = [r for r in results if r.certified and r.coverage >= min_coverage]
    return min(cands, key=lambda r: r.mean_cost) if cands else None
