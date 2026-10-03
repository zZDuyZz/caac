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
from math import comb

__all__ = [
    "binom_cdf",
    "pvalue_selective_risk",
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
