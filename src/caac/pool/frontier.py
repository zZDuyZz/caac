"""The Chow frontier -- the paper's central argument (CAAC_MASTER §10).

Claim: a policy that may only SERVE-or-ABSTAIN is capped at coverage
``min{1, pi/(1-alpha)}`` (Chow 1970), where ``pi`` is the share of CANDIDATE
answers that are correct. The cap binds hard in the small-model regime
(MATH-500 / Qwen-1.5B: pi~0.49 => cap 0.544 at alpha=0.10) and barely binds in
the regime PriceCheck operates in (pi~0.957 => cap ~0.97). Actions that CHANGE
the answer (sample-more, escalate) raise pi itself, so they lift the cap. That
is a qualitative difference, not an optimisation.

What this module computes, all from an existing pool (CPU only):

  r = P(some correcting action yields a correct answer | greedy is wrong)
      measured per (dataset, model) -- the ONLY empirical input to the sweep.

  pi_after(pi) = pi + (1 - pi) * r
  cap_serve(pi)   = min{1, pi / (1 - alpha)}
  cap_correct(pi) = min{1, pi_after(pi) / (1 - alpha)}

Sweeping ``pi`` over [~0.04, ~0.96] covers the span from "harder than our MATH
setting" to "the PriceCheck regime". Both curves are CAPS (an oracle attains
them; no real policy exceeds them), and ``cap_correct`` additionally assumes the
oracle picks the right action -- it is an upper bound on CAAC, never a claim
about what CAAC achieves. Realised operating points are added separately.

Honesty notes baked in:
  * ``r`` is estimated on the pool and is itself an upper bound on what a
    *deployed* policy gets, because it asks only whether SOME action works.
  * the correcting actions cost more; ``cost_ratio_at`` reports how much, so the
    lift is never shown for free.
  * ``r`` is reported with a Wilson interval: it is a proportion over the
    greedy-wrong questions only, which is a smaller sample than the pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from caac.pool.estimators import chow_coverage_bound
from caac.pool.ledger import Call, cost_profiles
from caac.pool.schema import PoolRecord

__all__ = [
    "RecoveryRate", "recovery_rate", "pi_after", "cap_serve", "cap_correct",
    "frontier_curve", "cost_ratio_at", "wilson",
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (better than normal at the edges)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass
class RecoveryRate:
    """P(a correcting action succeeds | greedy wrong), plus the breakdown."""

    n_total: int
    n_greedy_wrong: int
    n_recovered_any: int                  # sample-more OR escalate works
    by_action: dict[str, int] = field(default_factory=dict)
    pi_greedy: float = 0.0

    @property
    def r(self) -> float:
        return self.n_recovered_any / self.n_greedy_wrong if self.n_greedy_wrong else 0.0

    @property
    def r_ci(self) -> tuple[float, float]:
        return wilson(self.n_recovered_any, self.n_greedy_wrong)

    def as_dict(self) -> dict:
        lo, hi = self.r_ci
        return {"n_total": self.n_total, "n_greedy_wrong": self.n_greedy_wrong,
                "n_recovered_any": self.n_recovered_any, "r": self.r,
                "r_ci95": [lo, hi], "pi_greedy": self.pi_greedy,
                "by_action": dict(self.by_action)}


def _maj_correct(rec: PoolRecord, k: int, threshold: float = 0.5) -> bool:
    """Does majority voting over k samples land on the gold answer?

    Uses the exact maj@k estimator over all C(n,k) subsets and counts the
    question as recovered when gold wins more often than not. Deterministic;
    no permutation sampling, so it cannot be tuned by a lucky seed.
    """
    if rec.n_samples == 0:
        return False
    return rec.maj_at_k(k) > threshold


def recovery_rate(small: dict[str, PoolRecord], big: dict[str, PoolRecord] | None = None,
                  k: int = 8, maj_threshold: float = 0.5) -> RecoveryRate:
    """Measure r on a pool: of the questions the small model's greedy gets
    wrong, how many does SOME correcting action get right?

    Correcting actions considered: ``sample_more`` (maj@k of the small model)
    and, when ``big`` is given, ``escalate`` (the big model's greedy).
    """
    qids = sorted(small if big is None else set(small) & set(big))
    n_wrong = n_rec = 0
    by = {"sample_more": 0, "escalate": 0, "both": 0}
    n_greedy_ok = 0
    for q in qids:
        s = small[q]
        if s.greedy.correct:
            n_greedy_ok += 1
            continue
        n_wrong += 1
        a = _maj_correct(s, k, maj_threshold)
        b = bool(big and big[q].greedy.correct)
        if a:
            by["sample_more"] += 1
        if b:
            by["escalate"] += 1
        if a and b:
            by["both"] += 1
        if a or b:
            n_rec += 1
    return RecoveryRate(n_total=len(qids), n_greedy_wrong=n_wrong, n_recovered_any=n_rec,
                        by_action=by, pi_greedy=n_greedy_ok / len(qids) if qids else 0.0)


def pi_after(pi: float, r: float) -> float:
    """Candidate-correctness after correcting actions: pi + (1-pi)*r."""
    return min(1.0, pi + (1.0 - pi) * r)


def cap_serve(pi: float, alpha: float) -> float:
    """Chow cap for a serve-or-abstain-only policy (PriceCheck's action set)."""
    return chow_coverage_bound(pi, alpha)


def cap_correct(pi: float, alpha: float, r: float) -> float:
    """Chow cap once answer-correcting actions are allowed."""
    return chow_coverage_bound(pi_after(pi, r), alpha)


def frontier_curve(alpha: float, r: float, pis) -> list[dict]:
    """The two caps (and their gap) across a sweep of base correctness ``pi``."""
    out = []
    for pi in pis:
        cs, cc = cap_serve(pi, alpha), cap_correct(pi, alpha, r)
        out.append({"pi": pi, "base_error": 1 - pi, "cap_serve": cs, "cap_correct": cc,
                    "gap": cc - cs, "pi_after": pi_after(pi, r)})
    return out


def cost_ratio_at(small: dict[str, PoolRecord], big: dict[str, PoolRecord] | None,
                  k: int = 8, profile: str = "flops") -> dict[str, float]:
    """Mean cost of each action relative to answering now, on this pool.

    The frontier lift is not free: this is what the extra coverage costs. Prefill
    is re-paid for ``sample_more`` (pessimistic convention, CAAC_MASTER §9).
    """
    qids = sorted(small if big is None else set(small) & set(big))
    tot = {"answer_now": 0.0, "sample_more": 0.0, "escalate": 0.0}
    for q in qids:
        s = small[q]
        g = Call(s.model, s.prompt_tokens, [s.greedy.decode_tokens], 0)
        mean_dec = (sum(c.decode_tokens for c in s.samples) / len(s.samples)) if s.samples else 0.0
        tot["answer_now"] += cost_profiles([g])[profile]
        tot["sample_more"] += cost_profiles(
            [g, Call(s.model, s.prompt_tokens, [mean_dec] * k, 1)])[profile]
        if big is not None:
            b = big[q]
            tot["escalate"] += cost_profiles(
                [g, Call(b.model, b.prompt_tokens, [b.greedy.decode_tokens], 1)])[profile]
    base = tot["answer_now"] or 1.0
    out = {a: v / base for a, v in tot.items() if v}
    out["profile"] = profile
    return out
