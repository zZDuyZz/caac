"""Recoverability: can the next unit of compute actually fix this answer?

A selective cascade asks whether its current answer is trustworthy. This module
asks the different question the controller needs: GIVEN that the first answer is
wrong, which available action would repair it, and which of those is cheapest.

Three rules make the numbers here reportable rather than optimistic:

1. **Deployable, not oracle.** "Sampling repairs the answer" means the majority
   class of the drawn samples is a correct class -- the answer a deployed voter
   would actually serve. The weaker claim "some drawn sample happened to be
   right" is an oracle upper bound; ``oracle_rescue`` computes it separately so
   the gap between the two is visible instead of hidden.

2. **Four primitive groups before three regions.** Whether sampling or
   escalation repairs a question are two separate facts, so they are counted as
   four groups (both / sampling only / escalation only / neither). The three
   regions are a *cost-dependent* summary built on top: which repair is
   cheapest depends on the price ratio, so collapsing early would bake one
   price into what is meant to be a measurement.

3. **Unrecoverable is relative.** ``unrecoverable`` means "no action in THIS
   action set, model pool and sample budget repairs it", never "impossible".
   The field names and the docstrings keep saying so because a reader will
   otherwise hear the stronger claim.

Rescue is only half of the value of an action: escalation can also BREAK a
question the small model had right. ``harm_probs`` measures that, and
``delta_probs`` returns the net change this module exists to estimate,

    Delta_a = P(correct after a) - P(correct now) = P(gain_a) - P(harm_a),

an exact identity here because the greedy outcome is a constant per question.
"""

from __future__ import annotations

from caac.pool.schema import PoolRecord
from caac.pool.voting import greedy_vote_correct, greedy_vote_prob

__all__ = [
    "PRIMITIVE_GROUPS", "REGIONS", "CORRECTING_ACTIONS",
    "sample_correct_prob", "sample_correct_realised", "oracle_rescue",
    "rescue_probs", "harm_probs", "delta_probs",
    "group_probs", "region_of_group", "region_probs", "recovery_table",
]

CORRECTING_ACTIONS = ("sample_more", "escalate")
PRIMITIVE_GROUPS = ("both", "sampling_only", "escalation_only", "neither")
REGIONS = ("cheap_recoverable", "expensive_only", "unrecoverable")


def sample_correct_prob(small: PoolRecord, k: int) -> float:
    """P(a k-sample majority vote serves a correct answer), exact over draws.

    Exact rather than simulated: ``maj_at_k`` sums the probability of each
    correct answer class winning over all C(n, k) subsets, so there is no lucky
    permutation to average away. Answer classes are label-free, and the gold
    label only grades the class that would be served.
    """
    return greedy_vote_prob(small, k) if small.n_samples else 0.0


def sample_correct_realised(small: PoolRecord, order: list[int], k: int) -> float:
    """The same quantity for ONE concrete draw order: what a single run achieved."""
    idxs = [i for i in order if i < small.n_samples][:k]
    return greedy_vote_correct(small, idxs)


def oracle_rescue(small: PoolRecord, k: int) -> float:
    """Upper bound: P(greedy or at least one of k draws is correct), selector ignored.

    Reported next to the deployable number, never instead of it. A large gap
    says the samples contain the answer but the voter cannot find it, which is
    a statement about the selector, not about recoverability.
    """
    # The greedy answer is a vote, so a perfect selector keeps it. Its CLASS can be
    # graded correct while the completion itself was not (the matcher can merge
    # "0.2" with "20%", which the gold grader keeps apart): the vote then serves
    # a correct class, so the ceiling must count it too or it stops being one.
    if small.greedy.correct or small.label_correct(small.greedy_label()):
        return 1.0
    n = small.n_samples
    if not n:
        return 0.0
    k = min(k, n)
    good = sum(1 for lab in small.canonical_sample_answers() if small.label_correct(lab))
    if good == 0:
        return 0.0
    # 1 - C(n-good, k)/C(n, k), computed as a product to avoid big factorials
    p_none = 1.0
    for i in range(k):
        num = n - good - i
        if num <= 0:
            return 1.0
        p_none *= num / (n - i)
    return 1.0 if p_none < 1e-12 else 1.0 - p_none


def _escalate_correct(big: PoolRecord | None) -> float | None:
    return None if big is None else float(big.greedy.correct)


def rescue_probs(small: PoolRecord, big: PoolRecord | None, k: int) -> dict[str, float]:
    """P(action a serves a correct answer | the greedy answer is wrong), per action.

    Returns an empty dict when the greedy answer is already correct: there is
    nothing to rescue, and counting those questions would dilute the rates with
    a population the question is not about.
    """
    if small.greedy.correct:
        return {}
    out = {"sample_more": sample_correct_prob(small, k)}
    esc = _escalate_correct(big)
    if esc is not None:
        out["escalate"] = esc
    return out


def harm_probs(small: PoolRecord, big: PoolRecord | None, k: int) -> dict[str, float]:
    """P(action a serves a WRONG answer | the greedy answer is correct).

    The other half of an action's value. Empty when the greedy answer is wrong,
    for the same reason ``rescue_probs`` is empty when it is right.
    """
    if not small.greedy.correct:
        return {}
    out = {"sample_more": 1.0 - sample_correct_prob(small, k)}
    esc = _escalate_correct(big)
    if esc is not None:
        out["escalate"] = 1.0 - esc
    return out


def delta_probs(small: PoolRecord, big: PoolRecord | None, k: int) -> dict[str, float]:
    """Net value of each action: P(correct after a) - P(correct now), in [-1, 1].

    This is the quantity the allocator thresholds. Gating on the rescue rate
    alone would spend on actions whose expected damage exceeds their repair.
    """
    gain, harm = rescue_probs(small, big, k), harm_probs(small, big, k)
    acts = set(gain) | set(harm)
    return {a: gain.get(a, 0.0) - harm.get(a, 0.0) for a in acts}


def group_probs(small: PoolRecord, big: PoolRecord | None, k: int) -> dict[str, float]:
    """Distribution over the four primitive groups for one greedy-wrong question.

    Fractional because the sampling outcome is a distribution over draws, while
    escalation is a single recorded run. Empty for a greedy-correct question.
    """
    r = rescue_probs(small, big, k)
    if not r or "escalate" not in r:
        return {}
    ps, pe = r["sample_more"], r["escalate"]
    return {"both": ps * pe, "sampling_only": ps * (1.0 - pe),
            "escalation_only": (1.0 - ps) * pe, "neither": (1.0 - ps) * (1.0 - pe)}


def region_of_group(group: str, sample_cost: float, escalate_cost: float) -> str:
    """Name a group by the CHEAPEST action that repairs it.

    ``cheap_recoverable`` means sampling is both sufficient and no dearer than
    escalating; ``expensive_only`` means the repair has to be paid for at the
    big model's price, either because sampling cannot do it or because sampling
    that far costs more. Ties go to sampling, which keeps the big model idle.
    """
    if group == "neither":
        return "unrecoverable"
    if group == "sampling_only":
        return "cheap_recoverable"
    if group == "escalation_only":
        return "expensive_only"
    if group == "both":
        return "cheap_recoverable" if sample_cost <= escalate_cost else "expensive_only"
    raise ValueError(f"unknown group {group!r}")


def region_probs(small: PoolRecord, big: PoolRecord | None, k: int,
                 sample_cost: float, escalate_cost: float) -> dict[str, float]:
    out = {r: 0.0 for r in REGIONS}
    for g, p in group_probs(small, big, k).items():
        out[region_of_group(g, sample_cost, escalate_cost)] += p
    return out


def _action_cost(small: PoolRecord, big: PoolRecord | None, action: str, k: int,
                 price) -> float:
    from caac.pool.policies import calls_for_action
    order = list(range(small.n_samples))
    return price(calls_for_action(small, big, action, k, order))


def recovery_table(qids, small: dict, big: dict | None, k: int = 16,
                   price=None, cost_profile: str = "tokens") -> dict:
    """The Contribution-1 measurement: where do reasoning failures sit?

    Rates over the greedy-wrong questions (the population the question is
    about), plus the harm rate over the greedy-right ones and the mean net
    value of each action over ALL questions -- the allocator's own objective.

    ``price`` maps a list of Calls to a scalar; it decides the region split, so
    the caller passes the cost axis being reported and the result records which.
    """
    from caac.pool.ledger import cost_profiles
    price = price or (lambda calls: cost_profiles(calls)[cost_profile])

    qids = [q for q in qids if q in small and (big is None or q in big)]
    groups = {g: 0.0 for g in PRIMITIVE_GROUPS}
    regions = {r: 0.0 for r in REGIONS}
    delta = {a: 0.0 for a in CORRECTING_ACTIONS}
    harm = {a: 0.0 for a in CORRECTING_ACTIONS}
    oracle_gap = 0.0
    n_wrong = n_right = 0

    for q in qids:
        s, b = small[q], (big or {}).get(q)
        cs = _action_cost(s, b, "sample_more", k, price)
        ce = _action_cost(s, b, "escalate", k, price) if b is not None else float("inf")
        for a, d in delta_probs(s, b, k).items():
            delta[a] += d
        if s.greedy.correct:
            n_right += 1
            for a, h in harm_probs(s, b, k).items():
                harm[a] += h
            continue
        n_wrong += 1
        oracle_gap += oracle_rescue(s, k) - sample_correct_prob(s, k)
        for g, p in group_probs(s, b, k).items():
            groups[g] += p
            regions[region_of_group(g, cs, ce)] += p

    n = len(qids)
    frac = lambda d, m: {kk: (v / m if m else 0.0) for kk, v in d.items()}
    return {
        "n": n, "k": k, "cost_profile": cost_profile,
        "n_greedy_wrong": n_wrong, "n_greedy_right": n_right,
        # over the greedy-wrong questions
        "groups": frac(groups, n_wrong), "regions": frac(regions, n_wrong),
        "oracle_minus_deployable": (oracle_gap / n_wrong) if n_wrong else 0.0,
        # over the greedy-right questions
        "harm": frac(harm, n_right),
        # over every question: what the allocator maximises
        "mean_delta": frac(delta, n),
        "note": "unrecoverable = not repaired by THIS action set, model pool and "
                "sample budget; rescue uses the deployable majority selector",
    }
