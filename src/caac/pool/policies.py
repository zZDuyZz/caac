"""Policies as replay functions over the sample pool.

Each policy runs on ONE question and returns a PolicyOutcome: did it serve,
what answer, was it correct, and the list of model Calls it paid for (fed to
ledger.cost_profiles). Nothing here calls a GPU -- everything reads the pool.

Order-dependent policies (adaptive sampling, "sample k more") take an explicit
``order`` (a permutation of the small model's sample indices) so the replay
harness can average over many permutations with common random numbers.

Implemented:
  greedy_only            -- small greedy, always serve (lower anchor)
  self_consistency       -- maj vote over first k sampled (small), always serve
  serve_abstain          -- serve greedy iff the generator-confidence check
                            passes, else abstain. NO answer-correcting action.
  pricecheck             -- the faithful PriceCheck baseline: one or more
                            label-free CHECKS vote on the greedy answer; the
                            served answer is ALWAYS the greedy one.
  cascade                -- serve small greedy if confident, else escalate to
                            big greedy and serve that.
  caac                   -- the 4-action controller: {answer now, sample more,
                            escalate, abstain} chosen by a pluggable scorer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from caac.pool.ledger import Call
from caac.pool.schema import Completion, PoolRecord, GOLD

__all__ = [
    "PolicyOutcome",
    "greedy_only",
    "self_consistency",
    "serve_abstain",
    "cascade",
    "caac",
    "pricecheck",
    "agreement_with_greedy",
    "CHECKS",
    "ACTIONS",
]

ACTIONS = ("answer_now", "sample_more", "escalate", "abstain")


@dataclass
class PolicyOutcome:
    served: bool
    answer: str | None
    correct: bool
    calls: list[Call]
    action: str = ""          # which top-level action was taken (for CAAC)
    meta: dict = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


def _vote(answers: list[str | None]) -> str | None:
    cleaned = [a for a in answers if a is not None]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _greedy_call(rec: PoolRecord, round_index: int = 0) -> Call:
    return Call(rec.model, rec.prompt_tokens, [rec.greedy.decode_tokens], round_index)


def _samples_call(rec: PoolRecord, idxs: list[int], round_index: int) -> Call:
    return Call(rec.model, rec.prompt_tokens, [rec.samples[i].decode_tokens for i in idxs], round_index)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------


def greedy_only(small: PoolRecord) -> PolicyOutcome:
    g = small.greedy
    return PolicyOutcome(True, g.answer, g.correct, [_greedy_call(small)], "answer_now")


def self_consistency(small: PoolRecord, k: int, order: list[int]) -> PolicyOutcome:
    idxs = order[:k]
    canon = small.canonical_sample_answers()  # correct -> GOLD (see schema)
    answers = [canon[i] for i in idxs]
    voted = _vote(answers)
    correct = voted == GOLD
    return PolicyOutcome(True, voted, correct, [_samples_call(small, idxs, 0)], "sample_more")


def serve_abstain(small: PoolRecord, confidence_threshold: float) -> PolicyOutcome:
    """PriceCheck-style: a label-free check decides serve-vs-abstain on the
    greedy answer. Here the check is the greedy confidence clearing a threshold;
    swap in a re-solve-vote or backward-probe check to match PriceCheck exactly.
    Crucially: the served answer is ALWAYS the greedy one -- never corrected.
    """
    g = small.greedy
    conf = g.confidence if g.confidence is not None else 0.0
    calls = [_greedy_call(small)]
    if conf >= confidence_threshold:
        return PolicyOutcome(True, g.answer, g.correct, calls, "answer_now")
    return PolicyOutcome(False, None, False, calls, "abstain")


# --------------------------------------------------------------------------
# PriceCheck (2609.37493, Lee et al.) -- the closest competitor, as a policy.
#
# Every "action" in PriceCheck is a CHECK that returns an agree/disagree verdict
# on an answer that already exists; the output is always "serve the original" or
# "abstain". It can never produce a different answer. That single restriction is
# what the Chow frontier (pool/frontier.py) is about, so the baseline has to be
# faithful to it: below, the served answer is `small.greedy.answer` in every
# branch, no matter what the checks saw.
#
# Checks implemented (both FREE from the pool -- no extra GPU):
#   confidence    -- the generator's own mean token probability on the greedy run
#   resolve_vote  -- re-solve the question k times and measure what fraction of
#                    the re-solves AGREE with the greedy answer (not with gold)
# The backward probe of their paper needs fresh generations (A5) and is omitted;
# CAAC_MASTER §11 records that omission.
# --------------------------------------------------------------------------

CHECKS = ("confidence", "resolve_vote", "both")


def agreement_with_greedy(rec: PoolRecord, idxs: list[int]) -> float:
    """Fraction of the chosen samples whose answer matches the greedy answer.

    Matching convention is the schema's (see canonical_sample_answers): two
    completions graded correct are the same answer; a correct and an incorrect
    one differ; two incorrect ones match only if their normalized strings match
    (equivalent wrong answers stay split -- conservative, and it can only make
    the check look WEAKER, never stronger).

    This is label-free in deployment: the grader is used here only as a stand-in
    for "are these two strings the same answer", never to reveal the gold label
    to the policy. A deployed PriceCheck compares strings directly.
    """
    if not idxs:
        return 0.0
    g = rec.greedy
    n = 0
    for i in idxs:
        c = rec.samples[i]
        if g.correct and c.correct:
            n += 1
        elif (not g.correct) and (not c.correct) and c.answer is not None \
                and c.answer == g.answer:
            n += 1
    return n / len(idxs)


def pricecheck(small: PoolRecord, lam: float, check: str = "resolve_vote",
               k: int = 8, order: list[int] | None = None,
               conf_threshold: float | None = None) -> PolicyOutcome:
    """Serve-or-abstain on the GREEDY answer, decided by label-free check(s).

    lam   threshold the check score must clear to serve.
    check "confidence"   -- generator confidence only (no extra samples, no cost
                            beyond the greedy run)
          "resolve_vote" -- k re-solves must agree with greedy at rate >= lam
          "both"         -- run the cheap confidence check first; only pay for
                            the re-solves when it is inconclusive (this is the
                            cost-saving "schedule" idea of the paper). Serve iff
                            confidence >= conf_threshold OR the re-solve vote
                            clears lam.
    """
    if check not in CHECKS:
        raise ValueError(f"check must be one of {CHECKS}")
    g = small.greedy
    conf = g.confidence if g.confidence is not None else 0.0
    calls = [_greedy_call(small)]

    if check == "confidence":
        serve = conf >= lam
    else:
        run_votes = True
        if check == "both":
            ct = conf_threshold if conf_threshold is not None else 1.0
            if conf >= ct:          # cheap check already decides: serve, pay nothing more
                return PolicyOutcome(True, g.answer, g.correct, calls, "answer_now",
                                     {"check": check, "decided_by": "confidence"})
            run_votes = True
        idxs = (order or list(range(small.n_samples)))[:k]
        if run_votes:
            calls.append(_samples_call(small, idxs, 1))
        serve = agreement_with_greedy(small, idxs) >= lam

    if serve:
        # NOTE: the greedy answer, never a voted or escalated one. This is the point.
        return PolicyOutcome(True, g.answer, g.correct, calls, "answer_now",
                             {"check": check})
    return PolicyOutcome(False, None, False, calls, "abstain", {"check": check})


def cascade(small: PoolRecord, big: PoolRecord, confidence_threshold: float) -> PolicyOutcome:
    """Serve small greedy if confident, else escalate to big greedy (new round)."""
    g = small.greedy
    conf = g.confidence if g.confidence is not None else 0.0
    if conf >= confidence_threshold:
        return PolicyOutcome(True, g.answer, g.correct, [_greedy_call(small)], "answer_now")
    bg = big.greedy
    calls = [_greedy_call(small, 0), _greedy_call(big, 1)]
    return PolicyOutcome(True, bg.answer, bg.correct, calls, "escalate")


# --------------------------------------------------------------------------
# the CAAC controller (4 actions)
# --------------------------------------------------------------------------

# A scorer maps (features) -> estimated P(correct | action) for each action.
# In the paper this is the calibrated learned estimator trained on the
# train/calib split. For the CPU scaffold, ConfidenceScorer uses greedy
# confidence so the whole pipeline runs and is testable without GPU.
Scorer = Callable[[PoolRecord, PoolRecord], dict[str, float]]


@dataclass
class ConfidenceScorer:
    """Stand-in scorer: monotone maps of greedy confidence to per-action P(correct).

    Replace with the trained estimator. The INTERFACE is what matters: given
    the question's pool records, return a P(correct) estimate per action and
    the cost each action would incur.
    """

    sample_k: int = 8

    def __call__(self, small: PoolRecord, big: PoolRecord) -> dict[str, float]:
        c = small.greedy.confidence if small.greedy.confidence is not None else 0.5
        # crude monotone stand-ins; the real estimator is fit, not hand-set
        return {
            "answer_now": c,
            "sample_more": min(1.0, c + 0.12),
            "escalate": min(1.0, 0.55 + 0.4 * c),
            "abstain": 1.0,  # abstain never produces a wrong SERVED answer
        }


def _action_cost_scalar(small: PoolRecord, big: PoolRecord, action: str, k: int,
                        profile_fn) -> float:
    from caac.pool.ledger import cost_profiles
    calls = _calls_for_action(small, big, action, k, list(range(min(k, small.n_samples))))
    return cost_profiles(calls)[profile_fn]


def _calls_for_action(small, big, action, k, order):
    if action == "answer_now":
        return [_greedy_call(small)]
    if action == "sample_more":
        idxs = order[:k]
        return [_greedy_call(small, 0), _samples_call(small, idxs, 1)]
    if action == "escalate":
        return [_greedy_call(small, 0), _greedy_call(big, 1)]
    if action == "abstain":
        return [_greedy_call(small)]  # paid for the greedy before abstaining
    raise ValueError(action)


def caac(
    small: PoolRecord,
    big: PoolRecord,
    scorer: Scorer,
    lam: float,
    order: list[int],
    k: int = 8,
    utility: float = 1.0,
    cost_profile: str = "tokens",
    abstain_penalty: float = 0.0,
) -> PolicyOutcome:
    """Value-of-computation controller.

    For each action a: VOC(a) = utility * Phat(correct|a) - lam * cost(a),
    with abstain scored as -abstain_penalty (no correctness, no risk). Picks
    argmax VOC, then REPLAYS that action's real outcome from the pool.
    """
    from caac.pool.ledger import cost_profiles

    phat = scorer(small, big)
    voc = {}
    for a in ACTIONS:
        calls = _calls_for_action(small, big, a, k, order)
        cost = cost_profiles(calls)[cost_profile]
        if a == "abstain":
            voc[a] = -abstain_penalty - lam * cost
        else:
            voc[a] = utility * phat[a] - lam * cost
    action = max(ACTIONS, key=lambda a: voc[a])

    # replay the chosen action's true outcome
    if action == "answer_now":
        out = greedy_only(small)
    elif action == "sample_more":
        out = self_consistency(small, k, order)
        out.calls = [_greedy_call(small, 0), _samples_call(small, order[:k], 1)]
    elif action == "escalate":
        out = cascade_force_escalate(small, big)
    else:  # abstain
        out = PolicyOutcome(False, None, False, [_greedy_call(small)], "abstain")
    out.action = action
    out.meta["voc"] = voc
    return out


def cascade_force_escalate(small: PoolRecord, big: PoolRecord) -> PolicyOutcome:
    bg = big.greedy
    return PolicyOutcome(True, bg.answer, bg.correct,
                         [_greedy_call(small, 0), _greedy_call(big, 1)], "escalate")
