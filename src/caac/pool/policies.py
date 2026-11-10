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
                            escalate, abstain} chosen by a pluggable scorer
                            maximising value-of-computation (the ablation).
  caac_certified         -- CAAC proper: the CHEAPEST action whose calibrated
                            P(correct) clears tau, else abstain. One knob, on
                            the same scale as the risk LTT certifies.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from caac.pool.ledger import Call
from caac.pool.schema import Completion, PoolRecord
from caac.pool.voting import serve_with_greedy

__all__ = [
    "PolicyOutcome",
    "greedy_only",
    "self_consistency",
    "serve_abstain",
    "cascade",
    "caac",
    "caac_certified",
    "caac_staged",
    "pricecheck",
    "agreement_with_greedy",
    "calls_for_action",
    "CHECKS",
    "ACTIONS",
    "RHO_MODES",
]

ACTIONS = ("answer_now", "sample_more", "escalate", "abstain")

# What scale ``caac_staged``'s rho gate lives on. "phat" is the original rule;
# the other two gate on the estimated net value of the action, which is what
# lets the policy abstain before buying compute that cannot help.
RHO_MODES = ("phat", "delta", "value")


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
    canon = small.canonical_sample_answers()
    voted = _vote([canon[i] for i in idxs])
    correct = small.label_correct(voted)
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
    """Fraction of the chosen samples whose answer class equals the greedy one.

    Classes are label-free (schema._answer_labels): answers are compared with
    each other, never with gold, so this is the check a deployed system runs.
    An unparsable answer never agrees.
    """
    if not idxs:
        return 0.0
    g = rec.greedy_label()
    if g is None:
        return 0.0
    canon = rec.canonical_sample_answers()
    return sum(canon[i] == g for i in idxs) / len(idxs)


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


def calls_for_action(small, big, action, k, order):
    """What one action would pay for, so every module prices it the same way."""
    if action == "answer_now":
        return [_greedy_call(small)]
    if action == "sample_more":
        # clamp to the samples the pool actually holds: a record with fewer
        # than k (or none at all) must price the action it CAN take, not crash
        idxs = [i for i in order if i < small.n_samples][:k]
        return [_greedy_call(small, 0), _samples_call(small, idxs, 1)]
    if action == "escalate":
        return [_greedy_call(small, 0), _greedy_call(big, 1)]
    if action == "abstain":
        return [_greedy_call(small)]  # paid for the greedy before abstaining
    raise ValueError(action)


_calls_for_action = calls_for_action   # internal alias, kept for older callers


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


def caac_certified(
    small: PoolRecord,
    big: PoolRecord,
    scorer: Scorer,
    tau: float,
    order: list[int],
    k: int = 8,
    cost_profile: str = "flops",
    actions: tuple[str, ...] = ("answer_now", "sample_more", "escalate"),
    cost_fn=None,
) -> PolicyOutcome:
    """CAAC proper: **cheapest action predicted safe enough, else abstain.**

        eligible = { a : Phat(correct | a) >= tau }
        serve the CHEAPEST eligible action;  if none is eligible, abstain.

    One knob, ``tau``, and it is a probability on the same scale as the risk the
    guarantee is about -- so Learn-then-Test can certify it directly (§8.2) and
    the certified value is interpretable rather than an arbitrary score cut.

    Why "cheapest eligible" and not argmax-utility: the question CAAC answers is
    not "which action is best" but "how little compute suffices here". Spending
    more than needed on a question the small model already handles is exactly
    the waste the paper is about, and a cost-weighted VOC trades risk against
    cost on one axis, which would smear the risk guarantee. The VOC variant is
    kept in ``caac`` as the ablation.

    The contrast with ``pricecheck`` is one line of code: there the eligible set
    is only ``{answer_now}``, so coverage is capped by the Chow bound at
    ``pi/(1-alpha)``; here ``sample_more``/``escalate`` can make a question
    eligible that ``answer_now`` never could, which is what lifts the cap.

    ``actions`` restricts the action set -- the ablation table (answer-only,
    +sample_more, +escalate, all) is produced by varying just this argument.
    """
    from caac.pool.ledger import cost_profiles

    price = cost_fn or (lambda calls: cost_profiles(calls)[cost_profile])
    phat = scorer(small, big)
    costs = {a: price(_calls_for_action(small, big, a, k, order)) for a in actions}
    eligible = [a for a in actions if phat.get(a, 0.0) >= tau]
    action = min(eligible, key=lambda a: (costs[a], a)) if eligible else "abstain"

    if action == "answer_now":
        out = greedy_only(small)
    elif action == "sample_more":
        out = self_consistency(small, k, order)
        out.calls = [_greedy_call(small, 0), _samples_call(small, order[:k], 1)]
    elif action == "escalate":
        out = cascade_force_escalate(small, big)
    else:
        # the greedy turn is already paid for when we decide to abstain
        out = PolicyOutcome(False, None, False, [_greedy_call(small)], "abstain")
    out.action = action
    out.meta["phat"] = phat
    out.meta["eligible"] = eligible
    out.meta["tau"] = tau
    return out


def caac_staged(
    small: PoolRecord,
    big: PoolRecord,
    scorer: Scorer,
    stage2,
    tau: float,
    order: list[int],
    k: int = 8,
    cost_profile: str = "flops",
    actions: tuple[str, ...] = ("answer_now", "sample_more", "escalate"),
    k_steps: tuple[int, ...] = (),
    cost_fn=None,
    escalate_gate=None,
    rho: float | None = None,
    rho_mode: str = "phat",
    recovery=None,
    lam: float = 0.0,
) -> PolicyOutcome:
    """CAAC: pay for the cheapest promising step, re-decide on what it showed.

    After the greedy turn the policy holds a state (what it has paid for) and
    serves the first candidate whose POST-action estimate clears ``tau``:

        answer_now  -- the greedy answer, judged by the stage-1 estimate
        sample_more -- draw up to k_steps[0]; serve the majority if ``stage2``
                       clears tau, else draw up to the next step and ask again
        escalate    -- pay for the big model's answer; serve it if
                       ``escalate_gate`` clears tau given everything seen so far
                       (its confidence, and whether it agrees with the small
                       model's answers -- compared with each other, not gold)
        otherwise   -- abstain, having paid for everything it tried

    ``rho`` decides what is worth trying, and ``rho_mode`` decides on what
    scale, which is the difference between stopping on low confidence and
    stopping because nothing left can help:

      "phat"  -- the stage-1 estimate P(correct | a) reaches rho (the original
                 rule, kept as the ablation arm)
      "delta" -- the estimated NET gain over answering now reaches rho, so an
                 action whose expected damage cancels its repairs is not bought
      "value" -- the net gain minus ``lam`` times the extra cost reaches rho,
                 the economic form: a repair has to be worth its price

    The last two read ``recovery(small, big) -> {action: Delta_hat}``; with no
    estimator they score 0, so only a non-positive rho buys anything, which is
    the safe direction. When every correcting action fails the gate the policy
    abstains having paid for the greedy turn alone -- abstention BEFORE the
    expensive tier, not after it.

    Tried actions go cheapest first-step first under ``cost_fn``, which is
    where the price ratio enters; it never enters the guarantee. Without
    ``escalate_gate`` (or ``rho``) the older rule applies: escalate only when
    the stage-1 estimate already clears tau.
    """
    from caac.pool.ledger import cost_profiles

    if rho_mode not in RHO_MODES:
        raise ValueError(f"rho_mode must be one of {RHO_MODES}")
    price = cost_fn or (lambda calls: cost_profiles(calls)[cost_profile])
    steps = tuple(sorted(set(k_steps))) or (k,)
    phat = scorer(small, big)
    calls = [_greedy_call(small, 0)]
    tried: list[str] = []
    drawn: list[int] = []

    def done(out, gate, **extra):
        out.meta.update({"phat": phat, "tau": tau, "rho": rho, "rho_mode": rho_mode,
                         "tried": tried, "gate": gate, "n_drawn": len(drawn), **extra})
        return out

    if "answer_now" in actions and phat.get("answer_now", 0.0) >= tau:
        return done(PolicyOutcome(True, small.greedy.answer, small.greedy.correct, calls,
                                  "answer_now"), "stage1")

    candidates = [a for a in actions if a != "answer_now"]
    first_cost = {a: price(calls_for_action(small, big, a, steps[0], order))
                  for a in candidates}
    if rho is None:
        correcting = candidates
    elif rho_mode == "phat":
        correcting = [a for a in candidates if phat.get(a, 0.0) >= rho]
    else:
        delta = recovery(small, big) if recovery is not None else {}
        now_cost = price(calls_for_action(small, big, "answer_now", steps[0], order))
        def gate_score(a: str) -> float:
            d = delta.get(a, 0.0)
            return d if rho_mode == "delta" else d - lam * (first_cost[a] - now_cost)
        correcting = [a for a in candidates if gate_score(a) >= rho]
    rnd = 1
    for action in sorted(correcting, key=lambda a: (first_cost[a], a)):
        if action == "sample_more":
            avail = [i for i in order if i < small.n_samples]
            for kk in steps:
                new = avail[len(drawn):kk]
                if not new:
                    break
                calls.append(_samples_call(small, new, rnd))
                rnd += 1
                drawn += new
                if "sample_more" not in tried:
                    tried.append("sample_more")
                p2 = stage2(small, drawn, phat) if stage2 is not None else 0.0
                if p2 >= tau:
                    voted = serve_with_greedy(small, drawn)
                    return done(PolicyOutcome(True, voted, small.label_correct(voted), calls,
                                              "sample_more"), "stage2", p_stage2=p2)
        elif action == "escalate":
            if big is None:
                continue
            tried.append("escalate")
            if escalate_gate is None:
                if phat.get("escalate", 0.0) < tau:
                    continue
                p3 = phat["escalate"]
                calls.append(_greedy_call(big, rnd))
            else:
                calls.append(_greedy_call(big, rnd))
                p3 = escalate_gate(small, big, list(drawn), phat)
            rnd += 1
            if p3 >= tau:
                return done(PolicyOutcome(True, big.greedy.answer, big.greedy.correct, calls,
                                          "escalate"), "escalate_gate", p_escalate=p3)

    return done(PolicyOutcome(False, None, False, calls, "abstain"), "abstain")


def cascade_force_escalate(small: PoolRecord, big: PoolRecord) -> PolicyOutcome:
    bg = big.greedy
    return PolicyOutcome(True, bg.answer, bg.correct,
                         [_greedy_call(small, 0), _greedy_call(big, 1)], "escalate")
