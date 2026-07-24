"""Reproductions of published baselines (phase P1).

These are the methods CAAC must beat. Unlike the simplified decision rules in
``policy/baselines.py`` (which exist to test the control loop on CPU), these
follow the mechanism each paper actually describes, so their numbers can be
checked against the published ones before any comparison is claimed.

All of them run through the same backend and the same CostMeter as CAAC. That
is the point: a Pareto comparison is only meaningful if every method is charged
under one accounting rule. Methods that call a verifier pay for it here, just
as CAAC does.

Each baseline exposes the same method:

    run(backend, prompt, budget, meter) -> BaselineResult

so the evaluation harness can sweep them uniformly.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from caac.cost.accounting import CostMeter, CostSource
from caac.eval.answer_match import extract_answer, normalize_answer
from caac.types import CostWeights, Trajectory

__all__ = [
    "BaselineResult",
    "GreedyCoT",
    "SelfConsistency",
    "BudgetForcing",
    "DEER",
    "DeepConf",
    "BASELINE_REGISTRY",
    "build_baseline",
]


@dataclass
class BaselineResult:
    """Outcome of one baseline run on one query."""

    answer: str | None
    n_segments: int
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Helpers shared by the baselines
# --------------------------------------------------------------------------


def _generate(backend, prompt, prefix, k, meter, source=CostSource.REASONING, **kw) -> Trajectory:
    """Generate one segment and charge it to the meter."""
    with meter.track(source) as t:
        seg = backend.generate(prompt=prompt, prefix=prefix, max_tokens=k, **kw)
        t.add(tokens=seg.n_tokens, forward_passes=seg.n_tokens)
    return seg


def _mean_conf(logprobs: list[float], window: int | None = None) -> float:
    """Mean token probability, the confidence signal DeepConf/DEER rely on."""
    if not logprobs:
        return 0.5
    xs = logprobs[-window:] if window else logprobs
    return float(sum(math.exp(x) for x in xs) / len(xs))


def _run_to_completion(backend, prompt, meter, k, max_segments, **kw) -> Trajectory:
    """Roll out greedily until the model stops or the segment cap is hit."""
    traj = _generate(backend, prompt, "", k, meter, **kw)
    for _ in range(max_segments - 1):
        if traj.finished:
            break
        seg = _generate(backend, prompt, traj.text, k, meter, **kw)
        traj.extend(seg)
    return traj


# --------------------------------------------------------------------------
# Tier 1 -- non-adaptive
# --------------------------------------------------------------------------


@dataclass
class GreedyCoT:
    """Plain chain-of-thought, one trajectory, fixed cap.

    The lower anchor on cost. Sweeping ``max_segments`` traces its own curve.
    """

    segment_tokens: int = 64
    max_segments: int = 8
    name: str = "greedy_cot"

    def run(self, backend, prompt: str, budget: float, meter: CostMeter) -> BaselineResult:
        traj = _run_to_completion(
            backend, prompt, meter, self.segment_tokens, self.max_segments
        )
        return BaselineResult(answer=traj.answer, n_segments=self.max_segments)


@dataclass
class SelfConsistency:
    """Self-consistency: sample k trajectories, majority-vote the answer.

    The canonical parallel-scaling baseline. Cost grows linearly in k, which is
    exactly the inefficiency adaptive allocation is meant to remove.
    """

    k: int = 8
    segment_tokens: int = 64
    max_segments: int = 8
    temperature: float = 0.8
    name: str = "self_consistency"

    def run(self, backend, prompt: str, budget: float, meter: CostMeter) -> BaselineResult:
        answers = []
        for _ in range(self.k):
            traj = _run_to_completion(
                backend, prompt, meter, self.segment_tokens, self.max_segments,
                temperature=self.temperature,
            )
            if traj.answer is not None:
                answers.append(normalize_answer(traj.answer))
        if not answers:
            return BaselineResult(answer=None, n_segments=self.k * self.max_segments)
        answer, votes = Counter(answers).most_common(1)[0]
        return BaselineResult(
            answer=answer,
            n_segments=self.k * self.max_segments,
            meta={"votes": votes, "n_answers": len(answers)},
        )


# --------------------------------------------------------------------------
# Tier 3 -- sequential budget control
# --------------------------------------------------------------------------


@dataclass
class BudgetForcing:
    """Budget forcing (s1-style): force the model to think for a set budget.

    Two interventions from the paper: append a continuation token ("Wait") to
    push past an early stop, and truncate once the budget is spent. The budget
    is set externally, which is the limitation CAAC targets: it must be chosen
    before anything about the query's difficulty is known.
    """

    segment_tokens: int = 64
    target_segments: int = 8
    continuation: str = "\nWait"
    max_forces: int = 3
    name: str = "budget_forcing"

    def run(self, backend, prompt: str, budget: float, meter: CostMeter) -> BaselineResult:
        traj = _generate(backend, prompt, "", self.segment_tokens, meter)
        forces = 0
        for step in range(self.target_segments - 1):
            if traj.finished:
                # Force more thinking rather than accepting the early stop.
                if forces >= self.max_forces:
                    break
                traj.text += self.continuation
                traj.finished = False
                forces += 1
            seg = _generate(backend, prompt, traj.text, self.segment_tokens, meter)
            traj.extend(seg)
        return BaselineResult(
            answer=traj.answer,
            n_segments=self.target_segments,
            meta={"forces": forces},
        )


# --------------------------------------------------------------------------
# Tier 2 -- adaptive, threshold-based (the decisive comparison)
# --------------------------------------------------------------------------


@dataclass
class DEER:
    """Dynamic early exit in reasoning.

    At a transition point the model is prompted for a trial answer; if the
    confidence in that trial answer clears a threshold, reasoning stops early.
    Answers the *when* question with a fixed threshold and has no way to express
    *where*, which is precisely the gap CAAC addresses.

    ``threshold`` is the one knob; sweeping it produces DEER's cost/accuracy
    curve, and that curve is what CAAC's frontier is compared against.
    """

    threshold: float = 0.85
    segment_tokens: int = 64
    max_segments: int = 8
    min_segments: int = 1
    conf_window: int = 32
    name: str = "deer"

    def run(self, backend, prompt: str, budget: float, meter: CostMeter) -> BaselineResult:
        traj = _generate(backend, prompt, "", self.segment_tokens, meter)
        used = 1
        for step in range(self.max_segments - 1):
            if traj.finished:
                break
            if used >= self.min_segments and traj.answer is not None:
                # Confidence in the trial answer, measured on the recent window.
                conf = _mean_conf(traj.token_logprobs, self.conf_window)
                if conf >= self.threshold:
                    break  # early exit
            seg = _generate(backend, prompt, traj.text, self.segment_tokens, meter)
            traj.extend(seg)
            used += 1
        return BaselineResult(
            answer=traj.answer, n_segments=used, meta={"exited_early": used < self.max_segments}
        )


@dataclass
class DeepConf:
    """Confidence-filtered parallel thinking.

    Samples traces, terminates a trace early when its running confidence drops
    below a threshold, and aggregates the survivors with confidence weighting
    instead of a plain majority vote. Training-free, which is why it is the
    hardest baseline to beat: its overhead is essentially zero while CAAC must
    pay for its controller.
    """

    k: int = 8
    conf_threshold: float = 0.45
    segment_tokens: int = 64
    max_segments: int = 8
    temperature: float = 0.8
    conf_window: int = 32
    min_segments: int = 2   # warmup before pruning is allowed
    name: str = "deepconf"

    def run(self, backend, prompt: str, budget: float, meter: CostMeter) -> BaselineResult:
        scored: list[tuple[str, float]] = []
        pruned = 0

        for _ in range(self.k):
            traj = _generate(
                backend, prompt, "", self.segment_tokens, meter, temperature=self.temperature
            )
            alive = True
            for step in range(self.max_segments - 1):
                if traj.finished:
                    break
                conf = _mean_conf(traj.token_logprobs, self.conf_window)
                # Pruning only after a warmup: an early low reading reflects the
                # start of reasoning, not a bad trace.
                if step + 1 >= self.min_segments and conf < self.conf_threshold:
                    alive = False  # prune this trace early
                    pruned += 1
                    break
                seg = _generate(
                    backend, prompt, traj.text, self.segment_tokens, meter,
                    temperature=self.temperature,
                )
                traj.extend(seg)
            if alive and traj.answer is not None:
                scored.append(
                    (normalize_answer(traj.answer), _mean_conf(traj.token_logprobs))
                )

        if not scored:
            return BaselineResult(answer=None, n_segments=self.k, meta={"pruned": pruned})

        # Confidence-weighted vote.
        weights: dict[str, float] = {}
        for ans, conf in scored:
            weights[ans] = weights.get(ans, 0.0) + conf
        answer = max(weights, key=weights.get)
        return BaselineResult(
            answer=answer,
            n_segments=self.k,
            meta={"pruned": pruned, "survivors": len(scored)},
        )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

BASELINE_REGISTRY = {
    "greedy_cot": GreedyCoT,
    "self_consistency": SelfConsistency,
    "budget_forcing": BudgetForcing,
    "deer": DEER,
    "deepconf": DeepConf,
}


def build_baseline(name: str, **kwargs):
    """Factory used by the P1 reproduction script and the E2 sweep."""
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"unknown baseline '{name}'; choose {sorted(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](**kwargs)
