"""The inference loop -- Algorithm 1 from the proposal.

The controller is policy-agnostic: it executes whatever action a Policy returns
and does the bookkeeping (budget, branches, cost attribution). That separation
lets VOC and every baseline run through identical machinery, so a Pareto
comparison reflects the decision rule rather than harness differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from caac.adapters.base import Adapter
from caac.backends.base import GenerationBackend
from caac.cost.accounting import CostMeter, CostSource
from caac.policy.base import Policy
from caac.signals.features import FeatureExtractor
from caac.types import (
    Action,
    CostVector,
    CostWeights,
    ReasoningState,
    RolloutResult,
    Trajectory,
)
from caac.verifier.base import Verifier

__all__ = ["ControllerConfig", "CAACController"]


@dataclass
class ControllerConfig:
    """Knobs that shape the control loop, not the decision rule."""

    segment_tokens: int = 64        # k: decision-point spacing
    max_steps: int = 32             # hard cap on decision points per query
    initial_budget: float = 4096.0  # B_0 in scalar cost units
    aggregate: str = "belief"       # 'belief' | 'first'


@dataclass
class CAACController:
    """Runs one query to completion under a given policy."""

    backend: GenerationBackend
    policy: Policy
    adapter: Adapter
    features: FeatureExtractor
    weights: CostWeights
    verifier: Verifier | None = None
    config: ControllerConfig = field(default_factory=ControllerConfig)

    def run(self, query_id: str, prompt: str):
        """Execute the control loop.

        Returns (final_state, cost_meter, decisions). The decision trace is
        returned rather than logged because the distribution of actions across
        belief bins is an experimental result (RQ3).
        """
        meter = CostMeter()
        self.policy.reset()
        self.features.reset()

        state = self._initialise(query_id, prompt, meter)
        decisions = []

        for _ in range(self.config.max_steps):
            if state.budget_remaining <= 0 or state.active_trajectory.finished:
                break

            self._refresh(state, meter)

            with meter.track(CostSource.CONTROLLER):
                decision = self.policy.decide(state)
            decisions.append(decision)

            if decision.action is Action.STOP:
                break

            self._execute(state, decision.action, meter)
            state.step += 1

        return state, meter, decisions

    # -- loop internals ----------------------------------------------------

    def _initialise(self, query_id, prompt, meter) -> ReasoningState:
        state = ReasoningState(
            query_id=query_id,
            prompt=prompt,
            budget_remaining=self.config.initial_budget,
            initial_budget=self.config.initial_budget,
        )
        first = self._generate(prompt, "", meter, CostSource.REASONING)
        state.trajectories.append(first)
        return state

    def _refresh(self, state: ReasoningState, meter: CostMeter) -> None:
        """Recompute features and belief.

        Attributed to CONTROLLER because it is part of the cost of deciding --
        even though the underlying statistics were produced free during decode.
        """
        with meter.track(CostSource.CONTROLLER):
            state.features, state.meta = self.features.extract(state)
            state.belief = self.adapter.belief(state)

    def _execute(self, state: ReasoningState, action: Action, meter: CostMeter) -> None:
        if action is Action.CONTINUE:
            seg = self._generate(
                state.prompt, state.active_trajectory.text, meter, CostSource.REASONING
            )
            state.active_trajectory.extend(seg)
            self._charge(state, seg.n_tokens)

        elif action is Action.VERIFY:
            if self.verifier is None:
                raise RuntimeError("VERIFY chosen but no verifier configured")
            with meter.track(CostSource.VERIFIER) as t:
                res = self.verifier.verify(state.prompt, state.active_trajectory.text)
                t.add(tokens=res.n_tokens, forward_passes=res.n_forward_passes)
            state.n_verify_used += 1
            state.meta["verifier_score"] = res.score
            self._charge(state, res.n_tokens)

        elif action is Action.BRANCH:
            prefix = self._rollback_prefix(state)
            branch = self._generate(state.prompt, prefix, meter, CostSource.BRANCH)
            branch.parent_step = state.step
            state.trajectories.append(branch)
            state.active = len(state.trajectories) - 1
            state.n_branches += 1
            self._charge(state, branch.n_tokens)

    def _generate(self, prompt, prefix, meter, source) -> Trajectory:
        with meter.track(source) as t:
            seg = self.backend.generate(
                prompt=prompt, prefix=prefix, max_tokens=self.config.segment_tokens
            )
            t.add(tokens=seg.n_tokens, forward_passes=seg.n_tokens)
        return seg

    def _rollback_prefix(self, state: ReasoningState) -> str:
        """Choose where a branch restarts from.

        Restarting at the highest-belief checkpoint (not the current token)
        makes BRANCH a recovery move rather than a resample.
        """
        checkpoints = state.meta.get("belief_checkpoints", [])
        if not checkpoints:
            return state.active_trajectory.text
        best = max(checkpoints, key=lambda c: c["belief"])
        return best["prefix"]

    def _charge(self, state: ReasoningState, tokens: float) -> None:
        cost = CostVector(tokens=tokens, forward_passes=tokens)
        state.budget_remaining -= cost.scalar(self.weights)

    # -- results -----------------------------------------------------------

    def finalise(
        self,
        state: ReasoningState,
        meter: CostMeter,
        decisions: list,
        reference_answer: str | None,
        is_correct: Callable[[str | None, str | None], bool],
    ) -> RolloutResult:
        answer = self._aggregate(state)
        correct = bool(is_correct(answer, reference_answer)) if answer else False

        counts = {a: 0 for a in Action}
        for d in decisions:
            counts[d.action] += 1

        return RolloutResult(
            query_id=state.query_id,
            answer=answer,
            correct=correct,
            cost=meter.total(),
            n_decisions=len(decisions),
            action_counts=counts,
            beliefs=[d.belief for d in decisions],
        )

    def _aggregate(self, state: ReasoningState) -> str | None:
        """Confidence-weighted selection across branches.

        Plain majority vote is avoided: with 2-4 branches it is near-arbitrary,
        whereas belief-weighted selection uses signal already paid for.
        """
        candidates = [t for t in state.trajectories if t.answer is not None]
        if not candidates:
            return None
        if self.config.aggregate == "first":
            return candidates[0].answer
        return state.active_trajectory.answer or candidates[0].answer
