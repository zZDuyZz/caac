"""The value-of-computation decision rule -- the scientific core.

Everything else feeds the quantity computed here:

    VOC(a | s) = E[V(s') | s, a]  -  U_stop(s)  -  lambda * c(a, s)

    STOP                       if max_{a != STOP} VOC(a | s) <= tau
    argmax_{a != STOP} VOC     otherwise

The first branch answers *when*; the second answers *where*. Both come from
one expression -- the property that distinguishes this from a tuned threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from caac.policy.base import Policy
from caac.policy.estimators import CostEstimator, GainEstimator
from caac.types import Action, CostWeights, Decision, ReasoningState

__all__ = ["VOCPolicy"]


@dataclass
class VOCPolicy(Policy):
    """Analytic VOC policy: no learned parameters at the decision layer.

    Keeping the decision layer parameter-free is deliberate: it makes the
    Analytic-vs-Learned comparison a clean ablation. Any gap is attributable
    to learning; any gain Analytic already captures is attributable to the
    formulation itself.

    Parameters
    ----------
    gain : estimates E[V(s')] - U_stop(s) for each spending action.
    cost : estimates c(a, s) as a multi-dimensional CostVector.
    weights : deployment profile used to collapse cost to a scalar.
    lam : exchange rate between accuracy and compute. Sweeping it traces the
        Pareto frontier; it is not tuned per query.
    stop_threshold : tau. Zero is the decision-theoretic default; positive is
        the risk-controlled variant.
    max_branches : hard cap on concurrent trajectories (bounds KV growth).
    max_verify : optional cap on verifier calls per query.
    lazy : if True, skip VOC computation for actions that are clearly not
        worth evaluating given the current belief (a scheduler optimisation
        that never changes the chosen action).
    """

    gain: GainEstimator
    cost: CostEstimator
    weights: CostWeights
    lam: float = 1.0
    stop_threshold: float = 0.0
    max_branches: int = 2
    max_verify: int | None = None
    lazy: bool = False

    name: str = "voc"

    def decide(self, state: ReasoningState) -> Decision:
        voc: dict[Action, float] = {}
        blocked: dict[Action, str] = {}

        for action in Action.spending():
            reason = self._infeasible(state, action)
            if reason is not None:
                voc[action] = float("-inf")
                blocked[action] = reason
                continue

            if self.lazy and self._skip_lazy(state, action):
                voc[action] = float("-inf")
                blocked[action] = "lazy-skipped"
                continue

            expected_gain = self.gain.predict(state, action)
            scalar_cost = self.cost.predict(state, action).scalar(self.weights)
            voc[action] = float(expected_gain - self.lam * scalar_cost)

        voc[Action.STOP] = 0.0  # STOP is the reference point.

        best_action = max(Action.spending(), key=lambda a: voc[a])
        best_value = voc[best_action]

        if best_value <= self.stop_threshold:
            note = "no action clears its cost"
            if all(a in blocked for a in Action.spending()):
                note = "; ".join(f"{a.value}:{r}" for a, r in blocked.items())
            return Decision(Action.STOP, voc, state.belief, note)

        return Decision(best_action, voc, state.belief, f"voc={best_value:.4f}")

    def _infeasible(self, state: ReasoningState, action: Action) -> str | None:
        """Return a reason if the action is not admissible, else None.

        Checked before VOC so an infeasible action can never win an argmax.
        """
        if action is Action.BRANCH and state.n_branches >= self.max_branches:
            return "branch cap"
        if (
            action is Action.VERIFY
            and self.max_verify is not None
            and state.n_verify_used >= self.max_verify
        ):
            return "verify cap"
        predicted = self.cost.predict(state, action).scalar(self.weights)
        if predicted > state.budget_remaining:
            return "insufficient budget"
        return None

    def _skip_lazy(self, state: ReasoningState, action: Action) -> bool:
        """Cheap gate: skip evaluating VERIFY/BRANCH when far from the margin.

        When the belief is very low, CONTINUE dominates and paying to evaluate
        the information-acquiring actions is wasted controller compute. This
        only ever prunes actions that would not have been chosen anyway.
        """
        if action is Action.CONTINUE:
            return False
        return state.belief < 0.15 or state.belief > 0.97

    def reset(self) -> None:
        self.gain.reset()
        self.cost.reset()
